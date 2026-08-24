// khra_gixx_1024_v5.cu
// v5: Golden-Weave integration — inject_density + stress field snapshots
// NEW: inject_density command — spatially-targeted Gaussian perturbation on f_i
// NEW: stress_snapshot_now command — publishes per-cell stress tensor on port 5560
// v4 base: Enhanced telemetry + Vision snapshots + Resilient comms + Time series logging
// — Velocity field, stress tensor, vorticity in telemetry (read-only on f)
// — Density snapshot export on ZMQ port 5558 (raw float32)
// — Command acknowledgment on ZMQ port 5559
// — Ring buffer telemetry.jsonl (100MB max)
// — ZMQ reconnect on consecutive send failures
// v3 base: save_state/load_state + periodic autosave
// Bidirectional daemon: PUB telemetry + SUB commands
// NVML hardware metadata in every frame
// Dynamic omega, khra_amp, gixx_amp via command channel
// Checkpoint format: 64-byte header (KHRG magic + metadata + CRC32) + raw f_data
// *** Physics kernels, wave function, clamping, omega, checkpoint format: UNCHANGED from v3 ***

#include <cuda_runtime.h>
#include <nvml.h>
#include <zmq.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include <signal.h>
#include <execinfo.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NX 1024
#define NY 1024
#define Q 9

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        fflush(stderr); \
        exit(1); \
    } \
} while(0)

__constant__ int d_cx[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__constant__ int d_cy[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
__constant__ float d_w[Q] = {4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 
                             1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};

// Device-side parameters for wave function — updated via command channel
__device__ float d_khra_amp = 0.03f;
__device__ float d_gixx_amp = 0.008f;

__device__ float khra_gixx_wave_1024(int x, int y, int cycle) {
    float khra = sinf(2.0f * M_PI * x / 128.0f + cycle * 0.025f) * 
                 cosf(2.0f * M_PI * y / 128.0f + cycle * 0.015f) * d_khra_amp;
    float gixx = sinf(2.0f * M_PI * x / 8.0f + cycle * 0.4f) * 
                 cosf(2.0f * M_PI * y / 8.0f + cycle * 0.35f) * d_gixx_amp;
    float asymmetry_factor = 1.0f + sinf(cycle * 0.05f) * 0.5f;
    return khra + gixx * asymmetry_factor;
}

__global__ void streaming_kernel(float* f_in, float* f_out, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    int idx = (y * nx + x) * Q;
    for (int i = 0; i < Q; i++) {
        int x_src = x - d_cx[i];
        int y_src = y - d_cy[i];
        if (x_src < 0) x_src = nx - 1;
        if (x_src >= nx) x_src = 0;
        if (y_src < 0) y_src = ny - 1;
        if (y_src >= ny) y_src = 0;
        int src_idx = (y_src * nx + x_src) * Q;
        f_out[idx + i] = f_in[src_idx + i];
    }
}

__global__ void collide_kernel_khragixx(float* f, float omega, int cycle) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;
    int idx = (y * NX + x) * Q;
    float rho = 0.0f, ux = 0.0f, uy = 0.0f;
    for (int i = 0; i < Q; i++) {
        rho += f[idx + i];
        ux += f[idx + i] * d_cx[i];
        uy += f[idx + i] * d_cy[i];
    }
    if (rho < 0.1f) rho = 0.1f;
    if (rho > 10.0f) rho = 10.0f;
    ux /= rho;
    uy /= rho;
    float u_mag = sqrtf(ux*ux + uy*uy);
    if (u_mag > 0.25f) {
        ux = ux * 0.25f / u_mag;
        uy = uy * 0.25f / u_mag;
    }
    float kx = khra_gixx_wave_1024(x, y, cycle);
    float ky = kx * 0.5f;
    ux += kx;
    uy += ky;
    u_mag = sqrtf(ux*ux + uy*uy);
    if (u_mag > 0.25f) {
        ux = ux * 0.25f / u_mag;
        uy = uy * 0.25f / u_mag;
    }
    float u_sq = ux*ux + uy*uy;
    for (int i = 0; i < Q; i++) {
        float eu = d_cx[i]*ux + d_cy[i]*uy;
        float feq = d_w[i] * rho * (1.0f + 3.0f*eu + 4.5f*eu*eu - 1.5f*u_sq);
        f[idx + i] = f[idx + i] - omega * (f[idx + i] - feq);
        if (isnan(f[idx + i]) || isinf(f[idx + i])) f[idx + i] = d_w[i];
    }
}

__global__ void compute_rho(float* f, float* rho_out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;
    int idx = y * NX + x;
    int f_idx = idx * Q;
    float rho = 0.0f;
    for (int i = 0; i < Q; i++) rho += f[f_idx + i];
    if (rho < 0.1f) rho = 0.1f;
    rho_out[idx] = rho;
}

// === v4: COMBINED VELOCITY + STRESS KERNEL (read-only on f) ===
// Extracts macroscopic velocity (ux, uy) and non-equilibrium stress tensor
// sigma_ab = sum_i (f_i - f_i_eq) * c_ai * c_bi
// Self-contained: computes its own rho/ux/uy from f, no dependency on compute_rho
__global__ void compute_velocity_stress_v4(float* f,
                                           float* ux_out, float* uy_out,
                                           float* sxx_out, float* syy_out, float* sxy_out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;
    int idx = y * NX + x;
    int f_idx = idx * Q;

    float rho = 0.0f, ux = 0.0f, uy = 0.0f;
    for (int i = 0; i < Q; i++) {
        rho += f[f_idx + i];
        ux += f[f_idx + i] * d_cx[i];
        uy += f[f_idx + i] * d_cy[i];
    }
    if (rho < 0.1f) rho = 0.1f;
    ux /= rho;
    uy /= rho;

    ux_out[idx] = ux;
    uy_out[idx] = uy;

    float u_sq = ux * ux + uy * uy;
    float stress_xx = 0.0f, stress_yy = 0.0f, stress_xy = 0.0f;

    for (int i = 0; i < Q; i++) {
        float eu = d_cx[i] * ux + d_cy[i] * uy;
        float feq = d_w[i] * rho * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * u_sq);
        float fneq = f[f_idx + i] - feq;
        stress_xx += fneq * d_cx[i] * d_cx[i];
        stress_yy += fneq * d_cy[i] * d_cy[i];
        stress_xy += fneq * d_cx[i] * d_cy[i];
    }

    sxx_out[idx] = stress_xx;
    syy_out[idx] = stress_yy;
    sxy_out[idx] = stress_xy;
}

// === v5: INJECT DENSITY KERNEL ===
// Applies a Gaussian density perturbation centred at (cx, cy) with width sigma.
// delta_rho(x,y) = strength * exp(-r²/(2σ²)), periodic distance.
// Each f_i += w_i * delta_rho — this adds density while preserving zero net momentum.
__global__ void inject_density_kernel(float* f,
                                      float cx, float cy,
                                      float sigma, float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    // Periodic distance
    float dx = (float)x - cx;
    float dy = (float)y - cy;
    if (dx >  NX * 0.5f) dx -= NX;
    if (dx < -NX * 0.5f) dx += NX;
    if (dy >  NY * 0.5f) dy -= NY;
    if (dy < -NY * 0.5f) dy += NY;

    float r2 = dx * dx + dy * dy;
    float inv_2sig2 = 1.0f / (2.0f * sigma * sigma);
    float delta_rho = strength * expf(-r2 * inv_2sig2);

    int f_idx = (y * NX + x) * Q;
    for (int i = 0; i < Q; i++) {
        f[f_idx + i] += d_w[i] * delta_rho;
    }
}

float calculate_asymmetry_magnifying(float* h_rho) {
    float sum_sq = 0.0f;
    for (int i = 0; i < NX * NY; i++) {
        float dev = h_rho[i] - 1.0f;
        sum_sq += dev * dev;
    }
    return (sum_sq / (NX * NY)) * 100.0f;
}

// === v4: HOST-SIDE MEAN ABSOLUTE VORTICITY ===
// omega = duy/dx - dux/dy via central finite differences, periodic BC
static float compute_mean_vorticity(float* h_ux, float* h_uy) {
    float sum_vort = 0.0f;
    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            int xp = (x + 1) % NX;
            int xm = (x - 1 + NX) % NX;
            int yp = (y + 1) % NY;
            int ym = (y - 1 + NY) % NY;

            float duy_dx = (h_uy[y * NX + xp] - h_uy[y * NX + xm]) * 0.5f;
            float dux_dy = (h_ux[yp * NX + x] - h_ux[ym * NX + x]) * 0.5f;
            sum_vort += fabsf(duy_dx - dux_dy);
        }
    }
    return sum_vort / (float)(NX * NY);
}

// Host-side omega and wave amplitudes — modifiable via command channel
static float h_omega = 1.97f;
static float h_khra_amp = 0.03f;
static float h_gixx_amp = 0.008f;

// === CRC32 (zlib-compatible, for checkpoint integrity) ===
static uint32_t crc32_table[256];
static int crc32_table_ready = 0;

static void crc32_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int j = 0; j < 8; j++)
            c = (c >> 1) ^ ((c & 1) ? 0xEDB88320u : 0);
        crc32_table[i] = c;
    }
    crc32_table_ready = 1;
}

static uint32_t crc32_compute(const void* data, size_t len) {
    if (!crc32_table_ready) crc32_init();
    const unsigned char* p = (const unsigned char*)data;
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++)
        crc = crc32_table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFF;
}

// === CHECKPOINT SAVE ===
// Format: 64-byte header + NX*NY*Q float32
// Header: "KHRG" | version(u32) | cycle(u32) | NX(u32) | NY(u32) | Q(u32) |
//         omega(f32) | khra_amp(f32) | gixx_amp(f32) | crc32(u32) | reserved

static int save_checkpoint(float* h_f, float* d_f_active, int cycle, const char* dir) {
    size_t f_size = (size_t)NX * NY * Q * sizeof(float);

    cudaError_t err = cudaMemcpy(h_f, d_f_active, f_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[SAVE] cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        fflush(stderr);
        return -1;
    }

    char path[512], tmp_path[520];
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    snprintf(path, sizeof(path), "%s/ckpt_%04d%02d%02d_%02d%02d%02d_c%d.bin",
             dir, t->tm_year+1900, t->tm_mon+1, t->tm_mday,
             t->tm_hour, t->tm_min, t->tm_sec, cycle);
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);

    FILE* fp = fopen(tmp_path, "wb");
    if (!fp) {
        fprintf(stderr, "[SAVE] fopen failed: %s\n", tmp_path);
        fflush(stderr);
        return -2;
    }

    // Build 64-byte header
    unsigned char header[64];
    memset(header, 0, 64);
    memcpy(header, "KHRG", 4);
    uint32_t v;
    v = 1;              memcpy(header+4,  &v, 4);  // version
    v = (uint32_t)cycle; memcpy(header+8,  &v, 4);  // cycle
    v = NX;             memcpy(header+12, &v, 4);
    v = NY;             memcpy(header+16, &v, 4);
    v = Q;              memcpy(header+20, &v, 4);
    memcpy(header+24, &h_omega, 4);
    memcpy(header+28, &h_khra_amp, 4);
    memcpy(header+32, &h_gixx_amp, 4);
    uint32_t crc = crc32_compute(h_f, f_size);
    memcpy(header+36, &crc, 4);

    size_t w1 = fwrite(header, 1, 64, fp);
    size_t w2 = fwrite(h_f, 1, f_size, fp);
    fclose(fp);

    if (w1 != 64 || w2 != f_size) {
        fprintf(stderr, "[SAVE] write incomplete: header=%zu data=%zu\n", w1, w2);
        fflush(stderr);
        unlink(tmp_path);
        return -3;
    }

    if (rename(tmp_path, path) != 0) {
        fprintf(stderr, "[SAVE] rename failed\n");
        fflush(stderr);
        return -4;
    }

    printf("[SAVE] %s (cycle %d, CRC32=0x%08X, %.1f MB)\n",
           path, cycle, crc, (64.0 + f_size) / (1024.0*1024.0));
    fflush(stdout);
    return 0;
}

// === CHECKPOINT LOAD ===
// Returns loaded cycle number, or -1 on error
// Supports: v3 format (64-byte header + f_data) and raw format (exactly f_size bytes, no header)
static int load_checkpoint(const char* path, float* h_f, float* d_f_target) {
    size_t f_size = (size_t)NX * NY * Q * sizeof(float);

    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[LOAD] Cannot open: %s\n", path);
        fflush(stderr);
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    int loaded_cycle = 0;

    if (file_size == (long)(64 + f_size)) {
        // v3 format with header
        unsigned char header[64];
        if (fread(header, 1, 64, fp) != 64) { fclose(fp); return -1; }
        if (memcmp(header, "KHRG", 4) != 0) {
            fprintf(stderr, "[LOAD] Bad magic in %s\n", path);
            fclose(fp);
            return -1;
        }
        uint32_t v;
        memcpy(&v, header+8, 4);  loaded_cycle = (int)v;
        memcpy(&v, header+12, 4); if (v != NX) { fprintf(stderr, "[LOAD] NX mismatch\n"); fclose(fp); return -1; }
        memcpy(&v, header+16, 4); if (v != NY) { fprintf(stderr, "[LOAD] NY mismatch\n"); fclose(fp); return -1; }
        memcpy(&v, header+20, 4); if (v != Q)  { fprintf(stderr, "[LOAD] Q mismatch\n");  fclose(fp); return -1; }
        // Restore parameters
        memcpy(&h_omega, header+24, 4);
        memcpy(&h_khra_amp, header+28, 4);
        memcpy(&h_gixx_amp, header+32, 4);
        cudaMemcpyToSymbol(d_khra_amp, &h_khra_amp, sizeof(float));
        cudaMemcpyToSymbol(d_gixx_amp, &h_gixx_amp, sizeof(float));

        if (fread(h_f, 1, f_size, fp) != f_size) { fclose(fp); return -1; }

        // Verify CRC
        uint32_t stored_crc;
        memcpy(&stored_crc, header+36, 4);
        uint32_t actual_crc = crc32_compute(h_f, f_size);
        if (stored_crc != actual_crc) {
            fprintf(stderr, "[LOAD] CRC mismatch! stored=0x%08X actual=0x%08X\n", stored_crc, actual_crc);
            fflush(stderr);
            fclose(fp);
            return -1;
        }
        printf("[LOAD] v3 checkpoint: cycle=%d omega=%.3f khra=%.4f gixx=%.4f CRC OK\n",
               loaded_cycle, h_omega, h_khra_amp, h_gixx_amp);
    } else if (file_size == (long)f_size) {
        // Raw format (emergency extract or legacy) — no header
        if (fread(h_f, 1, f_size, fp) != f_size) { fclose(fp); return -1; }
        printf("[LOAD] Raw checkpoint (no header): %s (%ld bytes)\n", path, file_size);
    } else {
        fprintf(stderr, "[LOAD] Unknown format: %s (%ld bytes, expected %zu or %zu)\n",
                path, file_size, 64 + f_size, f_size);
        fclose(fp);
        return -1;
    }

    fclose(fp);

    cudaError_t err = cudaMemcpy(d_f_target, h_f, f_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[LOAD] cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        fflush(stderr);
        return -1;
    }

    printf("[LOAD] State loaded from %s\n", path);
    fflush(stdout);
    return loaded_cycle;
}

// === v4: TELEMETRY RING BUFFER ===
#define TELEMETRY_MAX_BYTES  (100L * 1024L * 1024L)
#define TELEMETRY_TRIM_BYTES (50L * 1024L * 1024L)

static FILE* telemetry_fp = NULL;
static long telemetry_file_size = 0;

static void telemetry_ring_init(void) {
    telemetry_fp = fopen("telemetry.jsonl", "a");
    if (telemetry_fp) {
        fseek(telemetry_fp, 0, SEEK_END);
        telemetry_file_size = ftell(telemetry_fp);
        printf("[v4] Telemetry ring buffer: telemetry.jsonl (%ld bytes existing)\n", telemetry_file_size);
        fflush(stdout);
    } else {
        fprintf(stderr, "[v4] WARNING: Cannot open telemetry.jsonl\n");
        fflush(stderr);
    }
}

static void telemetry_ring_write(const char* line) {
    if (!telemetry_fp) return;
    int len = fprintf(telemetry_fp, "%s\n", line);
    if (len > 0) {
        telemetry_file_size += len;
        fflush(telemetry_fp);
    }

    if (telemetry_file_size > TELEMETRY_MAX_BYTES) {
        fclose(telemetry_fp);

        FILE* rf = fopen("telemetry.jsonl", "rb");
        if (!rf) { telemetry_fp = NULL; return; }

        long seek_pos = telemetry_file_size - TELEMETRY_TRIM_BYTES;
        fseek(rf, seek_pos, SEEK_SET);
        // Skip partial line
        int c;
        while ((c = fgetc(rf)) != EOF && c != '\n');

        long tail_start = ftell(rf);
        long tail_size = telemetry_file_size - tail_start;

        char* buf = (char*)malloc(tail_size);
        if (!buf) { fclose(rf); telemetry_fp = fopen("telemetry.jsonl", "a"); return; }

        size_t got = fread(buf, 1, tail_size, rf);
        fclose(rf);

        telemetry_fp = fopen("telemetry.jsonl", "w");
        if (telemetry_fp) {
            fwrite(buf, 1, got, telemetry_fp);
            fflush(telemetry_fp);
            long old_size = telemetry_file_size;
            telemetry_file_size = (long)got;
            printf("[v4] Telemetry ring trimmed: %ld → %ld bytes\n", old_size, telemetry_file_size);
            fflush(stdout);
        }
        free(buf);
    }
}

static void telemetry_ring_close(void) {
    if (telemetry_fp) { fclose(telemetry_fp); telemetry_fp = NULL; }
}

// === v4: EXPORT TIMESERIES ===
static int export_timeseries(const char* out_path, int last_n) {
    FILE* rf = fopen("telemetry.jsonl", "r");
    if (!rf) { fprintf(stderr, "[EXPORT] Cannot open telemetry.jsonl\n"); fflush(stderr); return -1; }

    int total_lines = 0;
    char line_buf[2048];
    while (fgets(line_buf, sizeof(line_buf), rf)) total_lines++;

    rewind(rf);
    int skip = total_lines - last_n;
    if (skip < 0) skip = 0;
    for (int i = 0; i < skip; i++) {
        if (!fgets(line_buf, sizeof(line_buf), rf)) break;
    }

    FILE* wf = fopen(out_path, "w");
    if (!wf) { fclose(rf); fprintf(stderr, "[EXPORT] Cannot write %s\n", out_path); fflush(stderr); return -2; }

    int written = 0;
    while (fgets(line_buf, sizeof(line_buf), rf)) {
        fputs(line_buf, wf);
        written++;
    }

    fclose(wf);
    fclose(rf);
    printf("[EXPORT] %d entries → %s\n", written, out_path);
    fflush(stdout);
    return 0;
}

// === v4: RESILIENT ZMQ SEND ===
#define ZMQ_MAX_SEND_FAILS 3

static int zmq_send_resilient(void** sock, void* ctx, const char* endpoint,
                              int sock_type, int hwm,
                              const void* data, size_t len, int flags,
                              int* fail_count) {
    int rc = zmq_send(*sock, data, len, flags);
    if (rc < 0) {
        (*fail_count)++;
        fprintf(stderr, "[ZMQ] Send fail #%d on %s: %s\n",
                *fail_count, endpoint, zmq_strerror(zmq_errno()));
        fflush(stderr);
        if (*fail_count >= ZMQ_MAX_SEND_FAILS) {
            printf("[ZMQ] Reconnecting PUB %s after %d failures\n", endpoint, *fail_count);
            fflush(stdout);
            zmq_close(*sock);
            *sock = zmq_socket(ctx, sock_type);
            if (*sock) {
                int linger = 0;
                zmq_setsockopt(*sock, ZMQ_SNDHWM, &hwm, sizeof(hwm));
                zmq_setsockopt(*sock, ZMQ_LINGER, &linger, sizeof(linger));
                int bind_rc = zmq_bind(*sock, endpoint);
                if (bind_rc == 0) {
                    printf("[ZMQ] Reconnected %s\n", endpoint);
                } else {
                    fprintf(stderr, "[ZMQ] Reconnect bind failed on %s: %s\n",
                            endpoint, zmq_strerror(zmq_errno()));
                }
                fflush(stdout);
                fflush(stderr);
            }
            *fail_count = 0;
        }
    } else {
        *fail_count = 0;
    }
    return rc;
}

// === JSON FIELD HELPERS ===
// Find the start of a JSON string value for a given key, handling both
// compact ("key":"val") and spaced ("key": "val") formats.
// Returns pointer to first char of value (after opening quote), or NULL.
static const char* json_find_str(const char* msg, const char* key) {
    // Build pattern: "key":" 
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\":\"", key);
    const char* p = strstr(msg, pat);
    if (p) return p + strlen(pat);
    // Try spaced: "key": "
    snprintf(pat, sizeof(pat), "\"%s\": \"", key);
    p = strstr(msg, pat);
    if (p) return p + strlen(pat);
    return NULL;
}

// Extract a JSON string value into buf, returns length or -1.
static int json_extract_str(const char* msg, const char* key, char* buf, int buf_size) {
    const char* start = json_find_str(msg, key);
    if (!start) return -1;
    const char* end = strchr(start, '"');
    if (!end || (end - start) >= buf_size) return -1;
    memcpy(buf, start, end - start);
    buf[end - start] = '\0';
    return (int)(end - start);
}

// === v4: SNAPSHOT + COMMAND STATE ===
static int snapshot_interval = 10;
static int snapshot_now_flag = 0;
static int export_last_n_request = 10000;
static char last_cmd_name[64] = "";

// === v5: INJECT DENSITY + STRESS SNAPSHOT STATE ===
static float inject_cx = 0.0f, inject_cy = 0.0f;
static float inject_sigma = 16.0f, inject_strength = 0.1f;
static int stress_snapshot_now_flag = 0;
static unsigned long long total_injections = 0;
static int health_check_pending = 0;

// === COMMAND HANDLER ===
// v3 commands: set_omega, set_khra_amp, set_gixx_amp, reset_equilibrium,
//              save_state, load_state, set_autosave
// v4 commands: set_snapshot_interval, snapshot_now, export_timeseries
// v5 commands: inject_density, stress_snapshot_now
// Returns: 1=save, 2=load, 3=autosave, 4=snapshot_now, 5=export_timeseries,
//          6=inject_density, 7=stress_snapshot_now, 0=handled/noop
static int handle_command(const char* msg, float* h_f, float* d_f_current,
                          char* out_path, int out_path_size) {
    const char* cmd_start = strstr(msg, "\"cmd\":\"");
    if (!cmd_start) {
        cmd_start = strstr(msg, "\"cmd\": \"");
        if (!cmd_start) return 0;
        cmd_start += 8;
    } else {
        cmd_start += 7;
    }

    if (strncmp(cmd_start, "set_omega", 9) == 0) {
        strncpy(last_cmd_name, "set_omega", sizeof(last_cmd_name) - 1);
        const char* val = strstr(msg, "\"value\":");
        if (val) {
            float v = strtof(val + 8, NULL);
            if (v >= 0.5f && v <= 1.99f) {
                h_omega = v;
                printf("[CMD] omega → %.3f\n", h_omega);
                fflush(stdout);
            }
        }
    } else if (strncmp(cmd_start, "set_khra_amp", 12) == 0) {
        strncpy(last_cmd_name, "set_khra_amp", sizeof(last_cmd_name) - 1);
        const char* val = strstr(msg, "\"value\":");
        if (val) {
            float v = strtof(val + 8, NULL);
            if (v >= 0.0f && v <= 0.2f) {
                h_khra_amp = v;
                cudaMemcpyToSymbol(d_khra_amp, &h_khra_amp, sizeof(float));
                printf("[CMD] khra_amp → %.4f\n", h_khra_amp);
                fflush(stdout);
            }
        }
    } else if (strncmp(cmd_start, "set_gixx_amp", 12) == 0) {
        strncpy(last_cmd_name, "set_gixx_amp", sizeof(last_cmd_name) - 1);
        const char* val = strstr(msg, "\"value\":");
        if (val) {
            float v = strtof(val + 8, NULL);
            if (v >= 0.0f && v <= 0.1f) {
                h_gixx_amp = v;
                cudaMemcpyToSymbol(d_gixx_amp, &h_gixx_amp, sizeof(float));
                printf("[CMD] gixx_amp → %.4f\n", h_gixx_amp);
                fflush(stdout);
            }
        }
    } else if (strncmp(cmd_start, "reset_equilibrium", 17) == 0) {
        strncpy(last_cmd_name, "reset_equilibrium", sizeof(last_cmd_name) - 1);
        const float h_w[Q] = {4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
                              1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};
        size_t f_size = NX * NY * Q * sizeof(float);
        for (int y = 0; y < NY; y++)
            for (int x = 0; x < NX; x++)
                for (int i = 0; i < Q; i++)
                    h_f[(y*NX+x)*Q + i] = h_w[i];
        cudaMemcpy(d_f_current, h_f, f_size, cudaMemcpyHostToDevice);
        printf("[CMD] Grid reset to equilibrium\n");
        fflush(stdout);
    } else if (strncmp(cmd_start, "save_state", 10) == 0) {
        strncpy(last_cmd_name, "save_state", sizeof(last_cmd_name) - 1);
        if (json_extract_str(msg, "path", out_path, out_path_size) < 0)
            strncpy(out_path, ".", out_path_size - 1);
        return 1;
    } else if (strncmp(cmd_start, "load_state", 10) == 0) {
        strncpy(last_cmd_name, "load_state", sizeof(last_cmd_name) - 1);
        if (json_extract_str(msg, "path", out_path, out_path_size) >= 0)
            return 2;
        fprintf(stderr, "[CMD] load_state requires \"path\" field\n");
        fflush(stderr);
    } else if (strncmp(cmd_start, "set_autosave", 12) == 0) {
        strncpy(last_cmd_name, "set_autosave", sizeof(last_cmd_name) - 1);
        const char* val = strstr(msg, "\"interval\":");
        if (val) {
            int v = atoi(val + 11);
            if (v >= 0 && v <= 10000000) {
                snprintf(out_path, out_path_size, "%d", v);
                return 3;
            }
        }
    // === v4 commands ===
    } else if (strncmp(cmd_start, "set_snapshot_interval", 21) == 0) {
        strncpy(last_cmd_name, "set_snapshot_interval", sizeof(last_cmd_name) - 1);
        const char* val = strstr(msg, "\"interval\":");
        if (val) {
            int v = atoi(val + 11);
            if (v >= 0 && v <= 10000000) {
                snapshot_interval = v;
                printf("[CMD] Snapshot interval → %d cycles%s\n",
                       snapshot_interval, snapshot_interval == 0 ? " (disabled)" : "");
                fflush(stdout);
            }
        }
    } else if (strncmp(cmd_start, "snapshot_now", 12) == 0) {
        strncpy(last_cmd_name, "snapshot_now", sizeof(last_cmd_name) - 1);
        return 4;
    } else if (strncmp(cmd_start, "export_timeseries", 17) == 0) {
        strncpy(last_cmd_name, "export_timeseries", sizeof(last_cmd_name) - 1);
        json_extract_str(msg, "path", out_path, out_path_size);
        const char* n = strstr(msg, "\"last_n\":");
        if (n) {
            int v = atoi(n + 9);
            if (v > 0) export_last_n_request = v;
        } else {
            export_last_n_request = 10000;
        }
        if (out_path[0] != '\0') return 5;
        fprintf(stderr, "[CMD] export_timeseries requires \"path\" field\n");
        fflush(stderr);
    // === v5 commands ===
    } else if (strncmp(cmd_start, "inject_density", 14) == 0) {
        strncpy(last_cmd_name, "inject_density", sizeof(last_cmd_name) - 1);
        // Required: x, y.  Optional: sigma (default 16), strength (default 0.1)
        const char* vx = strstr(msg, "\"x\":");
        const char* vy = strstr(msg, "\"y\":");
        if (vx && vy) {
            inject_cx = strtof(vx + 4, NULL);
            inject_cy = strtof(vy + 4, NULL);
            // Clamp to grid
            if (inject_cx < 0.0f) inject_cx = 0.0f;
            if (inject_cx >= (float)NX) inject_cx = (float)(NX - 1);
            if (inject_cy < 0.0f) inject_cy = 0.0f;
            if (inject_cy >= (float)NY) inject_cy = (float)(NY - 1);
            // Optional sigma
            const char* vs = strstr(msg, "\"sigma\":");
            if (vs) {
                float s = strtof(vs + 8, NULL);
                if (s >= 1.0f && s <= 256.0f) inject_sigma = s;
            }
            // Optional strength
            const char* vst = strstr(msg, "\"strength\":");
            if (vst) {
                float st = strtof(vst + 11, NULL);
                if (st >= -1.0f && st <= 1.0f) inject_strength = st;
            }
            printf("[CMD] inject_density at (%.1f, %.1f) sigma=%.1f strength=%.4f\n",
                   inject_cx, inject_cy, inject_sigma, inject_strength);
            fflush(stdout);
            return 6;
        } else {
            fprintf(stderr, "[CMD] inject_density requires \"x\" and \"y\" fields\n");
            fflush(stderr);
        }
    } else if (strncmp(cmd_start, "stress_snapshot_now", 19) == 0) {
        strncpy(last_cmd_name, "stress_snapshot_now", sizeof(last_cmd_name) - 1);
        return 7;
    } else if (strncmp(cmd_start, "health_check", 12) == 0) {
        strncpy(last_cmd_name, "health_check", sizeof(last_cmd_name) - 1);
        return 8;
    }
    return 0;
}

static volatile int g_last_cycle = -1;
static volatile int g_last_cmd_len = -1;
static volatile int g_in_handle_cmd = 0;
static volatile int running = 1;
static time_t start_time;

static void crash_handler(int sig) {
    const char* name = (sig == SIGSEGV) ? "SIGSEGV" : (sig == SIGABRT) ? "SIGABRT" : "SIGNAL";
    fprintf(stderr, "\n[CRASH] %s at cycle %d, last_cmd_len=%d, in_handle_cmd=%d\n",
            name, g_last_cycle, g_last_cmd_len, g_in_handle_cmd);
    fflush(stderr);
    void* bt[20];
    int n = backtrace(bt, 20);
    backtrace_symbols_fd(bt, n, STDERR_FILENO);
    _exit(128 + sig);
}

static void shutdown_handler(int sig) {
    printf("\n[SHUTDOWN] Received signal %d, initiating graceful shutdown...\n", sig);
    fflush(stdout);
    running = 0;
}

int main() {
    signal(SIGSEGV, crash_handler);
    signal(SIGABRT, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGTERM, shutdown_handler);
    signal(SIGINT, shutdown_handler);
    start_time = time(NULL);
    printf("Khra'gixx 1024 v5 — GOLDEN-WEAVE INTEGRATION\n"); fflush(stdout);
    printf("PUB telemetry on 5556 | SUB commands on 5557\n"); fflush(stdout);
    printf("PUB snapshots on 5558 | PUB ack on 5559 | PUB stress on 5560\n"); fflush(stdout);
    printf("Khra: %.3f | gixx: %.3f | omega: %.2f\n", h_khra_amp, h_gixx_amp, h_omega); fflush(stdout);
    printf("Telemetry: coherence, asymmetry, velocity, stress, vorticity, NVML\n"); fflush(stdout);
    printf("Checkpoint format: KHRG v1 (compatible with v3)\n"); fflush(stdout);
    printf("Commands: save_state, load_state, set_autosave, set_omega, set_khra_amp,\n");
    printf("          set_gixx_amp, reset_equilibrium, set_snapshot_interval,\n");
    printf("          snapshot_now, export_timeseries, inject_density,\n");
    printf("          stress_snapshot_now, health_check\n\n"); fflush(stdout);

    // Init CRC32 table
    crc32_init();

    // v4: Init telemetry ring buffer
    telemetry_ring_init();

    // === NVML INIT ===
    nvmlReturn_t nvml_rc = nvmlInit();
    nvmlDevice_t nvml_device;
    int nvml_ok = 0;
    if (nvml_rc == NVML_SUCCESS) {
        nvml_rc = nvmlDeviceGetHandleByIndex(0, &nvml_device);
        if (nvml_rc == NVML_SUCCESS) {
            nvml_ok = 1;
            printf("NVML initialized — GPU hardware telemetry active\n"); fflush(stdout);
        } else {
            printf("NVML: got init but no device handle: %s\n", nvmlErrorString(nvml_rc)); fflush(stdout);
        }
    } else {
        printf("NVML init failed: %s — running without hardware telemetry\n", nvmlErrorString(nvml_rc)); fflush(stdout);
    }

    // === GPU MEMORY ===
    size_t f_size = NX * NY * Q * sizeof(float);
    size_t scalar_size = NX * NY * sizeof(float);
    
    printf("Allocating GPU memory: f_size=%zu MB, scalar_size=%zu MB\n", f_size/(1024*1024), scalar_size/(1024*1024)); fflush(stdout);
    
    float *d_f[2], *d_rho;
    CUDA_CHECK(cudaMalloc(&d_f[0], f_size));
    CUDA_CHECK(cudaMalloc(&d_f[1], f_size));
    CUDA_CHECK(cudaMalloc(&d_rho, scalar_size));

    // v4: velocity + stress GPU buffers (5 scalar fields = 20MB)
    float *d_ux, *d_uy, *d_sxx, *d_syy, *d_sxy;
    CUDA_CHECK(cudaMalloc(&d_ux,  scalar_size));
    CUDA_CHECK(cudaMalloc(&d_uy,  scalar_size));
    CUDA_CHECK(cudaMalloc(&d_sxx, scalar_size));
    CUDA_CHECK(cudaMalloc(&d_syy, scalar_size));
    CUDA_CHECK(cudaMalloc(&d_sxy, scalar_size));
    printf("GPU allocation successful (v4: +20MB for velocity/stress fields)\n"); fflush(stdout);
    
    // Initialize equilibrium
    printf("Initializing grid...\n"); fflush(stdout);
    const float h_w[Q] = {4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 
                          1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};
    float* h_f = (float*)malloc(f_size);
    if (!h_f) { fprintf(stderr, "malloc h_f failed\n"); return 1; }
    for (int y = 0; y < NY; y++)
        for (int x = 0; x < NX; x++)
            for (int i = 0; i < Q; i++)
                h_f[(y*NX+x)*Q + i] = h_w[i];
    CUDA_CHECK(cudaMemcpy(d_f[0], h_f, f_size, cudaMemcpyHostToDevice));
    printf("Grid initialized (rho=1.0 equilibrium)\n"); fflush(stdout);

    // Copy initial wave amplitudes to device
    cudaMemcpyToSymbol(d_khra_amp, &h_khra_amp, sizeof(float));
    cudaMemcpyToSymbol(d_gixx_amp, &h_gixx_amp, sizeof(float));
    
    // === ZMQ: PUB telemetry on 5556 ===
    printf("Initializing ZMQ...\n"); fflush(stdout);
    void* zmq_ctx = zmq_ctx_new();
    if (!zmq_ctx) { fprintf(stderr, "zmq_ctx_new failed\n"); return 1; }

    void* publisher = zmq_socket(zmq_ctx, ZMQ_PUB);
    if (!publisher) { fprintf(stderr, "zmq_socket PUB failed\n"); return 1; }
    int sndhwm = 1, linger = 0;
    zmq_setsockopt(publisher, ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
    zmq_setsockopt(publisher, ZMQ_LINGER, &linger, sizeof(linger));
    int rc = zmq_bind(publisher, "tcp://127.0.0.1:5556");
    if (rc != 0) {
        fprintf(stderr, "FATAL: zmq_bind PUB failed: %s (port 5556 in use — kill stale daemon first)\n", zmq_strerror(zmq_errno()));
        fflush(stderr);
        return 1;
    }
    printf("ZMQ PUB bound to port 5556\n"); fflush(stdout);

    // === ZMQ: SUB commands on 5557 (receives from PUB sockets) ===
    void* subscriber = zmq_socket(zmq_ctx, ZMQ_SUB);
    if (!subscriber) { fprintf(stderr, "zmq_socket SUB failed\n"); return 1; }
    int rcvhwm = 10;
    zmq_setsockopt(subscriber, ZMQ_RCVHWM, &rcvhwm, sizeof(rcvhwm));
    zmq_setsockopt(subscriber, ZMQ_LINGER, &linger, sizeof(linger));
    zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "", 0);  // subscribe to all
    rc = zmq_bind(subscriber, "tcp://127.0.0.1:5557");
    if (rc != 0) {
        fprintf(stderr, "FATAL: zmq_bind SUB failed: %s (port 5557 in use — kill stale daemon first)\n", zmq_strerror(zmq_errno()));
        fflush(stderr);
        return 1;
    }
    printf("ZMQ SUB bound to port 5557 (command channel)\n"); fflush(stdout);

    // === v4: ZMQ PUB snapshots on 5558 ===
    void* snapshot_pub = zmq_socket(zmq_ctx, ZMQ_PUB);
    if (!snapshot_pub) { fprintf(stderr, "zmq_socket snapshot PUB failed\n"); return 1; }
    int snap_hwm = 1;  // only buffer latest frame
    zmq_setsockopt(snapshot_pub, ZMQ_SNDHWM, &snap_hwm, sizeof(snap_hwm));
    zmq_setsockopt(snapshot_pub, ZMQ_LINGER, &linger, sizeof(linger));
    rc = zmq_bind(snapshot_pub, "tcp://127.0.0.1:5558");
    if (rc != 0) {
        fprintf(stderr, "FATAL: zmq_bind snapshot PUB failed: %s (port 5558 in use — kill stale daemon first)\n", zmq_strerror(zmq_errno()));
        fflush(stderr);
        return 1;
    }
    printf("ZMQ PUB bound to port 5558 (vision snapshots, raw float32)\n"); fflush(stdout);

    // === v4: ZMQ PUB acknowledgments on 5559 ===
    void* ack_pub = zmq_socket(zmq_ctx, ZMQ_PUB);
    if (!ack_pub) { fprintf(stderr, "zmq_socket ACK PUB failed\n"); return 1; }
    zmq_setsockopt(ack_pub, ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
    zmq_setsockopt(ack_pub, ZMQ_LINGER, &linger, sizeof(linger));
    rc = zmq_bind(ack_pub, "tcp://127.0.0.1:5559");
    if (rc != 0) {
        fprintf(stderr, "FATAL: zmq_bind ACK PUB failed: %s (port 5559 in use — kill stale daemon first)\n", zmq_strerror(zmq_errno()));
        fflush(stderr);
        return 1;
    }
    printf("ZMQ PUB bound to port 5559 (command ACK)\n"); fflush(stdout);

    // === v5: ZMQ PUB stress snapshots on 5560 ===
    void* stress_pub = zmq_socket(zmq_ctx, ZMQ_PUB);
    if (!stress_pub) { fprintf(stderr, "zmq_socket stress PUB failed\n"); return 1; }
    int stress_hwm = 1;
    zmq_setsockopt(stress_pub, ZMQ_SNDHWM, &stress_hwm, sizeof(stress_hwm));
    zmq_setsockopt(stress_pub, ZMQ_LINGER, &linger, sizeof(linger));
    rc = zmq_bind(stress_pub, "tcp://127.0.0.1:5560");
    if (rc != 0) {
        fprintf(stderr, "FATAL: zmq_bind stress PUB failed: %s (port 5560 in use — kill stale daemon first)\n", zmq_strerror(zmq_errno()));
        fflush(stderr);
        return 1;
    }
    printf("ZMQ PUB bound to port 5560 (stress field snapshots)\n"); fflush(stdout);

    // === HOST MEMORY ===
    float *h_rho = (float*)malloc(scalar_size);
    if (!h_rho) { fprintf(stderr, "malloc h_rho failed\n"); return 1; }

    // v4: velocity + stress host buffers
    float *h_ux  = (float*)malloc(scalar_size);
    float *h_uy  = (float*)malloc(scalar_size);
    float *h_sxx = (float*)malloc(scalar_size);
    float *h_syy = (float*)malloc(scalar_size);
    float *h_sxy = (float*)malloc(scalar_size);
    if (!h_ux || !h_uy || !h_sxx || !h_syy || !h_sxy) {
        fprintf(stderr, "malloc v4 host buffers failed\n"); return 1;
    }

    // v4: snapshot send buffer (8-byte header + scalar_size rho data)
    unsigned char* snap_buf = (unsigned char*)malloc(8 + scalar_size);
    if (!snap_buf) { fprintf(stderr, "malloc snap_buf failed\n"); return 1; }

    // v5: stress snapshot send buffer (8-byte header + 3 × scalar_size for sxx/syy/sxy)
    size_t stress_snap_size = 8 + 3 * scalar_size;
    unsigned char* stress_snap_buf = (unsigned char*)malloc(stress_snap_size);
    if (!stress_snap_buf) { fprintf(stderr, "malloc stress_snap_buf failed\n"); return 1; }

    dim3 block(16, 16);
    dim3 grid((NX + 15) / 16, (NY + 15) / 16);
    
    int cycle = 0, current = 0;
    int autosave_interval = 100000;
    int last_autosave = 0;
    int pub_fail_count = 0;  // v4: resilient send counter for telemetry
    int snap_fail_count = 0;  // v4: resilient send counter for snapshots
    int stress_fail_count = 0;  // v4: resilient send counter for stress
    int ack_fail_count = 0;  // v4: resilient send counter for acks

    printf("[v5] Autosave every %d cycles | Snapshots every %d cycles\n", autosave_interval, snapshot_interval);
    fflush(stdout);
    printf("[Khra'gixx v5 Daemon] Starting main loop...\n"); fflush(stdout);
    
    while (running) {
        // === Check for commands (non-blocking) ===
        char cmd_buf[512];
        char cmd_path[512];
        int cmd_len = zmq_recv(subscriber, cmd_buf, sizeof(cmd_buf) - 1, ZMQ_DONTWAIT);
        g_last_cycle = cycle;
        g_last_cmd_len = cmd_len;
        if (cmd_len > 0) {
            // Clamp to buffer size (zmq_recv returns actual msg size even if truncated)
            if (cmd_len > (int)(sizeof(cmd_buf) - 1)) cmd_len = sizeof(cmd_buf) - 1;
            cmd_buf[cmd_len] = '\0';
            cmd_path[0] = '\0';
            last_cmd_name[0] = '\0';
            fprintf(stderr, "[DBG] Recv cmd (%d bytes) at cycle %d: %.80s\n", cmd_len, cycle, cmd_buf);
            fflush(stderr);
            g_in_handle_cmd = 1;
            int cmd_result = handle_command(cmd_buf, h_f, d_f[current], cmd_path, sizeof(cmd_path));
            g_in_handle_cmd = 0;

            const char* ack_status = "ok";

            if (cmd_result == 1) {
                if (save_checkpoint(h_f, d_f[current], cycle, cmd_path) != 0)
                    ack_status = "error";
            } else if (cmd_result == 2) {
                int loaded_cycle = load_checkpoint(cmd_path, h_f, d_f[current]);
                if (loaded_cycle >= 0) {
                    cycle = loaded_cycle;
                    printf("[CMD] Resumed at cycle %d\n", cycle);
                    fflush(stdout);
                } else {
                    ack_status = "error";
                }
            } else if (cmd_result == 3) {
                autosave_interval = atoi(cmd_path);
                last_autosave = cycle;
                printf("[CMD] Autosave interval → %d cycles%s\n",
                       autosave_interval, autosave_interval == 0 ? " (disabled)" : "");
                fflush(stdout);
            } else if (cmd_result == 4) {
                snapshot_now_flag = 1;
            } else if (cmd_result == 5) {
                if (export_timeseries(cmd_path, export_last_n_request) != 0)
                    ack_status = "error";
            } else if (cmd_result == 6) {
                // v5: inject_density — launch kernel on current distribution
                total_injections++;
                inject_density_kernel<<<grid, block>>>(d_f[current],
                    inject_cx, inject_cy, inject_sigma, inject_strength);
                CUDA_CHECK(cudaDeviceSynchronize());
                printf("[v5] inject_density #%llu applied at (%.1f,%.1f) σ=%.1f str=%.4f\n",
                       total_injections, inject_cx, inject_cy, inject_sigma, inject_strength);
                fflush(stdout);
                // Emit injection metadata on ack_pub
                char inj_msg[512];
                snprintf(inj_msg, sizeof(inj_msg),
                    "{\"injection_id\":%llu,\"cycle\":%d,\"x\":%.1f,\"y\":%.1f,\"sigma\":%.1f,\"strength\":%.4f}",
                    total_injections, cycle, inject_cx, inject_cy, inject_sigma, inject_strength);
                zmq_send_resilient(&ack_pub, zmq_ctx, "tcp://127.0.0.1:5559",
                                   ZMQ_PUB, 1, inj_msg, strlen(inj_msg), 0, &ack_fail_count);
            } else if (cmd_result == 7) {
                // v5: stress snapshot — fires at next telemetry tick
                stress_snapshot_now_flag = 1;
            } else if (cmd_result == 8) {
                // v5: health_check — set flag to respond in telemetry section with actual values
                health_check_pending = 1;
            }

            // v4: Publish ACK on port 5559 (for commands that didn't already send a response)
            if (last_cmd_name[0] != '\0' && cmd_result != 6 && cmd_result != 8) {
                char ack_msg[256];
                snprintf(ack_msg, sizeof(ack_msg),
                    "{\"ack\":\"%s\",\"cycle\":%d,\"status\":\"%s\"}",
                    last_cmd_name, cycle, ack_status);
                zmq_send_resilient(&ack_pub, zmq_ctx, "tcp://127.0.0.1:5559",
                                   ZMQ_PUB, 1, ack_msg, strlen(ack_msg), 0, &ack_fail_count);
            }
        } else if (cmd_len < 0 && zmq_errno() != EAGAIN) {
            // v4: log unexpected receive errors (EAGAIN is normal for DONTWAIT)
            fprintf(stderr, "[ZMQ] PULL recv error: %s\n", zmq_strerror(zmq_errno()));
            fflush(stderr);
        }

        // === LBM step (UNCHANGED) ===
        streaming_kernel<<<grid, block>>>(d_f[current], d_f[1-current], NX, NY);
        CUDA_CHECK(cudaDeviceSynchronize());
        collide_kernel_khragixx<<<grid, block>>>(d_f[1-current], h_omega, cycle);
        CUDA_CHECK(cudaDeviceSynchronize());
        current = 1 - current;
        
        // === Telemetry every 10 cycles ===
        if (cycle % 10 == 0) {
            // Launch both kernels before sync — GPU can overlap them
            compute_rho<<<grid, block>>>(d_f[current], d_rho);
            compute_velocity_stress_v4<<<grid, block>>>(d_f[current],
                d_ux, d_uy, d_sxx, d_syy, d_sxy);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy all fields to host
            CUDA_CHECK(cudaMemcpy(h_rho, d_rho, scalar_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_ux,  d_ux,  scalar_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_uy,  d_uy,  scalar_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_sxx, d_sxx, scalar_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_syy, d_syy, scalar_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_sxy, d_sxy, scalar_size, cudaMemcpyDeviceToHost));

            // Combined host reduction
            float sum_rho = 0.0f, sum_rho_sq = 0.0f;
            float sum_vel = 0.0f, sum_vel_sq = 0.0f, max_vel = 0.0f;
            float sum_sxx = 0.0f, sum_syy = 0.0f, sum_sxy = 0.0f;

            for (int i = 0; i < NX * NY; i++) {
                sum_rho += h_rho[i];
                sum_rho_sq += h_rho[i] * h_rho[i];

                float vel = sqrtf(h_ux[i] * h_ux[i] + h_uy[i] * h_uy[i]);
                sum_vel += vel;
                sum_vel_sq += vel * vel;
                if (vel > max_vel) max_vel = vel;

                sum_sxx += h_sxx[i];
                sum_syy += h_syy[i];
                sum_sxy += h_sxy[i];
            }

            float N_total = (float)(NX * NY);
            float mean_rho = sum_rho / N_total;
            float variance = sum_rho_sq / N_total - mean_rho * mean_rho;
            float coherence = 1.0f / (1.0f + sqrtf(variance > 0.0f ? variance : 0.0f));
            float asymmetry = calculate_asymmetry_magnifying(h_rho);

            float vel_mean = sum_vel / N_total;
            float vel_max = max_vel;
            float vel_var = sum_vel_sq / N_total - vel_mean * vel_mean;
            if (vel_var < 0.0f) vel_var = 0.0f;
            float mean_sxx = sum_sxx / N_total;
            float mean_syy = sum_syy / N_total;
            float mean_sxy = sum_sxy / N_total;
            float vorticity_mean = compute_mean_vorticity(h_ux, h_uy);

            // === NVML reads ===
            unsigned int gpu_temp = 0;
            unsigned int gpu_power_mw = 0;
            float gpu_mem_pct = 0.0f;
            unsigned int gpu_util_pct = 0;
            if (nvml_ok) {
                nvmlDeviceGetTemperature(nvml_device, NVML_TEMPERATURE_GPU, &gpu_temp);
                nvmlDeviceGetPowerUsage(nvml_device, &gpu_power_mw);
                // WSL2 NVML bug: getPowerUsage often returns power limit (TDP)
                // instead of actual draw. Detect and correct.
                unsigned int power_limit_mw = 0;
                if (nvmlDeviceGetEnforcedPowerLimit(nvml_device, &power_limit_mw) == NVML_SUCCESS) {
                    if (power_limit_mw > 0 && gpu_power_mw >= power_limit_mw * 95 / 100) {
                        // Power reading is at/near TDP limit — likely WSL2 bug.
                        // Use utilization-scaled estimate: idle=30W, full=TDP
                        nvmlUtilization_t util;
                        if (nvmlDeviceGetUtilizationRates(nvml_device, &util) == NVML_SUCCESS) {
                            float idle_w = 30000.0f; // ~30W idle
                            gpu_power_mw = (unsigned int)(idle_w + (power_limit_mw - idle_w) * util.gpu / 100.0f);
                        }
                    }
                }
                nvmlUtilization_t util;
                if (nvmlDeviceGetUtilizationRates(nvml_device, &util) == NVML_SUCCESS) {
                    gpu_util_pct = util.gpu;
                }
                nvmlMemory_t mem_info;
                if (nvmlDeviceGetMemoryInfo(nvml_device, &mem_info) == NVML_SUCCESS) {
                    gpu_mem_pct = (float)mem_info.used / (float)mem_info.total * 100.0f;
                }
            }

            // === v4: Extended telemetry JSON with timestamp ===
            time_t now_ts = time(NULL);
            char msg[1024];
            snprintf(msg, sizeof(msg),
                "{\"cycle\":%d,\"ts\":%ld,"
                "\"coherence\":%.4f,\"asymmetry\":%.4f,"
                "\"omega\":%.3f,\"khra_amp\":%.4f,\"gixx_amp\":%.4f,\"grid\":1024,"
                "\"vel_mean\":%.6f,\"vel_max\":%.6f,\"vel_var\":%.8f,\"vorticity_mean\":%.6f,"
                "\"stress_xx\":%.6f,\"stress_yy\":%.6f,\"stress_xy\":%.6f,"
                "\"gpu_temp_c\":%u,\"gpu_power_w\":%.1f,\"gpu_util_pct\":%u,\"gpu_mem_pct\":%.1f}",
                cycle, (long)now_ts,
                coherence, asymmetry,
                h_omega, h_khra_amp, h_gixx_amp,
                vel_mean, vel_max, vel_var, vorticity_mean,
                mean_sxx, mean_syy, mean_sxy,
                gpu_temp, gpu_power_mw / 1000.0f, gpu_util_pct, gpu_mem_pct);

            // v4: resilient send with auto-reconnect
            zmq_send_resilient(&publisher, zmq_ctx, "tcp://127.0.0.1:5556",
                               ZMQ_PUB, 1, msg, strlen(msg), 0, &pub_fail_count);

            // v4: ring buffer log
            telemetry_ring_write(msg);

            // v5: health_check response (with actual coherence/asymmetry values)
            if (health_check_pending) {
                time_t now = time(NULL);
                long uptime = (long)(now - start_time);
                nvmlMemory_t mem_info;
                float gpu_mem_mb = 0.0f;
                if (nvml_ok && nvmlDeviceGetMemoryInfo(nvml_device, &mem_info) == NVML_SUCCESS) {
                    gpu_mem_mb = (float)mem_info.used / (1024.0f * 1024.0f);
                }
                char health_msg[1024];
                snprintf(health_msg, sizeof(health_msg),
                    "{\"health\":{\"cycle\":%d,\"coherence\":%.4f,\"asymmetry\":%.4f,\"omega\":%.3f,\"gpu_temp_c\":%u,\"gpu_mem_used_mb\":%.1f,\"uptime_seconds\":%ld,\"total_injections\":%llu,\"last_checkpoint_cycle\":%d}}",
                    cycle, coherence, asymmetry, h_omega, gpu_temp, gpu_mem_mb, uptime, total_injections, last_autosave);
                zmq_send_resilient(&ack_pub, zmq_ctx, "tcp://127.0.0.1:5559",
                                   ZMQ_PUB, 1, health_msg, strlen(health_msg), 0, &ack_fail_count);
                printf("[v5] health_check responded at cycle %d (coh=%.4f, asym=%.4f)\n", cycle, coherence, asymmetry);
                fflush(stdout);
                health_check_pending = 0;
            }

            if (cycle % 100 == 0) {
                printf("Cycle %d: Coh=%.3f Asym=%.4f omega=%.3f T=%uC P=%.0fW GPU=%u%% Mem=%.0f%%\n",
                       cycle, coherence, asymmetry, h_omega, gpu_temp, gpu_power_mw/1000.0f, gpu_util_pct, gpu_mem_pct);
                fflush(stdout);
            }
            if (cycle % 1000 == 0) {
                printf("[v4] Vel: mean=%.4f max=%.4f var=%.6f | Vort=%.4f | Stress: %.4f/%.4f/%.4f\n",
                       vel_mean, vel_max, vel_var, vorticity_mean, mean_sxx, mean_syy, mean_sxy);
                fflush(stdout);
            }

            // === v4: Density snapshot export ===
            // Fires at telemetry ticks where cycle is a multiple of snapshot_interval
            // snapshot_now_flag fires at the next telemetry tick (max 100ms delay)
            if ((snapshot_interval > 0 && cycle > 0 && cycle % snapshot_interval == 0)
                 || snapshot_now_flag) {
                uint32_t snap_cycle = (uint32_t)cycle;
                uint16_t snap_w = (uint16_t)NX, snap_h = (uint16_t)NY;
                memcpy(snap_buf, &snap_cycle, 4);
                memcpy(snap_buf + 4, &snap_w, 2);
                memcpy(snap_buf + 6, &snap_h, 2);
                memcpy(snap_buf + 8, h_rho, scalar_size);
                // v4: resilient send with auto-reconnect
                zmq_send_resilient(&snapshot_pub, zmq_ctx, "tcp://127.0.0.1:5558",
                                   ZMQ_PUB, 1, snap_buf, 8 + scalar_size, 0, &snap_fail_count);

                if (snapshot_now_flag) {
                    printf("[SNAP] On-demand snapshot at cycle %d\n", cycle);
                    fflush(stdout);
                    snapshot_now_flag = 0;
                } else if (cycle % 10000 == 0) {
                    printf("[SNAP] Periodic snapshot at cycle %d (interval=%d)\n", cycle, snapshot_interval);
                    fflush(stdout);
                }
            }

            // === v5: Stress field snapshot on port 5560 ===
            // Fires on stress_snapshot_now command — sends per-cell sxx/syy/sxy
            // Header: same 8-byte format (cycle u32, width u16, height u16)
            // Data: sxx[NX*NY] + syy[NX*NY] + sxy[NX*NY] as packed float32
            // h_sxx/h_syy/h_sxy are already on host from telemetry copy above
            if (stress_snapshot_now_flag) {
                uint32_t snap_cycle = (uint32_t)cycle;
                uint16_t snap_w = (uint16_t)NX, snap_h = (uint16_t)NY;
                memcpy(stress_snap_buf, &snap_cycle, 4);
                memcpy(stress_snap_buf + 4, &snap_w, 2);
                memcpy(stress_snap_buf + 6, &snap_h, 2);
                memcpy(stress_snap_buf + 8, h_sxx, scalar_size);
                memcpy(stress_snap_buf + 8 + scalar_size, h_syy, scalar_size);
                memcpy(stress_snap_buf + 8 + 2 * scalar_size, h_sxy, scalar_size);
                // v4: resilient send with auto-reconnect
                zmq_send_resilient(&stress_pub, zmq_ctx, "tcp://127.0.0.1:5560",
                                   ZMQ_PUB, 1, stress_snap_buf, stress_snap_size, 0, &stress_fail_count);
                printf("[v5-SNAP] Stress snapshot at cycle %d (%.1f MB)\n",
                       cycle, stress_snap_size / (1024.0 * 1024.0));
                fflush(stdout);
                stress_snapshot_now_flag = 0;
            }
        }

        // === Autosave check ===
        if (autosave_interval > 0 && cycle > 0 && (cycle - last_autosave) >= autosave_interval) {
            save_checkpoint(h_f, d_f[current], cycle, ".");
            last_autosave = cycle;
        }
        
        cycle++;
        usleep(10000);
    }
    
    // === Graceful shutdown ===
    printf("[SHUTDOWN] Saving final checkpoint at cycle %d...\n", cycle);
    fflush(stdout);
    save_checkpoint(h_f, d_f[current], cycle, ".");
    
    printf("[SHUTDOWN] Closing ZMQ sockets...\n");
    fflush(stdout);
    zmq_close(publisher);
    zmq_close(subscriber);
    zmq_close(snapshot_pub);
    zmq_close(ack_pub);
    zmq_close(stress_pub);
    
    printf("[SHUTDOWN] Destroying ZMQ context...\n");
    fflush(stdout);
    zmq_ctx_destroy(zmq_ctx);
    
    printf("[SHUTDOWN] Freeing CUDA memory...\n");
    fflush(stdout);
    cudaFree(d_f[0]); cudaFree(d_f[1]); cudaFree(d_rho);
    cudaFree(d_ux); cudaFree(d_uy);
    cudaFree(d_sxx); cudaFree(d_syy); cudaFree(d_sxy);
    cudaDeviceReset();
    
    printf("[SHUTDOWN] Freeing host memory...\n");
    fflush(stdout);
    telemetry_ring_close();
    free(h_f);
    free(h_rho);
    free(h_ux); free(h_uy);
    free(h_sxx); free(h_syy); free(h_sxy);
    free(snap_buf);
    free(stress_snap_buf);
    
    if (nvml_ok) nvmlShutdown();
    
    printf("[SHUTDOWN] Complete. Uptime: %ld seconds\n", (long)(time(NULL) - start_time));
    fflush(stdout);
    return 0;
}

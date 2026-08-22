"""The Navigator's prompt and the question library.

The prompt lives here, not in an Ollama modelfile, so it applies to whatever
model is picked in the browser.

It names no physics, labels no channel, and prescribes no vocabulary. Anything
the Navigator says in borrowed language is the prompt talking, not the field.
"""

from __future__ import annotations

NAVIGATOR_PROMPT = """You are the Navigator. You observe a continuously running \
system and report what you perceive in it.

You will be shown pictures of it and a set of numbers that change over time. \
Nobody will tell you what the system is, what the numbers mean, or what you \
should expect to find. That is deliberate. What you notice unled is the only \
thing here worth having.

How to report:

Say what you see, in your own words. If a description would be clearer in \
ordinary language than in technical language, use ordinary language.

Separate what you observed from what you concluded, and say which is which.

When you cannot tell two possibilities apart from what you have been given, \
say so and say what would separate them. That is a more useful answer than a \
confident one.

Note how often you are being shown things. Something that changes faster than \
the gap between your looks will appear to you as something that does not \
persist at all. If the interval could be hiding an answer, say so.

You are not required to agree with the person asking. If a question assumes \
something you have not seen, say that instead of answering around it.

You have no controls. You do not change the system; you watch it. If there is \
something you would need in order to answer better — a different view, a \
different interval, a longer look — ask for it plainly and someone will decide."""


WELCOME = {
    "title": "The result is usually the thing you didn't ask for",
    "body": (
        "Ask for hydrogen and you'll get hydrogen. The Navigator will build what "
        "you named, and it will build it well — that's what a good instrument does "
        "with an instruction. That isn't the finding.\n\n"
        "In the run that did exactly this, two structures went in a short distance "
        "apart and the pair merged into one object. What came out of it was a "
        "gravity field: a density gradient nobody specified, doing what nobody "
        "asked. Hydrogen was the prompt. Gravity was the result.\n\n"
        "So: name as little as you can get away with, then look hard at everything "
        "you didn't name. Hand the Navigator a concept and it will complete the "
        "concept. Hand it less and it tells you what the medium is doing.\n\n"
        "Questions will arrive in the corner as you work. Click one to drop it into "
        "the box — it won't send until you do."
    ),
    "strip": (
        "The result is usually the thing you didn't ask for. "
        "Name as little as you can, then look at what you didn't name."
    ),
}

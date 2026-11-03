"""The completion-check prompt (T-0207 group A1, docs/tasks/T-0207.md A1.1-A1.4).

When a live turn that executed browser actions ends with an answer and no call, the session (A2) holds that answer
and asks once more with a fresh browser frame and a fixed instruction. ``check_prompt`` builds that request by
composing ``announced_call.reask_prompt``: the step's exact prompt, the model's words appended as its message, then a
user turn holding the fresh frame and the instruction, in that order. The whole step prompt stays a byte prefix of the
rendered check (A1.3), so the prefix cache holds.

T-0227 R-3: the request never carries more than ``max_images`` images (the engine's limit, HTTP 400 above it). The
fresh frame is always kept; when the step's images plus the fresh one would exceed the limit, the oldest earlier
images (first in message order) are replaced in place by ``NOT_SHOWN_TEXT`` until the total fits. Only then does the
step prompt stop being a byte prefix of the check; a check within the limit is unchanged.

The instruction is never stored in history and never published. Pure: standard library, ``local_astra.contracts`` and
``toolcall.announced_call`` only.
"""

from __future__ import annotations

import base64

from local_astra.contracts.llm import ImagePart, Message, Part, Prompt, TextPart
from local_astra.contracts.perception import Frame
from local_astra.toolcall import announced_call

#: T-0227 R-3: what an earlier image dropped to keep the check within the image limit becomes, at its position.
NOT_SHOWN_TEXT = "(an earlier screen, not shown)"

#: T-0227 R-3: the engine's image limit per request (``--limit-mm-per-prompt`` in ``services/vllm.sh``, mirrored by
#: ``prompt_builder.Budget.max_images``; this module imports neither). The session passes its builder's value.
DEFAULT_MAX_IMAGES = 4

#: H-T207-3: the instruction after a step that still had actions left.
CHECK_INSTRUCTION = (
    "This is the browser screen now. Before anything reaches the user, check what you just wrote against this "
    "screen. If the task is not done, continue with a tool call. If it is done, answer the user again in one or two "
    "sentences, saying only what this screen shows."
)

#: H-T207-3 / H-T207-4: the instruction at the report step (no actions left).
CHECK_INSTRUCTION_REPORT = (
    "This is the browser screen now. You have no actions left. Check what you just wrote against this screen, then "
    "tell the user in one or two sentences what this screen shows is done and where you stopped. Do not call a tool."
)


def check_prompt(
    sent: Prompt, said: str, frame: Frame, *, report: bool, max_images: int = DEFAULT_MAX_IMAGES
) -> Prompt:
    """``reask_prompt(sent, said)`` with its final user message replaced by ``(ImagePart(frame), TextPart(<instr>))``.

    The image is encoded as the prompt builder encodes a current frame (``data:image/jpeg;base64,...``,
    ``phash=frame.phash``). The instruction is ``CHECK_INSTRUCTION_REPORT`` when ``report`` else
    ``CHECK_INSTRUCTION``. Every other field is ``sent``'s; ``continue_final_message`` is False.

    T-0227 R-3: at most ``max_images`` images in all, the fresh one always among them: while the total is over, the
    oldest earlier ``ImagePart`` is replaced in place by ``TextPart(NOT_SHOWN_TEXT)``. Within the limit nothing else
    changes.
    """
    reasked = announced_call.reask_prompt(sent, said)
    image = ImagePart(
        url="data:image/jpeg;base64," + base64.b64encode(frame.jpeg).decode(), phash=frame.phash
    )
    text = CHECK_INSTRUCTION_REPORT if report else CHECK_INSTRUCTION
    last = Message(role="user", content=(image, TextPart(text=text)))
    earlier = reasked.messages[:-1]
    held = sum(1 for m in earlier for p in m.content if isinstance(p, ImagePart))
    excess = held + 1 - max(1, max_images)
    if excess > 0:
        replaced: list[Message] = []
        for message in earlier:
            parts: list[Part] = []
            for part in message.content:
                if isinstance(part, ImagePart) and excess > 0:
                    parts.append(TextPart(text=NOT_SHOWN_TEXT))
                    excess -= 1
                else:
                    parts.append(part)
            replaced.append(message.model_copy(update={"content": tuple(parts)}))
        earlier = tuple(replaced)
    return reasked.model_copy(update={"messages": (*earlier, last)})

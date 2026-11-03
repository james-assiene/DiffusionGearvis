"""Excerpt of diffgemma-fa ``diffgemma_fa/eval/run.py`` at 27c7d0d: ``PROMPT_STYLES`` and
``build_prompt`` only, verbatim (the rest of that module imports JAX). See PROVENANCE.md."""

from __future__ import annotations


#: Experiment E2. The stock prompt says nothing about rendering, and the model
#: fills the vacuum with fenced, pretty-printed multi-line JSON — 126/130 of the
#: unconstrained outputs. The compiled grammar admits none of that (0/130
#: verbatim acceptance), so every constrained decode is forced off the model's
#: plan at each value boundary. `compact` tells the model to render the way the
#: grammar reads, which is the zero-code test of that whole hypothesis; the
#: "every key" clause additionally targets the shortest-member collapse, since
#: the unconstrained arm already emits every key in 125/126 outputs.
PROMPT_STYLES = {
    "stock": "",
    #: Drops the key-order hint entirely. The stock prompt ends with "keys in
    #: this order: [...]", and 98/98 unconstrained outputs obeyed it -- so the
    #: grammar's key-order over-constraint currently costs nothing. But that
    #: measurement cannot tell whether the model would reorder WITHOUT the
    #: hint, and anyone reusing the grammar without it inherits the risk. This
    #: style removes the hint so the question can be answered.
    "noorder": "",
    "compact": (
        " Output compact single-line JSON with no newlines and no code "
        "fences, and include every listed key."
    ),
}


def build_prompt(rec, fn, style: str = "stock") -> str:
    question = (rec.question[0][0]["content"] if rec.question
                else "Call the function.")
    props = list((fn.get("parameters") or {}).get("properties", {}))
    if style == "noorder":
        return (f"{question}\n\nRespond with a JSON object of arguments for "
                f"`{fn.get('name')}` using these keys: {props}.")
    return (f"{question}\n\nRespond with a JSON object of arguments for "
            f"`{fn.get('name')}`, keys in this order: {props}."
            + PROMPT_STYLES[style])

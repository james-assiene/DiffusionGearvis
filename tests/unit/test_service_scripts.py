"""``services/tts.sh``: Pocket TTS starts through ``uvx`` at the pinned version (T-0041 crit. 1).

The bug this module pins: the first real ``astra up --profile lean`` on this box (2026-09-15,
after a reboot) printed a FAILED row for ``tts`` because the script ``exec``s
``taskset -c 2,3 pocket-tts serve …`` and ``pocket-tts`` has never been on PATH here — every
hand start used ``uvx pocket-tts …`` (PROGRESS *Runs*, 2026-09-14). So the program ``taskset``
is asked to execute must be ``uvx``, not ``pocket-tts``, and the version it resolves must be the
one ``services/versions.lock`` pins rather than "whatever PyPI serves today".

Static assertions over the script plus ``bash -n``; nothing here starts a server or reaches the
network (a real ``uvx`` run would download a wheel). Two groups:

- unmarked rows — what the edit must **not** break (``set -euo pipefail``, ``taskset -c`` with
  ``ASTRA_TTS_CORES``, ``--host 127.0.0.1``, one ``exec``). They pass against today's script and
  must still pass after it;
- ``pending(task="T-0041")`` rows — the uvx invocation and the version pin, red today.

Regime: static; no box, no GPU, no network. The lock value is read at test time, so the row that
compares the two moves with ``services/versions.lock`` instead of hard-coding ``3.1.0``.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import tomllib
import warnings

from tests.unit import _box_rig as rig

SCRIPT = rig.REPO / "services" / "tts.sh"
LOCK = rig.REPO / "services" / "versions.lock"

#: ``x.y.z``-shaped, which is what a pin has to be to be a pin: ``pocket-tts==3`` or a bare
#: ``pocket-tts`` both resolve to a different wheel next week.
VERSION = re.compile(r"^\d+\.\d+\.\d+")


def source() -> str:
    return SCRIPT.read_text()


def code_lines() -> list[str]:
    """The script without comments and blank lines — a claim in a comment is not an invocation."""
    out = []
    for line in source().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            out.append(line)
    return out


def raw_exec_line() -> str:
    """The single ``exec …`` line that becomes the service process, verbatim.

    Every ``services/*.sh`` ends in a foreground server reached through ``exec`` (T-0033 P0-1:
    the pid ``up`` records must be the server's). If that ever becomes two lines this helper
    fails loudly rather than silently checking the wrong one.
    """
    execs = [line for line in code_lines() if line.strip().startswith("exec ")]
    assert len(execs) == 1, (
        f"services/tts.sh starts the server with exactly one `exec` line (the pid `astra up` "
        f"records is the server's, T-0033 P0-1); found {execs}"
    )
    return execs[0].strip()


#: ``NAME=value`` / ``export NAME=value`` at the start of a code line.
ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
#: ``$NAME`` or ``${NAME}``.
REFERENCE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")
#: ``${NAME:-default}`` — the value on a box where nothing is exported, which is this one.
DEFAULTED = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*:-(.*)\}$")


def assignments() -> dict[str, str]:
    """The script's plain ``NAME=value`` variables, with ``${NAME:-default}`` read as *default*.

    Not a shell: it exists so the rows below read the invocation as it runs with nothing
    exported, whether the script writes the version inline or through ``POCKET_TTS_VERSION=``.
    Anything it cannot resolve (a command substitution, say) it leaves alone, and the row that
    needed it then fails with the unexpanded text in its message.
    """
    out: dict[str, str] = {}
    for line in code_lines():
        found = ASSIGNMENT.match(line)
        if not found:
            continue
        name, raw = found.group(1), found.group(2).strip()
        if raw[:1] in ("'", '"') and raw[-1:] == raw[:1]:
            raw = raw[1:-1]
        defaulted = DEFAULTED.match(raw)
        out[name] = defaulted.group(1) if defaulted else raw
    return out


def exec_line() -> str:
    """The exec line with the script's own variables substituted (see :func:`assignments`)."""
    line = raw_exec_line()
    known = assignments()
    for _ in range(4):  # a few levels of indirection, then give up rather than loop
        expanded = REFERENCE.sub(lambda hit: known.get(hit.group(1), hit.group(0)), line)
        if expanded == line:
            break
        line = expanded
    return line


def exec_words() -> list[str]:
    """The expanded exec line as argv words."""
    return shlex.split(exec_line(), comments=True)


def locked_pocket_tts() -> str:
    """``[speech] pocket_tts`` from ``services/versions.lock``, or ``"TBD"`` while unpinned.

    Criterion 4 is a *human* edit (the lock is hook-denied to agents), so the value may still be
    ``TBD`` when these rows first run; the version row below says so out loud instead of
    silently checking nothing.
    """
    with LOCK.open("rb") as handle:
        data = tomllib.load(handle)
    value = data.get("speech", {}).get("pocket_tts")
    assert isinstance(value, str) and value, (
        "services/versions.lock [speech] must carry a pocket_tts line (T-0041 criterion 4); "
        f"it has {value!r}"
    )
    return value


# =================================================================================================
# Unmarked: true of today's script, and the edit must keep them true.
# =================================================================================================


def test_tts_script_parses_under_bash_n():
    """``bash -n``: a syntax error in the start script is a FAILED row with no log to read."""
    done = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, (
        f"criterion 1: `bash -n services/tts.sh` must be clean; it said:\n{done.stderr}"
    )


def test_tts_script_stays_loopback_only_on_the_pinned_cores():
    """The three properties the uvx rewrite must carry over: loopback, cores, strict mode."""
    line = exec_line()
    assert "--host 127.0.0.1" in line, (
        f"docs/02 §9: Pocket TTS binds loopback only; the exec line is:\n{line}"
    )
    assert "taskset -c" in line, (
        f"docs/01 §9: Pocket TTS runs on pinned cores so it cannot fight the engine; got:\n{line}"
    )
    assert "ASTRA_TTS_CORES" in source(), (
        "the pinned cores are overridable through ASTRA_TTS_CORES (services/tts.sh today)"
    )
    assert "set -euo pipefail" in source(), (
        "services/tts.sh runs under `set -euo pipefail` like every other service script"
    )


def test_the_exec_line_still_names_the_service_port():
    """8081 is the TTS endpoint every profile in ``config.DEFAULTS`` probes."""
    assert "8081" in exec_line(), (
        f"the tts endpoint is 127.0.0.1:8081 (local_astra/config.py); the exec line is:\n"
        f"{exec_line()}"
    )


# =================================================================================================
# T-0041 criterion 1: through uvx, at the locked version, without pocket-tts on PATH.
# =================================================================================================


def test_pocket_tts_is_started_through_uvx():
    """Criterion 1: the invocation goes through ``uvx``, so no ``pocket-tts`` on PATH is needed."""
    line = exec_line()
    words = exec_words()
    assert any(word == "uvx" or word.endswith("/uvx") for word in words), (
        "criterion 1: Pocket TTS has only ever started here through `uvx` (PROGRESS Runs "
        "2026-09-14); the script must use it rather than a `pocket-tts` on PATH. The exec line "
        f"is:\n{line}"
    )
    assert "pocket-tts" in line, (
        f"criterion 1: …and it is Pocket TTS that uvx runs; the exec line is:\n{line}"
    )


def test_taskset_execs_uvx_and_never_a_bare_pocket_tts():
    """Adversarial, the exact reboot failure: ``taskset: failed to execute pocket-tts``.

    ``taskset -c "$CORES" <program>`` — whatever ``<program>`` is has to exist on PATH, because
    ``taskset`` execs it directly. A rewrite that leaves ``pocket-tts`` in that position (say by
    adding ``uvx`` somewhere else on the line, or in a comment) reproduces the FAILED row
    verbatim, so this row reads the *position*, not the presence of a word.
    """
    words = exec_words()
    assert words and words[0] == "exec", f"the exec line starts with `exec`; it is {words}"
    assert "taskset" in words, f"criterion 1: taskset still pins the cores; the words are {words}"
    rest = words[words.index("taskset") + 1 :]
    # Skip taskset's own options: `-c` takes the core list, other flags do not.
    while rest and rest[0].startswith("-"):
        rest = rest[2:] if rest[0] == "-c" else rest[1:]
    assert rest, f"criterion 1: taskset is given a program to run; the words are {words}"
    program = rest[0]
    assert program != "pocket-tts" and not program.endswith("/pocket-tts"), (
        "criterion 1: this is the 2026-09-15 reboot failure exactly — "
        "`taskset: failed to execute pocket-tts: No such file or directory`, because taskset "
        f"execs {program!r} directly and pocket-tts is not on PATH on this box. It must exec "
        f"uvx. The exec line is:\n{exec_line()}"
    )
    assert program == "uvx" or program.endswith("/uvx"), (
        f"criterion 1: …and the program taskset execs is uvx; it is {program!r} in:\n{exec_line()}"
    )


def test_the_pocket_tts_version_is_pinned_and_matches_the_lock():
    """Criterion 1: ``pocket-tts==<x.y.z>`` on the exec line, and the x.y.z is the lock's.

    An unpinned ``uvx pocket-tts`` resolves to whatever PyPI serves on the day, which is the
    class of drift ``services/versions.lock`` exists to stop. While the lock still says ``TBD``
    (criterion 4 is a **human** edit — the lock is hook-denied to agents) the equality cannot be
    checked, so this row warns and checks the pin's shape alone; the lock line has its own row
    below.
    """
    line = exec_line()
    pins = re.findall(r"pocket-tts==([^\s\"']+)", line)
    assert pins, (
        "criterion 1: the script pins the Pocket TTS version (`pocket-tts==<x.y.z>`); a bare "
        f"`uvx pocket-tts` resolves to today's PyPI release. The exec line is:\n{line}"
    )
    assert len(set(pins)) == 1, f"criterion 1: one version, not {sorted(set(pins))}, in:\n{line}"
    pinned = pins[0]
    assert VERSION.match(pinned), (
        f"criterion 1: the pin is a full x.y.z version; the script says pocket-tts=={pinned!r}"
    )
    locked = locked_pocket_tts()
    if locked == "TBD":
        warnings.warn(
            f'services/versions.lock still says pocket_tts = "TBD", so nothing checks that the '
            f"{pinned!r} in services/tts.sh is the version this box measured (T-0041 criterion 4 "
            "records 3.1.0 and is a human edit). See test_versions_lock_pins_pocket_tts.",
            stacklevel=1,
        )
        return
    assert pinned == locked, (
        f"criterion 1: the script must run the version the lock pins — versions.lock says "
        f"pocket_tts = {locked!r} and services/tts.sh runs {pinned!r}. Two numbers that can "
        f"drift are not a pin. The exec line is:\n{line}"
    )


def test_versions_lock_pins_pocket_tts():
    """Criterion 4: ``[speech] pocket_tts`` is a version, not ``TBD``.

    The human's edit, recorded here so the task cannot be called done without it: ``TBD`` in the
    lock means the pin in ``services/tts.sh`` is a number nobody has checked against the box
    (T-0041 records 3.1.0, measured 2026-09-15 via ``importlib.metadata``). Agents cannot write
    this file — ``.claude/hooks/protect_paths.sh`` denies it — which is why it is its own row.
    """
    locked = locked_pocket_tts()
    assert locked != "TBD", (
        "criterion 4: services/versions.lock must pin pocket_tts (T-0041 measured 3.1.0 on this "
        "box on 2026-09-15). The lock is hook-denied to agents: this line is a human edit."
    )
    assert VERSION.match(locked), (
        f"criterion 4: the pin is a full x.y.z version; the lock says pocket_tts = {locked!r}"
    )

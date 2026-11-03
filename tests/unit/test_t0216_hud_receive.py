"""T-0216 criterion 4 (and 3's ``wire.in`` half): ``HudConnection.receive`` stays total on a huge
int literal and on deep nesting.

``receive`` promises "nothing a client can send raises out of here". Today the codec lets a
``ValueError`` (int over 4300 digits) and a ``RecursionError`` (deep nesting) escape it. After
T-0216 each such frame counts one ``malformed``, queues one ``astra.error(code="malformed")``,
leaves the connection open, and the session log's ``wire.in`` row records the codec's fixed
reason, never the payload.

Regime: no socket (``server.open()``), the T-0028 rig over ``tests/fakes/``, ``SessionLog`` in a
tmp dir; no measurement.
"""

from __future__ import annotations

import random
import re
from typing import Any

import pytest

from local_astra.contracts.codec import encode
from local_astra.contracts.protocol import AstraError, AstraHello
from tests.unit._hud_rig import Recorder, require_symbol, running
from tests.unit._session_log_rig import make_log, of_kind, read_records

#: T-0216 re-pin: R-1 (docs/tasks/T-0216.md Status; both ends pinned, the T-0193 A0 way). T-0216 deliberately edits
#: ``contracts/codec.py``, which two earlier census rows pin byte-identical
#: (``test_session_engine_empty_retry.test_turn_machine_untouched`` and
#: ``test_t0181_mic_states.test_a11_the_machine_files_are_pinned_and_only_mic_muted_is_new``). The base end: the file at
#: T-0216's literal base 423362b7 is the content those rows pinned (``git show 423362b7:<path> | sha256sum``). The tree
#: end: the working tree is exactly the implementation commit e705e79b's blob (``git show e705e79b:<path> |
#: sha256sum``, 2026-10-31). Both are literals, so they hold whatever later ref is checked out.
T0216_BASE = "423362b79666268e313377025c1f359081199ac1"
T0216_CODEC = "local_astra/contracts/codec.py"
T0216_BASE_SHA256 = "ec10fa7c84ec4845197321c37111b91a65ec263cc2fa2228baf9ff6fb3f92338"
T0216_MOVED_SHA256 = "97383bb0e652c932b5fdf27a6f634e58276892626cc87b08253d292621da651d"

#: T-0216c re-pin: R-1 (docs/tasks/T-0216c.md Owned files; the T-0216 R-1 way, both ends pinned). T-0216c
#: deliberately edits ``contracts/codec.py`` again. The base end: the file at T-0216c's literal base 7489b9ed is
#: exactly the T-0216 tree end above (``T0216_MOVED_SHA256``). The tree end: ``T0216C_MOVED_SHA256``, the sha256 of
#: the T-0216c implementation's codec.py (be4fd01c; ``git show be4fd01c:local_astra/contracts/codec.py |
#: sha256sum`` checked 2026-10-31). ``T0216C_TREE_ENDS`` is what the two census rows accept for the working tree:
#: the T-0216c blob alone. (While T-0216c was pending it held both contents; activation narrowed it.) The row
#: ``test_t0216c_codec_census_tree_end_is_the_implementation_blob`` below pins both: the tree is the new blob AND
#: the tuple holds only it.
T0216C_BASE = "7489b9eda75f382f88ce62e2a0b56ae4719d3a29"
T0216C_MOVED_SHA256 = "e9bf2432d43b35ab6fc6e891dc45c04cc25e632f9328be64bc7a39975e34222d"
T0216C_TREE_ENDS: tuple[str, ...] = (T0216C_MOVED_SHA256,)


class RecordingSink:
    def __init__(self) -> None:
        self.ingested: list[Any] = []

    def ingest(self, message: Any) -> Any:
        self.ingested.append(message)
        return None


def _digits(n: int) -> str:
    rng = random.Random(216)
    return str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(n - 1))


HUGE_INT_MSG = '{"type":"astra.text","text":"hi","n":' + _digits(5000) + "}"
DEEP_MSG = '{"type":"astra.text","a":' + "[" * 100_000 + "]" * 100_000 + "}"


@pytest.mark.parametrize(
    ("raw", "run"),
    [
        (HUGE_INT_MSG, r"\d{50}"),
        (HUGE_INT_MSG.encode(), r"\d{50}"),
        (DEEP_MSG, r"[\[\]]{50}"),
    ],
    ids=["huge_int_str", "huge_int_bytes", "deep_nesting"],
)
async def test_receive_counts_one_malformed_and_stays_open(tmp_path, raw: str | bytes, run: str):
    log = make_log(tmp_path / "logs")
    async with running(tmp_path / "data", session_log=log, publish=Recorder()) as rig:
        server = require_symbol("HudServer")(rig.session, frame_sink=RecordingSink(), port=0)
        conn = server.open()
        before = dict(server.counts)
        try:
            conn.receive(raw)
        except Exception as e:
            raise AssertionError(
                f"criterion 4: receive is total but raised {type(e).__name__}: {str(e)[:120]}"
            ) from None

        assert server.counts["malformed"] - before["malformed"] == 1, (
            f"criterion 4: one malformed counted; counts went {before} -> {dict(server.counts)}"
        )
        errors = [m for m in conn.sent() if isinstance(m, AstraError)]
        assert [e.code for e in errors] == ["malformed"], (
            f"criterion 4: exactly one astra.error(code='malformed') queued; sent={conn.sent()!r:.300}"
        )
        assert not re.search(run, errors[0].message), (
            f"the error message quotes the payload: {errors[0].message!r:.200}"
        )
        assert not conn.closed and conn.close_reason is None, (
            f"criterion 4: the connection stays open; close_reason={conn.close_reason!r}"
        )
        # Still serving: a hello after the bad frame is answered.
        n_sent = len(conn.sent())
        conn.receive(encode(AstraHello()))
        assert conn.greeted and len(conn.sent()) > n_sent, (
            "criterion 4: after the malformed frame the connection still takes a hello"
        )

    inbound = of_kind(read_records(log.path), "wire.in")
    assert len(inbound) == 2, f"one wire.in per frame; got {len(inbound)}"
    row = inbound[0]["data"]
    assert row["outcome"] == "malformed", f"criterion 3/4: the row says {row!r:.300}"
    assert isinstance(row["msg"], str) and row["msg"], f"the codec's reason; got {row!r:.300}"
    assert not re.search(r"\d{50}", row["msg"]), f"a digit run in wire.in msg {row['msg']!r:.200}"
    assert not re.search(r"[\[\]{}]{50}", row["msg"]), f"a bracket run in {row['msg']!r:.200}"
    assert row["n_bytes"] == len(raw if isinstance(raw, bytes) else raw.encode())


def test_t0216c_codec_census_base_end_is_the_t0216_tree_end() -> None:
    """T-0216c re-pin: R-1, the base end: codec.py at T-0216c's literal base 7489b9ed is the T-0216 tree end."""
    from tests.unit.test_t0192_hand_back_machine import t0192_end_sha256

    assert t0192_end_sha256(T0216_CODEC, T0216C_BASE) == T0216_MOVED_SHA256, (
        f"T-0216c re-pin: R-1: {T0216_CODEC} at {T0216C_BASE[:8]} is not the e705e79b blob"
    )
    assert T0216C_MOVED_SHA256 in T0216C_TREE_ENDS, (
        "T-0216c re-pin: R-1: the census rows accept the new content"
    )
    assert set(T0216C_TREE_ENDS) <= {T0216_MOVED_SHA256, T0216C_MOVED_SHA256}, (
        "T-0216c re-pin: R-1: the census rows accept no content but the two named blobs"
    )


def test_t0216c_codec_census_tree_end_is_the_implementation_blob() -> None:
    """T-0216c re-pin: R-1, the tree end, strict: the working tree is the T-0216c blob and the census rows accept
    only it (``T0216C_TREE_ENDS`` was narrowed at activation)."""
    from tests.unit.test_t0192_hand_back_machine import t0192_end_sha256

    got = t0192_end_sha256(T0216_CODEC)
    assert got == T0216C_MOVED_SHA256, (
        f"T-0216c re-pin: R-1: {T0216_CODEC} is not the T-0216c blob ({got} != {T0216C_MOVED_SHA256})"
    )
    assert T0216C_TREE_ENDS == (T0216C_MOVED_SHA256,), (
        "T-0216c re-pin: R-1: at activation T0216C_TREE_ENDS is narrowed to the T-0216c blob alone"
    )

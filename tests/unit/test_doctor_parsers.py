"""The doctor's parsers, against output this box really produced (criteria 5 and 7).

``tests/fixtures/box/`` is a recording of the 5090 box on 2026-09-14 (see its ``MANIFEST.md``).
The expectations below were read off those files by hand — the webcam-only source list, the
``Dummy Output`` default sink, ``27433/32607`` MiB, the ``--cpu`` in the STT container argv, the
missing ``--revision`` on the engine one — so a parser written to match the test rather than the
box fails.

Regime: pure text/JSON parsing. No box is touched; no measurement.
"""

from __future__ import annotations

import json

from tests.unit import _box_rig as rig


def parse(name: str, text: str):
    preflight = rig.module("preflight")
    return rig.attr(preflight, name)(text)


# --- `wpctl status` (criterion 2 facts) ----------------------------------------------------------


def test_wpctl_status_yields_the_dummy_sink_and_the_webcam_source():
    """Hand-read from the fixture: ``* 123. Dummy Output`` and one Analog Stereo source."""
    got = parse("parse_wpctl_status", rig.fixture("wpctl_status.txt"))
    assert got["default_sink"] == "Dummy Output", (
        "criterion 2: the default sink is the one wpctl marks with a star under Sinks; the "
        f"fixture has `Dummy Output` and the parser said {got['default_sink']!r}"
    )
    sources = tuple(got["sources"])
    assert sources == ("Logitech Webcam C925e Analog Stereo",), (
        "criterion 2: the *audio* sources, not the video ones — the fixture Video section also "
        f"lists a `Logitech Webcam C925e (V4L2)` source; the parser said {sources!r}"
    )
    assert "vol:" not in "".join(sources) and "52." not in "".join(sources), (
        f"…as names, without the wpctl id and volume columns; it said {sources!r}"
    )


def test_the_dummy_sink_from_the_fixture_is_a_failure():
    got = parse("parse_wpctl_status", rig.fixture("wpctl_status.txt"))
    rows = rig.by_name(
        rig.evaluate(
            **rig.good_box(default_sink=got["default_sink"], sources=tuple(got["sources"]))
        )
    )
    assert rows["audio.sink"].status == "FAIL", (
        f"criterion 2: a `Dummy Output` default sink is FAIL; got {rows['audio.sink']}"
    )
    assert "Dummy Output" in rows["audio.sink"].detail, "…naming it"


# --- nvidia-smi memory.used,memory.total csv (criterion 6) ---------------------------------------


def test_nvidia_smi_csv_is_read_as_mebibytes():
    got = parse("parse_nvidia_smi_csv", rig.fixture("nvidia_smi.csv"))
    assert (got["vram_used_mib"], got["vram_total_mib"]) == (27433, 32607), (
        "criterion 6: the fixture row is `27433 MiB, 32607 MiB` (header line included, units in "
        f"the cells); the parser said {got!r}"
    )


# --- container inspection (criterion 5) ----------------------------------------------------------


def test_the_stt_container_argv_shows_the_cpu_flag():
    got = parse("parse_docker_inspect", rig.fixture("docker_inspect_stt.json"))
    argv = tuple(got["argv"])
    assert "--cpu" in argv, (
        f"criterion 5: the fixture container was started with --cpu; the parser said {argv!r}"
    )
    assert argv[:2] == rig.STT_ARGV_CPU[:2], (
        "criterion 5: the argv is Path + Args (the entrypoint and its arguments), not Config.Cmd "
        f"alone; it said {argv!r}"
    )
    assert got["name"] == "astra-stt", f"got {got['name']!r}"
    assert "ASTRA_STT_CPU" not in json.dumps(got), (
        "criterion 5, negative: the *argv* is the fact; ASTRA_STT_CPU is what somebody meant to "
        "type and does not appear in the container description at all"
    )


def test_a_cpu_container_under_a_gpu_profile_is_the_warning_with_the_measured_cost():
    got = parse("parse_docker_inspect", rig.fixture("docker_inspect_stt.json"))
    rows = rig.by_name(
        rig.evaluate(**rig.good_box(stt_argv=tuple(got["argv"]), profile_stt_backend="kyutai-gpu"))
    )
    check = rows["container.stt_device"]
    assert check.status == "WARN", f"criterion 5: the live P0-2 state is a WARN; got {check}"
    assert "1.3" in check.detail, f"…carrying the measured cost; it said {check.detail!r}"

    agreeing = rig.by_name(
        rig.evaluate(**rig.good_box(stt_argv=rig.STT_ARGV_GPU, profile_stt_backend="kyutai-gpu"))
    )["container.stt_device"]
    assert agreeing.status == "PASS", (
        f"…and a container that agrees with the profile passes; got {agreeing}"
    )


def test_the_engine_container_argv_has_no_revision_and_that_is_the_carve_out_warning():
    got = parse("parse_docker_inspect", rig.fixture("docker_inspect_vllm.json"))
    argv = tuple(got["argv"])
    assert "--revision" not in argv, (
        "the 2026-09-14 recording has none (the launch script belongs to T-0014/T-0015); if this "
        f"fires the fixture was re-recorded after that follow-up landed. argv={argv!r}"
    )
    assert "--port" in argv and "8001" in argv, (
        f"criterion 5: the engine argv is read out of the container command; it said {argv!r}"
    )
    assert "--kv-cache-dtype" in argv and "bfloat16" in argv, (
        "criterion 5: the flags are compared to the launch script, so they must survive parsing; "
        f"the parser said {argv!r}"
    )
    rows = rig.by_name(rig.evaluate(**rig.good_box(vllm_argv=argv)))
    assert rows["container.vllm_argv"].status == "WARN", (
        f"criterion 5: a missing --revision is a WARN; got {rows['container.vllm_argv']}"
    )


# --- `ss -tlnp` (criterion 7) --------------------------------------------------------------------


def test_ss_tlnp_is_read_by_local_address_and_the_non_loopback_row_fails():
    """Criterion 7: the last fixture row is a deliberate ``0.0.0.0:8081`` listener.

    Every row of ``ss -tlnp`` carries ``0.0.0.0:*`` in its *peer* column, so a check that greps
    the output for ``0.0.0.0`` calls a loopback-only box non-loopback, and one that greps for
    ``127.0.0.1`` calls this fixture fine. Both are wrong, and both fail here.
    """
    listeners = parse("parse_ss_tlnp", rig.fixture("ss_tlnp.txt"))
    rows = {(row["address"], row["port"]) for row in listeners}
    assert ("127.0.0.1", 8001) in rows and ("127.0.0.1", 8080) in rows, (
        f"criterion 7: the engine and STT listeners are in the fixture; parsed {sorted(rows)}"
    )
    assert ("0.0.0.0", 8081) in rows, (
        f"criterion 7: the deliberate non-loopback row is parsed too; parsed {sorted(rows)}"
    )
    assert ("0.0.0.0", 0) not in rows, "the peer column is not a listening address"
    assert all(isinstance(row["port"], int) for row in listeners)

    check = rig.by_name(rig.evaluate(**rig.good_box(listeners=tuple(listeners))))["net.loopback"]
    assert check.status == "FAIL", (
        "criterion 7 (docs/02 §9, docs/03 E8, CLAUDE.md): a listener on 0.0.0.0 is FAIL; "
        f"got {check}"
    )
    assert "8081" in check.detail and "0.0.0.0" in check.detail, (
        f"…naming the socket; it said {check.detail!r}"
    )

    loopback_only = tuple(row for row in listeners if row["address"] != "0.0.0.0")
    ok = rig.by_name(rig.evaluate(**rig.good_box(listeners=loopback_only)))["net.loopback"]
    assert ok.status == "PASS", f"…and a loopback-only box passes; got {ok}"


def test_an_ipv6_wildcard_listener_is_also_a_failure():
    """Criterion 7: ``[::]`` is the same hole written differently (the fixture has ``[::1]``)."""
    listeners = (
        {"address": "::1", "port": 631, "process": "cupsd"},
        {"address": "::", "port": 8000, "process": "python3"},
    )
    check = rig.by_name(rig.evaluate(**rig.good_box(listeners=listeners)))["net.loopback"]
    assert check.status == "FAIL", f"criterion 7: `[::]` is not loopback; got {check}"
    assert "8000" in check.detail, f"…naming the socket; it said {check.detail!r}"

    only_v6_loopback = (listeners[0],)
    ok = rig.by_name(rig.evaluate(**rig.good_box(listeners=only_v6_loopback)))["net.loopback"]
    assert ok.status == "PASS", f"…while `::1` is loopback; got {ok}"


# =================================================================================================
# T-0058 criteria 6 and 8: the facts the engine-start row is a function of.
#
# `tests/fixtures/box/nvidia_smi_memory_coldboot.csv` and `..._compute_apps_coldboot.csv` are
# verbatim slices of `logs/vram-coldboot-73a85dd.before.txt`, recorded on this box at
# 2026-09-16 22:17:23+01:00 (commit 73a85dd) seconds before the cold boot that died 21 s later
# with `ValueError: Free memory on device cuda:0 (27.23/31.33 GiB) on startup is less than
# desired GPU memory utilization (0.92, 28.83 GiB)`. Expectations below are read off those files
# by hand; see `tests/fixtures/box/MANIFEST.md` for the commands that produced them.
# =================================================================================================


def test_the_free_column_is_read_and_is_not_total_minus_used():
    """Criterion 8: the engine's check compares against `memory.free`, which is not `total - used`.

    The recorded row is ``32607 MiB, 3696 MiB, 28391 MiB``. Subtracting gives **28 911 MiB**, a
    figure 520 MiB more generous than the driver's own free reading — and generous in the
    direction that costs a boot: a row that subtracts predicts room the engine will not find.
    The same 525 MiB gap is why vLLM's `mem_get_info` total (≈ 32 082) is not `nvidia-smi`'s
    32 607, which is the other half of criterion 8.
    """
    got = parse("parse_nvidia_smi_memory", rig.fixture("nvidia_smi_memory_coldboot.csv"))
    assert (got["vram_total_mib"], got["vram_used_mib"], got["vram_free_mib"]) == (
        32607,
        3696,
        28391,
    ), f"the fixture row is `32607 MiB, 3696 MiB, 28391 MiB`; the parser said {got!r}"
    assert got["vram_free_mib"] != got["vram_total_mib"] - got["vram_used_mib"], (
        "criterion 8: on this device free != total - used (28391 vs 28911, a 520 MiB gap the "
        "driver keeps out of both columns). A parser that derives the free figure reports "
        "520 MiB of room that is not there, which is exactly the error that costs a 21 s boot"
    )


def test_the_memory_columns_are_read_by_name_not_by_position():
    """Criterion 8, adversarial: the column order is the caller's choice, not a constant.

    ``--query-gpu`` prints the fields in the order they were asked for, so a positional parser
    reads whichever order the collector happens to use today and silently swaps free with total
    when someone reorders the query. The input below is **synthetic** (the same numbers as the
    recording, asked in a different order); the fixture above is the verbatim one.
    """
    reordered = (
        "memory.free [MiB], memory.total [MiB], memory.used [MiB]\n28391 MiB, 32607 MiB, 3696 MiB\n"
    )
    got = parse("parse_nvidia_smi_memory", reordered)
    assert (got["vram_total_mib"], got["vram_used_mib"], got["vram_free_mib"]) == (
        32607,
        3696,
        28391,
    ), (
        "criterion 8: the header names the columns; reading them by position makes the free "
        f"figure whatever the query's word order was. The parser said {got!r}"
    )


def test_compute_apps_rows_name_the_holders_with_their_megabytes():
    """Criterion 6: the row must say *which processes hold the memory*, so they must parse.

    Hand-read from the fixture: Chrome's GPU process at 50 MiB and `moshi-server` — the
    recogniser whose residency is the whole failure — at 3 180 MiB.
    """
    rows = parse(
        "parse_nvidia_smi_compute_apps", rig.fixture("nvidia_smi_compute_apps_coldboot.csv")
    )
    by_mib = {row["used_mib"]: row for row in rows}
    assert sorted(by_mib) == [50, 3180], (
        f"criterion 6: two compute apps were resident, at 50 and 3180 MiB; the parser said {rows!r}"
    )
    assert by_mib[3180]["process_name"] == "moshi-server", (
        f"…and the recogniser is named as such; the parser said {by_mib[3180]!r}"
    )
    assert by_mib[3180]["pid"] == 3380309, f"…with its pid; the parser said {by_mib[3180]!r}"
    chrome = by_mib[50]
    assert "--type=gpu-process" in chrome["process_name"], (
        "criterion 6: the process_name column holds Chrome's whole argv, and the used_gpu_memory "
        f"column is the *last* one; the parser said {chrome!r}"
    )
    assert rig.RECOGNISER_MIB == 3180 and rig.CHROME_MIB == 50, (
        "fixture check: the constants the verdict rows compute with are the ones recorded here"
    )


def test_a_process_name_containing_a_comma_is_not_split():
    """Criterion 6, adversarial: CSV with an unquoted, comma-bearing field.

    ``nvidia-smi`` does not quote the process name, and a Chrome argv routinely carries
    ``--disable-features=A,B``. A parser that reads field 1 as the name and field 2 as the
    memory then reports a *flag* as the number of megabytes a process holds. Synthetic input,
    written here rather than appended to the verbatim recording.
    """
    text = (
        "pid, process_name, used_gpu_memory [MiB]\n"
        "4242, /opt/google/chrome/chrome --type=gpu-process --disable-features=A,B, 77 MiB\n"
    )
    rows = parse("parse_nvidia_smi_compute_apps", text)
    assert len(rows) == 1, f"one compute app, not one per comma; the parser said {rows!r}"
    assert rows[0]["used_mib"] == 77, (
        "criterion 6: the memory is the last column, whatever the name contains; the parser said "
        f"{rows[0]!r}"
    )
    assert rows[0]["process_name"].endswith("--disable-features=A,B"), (
        f"…and the name keeps its commas; the parser said {rows[0]!r}"
    )


# --- the utilisation the engine's argv will carry (criteria 6 and 8) -----------------------------

#: A script that passes the flag, one that only mentions it in a comment, and one with no flag at
#: all -- which is the state `services/vllm.sh` was in on 2026-09-16 and the reason the 0.92
#: default applied. Written here, not read from the repository, so the parser is tested against
#: all three shapes whatever the script says today.
SCRIPT_WITH_FLAG = """#!/usr/bin/env bash
# --gpu-memory-utilization used to be absent here (logs/vllm-coldboot-73a85dd.log).
exec docker run "$CONTAINER" -c "vllm serve $MODEL \\
  --gpu-memory-utilization 0.86 --kv-cache-memory-bytes 4294967296"
"""
SCRIPT_WITH_ONLY_A_COMMENT = """#!/usr/bin/env bash
# We could pass --gpu-memory-utilization 0.86 here, but we do not.
exec docker run "$CONTAINER" -c "vllm serve $MODEL --kv-cache-memory-bytes 4294967296"
"""
SCRIPT_WITHOUT_FLAG = """#!/usr/bin/env bash
exec docker run "$CONTAINER" -c "vllm serve $MODEL --kv-cache-memory-bytes 4294967296"
"""


def test_the_utilisation_is_parsed_from_the_executable_text_only():
    """Criterion 6/8: the flag's value, and `None` when only a comment mentions it.

    `None` means *the flag is absent*, which is not the same as "the script could not be read":
    the row turns the first into vLLM's own default (naming the version it belongs to) and the
    second into "not gathered".
    """
    assert parse("parse_vllm_utilization", SCRIPT_WITH_FLAG) == 0.86, (
        "the flag's value is read from the exec line"
    )
    assert parse("parse_vllm_utilization", SCRIPT_WITH_ONLY_A_COMMENT) is None, (
        "a commented-out flag is not passed to vLLM: the engine would still take its 0.92 "
        "default, so reading comments is how a doctor tells a human their box is fine when it "
        "is not (`tests/unit/test_vllm_launch_flags.py` makes the same distinction)"
    )
    assert parse("parse_vllm_utilization", SCRIPT_WITHOUT_FLAG) is None, (
        "no flag at all is the 2026-09-16 state"
    )

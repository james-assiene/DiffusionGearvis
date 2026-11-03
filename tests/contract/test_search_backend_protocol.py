"""T-0113 criterion 15: the ``SearchBackend`` protocol, the ``SearchResult``/``SearchResults`` contracts, and the fake.

- ``SearchBackend`` is a ``runtime_checkable`` protocol in ``contracts/protocols.py`` with ``async search(query) ->
  SearchResults``; ``tests/fakes/search.py::FakeSearch`` is one, and an object without ``search`` is not;
- the contracts follow the house rules (``contracts/base.Contract``: pydantic v2, frozen, unknown fields refused,
  hashable) and round-trip through JSON;
- ``SearchResults`` holds as many results as the backend returned (the renderer's bound drops the sixth, criterion
  12, so the contract must be able to carry it);
- the fake echoes the query, answers with its canned rows in rank order, and logs every query in order.

The ``real`` side builds ``DdgsSearch()`` (T-0111b); it is ``net``-marked and skips unless ``ASTRA_LIVE_SEARCH=1``.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from local_astra.contracts.base import Contract
from tests.fakes.search import DEFAULT_ROWS, FakeSearch


def _net_mark() -> list[pytest.MarkDecorator]:
    """T-0111b criterion 9: the rows carry ``net`` once pyproject registers it (``--strict-markers`` refuses it
    before); the env gate in ``_real_not_wired`` keeps the real side off the network either way."""
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    ini = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]
    return (
        [pytest.mark.net] if any(m.split(":")[0].strip() == "net" for m in ini["markers"]) else []
    )


@pytest.fixture(params=["fake", pytest.param("real", marks=[pytest.mark.gpu, *_net_mark()])])
def backend(request: pytest.FixtureRequest) -> str:
    """``tests/contract/conftest.py``'s ``backend``, overridden here only to add ``net`` to the ``real`` param: the
    fake rows stay unmarked and selected offline (``-m "not gpu and not net"``)."""
    return str(request.param)


def _real_not_wired(backend: str) -> None:
    """T-0111b: the ``real`` side builds ``DdgsSearch()`` (opt-in: ``ASTRA_LIVE_SEARCH=1``, ``make search-live``)."""
    if backend != "real":
        return
    import importlib
    import os

    if os.environ.get("ASTRA_LIVE_SEARCH") != "1":
        pytest.skip(
            "the real search backend reaches the network: set ASTRA_LIVE_SEARCH=1 (make search-live)"
        )
    try:
        mod = importlib.import_module("local_astra.search.ddgs_backend")
    except ModuleNotFoundError as exc:
        raise NotImplementedError(
            "T-0111b: local_astra.search.ddgs_backend does not exist yet"
        ) from exc
    assert isinstance(mod.DdgsSearch(), _protocol())


def _contract(name: str) -> type[BaseModel]:
    from local_astra.contracts import toolcall as ct

    obj = getattr(ct, name, None)
    if obj is None:
        raise NotImplementedError(f"T-0113: contracts.toolcall.{name} does not exist yet")
    assert isinstance(obj, type)
    return obj


def _protocol() -> type:
    from local_astra.contracts import protocols

    obj = getattr(protocols, "SearchBackend", None)
    if obj is None:
        raise NotImplementedError("T-0113: contracts.protocols.SearchBackend does not exist yet")
    assert isinstance(obj, type)
    return obj


def test_the_fake_satisfies_the_runtime_checkable_protocol(backend: str) -> None:
    _real_not_wired(backend)
    proto = _protocol()
    assert isinstance(FakeSearch(), proto)

    class NoSearch:
        async def lookup(self, query: str) -> None: ...

    assert not isinstance(NoSearch(), proto)  # the check has power: a renamed method is caught


def test_the_contracts_follow_the_house_rules() -> None:
    result_t, results_t = _contract("SearchResult"), _contract("SearchResults")
    assert issubclass(result_t, Contract) and issubclass(results_t, Contract)
    r = result_t(title="Paris weather", url="https://example.org/paris", snippet="14C")
    with pytest.raises(ValidationError):
        result_t(title="t", url="u", snippet="s", rank=1)  # extra="forbid"
    with pytest.raises(ValidationError):
        r.title = "changed"  # frozen
    rs = results_t(query="weather in Paris", results=(r,))
    assert hash(rs) == hash(results_t(query="weather in Paris", results=(r,)))
    assert results_t.model_validate_json(rs.model_dump_json()) == rs
    assert isinstance(rs.results, tuple)


def test_search_results_carry_more_than_five_results() -> None:
    result_t, results_t = _contract("SearchResult"), _contract("SearchResults")
    six = tuple(
        result_t(title=f"t{i}", url=f"https://example.org/{i}", snippet="s") for i in range(6)
    )
    assert len(results_t(query="q", results=six).results) == 6


async def test_the_fake_echoes_the_query_and_answers_in_rank_order(backend: str) -> None:
    _real_not_wired(backend)
    results_t = _contract("SearchResults")
    canned = {
        "tide times": [
            {"title": "Tides", "url": "https://tides.example.org", "snippet": "High 06:12"}
        ]
    }
    fake = FakeSearch(canned)
    got = await fake.search("tide times")
    assert isinstance(got, results_t)
    assert got.query == "tide times"
    assert [(r.title, r.url, r.snippet) for r in got.results] == [
        ("Tides", "https://tides.example.org", "High 06:12")
    ]
    other = await fake.search("weather in Paris")
    assert [r.title for r in other.results] == [row["title"] for row in DEFAULT_ROWS]
    assert fake.queries == ["tide times", "weather in Paris"]


#: T-0176 criterion 8: the kinds a ``last_attempts`` entry may carry, and the engine-label shape (T-0125's)
_ATTEMPT_KINDS = frozenset({"error", "rate_limited", "network", "timeout", "closed"})
_ENGINE_SHAPE = re.compile(r"^[a-z][a-z0-9_]{0,23}$")


@pytest.mark.parametrize(
    "side",
    [
        "fake",
        pytest.param("real", marks=[pytest.mark.gpu, *_net_mark()]),
    ],
)
async def test_last_attempts_shape_after_an_answered_search(side: str) -> None:
    """T-0176 criterion 8: the fake's additive ``last_attempts`` behaves as the real backend's: a tuple of
    ``(engine, kind)`` string pairs after an answered search, kinds from the closed set, never the answering engine.
    The real side runs one live query through ``DdgsSearch()`` (opt-in: ``ASTRA_LIVE_SEARCH=1``) and is pending for
    T-0176 (at BASE a0badcd1 the attribute does not exist)."""
    _real_not_wired(side)
    if side == "fake":
        backend: Any = FakeSearch(
            last_engine="mojeek", last_attempts=(("duckduckgo", "error"), ("brave", "rate_limited"))
        )
    else:
        import importlib

        backend = importlib.import_module("local_astra.search.ddgs_backend").DdgsSearch()
    assert getattr(backend, "last_attempts", None) == (), "() before any search"
    await backend.search("weather in Paris")
    got = getattr(backend, "last_attempts", None)
    assert isinstance(got, tuple), repr(got)
    for entry in got:
        assert isinstance(entry, tuple) and len(entry) == 2, repr(entry)
        engine, kind = entry
        assert isinstance(engine, str) and _ENGINE_SHAPE.fullmatch(engine), repr(entry)
        assert kind in _ATTEMPT_KINDS, repr(entry)
    assert getattr(backend, "last_engine", None) not in {e for e, _ in got}, (got, backend)

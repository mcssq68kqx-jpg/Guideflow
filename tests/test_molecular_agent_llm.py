"""Tests for the true LLM molecular agent (tool-calling loop)."""
from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.app.services.molecular_evidence import (
    MockEvidenceProvider,
    MolecularInputParser,
    _ProviderView,
)


class FakeQwen:
    def __init__(self, api_key: str | None, script: list[dict]):
        self.api_key = api_key
        self._script = list(script)
        self._idx = 0

    def run_tool_turn(self, messages, *, tools=None, timeout=45):
        if self._idx >= len(self._script):
            return {"role": "assistant", "content": '{"ready": true}'}, [], None
        entry = self._script[self._idx]
        self._idx += 1
        tool_calls = entry.get("tool_calls") or []
        content = entry.get("content") or ""
        # normalize to the shape QwenClient returns
        norm_calls = []
        for c in tool_calls:
            fn = c.get("function") or {}
            norm_calls.append({
                "id": c.get("id") or fn.get("name") or "",
                "type": "function",
                "function": {"name": fn.get("name") or "", "arguments": fn.get("arguments") or "{}"},
            })
        msg: dict = {"role": "assistant", "content": content}
        if norm_calls:
            msg["tool_calls"] = norm_calls
        return msg, norm_calls, entry.get("degraded")


def _mock_providers():
    from backend.app.services.molecular_evidence import MockGnomADProvider, MockVEPProvider
    mock = MockEvidenceProvider()
    return [_ProviderView(mock, "ClinVar"), _ProviderView(mock, "CIViC"), _ProviderView(mock, "MyVariant"), MockGnomADProvider(), MockVEPProvider()]


def _providers_without_myvariant():
    mock = MockEvidenceProvider()
    return [_ProviderView(mock, "ClinVar"), _ProviderView(mock, "CIViC")]


def test_llm_agent_resolves_then_searches_and_finishes():
    from backend.app.services.molecular_agent_llm import MolecularLLMAgent
    from backend.app.services.molecular_evidence import MockVariantResolver

    variants = MolecularInputParser().parse("MYD88 p.L265P", disease="DLBCL", sample_type="tumor tissue")
    assert len(variants) == 1
    # Force protein_level_only so the LLM has a reason to resolve; then it should search
    # This also exercises the enrichment path without relying on the deterministic agent's auto-resolve
    var_id = variants[0].variant_id

    qwen = FakeQwen("fake-key", [
        {"tool_calls": [{"id": "t1", "function": {"name": "resolve_variant_coordinates", "arguments": f'{{"variant_id": "{var_id}"}}'}}]},
        {"tool_calls": [{"id": "t2", "function": {"name": "search_clinvar", "arguments": f'{{"variant_id": "{var_id}"}}'}}]},
        {"tool_calls": [{"id": "t3", "function": {"name": "search_civic", "arguments": f'{{"variant_id": "{var_id}"}}'}}]},
        {"tool_calls": [{"id": "t4", "function": {"name": "finish_evidence_collection", "arguments": '{"reason": "evidence sufficient"}'}}]},
    ])
    agent = MolecularLLMAgent(qwen_client=qwen, providers=_mock_providers(), resolver=MockVariantResolver(), max_steps=8)

    records, trace, status, resolver_status = agent.run(variants, question="是否支持MCD亚型？", trace=[])

    # Should have resolved and queried required providers
    assert resolver_status["queried"] is True
    assert resolver_status["resolved_count"] == 1
    assert status["ClinVar"]["queried"] is True
    assert status["CIViC"]["queried"] is True
    assert status["ClinVar"]["record_count"] >= 1
    assert status["CIViC"]["record_count"] >= 1
    assert any(s["step"] == "resolve_variant" for s in trace)
    assert any(s["step"] == "agent_tool_call" and s["tool"] == "search_clinvar" for s in trace)
    assert any(s["step"] == "agent_finish_tool" for s in trace)
    assert trace[0]["agent"] == "llm"
    assert trace[-1]["step"] == "complete"


def test_llm_agent_supplements_required_provider_when_llm_skips():
    from backend.app.services.molecular_agent_llm import MolecularLLMAgent
    from backend.app.services.molecular_evidence import MockVariantResolver

    variants = MolecularInputParser().parse("MYD88 p.L265P", disease="DLBCL", sample_type="tumor tissue")
    var_id = variants[0].variant_id
    # LLM only searches ClinVar then finishes — CIViC is skipped and must be supplemented deterministically
    qwen = FakeQwen("fake-key", [
        {"tool_calls": [{"id": "t1", "function": {"name": "search_clinvar", "arguments": f'{{"variant_id": "{var_id}"}}'}}]},
        {"tool_calls": [{"id": "t2", "function": {"name": "finish_evidence_collection", "arguments": '{"reason": "done"}'}}]},
    ])
    agent = MolecularLLMAgent(qwen_client=qwen, providers=_mock_providers(), resolver=MockVariantResolver(), max_steps=6)

    records, trace, status, resolver_status = agent.run(variants, trace=[])

    assert status["ClinVar"]["queried"] is True
    assert status["CIViC"]["queried"] is True  # supplemented
    assert any(s["step"] == "supplement_required_query" and s["provider"] == "CIViC" for s in trace)
    # Records should include both providers despite LLM skipping CIViC
    providers_hit = {r.provider for r in records}
    assert "ClinVar" in providers_hit
    assert "CIViC" in providers_hit


def test_llm_agent_fallback_without_api_key_is_deterministic():
    from backend.app.services.molecular_agent_llm import MolecularLLMAgent
    from backend.app.services.molecular_evidence import MockVariantResolver

    variants = MolecularInputParser().parse("MYD88 p.L265P", disease="DLBCL", sample_type="tumor tissue")
    qwen = FakeQwen(None, [])  # no key
    agent = MolecularLLMAgent(qwen_client=qwen, providers=_mock_providers(), resolver=MockVariantResolver(), max_steps=6)

    records, trace, status, resolver_status = agent.run(variants, trace=[])

    assert any(s["step"] == "llm_fallback_to_deterministic" for s in trace)
    assert any(s["step"] == "fallback_complete" for s in trace)
    # Deterministic fallback still queries required providers
    assert status["ClinVar"]["queried"] is True
    assert status["CIViC"]["queried"] is True
    assert len(records) >= 1


def test_llm_agent_handles_degraded_tool_turn_then_supplements():
    from backend.app.services.molecular_agent_llm import MolecularLLMAgent
    from backend.app.services.molecular_evidence import MockVariantResolver

    variants = MolecularInputParser().parse("MYD88 p.L265P", disease="DLBCL", sample_type="tumor tissue")
    qwen = FakeQwen("fake-key", [
        {"degraded": "agent_tool_turn_failed:TimeoutError"},
    ])
    agent = MolecularLLMAgent(qwen_client=qwen, providers=_mock_providers(), resolver=MockVariantResolver(), max_steps=6)

    records, trace, status, resolver_status = agent.run(variants, trace=[])

    assert any(s["step"] == "agent_degraded" for s in trace)
    # Even on degraded, required providers should be supplemented or at least the trace completes
    assert trace[-1]["step"] == "complete"
    assert trace[-1]["completion_state"] == "degraded"


def test_service_auto_uses_llm_when_key_present_and_live_mode(monkeypatch, tmp_path):
    db_path = tmp_path / f"test_{uuid.uuid4().hex}.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("AUTH_SECRET", "test-secret-not-for-prod")
    monkeypatch.setenv("COOKIE_SECURE", "0")
    monkeypatch.setenv("CORS_ORIGINS", "http://testserver,http://127.0.0.1:5173")
    monkeypatch.setenv("QWEN_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("MOLECULAR_AGENT_MODE", "auto")
    monkeypatch.setenv("MOLECULAR_EVIDENCE_PROVIDER_MODE", "live")

    import backend.app.db as dbmod
    from backend.app.db import init_db
    from backend.app.web_config import reset_web_config_cache

    reset_web_config_cache()
    if dbmod._engine is not None:
        dbmod._engine.dispose()
    dbmod._engine = None
    dbmod._SessionLocal = None
    init_db()

    from backend.app.services.molecular_evidence import MolecularEvidenceService

    service = MolecularEvidenceService(provider_mode="live")
    # auto should have picked llm when key is present in live mode
    assert service.agent_mode_effective == "llm"

    # Cleanup
    if dbmod._engine is not None:
        dbmod._engine.dispose()
    dbmod._engine = None
    dbmod._SessionLocal = None
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("MOLECULAR_AGENT_MODE", raising=False)


def test_service_mock_mode_forces_deterministic_even_with_key(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("MOLECULAR_AGENT_MODE", "llm")
    monkeypatch.setenv("MOLECULAR_EVIDENCE_PROVIDER_MODE", "mock")

    from backend.app.services.molecular_evidence import MolecularEvidenceService

    service = MolecularEvidenceService(provider_mode="mock")
    assert service.agent_mode_effective == "deterministic"

    monkeypatch.delenv("QWEN_API_KEY", raising=False)


def test_api_query_mock_still_deterministic_and_structured(monkeypatch, tmp_path):
    db_path = tmp_path / f"test_{uuid.uuid4().hex}.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("AUTH_SECRET", "test-secret-not-for-prod")
    monkeypatch.setenv("COOKIE_SECURE", "0")
    monkeypatch.setenv("CORS_ORIGINS", "http://testserver,http://127.0.0.1:5173")

    import backend.app.db as dbmod
    from backend.app.web_config import reset_web_config_cache

    reset_web_config_cache()
    if dbmod._engine is not None:
        dbmod._engine.dispose()
    dbmod._engine = None
    dbmod._SessionLocal = None

    from backend.app.db import init_db
    from backend.api.server import create_app

    init_db()
    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/molecular-evidence/query",
            json={
                "variants_text": "MYD88 p.L265P",
                "disease": "DLBCL",
                "sample_type": "tumor tissue",
                "provider_mode": "mock",
                "agent_mode": "auto",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        # mock forces deterministic — trace should NOT contain llm agent marker
        assert body["meta"]["agent_mode"] == "deterministic"
        assert body["audit"]["agent_trace"][0]["step"] == "plan"
        assert len(body["evidence"]) >= 1

    if dbmod._engine is not None:
        dbmod._engine.dispose()
    dbmod._engine = None
    dbmod._SessionLocal = None

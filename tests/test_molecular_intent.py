"""Tests for deterministic molecular-evidence intent detection and wiring."""
from __future__ import annotations

import pytest

from backend.app.models import EvidenceBundle, QAResult
from backend.app.prompts import build_evidence_prompt
from backend.app.services.molecular_intent import (
    MEDIUM,
    NONE,
    STRONG,
    detect_molecular_intent,
)


@pytest.mark.parametrize(
    "question",
    [
        "MYD88 p.L265P 在 DLBCL 中有什么临床意义？",
        "MYD88 p.L265P, CD79B p.Y196H 的分子证据",
        "TP53 c.743G>A 这个变异的致病性如何？",
    ],
)
def test_strong_intent_on_gene_with_hgvs(question):
    intent = detect_molecular_intent(question)
    assert intent.level == STRONG
    assert intent.should_run is True
    assert intent.variants_text


def test_medium_intent_on_gene_with_variant_context():
    intent = detect_molecular_intent("患者检出 TP53 突变，VAF 32%，有什么靶向药提示？")
    assert intent.level == MEDIUM
    assert intent.should_run is False
    assert intent.should_hint is True
    assert "TP53" in intent.genes


def test_medium_intent_on_structural_wording():
    intent = detect_molecular_intent("BCL2 融合是否提示预后不良？")
    assert intent.level == MEDIUM
    assert intent.should_hint is True


def test_no_intent_on_plain_guideline_question():
    for question in ("DLBCL 一线治疗推荐什么方案？", "IPI 4 分的患者能用 R-CHOP 吗？"):
        intent = detect_molecular_intent(question)
        assert intent.level == NONE
        assert intent.should_run is False


def test_bare_gene_mention_stays_guideline():
    """A gene name alone should not trigger the molecular sub-agent."""
    intent = detect_molecular_intent("CD20 表达是否影响方案选择？")
    assert intent.level in (NONE, MEDIUM)


def test_empty_question_is_none():
    assert detect_molecular_intent("").level == NONE


def test_anaphora_inherits_variant_from_history():
    """A follow-up with no HGVS of its own should reuse the previous variant."""
    history = [
        {"role": "user", "content": "MYD88 p.L265P 在 DLBCL 中有什么临床意义？"},
        {"role": "assistant", "content": "MYD88 p.L265P 是 DLBCL 中常见的复发突变，证据多来自回顾性研究。"},
    ]
    intent = detect_molecular_intent("这个变异会影响一线方案选择吗？", history)
    assert intent.level == STRONG
    assert intent.should_run is True
    assert "MYD88" in intent.variants_text
    assert "MYD88" in intent.genes
    assert "anaphora" in intent.signals


def test_anaphora_without_history_stays_hint_only():
    """Pointing back with nothing to inherit must not invent a variant."""
    intent = detect_molecular_intent("这个变异有什么临床意义？")
    assert intent.level != STRONG
    assert intent.variants_text == "" or intent.level == MEDIUM


def test_medium_variant_text_is_trimmed():
    """Medium intent must not ship the whole prose sentence to the parser."""
    intent = detect_molecular_intent("患者检出 TP53 突变，VAF 32%，有什么靶向药提示？")
    assert intent.level == MEDIUM
    assert "靶向药提示" not in intent.variants_text
    assert "TP53" in intent.variants_text


def test_prompt_includes_molecular_section_with_mn_citations():
    bundle = EvidenceBundle(
        primary_hits=[],
        molecular_evidence={
            "provider_mode": "mock",
            "variants": [{"gene": "MYD88", "protein_hgvs": "p.L265P", "mapping_status": "insufficient"}],
            "evidence": [
                {
                    "provider": "CIViC",
                    "source_title": "MYD88 L265P in DLBCL",
                    "evidence_level": "B",
                    "disease": "DLBCL",
                    "claim": "可作为通路异常线索",
                }
            ],
            "global_warnings": ["不能单独决定诊断或治疗"],
        },
    )
    prompt = build_evidence_prompt("MYD88 p.L265P 意义？", bundle)
    assert "[M1]" in prompt
    assert "[M2]" in prompt
    assert "MYD88 p.L265P" in prompt
    assert "不能单独决定诊断或治疗" in prompt
    assert "不得据此给出诊疗或预后结论" in prompt
    # mock mode must be disclosed so nobody treats it as clinical evidence
    assert "mock" in prompt


def test_prompt_without_molecular_evidence_is_unchanged():
    prompt = build_evidence_prompt("DLBCL 一线治疗？", EvidenceBundle(primary_hits=[]))
    assert "[M1]" not in prompt
    assert "分子数据库证据" not in prompt


def test_web_payload_carries_molecular_evidence():
    result = QAResult(
        question="MYD88 p.L265P 意义？",
        answer="测试回答",
        sources=[],
        verification={},
        run_id="r1",
        trace_path="/tmp/r1.jsonl",
        molecular_evidence={"variant_count": 1, "variants": [{"gene": "MYD88"}]},
    )
    payload = result.to_web_payload()
    assert payload["molecular_evidence"]["variant_count"] == 1
    assert payload["molecular_evidence"]["variants"][0]["gene"] == "MYD88"

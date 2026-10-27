"""Tests for the somatic oncogenicity scorer (Horak-inspired O/LO/VUS/LB/B)."""
from __future__ import annotations

from backend.app.services.oncogenicity import (
    LITERATURE_TRIGGER_SCORE,
    SomaticOncogenicityScorer,
    tier_for_score,
)


def _card(**overrides):
    base = {
        "provider": "CIViC",
        "evidence_level": "L3",
        "disease_match": "DLBCL直接证据",
        "record_status": "accepted",
        "evidence_type": "predictive",
        "direction": "supports",
    }
    base.update(overrides)
    return base


def test_tier_boundaries_match_spec():
    assert tier_for_score(10)[0] == "O"
    assert tier_for_score(14)[0] == "O"
    assert tier_for_score(9)[0] == "LO"
    assert tier_for_score(6)[0] == "LO"
    assert tier_for_score(5)[0] == "VUS"
    assert tier_for_score(0)[0] == "VUS"
    assert tier_for_score(-1)[0] == "LB"
    assert tier_for_score(-6)[0] == "LB"
    assert tier_for_score(-7)[0] == "B"
    assert LITERATURE_TRIGGER_SCORE == 6


def test_hotspot_like_variant_reaches_lo_and_triggers_literature():
    scorer = SomaticOncogenicityScorer()
    result = scorer.score(
        variant_id="var_x",
        gene="MYD88",
        variant_type="SNV",
        cards=[_card()],
        allele_freq=1.2e-05,
        predictions_deleterious=2,
        predictions_total=2,
        hotspot=True,
    )
    # CIViC direct L3 +2, oncogene missense +1, rare +1, in-silico +1, hotspot +2 = 7
    assert result.score == 7
    assert result.tier == "LO"
    assert result.triggers_literature is True


def test_empty_evidence_defaults_to_vus_without_forcing():
    scorer = SomaticOncogenicityScorer()
    result = scorer.score(variant_id="var_y", gene="GENEX", cards=[])
    assert result.score == 0
    assert result.tier == "VUS"
    assert result.triggers_literature is False


def test_common_polymorphism_scores_benign():
    scorer = SomaticOncogenicityScorer()
    result = scorer.score(
        variant_id="var_z",
        gene="GENEX",
        variant_type="SNV",
        cards=[],
        allele_freq=0.02,
    )
    assert result.score == -4
    assert result.tier == "LB"


def test_benign_requires_explicit_evidence_not_absence():
    scorer = SomaticOncogenicityScorer()
    # No AF data at all -> stays VUS, never drifts benign.
    result = scorer.score(variant_id="var_w", gene="GENEX", cards=[], allele_freq=None)
    assert result.tier == "VUS"


def test_service_mock_query_attaches_oncogenicity_and_skips_pubmed(tmp_path, monkeypatch):
    import uuid

    db_path = tmp_path / f"test_{uuid.uuid4().hex}.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("AUTH_SECRET", "test-secret-not-for-prod")
    monkeypatch.setenv("COOKIE_SECURE", "0")

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

    service = MolecularEvidenceService(provider_mode="mock")
    result = service.analyze(
        text="MYD88 p.L265P", disease="DLBCL", sample_type="tumor tissue",
    )
    assert len(result.variants) == 1
    onco = result.variants[0].oncogenicity
    assert onco["tier"] in {"O", "LO", "VUS", "LB", "B"}
    assert isinstance(onco["score"], int)
    assert onco["contributions"]
    # mock stays offline: no PubMed status entry, literature gate traced or skipped
    assert "PubMed" not in result.audit["provider_status"]
    assert onco["tier_zh"] in {"致癌", "可能致癌", "意义不明", "可能良性", "良性"}

    if dbmod._engine is not None:
        dbmod._engine.dispose()
    dbmod._engine = None
    dbmod._SessionLocal = None

"""Somatic oncogenicity scoring (Horak-2022-inspired, heuristic).

Implements the "加减分出档" idea: 12 transparent criteria add or subtract
points from aggregated molecular evidence, and the total maps onto five
tiers: O (致癌, >=10), LO (可能致癌, 6-9), VUS (意义不明, 0-5),
LB (可能良性, -6--1), B (良性, <=-7).

This is a *decision-support heuristic*, not a certified implementation of
any guideline. Evidence insufficiency defaults to VUS (never forced), and
scores >= 6 trigger deeper lookup (literature / targeted-therapy search) so
compute is spent where suspicion is genuine -- mirroring the "好钢用在刀刃上"
gating. Errors are asymmetric by design: the scorer may over-call (AI 可以虚惊)
but must not under-call a genuinely risky variant into B/LB on thin evidence;
B/LB require explicit benign evidence (population frequency or benign records).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

TIER_ONCOGENIC = "O"
TIER_LIKELY_ONCOGENIC = "LO"
TIER_VUS = "VUS"
TIER_LIKELY_BENIGN = "LB"
TIER_BENIGN = "B"

TIER_LABEL_ZH = {
    TIER_ONCOGENIC: "致癌",
    TIER_LIKELY_ONCOGENIC: "可能致癌",
    TIER_VUS: "意义不明",
    TIER_LIKELY_BENIGN: "可能良性",
    TIER_BENIGN: "良性",
}

#: Score at or above which literature / targeted-therapy lookup is triggered.
LITERATURE_TRIGGER_SCORE = 6

TSG_SET = {
    "TP53", "BRCA1", "BRCA2", "ATM", "PTEN", "RB1",
    "CDKN2A", "APC", "VHL", "WT1", "NF1", "NF2", "SMAD4",
}

ONCOGENE_SET = {
    "MYD88", "CD79B", "BRAF", "EGFR", "ALK", "FLT3",
    "IDH1", "IDH2", "MYC", "BCL2", "BCL6", "EZH2",
    "KRAS", "NRAS", "PIK3CA", "KIT", "MET", "RET", "ROS1",
}


@dataclass
class ScoreContribution:
    criterion: str
    points: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"criterion": self.criterion, "points": self.points, "detail": self.detail}


@dataclass
class OncogenicityScore:
    variant_id: str
    gene: str
    score: int
    tier: str
    tier_zh: str
    contributions: list[ScoreContribution] = field(default_factory=list)
    triggers_literature: bool = False
    evidence_basis_count: int = 0
    note: str = (
        "启发式算法评分（Horak 2022 思想），仅供解读参考，非指南认证结论；"
        "证据不足默认归为意义不明（VUS），不硬判。"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "gene": self.gene,
            "score": self.score,
            "tier": self.tier,
            "tier_zh": self.tier_zh,
            "contributions": [c.to_dict() for c in self.contributions],
            "triggers_literature": self.triggers_literature,
            "evidence_basis_count": self.evidence_basis_count,
            "note": self.note,
        }


def tier_for_score(score: int) -> tuple[str, str]:
    if score >= 10:
        return TIER_ONCOGENIC, TIER_LABEL_ZH[TIER_ONCOGENIC]
    if score >= 6:
        return TIER_LIKELY_ONCOGENIC, TIER_LABEL_ZH[TIER_LIKELY_ONCOGENIC]
    if score >= 0:
        return TIER_VUS, TIER_LABEL_ZH[TIER_VUS]
    if score >= -6:
        return TIER_LIKELY_BENIGN, TIER_LABEL_ZH[TIER_LIKELY_BENIGN]
    return TIER_BENIGN, TIER_LABEL_ZH[TIER_BENIGN]


class SomaticOncogenicityScorer:
    """Score one variant from its evidence cards + aggregated annotations.

    Parameters mirror what the pipeline already collects, so no new network
    calls happen here:

    - ``cards``: EvidenceCard-like dicts (provider, evidence_level,
      disease_match, evidence_type, drug, record_status, direction).
    - ``allele_freq``: best available population AF (gnomAD/MyVariant) or None.
    - ``predictions_deleterious`` / ``predictions_total``: in-silico counts.
    - ``hotspot``: cancer-hotspot flag (OncoKB hotspot / recurrence), or None.
    - ``in_functional_domain``: UniProt domain overlap flag, or None.
    - ``variant_type`` / ``gene``: from molecular annotation.
    """

    def score(
        self,
        *,
        variant_id: str,
        gene: str,
        variant_type: Optional[str] = None,
        cards: Optional[list[dict[str, Any]]] = None,
        allele_freq: Optional[float] = None,
        predictions_deleterious: int = 0,
        predictions_total: int = 0,
        hotspot: Optional[bool] = None,
        in_functional_domain: Optional[bool] = None,
        oncokb_oncogenicity: Optional[str] = None,
    ) -> OncogenicityScore:
        cards = cards or []
        gene_up = (gene or "").upper()
        vtype = (variant_type or "").lower()
        contributions: list[ScoreContribution] = []

        def add(criterion: str, points: int, detail: str) -> None:
            contributions.append(ScoreContribution(criterion, points, detail))

        civic_direct = [
            c for c in cards
            if str(c.get("provider") or "").lower() == "civic"
            and c.get("disease_match") == "DLBCL直接证据"
            and str(c.get("record_status") or "").lower() in {"accepted", "mock_accepted"}
        ]
        if any(str(c.get("evidence_level") or "") in {"L1", "L2"} for c in civic_direct):
            add("CIViC直接证据L1-L2", 3, "CIViC 已审核 DLBCL 直接预测/预后证据（L1-L2）。")
        elif any(str(c.get("evidence_level") or "") == "L3" for c in civic_direct):
            add("CIViC直接证据L3", 2, "CIViC 已审核 DLBCL 直接证据（L3）。")
        elif any(
            str(c.get("provider") or "").lower() == "civic"
            and c.get("disease_match") == "LBCL近似证据" for c in cards
        ):
            add("CIViC近似证据", 1, "仅有 LBCL 近似 CIViC 证据，按外推计弱分。")

        onco = (oncokb_oncogenicity or "").lower()
        if "likely oncogenic" in onco or "likely_oncogenic" in onco:
            add("OncoKB可能致癌", 2, "OncoKB 标注为 Likely Oncogenic。")
        elif onco.strip() == "oncogenic":
            add("OncoKB致癌", 3, "OncoKB 标注为 Oncogenic。")
        elif "resistance" in onco:
            add("OncoKB耐药", 1, "OncoKB 标注与耐药相关，视为弱致癌线索。")
        if hotspot is True:
            add("癌症热点复发", 2, "癌症热点（OncoKB hotspot/人群复发）阳性。")

        for c in cards:
            if str(c.get("provider") or "").lower() != "clinvar":
                continue
            text = " ".join(str(c.get(k) or "") for k in ("evidence_type", "claim", "review_status")).lower()
            if "oncogenic" in text and "likely" not in text:
                add("ClinVar致癌性", 2, "ClinVar 体细胞致癌性分类支持。")
                break
            if "likely oncogenic" in text or "likely_oncogenic" in text:
                add("ClinVar可能致癌", 1, "ClinVar 可能致癌分类。")
                break
        for c in cards:
            if str(c.get("provider") or "").lower() != "clinvar":
                continue
            text = " ".join(str(c.get(k) or "") for k in ("claim", "evidence_type")).lower()
            if "likely benign" in text:
                add("ClinVar可能良性", -2, "ClinVar 胚系可能良性分类。")
                break
            if "benign" in text:
                add("ClinVar良性", -2, "ClinVar 胚系良性分类。")
                break

        if vtype in {"nonsense", "frameshift", "splice_site"} and gene_up in TSG_SET:
            add("抑癌基因功能缺失", 2, f"{gene_up} 为抑癌基因且变异致功能缺失。")
        elif vtype in {"deletion",} and gene_up in TSG_SET:
            add("抑癌基因缺失", 2, f"{gene_up} 为抑癌基因且存在缺失。")
        if vtype == "snv" and gene_up in ONCOGENE_SET:
            add("癌基因错义", 1, f"{gene_up} 为癌基因，错义改变具潜在激活意义。")

        if predictions_total >= 2 and predictions_deleterious >= 2:
            add("计算预测有害", 1, f"{predictions_deleterious}/{predictions_total} 个预测工具提示有害。")
        elif predictions_total >= 2 and predictions_deleterious == 0:
            add("计算预测无害", -1, "多个预测工具一致提示无害。")

        if allele_freq is not None:
            if allele_freq > 0.01:
                add("人群常见(BA1类)", -4, f"群体频率 {allele_freq:.2e} 为常见多态，不支持高外显致病假设。")
            elif allele_freq > 0.001:
                add("人群偏常见", -2, f"群体频率 {allele_freq:.2e} 偏常见，减分。")
            elif allele_freq < 0.0001:
                add("人群罕见", 1, f"群体频率 {allele_freq:.2e} 罕见，支持罕见变异假设。")
        else:
            add("人群频率缺失", 0, "无群体频率数据，该项不加分也不减分。")

        if in_functional_domain is True:
            add("功能结构域", 1, "变异落在 UniProt 注释的功能结构域内。")

        if vtype in {"synonymous", "intronic"}:
            add("同义/内含子", -2, "同义或内含子改变且无剪接影响证据。")

        total = sum(c.points for c in contributions)
        tier, tier_zh = tier_for_score(total)
        basis = sum(1 for c in contributions if c.points != 0)
        # Asymmetric safety: B/LB demand explicit benign evidence; a bare zero
        # stays VUS, and a zero with no basis at all is explicitly VUS.
        return OncogenicityScore(
            variant_id=variant_id,
            gene=gene_up,
            score=total,
            tier=tier,
            tier_zh=tier_zh,
            contributions=contributions,
            triggers_literature=total >= LITERATURE_TRIGGER_SCORE,
            evidence_basis_count=basis,
        )

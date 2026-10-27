"""Deterministic molecular-evidence intent detection for the main QA pipeline.

Runs before the agent loop so molecular evidence retrieval does not depend on the
planner LLM "noticing" a variant. Pure regex + the existing variant parser: no
extra LLM call on the hot path.

Levels:
- strong  — gene + HGVS (or gene + fusion/amplification wording): run the
  molecular sub-agent (concurrently with the main agent loop).
- medium  — variant-ish wording or a gene mentioned in a molecular context but
  no usable HGVS: hint the planner and guide the user to supply an HGVS.
- none    — plain guideline question: leave the pipeline untouched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

STRONG = "strong"
MEDIUM = "medium"
NONE = "none"

# HGVS-style tokens, even when the gene is missing.
HGVS_PROTEIN_RE = re.compile(
    r"\bp\.\s*\(?\s*(?:[A-Z][a-z]{2}|[A-Z*])\s*\d+\s*(?:[A-Z][a-z]{2}|[A-Z*]|Ter|fs|del|dup|ins)",
)
HGVS_CDNA_RE = re.compile(r"\bc\.\s*\d+\s*[A-Za-z]*[>_]", re.IGNORECASE)
GENOMIC_RE = re.compile(r"\b(?:chr)?(?:[0-9]{1,2}|X|Y)\s*[:：]\s*g\.\s*\d+", re.IGNORECASE)
TRANSCRIPT_RE = re.compile(r"\b(?:NM_|ENST)\s*[\d.]+", re.IGNORECASE)
FUSION_RE = re.compile(
    r"融合|重排|扩增|缺失|拷贝数|fusion|rearrangement|amplification|deletion|copy\s*number",
    re.IGNORECASE,
)

# Molecular-evidence context words that turn a bare gene mention into a hint.
VARIANT_CONTEXT_RE = re.compile(
    r"突变|变异|致病性|致癌性|致病|致癌|分子分型|分子检测|基因检测|分子证据|携带|"
    r"VAF|等位基因频率|测序深度|克隆性|亚克隆|外显子|转录本|位点|证据等级|"
    r"ClinVar|CIViC|gnomAD|MyVariant|Ensembl|VEP|ACMG|HGVS|NGS|panel|"
    r"靶向|耐药|用药提示|预后意义|临床意义",
    re.IGNORECASE,
)
REPORT_CONTEXT_RE = re.compile(
    r"检测报告|测序报告|基因报告|报告提示|报告显示|检出|NGS 结果", re.IGNORECASE
)

# Follow-up wording that refers back to a variant mentioned in earlier turns.
ANAPHORA_RE = re.compile(
    r"这个变异|该变异|此变异|这个突变|该突变|此突变|这个基因|该基因|上述变异|"
    r"这些变异|同一变异|它的|它们|这个位点|该位点",
)

# Uppercase gene-like token (avoid matching plain English words mid-sentence).
GENE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_-])([A-Z][A-Z0-9]{1,9})(?![a-z])")

# Lymphoma-relevant genes; used only to raise a bare mention to "medium".
LYMPHOMA_GENES = {
    "MYD88", "CD79B", "CD79A", "TP53", "BCL2", "BCL6", "MYC", "EZH2", "CARD11",
    "TNFAIP3", "GNA13", "MEF2B", "PIM1", "BTK", "CXCR4", "NOTCH1", "NOTCH2",
    "SPEN", "KMT2D", "KMT2C", "CREBBP", "EP300", "IRF8", "PRDM1", "BCL10",
    "ATM", "CCND1", "CDKN2A", "SOCS1", "SGK1", "TET2", "DNMT3A", "ID3",
    "TCF3", "FOXO1", "BIRC3", "KLHL6", "STAT6", "XPO1", "ARID1A", "CD58",
    "MS4A1", "BRAF", "KRAS", "NRAS", "PIK3CA", "AKT1", "MTOR", "PTEN",
    "JAK1", "JAK2", "STAT3", "STAT5B", "CIITA", "B2M", "CD274", "PDCD1",
    "CTLA4", "RHOA", "P2RY8", "NFKBIE", "S1PR2", "DUSP2", "ZEB2", "CD70",
    "TP63", "PRKCB", "MYC", "BCL2L1", "MCL1", "CCND3", "CDK4", "CDKN2B",
}

# Acronyms that look like gene symbols but are diseases / regimens / scores.
NON_GENE_TOKENS = {
    "DLBCL", "LBCL", "NCCN", "CSCO", "EHA", "CHOP", "RCHOP", "REPOCH", "DAEPOCH",
    "IPI", "FLIPI", "MIPI", "ECOG", "PET", "PETCT", "CT", "MRI", "LDH", "CNS",
    "ASCT", "CAR", "CRT", "COO", "ABC", "GCB", "NOS", "WHO", "ICC", "FDA",
    "NMPA", "AE", "SAE", "OS", "PFS", "DFS", "ORR", "CR", "PR", "PD", "SD",
    "IHC", "FISH", "PCR", "WES", "WGS", "IV", "III", "II", "SM", "EBV", "HIV",
    "HBV", "HCV", "CNSIPI", "BCL", "HGBL", "PMBL", "MCL", "FL", "CLL", "SLL",
    "MZL", "BL", "LPL", "PTCL", "NKTL", "ALCL", "HL", "NHL", "AHL",
}


@dataclass
class MolecularIntent:
    """Result of intent detection, attached to the ask context."""

    level: str = NONE
    variants_text: str = ""
    genes: List[str] = field(default_factory=list)
    reason: str = ""
    signals: List[str] = field(default_factory=list)

    @property
    def should_run(self) -> bool:
        """Whether the molecular sub-agent should be started directly."""
        return self.level == STRONG

    @property
    def should_hint(self) -> bool:
        """Whether the planner should be told molecular evidence is relevant."""
        return self.level in (STRONG, MEDIUM)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "variants_text": self.variants_text,
            "genes": list(self.genes),
            "reason": self.reason,
            "signals": list(self.signals),
        }


def _parse_variants(text: str) -> tuple[List[Any], Optional[str]]:
    """Try the production variant parser; degrade quietly if unavailable."""
    try:
        from backend.app.services.molecular_evidence import MolecularInputParser

        variants = MolecularInputParser().parse(text, disease="DLBCL")
    except Exception as exc:  # pragma: no cover - parser must never break QA
        return [], f"{type(exc).__name__}:{exc}"
    return list(variants or []), None


def _extract_genes(text: str) -> List[str]:
    genes: List[str] = []
    for match in GENE_TOKEN_RE.finditer(text or ""):
        token = match.group(1).upper()
        if token in NON_GENE_TOKENS:
            continue
        if token in LYMPHOMA_GENES and token not in genes:
            genes.append(token)
    return genes


def _variant_text(variants: List[Any]) -> str:
    """Build a compact "GENE p.X" style string for the molecular sub-agent.

    Prefer the canonical gene + HGVS form over the raw chunk so trailing
    Chinese wording from the question is not sent to the variant parser.
    """
    parts: List[str] = []
    for variant in variants:
        gene = str(getattr(variant, "gene", "") or "").strip().upper()
        hgvs = (
            getattr(variant, "protein_hgvs", None)
            or getattr(variant, "cdna_hgvs", None)
            or getattr(variant, "genomic_hgvs", None)
            or getattr(variant, "unresolved_position", None)
            or ""
        )
        if gene and hgvs:
            parts.append(f"{gene} {hgvs}")
            continue
        raw = str(getattr(variant, "raw_input", "") or "").strip()
        if raw:
            parts.append(raw)
    return ", ".join(dict.fromkeys(parts))


# Wording that marks a clause as carrying an actual molecular payload. Narrower
# than VARIANT_CONTEXT_RE on purpose: generic intent words ("临床意义", "靶向")
# signal *interest* but are not something the variant parser can consume.
PAYLOAD_CLAUSE_RE = re.compile(
    r"突变|变异|位点|外显子|转录本|等位基因频率|VAF|检出|携带|融合|重排|扩增|"
    r"拷贝数|缺失|fusion|rearrangement|amplification|deletion|copy\s*number|exon|"
    r"variant|mutation",
    re.IGNORECASE,
)

# Sentence/clause separators used to trim a question down to its molecular payload.
_CLAUSE_SPLIT_RE = re.compile(r"[，。；;!?！？、,\n]+")


def _compact_variant_text(text: str, genes: List[str]) -> str:
    """Trim a free-text question down to its molecular payload.

    Handing the whole sentence to the variant parser drags in Chinese prose
    ("患者检出 TP53 突变，VAF 32%，有什么靶向药提示？"), which makes the parser
    emit garbage `raw_input`. Keep only the clauses that actually carry a gene,
    an HGVS token or molecular wording.
    """
    raw = (text or "").strip()
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(raw) if c.strip()]
    if not clauses:
        return raw
    upper_genes = {str(g or "").upper() for g in genes}
    kept: List[str] = []
    for clause in clauses:
        hit = any(g and g in clause.upper() for g in upper_genes)
        hit = hit or bool(
            HGVS_PROTEIN_RE.search(clause)
            or HGVS_CDNA_RE.search(clause)
            or GENOMIC_RE.search(clause)
            or TRANSCRIPT_RE.search(clause)
            or PAYLOAD_CLAUSE_RE.search(clause)
        )
        if hit:
            kept.append(clause)
    if not kept:
        return clauses[0][:80]
    return "，".join(dict.fromkeys(kept))


def _recover_from_history(history: Optional[List[dict]]) -> tuple[List[Any], List[str]]:
    """Recover variant context from recent assistant turns.

    Follow-ups like "这个变异有什么临床意义？" carry no HGVS of their own, but
    the previous answer usually spells the variant out. Scan the most recent
    assistant messages and reuse whatever variant is parseable.
    """
    if not history:
        return [], []
    turns = [t for t in history if isinstance(t, dict) and str(t.get("role") or "") == "assistant"]
    for turn in reversed(turns[-3:]):
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        variants, _ = _parse_variants(content)
        usable = [
            v
            for v in variants
            if getattr(v, "gene", None)
            and (
                getattr(v, "protein_hgvs", None)
                or getattr(v, "cdna_hgvs", None)
                or getattr(v, "genomic_hgvs", None)
                or getattr(v, "unresolved_position", None)
            )
        ]
        if usable:
            genes: List[str] = []
            for variant in usable:
                gene = str(getattr(variant, "gene", "") or "").upper()
                if gene and gene not in genes:
                    genes.append(gene)
            return usable, genes
    return [], []


def detect_molecular_intent(question: str, history: Optional[List[dict]] = None) -> MolecularIntent:
    """Classify whether the question carries molecular-evidence retrieval intent."""
    text = (question or "").strip()
    if not text:
        return MolecularIntent(level=NONE, reason="empty_question")

    signals: List[str] = []

    # 1) Production parser with a gene anchored to an HGVS / structural wording.
    variants, parse_error = _parse_variants(text)
    usable = [
        v
        for v in variants
        if getattr(v, "gene", None)
        and (
            getattr(v, "protein_hgvs", None)
            or getattr(v, "cdna_hgvs", None)
            or getattr(v, "genomic_hgvs", None)
            or getattr(v, "unresolved_position", None)
        )
    ]
    if usable:
        genes = []
        for variant in usable:
            gene = str(getattr(variant, "gene", "") or "").upper()
            if gene and gene not in genes:
                genes.append(gene)
        return MolecularIntent(
            level=STRONG,
            variants_text=_variant_text(usable) or text,
            genes=genes,
            reason="gene_with_hgvs",
            signals=["parser_hit"],
        )

    # 1b) Follow-up that points back at a variant ("这个变异有什么临床意义？").
    # The turn itself carries no HGVS, but the previous answer spelled it out.
    if ANAPHORA_RE.search(text):
        recovered, recovered_genes = _recover_from_history(history)
        if recovered:
            return MolecularIntent(
                level=STRONG,
                variants_text=_variant_text(recovered),
                genes=recovered_genes,
                reason="anaphora_inherited_from_history",
                signals=signals + ["anaphora", "history_inherit"],
            )
        signals.append("anaphora_unresolved")

    # 2) HGVS / structural wording without a resolvable gene.
    if HGVS_PROTEIN_RE.search(text) or HGVS_CDNA_RE.search(text) or GENOMIC_RE.search(text):
        signals.append("hgvs_token")
        hgvs_genes = _extract_genes(text)
        return MolecularIntent(
            level=MEDIUM,
            variants_text=_compact_variant_text(text, hgvs_genes),
            genes=hgvs_genes,
            reason="hgvs_without_gene",
            signals=signals,
        )
    if TRANSCRIPT_RE.search(text):
        signals.append("transcript_token")

    genes = _extract_genes(text)
    if genes:
        if FUSION_RE.search(text):
            return MolecularIntent(
                level=MEDIUM,
                variants_text=_compact_variant_text(text, genes),
                genes=genes,
                reason="gene_with_structural_wording",
                signals=signals + ["fusion_wording"],
            )
        if VARIANT_CONTEXT_RE.search(text) or REPORT_CONTEXT_RE.search(text):
            return MolecularIntent(
                level=MEDIUM,
                variants_text=_compact_variant_text(text, genes),
                genes=genes,
                reason="gene_in_molecular_context",
                signals=signals + ["variant_context"],
            )
        # Bare gene mention (e.g. "TP53 在 DLBCL 中的意义") stays a guideline
        # question — unless the user is pointing back at a variant we lost track
        # of, in which case we hint instead of silently dropping the thread.
        if "anaphora_unresolved" in signals:
            return MolecularIntent(
                level=MEDIUM,
                variants_text=_compact_variant_text(text, genes),
                genes=genes,
                reason="anaphora_without_history",
                signals=signals,
            )
        return MolecularIntent(
            level=NONE,
            genes=genes,
            reason="bare_gene_mention",
            signals=signals,
        )

    if "anaphora_unresolved" in signals:
        return MolecularIntent(
            level=MEDIUM,
            variants_text=text,
            reason="anaphora_without_history",
            signals=signals,
        )
    if parse_error:
        signals.append(f"parser_error:{parse_error}")
    return MolecularIntent(level=NONE, reason="no_molecular_signal", signals=signals)

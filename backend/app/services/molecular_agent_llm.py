"""LLM-driven molecular evidence agent: tool-calling loop over resolvers and evidence providers.

Optimized: dedup, early-stop hint, grouped parallel search, retry/backoff,
invalid-variant skip, budget guard, and invalid-variant short-circuit.
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

from backend.app.prompts import MOLECULAR_AGENT_SYSTEM_PROMPT

MOLECULAR_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "resolve_variant_coordinates",
            "description": (
                "为单个变异补充缺失的转录本 / cDNA HGVS / 基因组坐标 / 参考基因组版本。"
                "当变异为 protein_level_only 或缺坐标时应优先调用；已是 unique 且带坐标的变异无需调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "variant_id": {"type": "string", "description": "待补坐标的 variant_id，如 var_abc123"},
                    "reason": {"type": "string", "description": "为什么需要补坐标（简述）"},
                },
                "required": ["variant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_clinvar",
            "description": "为单个变异查询 ClinVar 数据库（胚系致病性 / 体细胞致癌性线索）。已查过的 variant_id+provider 会去重。",
            "parameters": {
                "type": "object",
                "properties": {
                    "variant_id": {"type": "string", "description": "要查询的 variant_id"},
                },
                "required": ["variant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_civic",
            "description": "为单个变异查询 CIViC 数据库（临床证据，含证据等级与疾病匹配）。必查库之一。",
            "parameters": {
                "type": "object",
                "properties": {
                    "variant_id": {"type": "string", "description": "要查询的 variant_id"},
                },
                "required": ["variant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_myvariant",
            "description": "为单个变异查询 MyVariant.info 聚合注释（群体频率、功能预测、rsID）。必查库之一，每个可检索变异都必须调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "variant_id": {"type": "string", "description": "要查询的 variant_id"},
                },
                "required": ["variant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_gnomad",
            "description": (
                "为单个变异查询 gnomAD 群体频率（AF/AC/AN，用于罕见度与 ACMG 频率证据）。"
                "必查库之一；需已解析出 SNV 基因组坐标，否则返回空（空结果也算已覆盖，仍必须调用一次）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "variant_id": {"type": "string", "description": "要查询的 variant_id"},
                },
                "required": ["variant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_vep",
            "description": (
                "为单个变异查询 Ensembl VEP 功能后果与计算预测（consequence/SIFT/PolyPhen/外显子）。"
                "必查库之一；需转录本+cDNA 或基因组 HGVS，否则返回空（空结果也算已覆盖，仍必须调用一次）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "variant_id": {"type": "string", "description": "要查询的 variant_id"},
                },
                "required": ["variant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_evidence_collection",
            "description": "证据收集完成后调用，声明可以进入建卡与安全门控。所有关注的变异必须已查过全部 5 个库（ClinVar、CIViC、MyVariant、gnomAD、EnsemblVEP）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "为什么认为证据已足够"},
                },
                "required": [],
            },
        },
    },
]

SKIP_STATUSES = {"insufficient", "conflicting"}

# Maps a search tool name to the canonical provider_name used in provider_status
# and the agent provider map. Keeps dedup accounting and dispatch consistent for
# the extended (non-required) sources.
_TOOL_TO_PROVIDER = {
    "search_clinvar": "ClinVar",
    "search_civic": "CIViC",
    "search_myvariant": "MyVariant",
    "search_gnomad": "gnomAD",
    "search_vep": "EnsemblVEP",
}


def _provider_key_for_tool(name: str) -> str:
    return _TOOL_TO_PROVIDER.get(name, "MyVariant")


def status_for_molecular_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    labels = {
        "resolve_variant_coordinates": "补全变异坐标中…",
        "search_clinvar": "查询 ClinVar 中…",
        "search_civic": "查询 CIViC 中…",
        "search_myvariant": "查询 MyVariant 注释中…",
        "search_gnomad": "查询 gnomAD 群体频率中…",
        "search_vep": "查询 Ensembl VEP 功能注释中…",
        "finish_evidence_collection": "完成证据收集…",
    }
    label = labels.get(name, f"执行 {name} 中…")
    variant = arguments.get("variant_id") or ""
    detail: Dict[str, Any] = {"tool": name, "arguments": arguments}
    if variant:
        detail["variant_id"] = variant
    return {"type": "status", "stage": "molecular_tool", "label": label, "detail": detail}


@dataclass
class MolecularAgentState:
    steps: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    finished: bool = False


def _summarize_variants(variants: list) -> str:
    lines: List[str] = []
    for v in variants:
        ann = getattr(v, "molecular_annotation", {}) or {}
        exhausted = getattr(v, "mapping_status", "") in SKIP_STATUSES and not getattr(v, "gene", None)
        hint = " ⚠ 不可检索（需先补基因/HGVS）" if exhausted else ""
        lines.append(
            f"- {v.variant_id}: {v.gene} {v.protein_hgvs or v.cdna_hgvs or v.genomic_hgvs or v.unresolved_position or '未明确位点'}"
            f" | disease={v.disease} sample={v.sample_type or 'unknown'}"
            f" | mapping={v.mapping_status} transcript={v.transcript or '-'} build={v.genome_build or '-'}"
            f" | warnings={'; '.join(v.warnings) if v.warnings else '无'}"
            f" | effect={ann.get('protein_effect') or '-'}"
            f"{hint}"
        )
    return "\n".join(lines) if lines else "(无变异)"


def _parse_tool_args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _extract_finish_payload(content: str) -> Optional[dict]:
    if not content:
        return None
    m = re.search(r"\{.*\}", content or "", re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def build_molecular_agent_messages(variants: list, question: Optional[str] = None) -> List[Dict[str, Any]]:
    skipped = [v.variant_id for v in variants if getattr(v, "mapping_status", "") in SKIP_STATUSES and not getattr(v, "gene", None)]
    skip_note = ""
    if skipped:
        skip_note = f"\n注意：以下变异不可检索（缺少基因与位点），请勿对它们调用任何 search 工具，直接在 finish 理由中说明：{', '.join(skipped)}。\n"
    cov_hint = f"需覆盖的必查对：{len(variants)} 变异 × {len(MolecularLLMAgent.REQUIRED_PROVIDERS)} 库 = {len(variants)*len(MolecularLLMAgent.REQUIRED_PROVIDERS)} 次查询。"
    user_block = (
        f"待分析变异（共 {len(variants)} 个）：\n{_summarize_variants(variants)}\n\n"
        + skip_note
        + cov_hint + "\n"
        + (f"用户补充问题：{question}\n\n" if question else "")
        + "请按需调用工具完成坐标补全与证据检索。完成后调用 finish_evidence_collection。"
    )
    return [
        {"role": "system", "content": MOLECULAR_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]


def _should_skip_search(variant: Any) -> Optional[str]:
    if not getattr(variant, "gene", None) and getattr(variant, "mapping_status", "") in {"insufficient", "ambiguous", "conflicting"}:
        return "variant缺少基因且位点不唯一，无法发起有意义的数据库查询；请在 finish 中提示补充标准 HGVS"
    return None


class MolecularLLMAgent:
    """True LLM agent: Qwen decides resolve/search order and when to finish."""

    REQUIRED_PROVIDERS = ("ClinVar", "CIViC", "MyVariant", "gnomAD", "EnsemblVEP")

    def __init__(
        self,
        *,
        qwen_client: Any,
        providers: List[Any],
        resolver: Optional[Any] = None,
        max_steps: int = 8,
        settings: Optional[Any] = None,
    ) -> None:
        self.qwen = qwen_client
        self.providers = providers
        self.resolver = resolver
        self.max_steps = max(1, int(max_steps))
        self.settings = settings
        self._provider_map: Dict[str, Any] = {p.provider_name: p for p in providers}
        self._provider_map_lower: Dict[str, Any] = {k.lower(): v for k, v in self._provider_map.items()}
        self._max_parallel = max(2, min(6, int(os.getenv("MOLECULAR_SEARCH_PARALLELISM", "4"))))
        self._tool_timeout_s = float(os.getenv("MOLECULAR_TOOL_TIMEOUT_S", "12"))

    def _required_coverage_complete(self, variants: list, queried_pairs: set) -> bool:
        for v in variants:
            if _should_skip_search(v):
                continue
            for req in self.REQUIRED_PROVIDERS:
                if (v.variant_id, req.lower()) not in queried_pairs:
                    return False
        return True

    def _coverage_hint(self, variants: list, queried_pairs: set) -> Optional[str]:
        uncovered = []
        for v in variants:
            if _should_skip_search(v):
                continue
            for req in self.REQUIRED_PROVIDERS:
                if (v.variant_id, req.lower()) not in queried_pairs:
                    uncovered.append(f"{v.gene or v.variant_id}/{req}")
        if not uncovered:
            return "所有必查库已覆盖，可调用 finish_evidence_collection 结束"
        if len(uncovered) <= 4:
            return "尚未覆盖： " + "；".join(uncovered)
        return f"尚有 {len(uncovered)} 个必查对未覆盖"

    # non-streaming entry (backwards compat)
    def run(
        self,
        variants: list,
        *,
        question: Optional[str] = None,
        trace: Optional[list] = None,
    ) -> tuple[list, list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
        events: list[dict[str, Any]] = []
        gen = self.run_streaming(variants, question=question, trace=trace)
        try:
            while True:
                ev = next(gen)
                events.append(ev)
        except StopIteration as stop:
            if stop.value is not None:
                return stop.value  # type: ignore[return-value]
        return events[-1].get("_final_tuple") if events else ([], trace or [], {}, {})  # type: ignore

    def run_streaming(
        self,
        variants: list,
        *,
        question: Optional[str] = None,
        trace: Optional[list] = None,
    ) -> Generator[Dict[str, Any], None, Tuple[list, list[dict[str, Any]], Dict[str, dict[str, Any]], Dict[str, Any]]]:
        trace = trace if trace is not None else []
        trace.append({
            "step": "plan",
            "agent": "llm",
            "required_providers": list(self.REQUIRED_PROVIDERS),
            "variant_count": len(variants),
            "policy": "llm_decides_tool_order_then_finish",
            "max_steps": self.max_steps,
            "max_parallel": self._max_parallel,
        })
        yield {"type": "status", "stage": "planning", "label": "分子证据智能体规划中…", "detail": {"variant_count": len(variants), "max_steps": self.max_steps}}

        if not variants:
            resolver_status: Dict[str, Any] = {"enabled": self.resolver is not None, "state": "no_variants", "mode": "llm_tool"}
            self._finalize_status({}, resolver_status, trace, degraded=False)
            return [], trace, {}, resolver_status  # type: ignore[return-value]

        if self.qwen is None or not getattr(self.qwen, "api_key", None):
            recs, fb_trace, fb_status, fb_resolver = self._linear_fallback(variants, trace, reason="agent_no_api_key")
            yield {"type": "status", "stage": "fallback", "label": "无 LLM Key，回退确定性检索…", "detail": {"reason": "agent_no_api_key"}}
            return recs, fb_trace, fb_status, fb_resolver  # type: ignore[return-value]

        variant_by_id: Dict[str, Any] = {v.variant_id: v for v in variants}
        resolver_status: Dict[str, Any] = {
            "enabled": self.resolver is not None,
            "resolver": getattr(self.resolver, "resolver_name", None) if self.resolver else None,
            "queried": False,
            "attempted_count": 0,
            "resolved_count": 0,
            "unresolved_count": 0,
            "state": "disabled" if self.resolver is None else "pending",
            "mode": "llm_tool",
        }
        provider_status: Dict[str, Dict[str, Any]] = {
            name: {"required": True, "queried": False, "query_count": 0, "record_count": 0, "error_count": 0, "state": "pending", "dedup_skipped": 0}
            for name in self.REQUIRED_PROVIDERS
        }
        for p in self.providers:
            if p.provider_name not in provider_status:
                provider_status[p.provider_name] = {
                    "required": False, "queried": False, "query_count": 0, "record_count": 0, "error_count": 0, "state": "pending", "dedup_skipped": 0
                }

        records: List[Any] = []
        seen_records: set[tuple[str, str]] = set()
        state = MolecularAgentState()
        messages = build_molecular_agent_messages(variants, question=question)
        queried_pairs: set[tuple[str, str]] = set()
        dedup_cache: Dict[tuple[str, str], str] = {}  # pair -> json result text
        total_tool_calls = 0

        for step_idx in range(self.max_steps):
            total_tool_calls += 1
            if total_tool_calls > self.max_steps * 6:
                trace.append({"step": "budget_guard", "at_step": step_idx + 1, "reason": "tool_call_budget_exceeded"})
                yield {"type": "status", "stage": "budget", "label": "工具调用已达上限，准备收敛…", "detail": {"max_steps": self.max_steps}}
                break

            # early-stop hint injection: if all required covered, nudge the model toward finish
            if step_idx > 0 and self._required_coverage_complete(variants, queried_pairs):
                hint = self._coverage_hint(variants, queried_pairs)
                messages.append({
                    "role": "system",
                    "content": f"[系统提示] {hint}；若已无额外问题请直接调用 finish_evidence_collection。",
                })

            assistant_msg, tool_calls, degraded = self.qwen.run_tool_turn(
                messages, tools=MOLECULAR_TOOL_DEFINITIONS, timeout=45
            )
            if degraded:
                trace.append({"step": "agent_degraded", "reason": degraded, "at_step": step_idx + 1})
                yield {"type": "status", "stage": "degraded", "label": f"智能体异常：{degraded}", "detail": {"reason": degraded}}
                self._supplement_missing_required(variants, records, seen_records, provider_status, queried_pairs, trace)
                self._finalize_status(provider_status, resolver_status, trace, degraded=True)
                return records, trace, provider_status, resolver_status  # type: ignore[return-value]

            messages.append(assistant_msg)
            content = str(assistant_msg.get("content") or "")
            finish_payload = _extract_finish_payload(content) if content and not tool_calls else None
            if finish_payload is not None and (finish_payload.get("ready") is True or "finish" in str(finish_payload).lower()):
                state.finished = True
                state.steps.append({"step": step_idx + 1, "type": "ready", "payload": finish_payload})
                trace.append({"step": "agent_ready", "at_step": step_idx + 1, "payload": finish_payload})
                yield {"type": "status", "stage": "ready", "label": "证据收集完成，准备生成报告…", "detail": finish_payload}
                break

            if not tool_calls:
                state.finished = True
                state.steps.append({"step": step_idx + 1, "type": "stop", "content": content[:400]})
                trace.append({"step": "agent_stop_without_tool", "at_step": step_idx + 1, "content": content[:400]})
                if content:
                    yield {"type": "status", "stage": "stop", "label": "模型结束工具调用", "detail": {"content": content[:200]}}
                break

            finish_calls = [c for c in tool_calls if (c.get("function") or {}).get("name") == "finish_evidence_collection"]
            if finish_calls:
                fn = finish_calls[0].get("function") or {}
                args = _parse_tool_args(fn.get("arguments"))
                yield status_for_molecular_tool("finish_evidence_collection", args)
                state.finished = True
                trace.append({"step": "agent_finish_tool", "at_step": step_idx + 1, "reason": args.get("reason")})
                messages.append({"role": "tool", "tool_call_id": finish_calls[0].get("id") or "finish", "name": "finish_evidence_collection", "content": json.dumps({"ok": True}, ensure_ascii=False)})
                break

            resolve_calls = [c for c in tool_calls if (c.get("function") or {}).get("name") == "resolve_variant_coordinates"]
            search_calls = [c for c in tool_calls if (c.get("function") or {}).get("name") in {"search_clinvar", "search_civic", "search_myvariant", "search_gnomad", "search_vep"}]
            other_calls = [c for c in tool_calls if c not in resolve_calls and c not in search_calls and c not in finish_calls]

            # dedup search calls inside the same turn (LLM may repeat)
            deduped_search: list = []
            for c in search_calls:
                fn = c.get("function") or {}
                name = str(fn.get("name") or "")
                args = _parse_tool_args(fn.get("arguments"))
                vid = str(args.get("variant_id") or "").strip()
                pair = (vid, name.lower())
                if pair in queried_pairs and pair in dedup_cache:
                    trace.append({"step": "dedup_skip", "tool": name, "variant_id": vid, "reason": "already_queried_this_run"})
                    provider_key = _provider_key_for_tool(name)
                    st = provider_status.get(provider_key)
                    if st is not None:
                        st["dedup_skipped"] = int(st.get("dedup_skipped", 0)) + 1
                    yield {"type": "tool_result", "tool": name, "arguments": args, "result": dedup_cache[pair], "stage": "search", "dedup": True}
                    messages.append({"role": "tool", "tool_call_id": c.get("id") or name, "name": name, "content": dedup_cache[pair]})
                    continue
                deduped_search.append(c)
            search_calls = deduped_search

            # prioritize urgent variants: protein_level_only + missing coords first
            def _resolve_priority(c):
                fn = c.get("function") or {}
                args = _parse_tool_args(fn.get("arguments"))
                v = variant_by_id.get(str(args.get("variant_id") or ""))
                if v is None:
                    return 9
                return 0 if getattr(v, "mapping_status", "") == "protein_level_only" else 1

            resolve_calls.sort(key=_resolve_priority)

            for call in resolve_calls:
                fn = call.get("function") or {}
                name = str(fn.get("name") or "")
                args = _parse_tool_args(fn.get("arguments"))
                yield status_for_molecular_tool(name, args)
                result_text, local_records = self._dispatch(
                    name, args, variant_by_id=variant_by_id, resolver_status=resolver_status, provider_status=provider_status,
                    queried_pairs=queried_pairs, trace=trace, step_idx=step_idx,
                )
                for rec in local_records:
                    key = (rec.provider, rec.provider_record_id)
                    is_error = rec.provider_record_id.endswith("_QUERY_ERROR") or str(rec.raw_response.get("record_status") or "").lower() == "error"
                    if is_error:
                        records.append(rec)
                        continue
                    if key in seen_records:
                        continue
                    seen_records.add(key)
                    records.append(rec)
                state.steps.append({"step": step_idx + 1, "tool": name, "arguments": args, "result_preview": result_text[:600]})
                trace.append({"step": "agent_tool_call", "at_step": step_idx + 1, "tool": name, "arguments": args, "result_preview": result_text[:900]})
                yield {"type": "tool_result", "tool": name, "arguments": args, "result": result_text[:1200], "stage": "resolve"}
                messages.append({"role": "tool", "tool_call_id": call.get("id") or name, "name": name, "content": result_text})
                if state.finished:
                    break
            if state.finished:
                break

            if search_calls:
                # sort: required providers first, then unique-mapping variants first, then skipped last
                def _search_priority(c):
                    fn = c.get("function") or {}
                    n = str(fn.get("name") or "").lower()
                    req = 0 if ("clinvar" in n or "civic" in n) else 1
                    args = _parse_tool_args(fn.get("arguments"))
                    v = variant_by_id.get(str(args.get("variant_id") or ""))
                    if v is not None and _should_skip_search(v):
                        return 9
                    if v is not None and getattr(v, "mapping_status", "") == "unique":
                        return req
                    return req + 1

                search_calls.sort(key=_search_priority)
                for call in search_calls:
                    fn = call.get("function") or {}
                    yield status_for_molecular_tool(str(fn.get("name") or ""), _parse_tool_args(fn.get("arguments")))

                search_results: list[tuple[dict, str, list, tuple[str, str]]] = []
                if len(search_calls) > 1:
                    with ThreadPoolExecutor(max_workers=min(self._max_parallel, len(search_calls))) as pool:
                        future_map = {}
                        for call in search_calls:
                            fn = call.get("function") or {}
                            name = str(fn.get("name") or "")
                            args = _parse_tool_args(fn.get("arguments"))
                            fut = pool.submit(
                                self._dispatch, name, args,
                                variant_by_id=variant_by_id, resolver_status=resolver_status,
                                provider_status=provider_status, queried_pairs=queried_pairs,
                                trace=trace, step_idx=step_idx,
                            )
                            future_map[fut] = (call, name, args)
                        for fut in as_completed(future_map):
                            call, name, args = future_map[fut]
                            try:
                                result_text, local_records = fut.result(timeout=self._tool_timeout_s + 2)
                            except Exception as exc:
                                result_text, local_records = json.dumps({"ok": False, "error": f"parallel_dispatch:{type(exc).__name__}: {exc}"}, ensure_ascii=False), []
                            vid = str(args.get("variant_id") or "").strip()
                            search_results.append((call, result_text, local_records, (vid, name.lower())))
                else:
                    for call in search_calls:
                        fn = call.get("function") or {}
                        name = str(fn.get("name") or "")
                        args = _parse_tool_args(fn.get("arguments"))
                        result_text, local_records = self._dispatch(
                            name, args, variant_by_id=variant_by_id, resolver_status=resolver_status, provider_status=provider_status,
                            queried_pairs=queried_pairs, trace=trace, step_idx=step_idx,
                        )
                        search_results.append((call, result_text, local_records, (str(args.get("variant_id") or "").strip(), name.lower())))

                for call, result_text, local_records, pair in search_results:
                    dedup_cache[pair] = result_text
                    fn = call.get("function") or {}
                    name = str(fn.get("name") or "")
                    args = _parse_tool_args(fn.get("arguments"))
                    for rec in local_records:
                        key = (rec.provider, rec.provider_record_id)
                        is_error = rec.provider_record_id.endswith("_QUERY_ERROR") or str(rec.raw_response.get("record_status") or "").lower() == "error"
                        if is_error:
                            records.append(rec)
                            continue
                        if key in seen_records:
                            continue
                        seen_records.add(key)
                        records.append(rec)
                    state.steps.append({"step": step_idx + 1, "tool": name, "arguments": args, "result_preview": result_text[:600]})
                    trace.append({"step": "agent_tool_call", "at_step": step_idx + 1, "tool": name, "arguments": args, "result_preview": result_text[:900]})
                    yield {"type": "tool_result", "tool": name, "arguments": args, "result": result_text[:1500], "stage": "search"}
                    messages.append({"role": "tool", "tool_call_id": call.get("id") or name, "name": name, "content": result_text})
                    # early-stop check mid-batch: if all required done and model queued extra MyVariant, let it still flush

            for call in other_calls:
                fn = call.get("function") or {}
                name = str(fn.get("name") or "")
                args = _parse_tool_args(fn.get("arguments"))
                yield status_for_molecular_tool(name, args)
                result_text, local_records = self._dispatch(
                    name, args, variant_by_id=variant_by_id, resolver_status=resolver_status, provider_status=provider_status,
                    queried_pairs=queried_pairs, trace=trace, step_idx=step_idx,
                )
                trace.append({"step": "agent_tool_call", "at_step": step_idx + 1, "tool": name, "arguments": args, "result_preview": result_text[:900]})
                yield {"type": "tool_result", "tool": name, "arguments": args, "result": result_text[:1200], "stage": "unknown"}
                messages.append({"role": "tool", "tool_call_id": call.get("id") or name, "name": name, "content": result_text})

            if state.finished:
                break

        # supplement only skippable variants are excluded
        needs_supplement = False
        for v in variants:
            if _should_skip_search(v):
                continue
            for req in self.REQUIRED_PROVIDERS:
                if (v.variant_id, req.lower()) not in queried_pairs:
                    needs_supplement = True
                    break
        if needs_supplement:
            yield {"type": "status", "stage": "supplement", "label": "补充必查库收敛中…", "detail": {"uncovered": self._coverage_hint(variants, queried_pairs)}}
        self._supplement_missing_required(variants, records, seen_records, provider_status, queried_pairs, trace)
        self._finalize_status(provider_status, resolver_status, trace, degraded=False)
        yield {"type": "status", "stage": "complete", "label": "检索完成，生成证据卡片中…", "detail": {"record_count": len(records), "dedup_skipped": sum(int(s.get("dedup_skipped", 0)) for s in provider_status.values())}}
        return records, trace, provider_status, resolver_status  # type: ignore[return-value]

    def _dispatch(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        variant_by_id: Dict[str, Any],
        resolver_status: Dict[str, Any],
        provider_status: Dict[str, Any],
        queried_pairs: set[tuple[str, str]],
        trace: list,
        step_idx: int,
    ) -> tuple[str, list]:
        if name == "resolve_variant_coordinates":
            return self._tool_resolve(args, variant_by_id=variant_by_id, resolver_status=resolver_status, trace=trace, step_idx=step_idx)
        if name == "search_clinvar":
            return self._tool_search(args, variant_by_id=variant_by_id, provider_name="ClinVar", provider_status=provider_status, queried_pairs=queried_pairs)
        if name == "search_civic":
            return self._tool_search(args, variant_by_id=variant_by_id, provider_name="CIViC", provider_status=provider_status, queried_pairs=queried_pairs)
        if name == "search_myvariant":
            return self._tool_search(args, variant_by_id=variant_by_id, provider_name="MyVariant", provider_status=provider_status, queried_pairs=queried_pairs)
        if name == "search_gnomad":
            return self._tool_search(args, variant_by_id=variant_by_id, provider_name="gnomAD", provider_status=provider_status, queried_pairs=queried_pairs)
        if name == "search_vep":
            return self._tool_search(args, variant_by_id=variant_by_id, provider_name="EnsemblVEP", provider_status=provider_status, queried_pairs=queried_pairs)
        if name == "finish_evidence_collection":
            return json.dumps({"ok": True, "message": "证据收集结束，进入建卡与门控"}, ensure_ascii=False), []
        return json.dumps({"ok": False, "error": f"unknown_tool:{name}"}, ensure_ascii=False), []

    def _tool_resolve(self, args: Dict[str, Any], *, variant_by_id: Dict[str, Any], resolver_status: Dict[str, Any], trace: list, step_idx: int) -> tuple[str, list]:
        variant_id = str(args.get("variant_id") or "").strip()
        variant = variant_by_id.get(variant_id)
        if variant is None:
            return json.dumps({"ok": False, "error": f"variant_id not found: {variant_id}", "known": list(variant_by_id.keys())}, ensure_ascii=False), []
        if self.resolver is None:
            return json.dumps({"ok": False, "error": "resolver not configured"}, ensure_ascii=False), []
        resolver_status["queried"] = True
        resolver_status["attempted_count"] += 1
        if not variant.gene:
            resolver_status["unresolved_count"] += 1
            return json.dumps({"ok": False, "error": "variant has no gene, nothing to resolve; 请在 finish 中提示补充基因与标准 HGVS"}, ensure_ascii=False), []
        resolved = None
        try:
            resolved = self.resolver.resolve(variant)
        except Exception as exc:
            resolver_status["unresolved_count"] += 1
            return json.dumps({"ok": False, "error": f"resolver exception: {type(exc).__name__}: {exc}"}, ensure_ascii=False), []
        if not resolved or resolved.get("error"):
            resolver_status["unresolved_count"] += 1
            msg = resolved.get("error") if isinstance(resolved, dict) and resolved.get("error") else "resolver returned no mapping"
            return json.dumps({"ok": False, "error": msg, "variant": f"{variant.gene} {variant.protein_hgvs or variant.cdna_hgvs or ''}"}, ensure_ascii=False), []
        for fld in ("protein_hgvs", "cdna_hgvs", "genomic_hgvs", "genome_build", "transcript"):
            val = resolved.get(fld)
            if val:
                setattr(variant, fld, str(val))
        resolved_fields = {fld: getattr(variant, fld) for fld in ("protein_hgvs", "cdna_hgvs", "genomic_hgvs", "genome_build", "transcript") if getattr(variant, fld)}
        # Carry the ClinGen CA ID through so downstream providers / audit can see
        # which canonical allele the writing was aligned to.
        ca_id = resolved.get("ca_id")
        if ca_id:
            resolved_fields["ca_id"] = str(ca_id)
            enrichment_extra: Dict[str, Any] = {"ca_id": str(ca_id)}
            if resolved.get("version"):
                enrichment_extra["ca_version"] = resolved.get("version")
        else:
            enrichment_extra = {}
        try:
            from backend.app.services.molecular_evidence import MappingValidator
            MappingValidator().validate(variant)
        except Exception:
            pass
        variant.enrichment = {
            "source": resolved.get("source") or getattr(self.resolver, "resolver_name", "unknown"),
            "query": resolved.get("query") or variant.raw_input,
            "resolved_fields": resolved_fields,
            "warnings": resolved.get("warnings") or [],
            **enrichment_extra,
        }
        resolver_status["resolved_count"] += 1
        trace.append({
            "step": "resolve_variant", "resolver": getattr(self.resolver, "resolver_name", "unknown"),
            "variant_id": variant_id, "status": "resolved", "mapping_status": variant.mapping_status,
            "resolved_fields": resolved_fields,
        })
        return json.dumps({
            "ok": True, "variant_id": variant_id, "mapping_status": variant.mapping_status,
            "resolved_fields": resolved_fields, "warnings": variant.warnings,
        }, ensure_ascii=False), []

    def _tool_search(self, args: Dict[str, Any], *, variant_by_id: Dict[str, Any], provider_name: str, provider_status: Dict[str, Any], queried_pairs: set[tuple[str, str]]) -> tuple[str, list]:
        variant_id = str(args.get("variant_id") or "").strip()
        variant = variant_by_id.get(variant_id)
        if variant is None:
            return json.dumps({"ok": False, "error": f"variant_id not found: {variant_id}"}, ensure_ascii=False), []
        skip_reason = _should_skip_search(variant)
        if skip_reason:
            return json.dumps({"ok": False, "skipped": True, "reason": skip_reason, "variant_id": variant_id}, ensure_ascii=False), []
        provider = self._provider_map.get(provider_name) or self._provider_map_lower.get(provider_name.lower())
        if provider is None:
            return json.dumps({"ok": False, "error": f"provider not configured: {provider_name}"}, ensure_ascii=False), []
        key = (variant_id, provider_name.lower())
        queried_pairs.add(key)
        st = provider_status.setdefault(provider_name, {"required": provider_name in self.REQUIRED_PROVIDERS, "queried": False, "query_count": 0, "record_count": 0, "error_count": 0, "state": "pending", "dedup_skipped": 0})
        st["queried"] = True
        st["query_count"] += 1
        recs: list = []
        last_exc: Optional[Exception] = None
        for attempt in range(2):
            try:
                recs = provider.search(variant)  # type: ignore
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                transient = any(k in msg for k in ("timeout", "timed out", "connection", "temporarily", "503", "502", "429"))
                if transient and attempt == 0:
                    time.sleep(0.6 * (attempt + 1))
                    continue
                break
        if last_exc is not None:
            st["error_count"] += 1
            return json.dumps({"ok": False, "error": f"{type(last_exc).__name__}: {last_exc}"}, ensure_ascii=False), []
        for r in recs:
            r.raw_response.setdefault("variant_id", variant_id)
        usable = sum(1 for r in recs if not (r.provider_record_id.endswith("_QUERY_ERROR") or str(r.raw_response.get("record_status") or "").lower() == "error"))
        errors = len(recs) - usable
        st["record_count"] += usable
        st["error_count"] += errors
        preview = []
        for r in recs[:3]:
            preview.append({
                "provider": r.provider, "id": r.provider_record_id,
                "title": str(r.raw_response.get("source_title") or r.raw_response.get("source_title_zh") or "")[:80],
                "level": r.raw_response.get("evidence_level"), "disease": r.raw_response.get("disease"),
            })
        return json.dumps({
            "ok": True, "variant_id": variant_id, "provider": provider_name,
            "record_count": usable, "error_count": errors, "preview": preview,
        }, ensure_ascii=False), recs

    def _supplement_missing_required(self, variants: list, records: list, seen: set, provider_status: Dict[str, Any], queried_pairs: set, trace: list) -> None:
        for v in variants:
            if _should_skip_search(v):
                continue
            for req in self.REQUIRED_PROVIDERS:
                if (v.variant_id, req.lower()) in queried_pairs:
                    continue
                provider = self._provider_map.get(req) or self._provider_map_lower.get(req.lower())
                if provider is None:
                    continue
                trace.append({"step": "supplement_required_query", "provider": req, "variant_id": v.variant_id, "reason": "llm_skipped_required_provider"})
                st = provider_status.setdefault(req, {"required": True, "queried": False, "query_count": 0, "record_count": 0, "error_count": 0, "state": "pending", "dedup_skipped": 0})
                st["queried"] = True
                st["query_count"] += 1
                recs: list = []
                try:
                    recs = provider.search(v)  # type: ignore
                except Exception as exc:
                    st["error_count"] += 1
                    trace.append({"step": "supplement_error", "provider": req, "variant_id": v.variant_id, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                for r in recs:
                    r.raw_response.setdefault("variant_id", v.variant_id)
                    is_error = r.provider_record_id.endswith("_QUERY_ERROR") or str(r.raw_response.get("record_status") or "").lower() == "error"
                    if is_error:
                        records.append(r)
                        st["error_count"] += 1
                        continue
                    key = (r.provider, r.provider_record_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(r)
                    st["record_count"] += 1
                queried_pairs.add((v.variant_id, req.lower()))

    def _finalize_status(self, provider_status: Dict[str, Any], resolver_status: Dict[str, Any], trace: list, *, degraded: bool) -> None:
        if resolver_status.get("queried"):
            resolver_status["state"] = "resolved" if resolver_status.get("resolved_count") else "unresolved"
        elif resolver_status.get("enabled"):
            resolver_status["state"] = "pending"
        for name, st in provider_status.items():
            if st.get("error_count"):
                st["state"] = "degraded"
            elif st.get("record_count", 0) == 0 and st.get("queried"):
                st["state"] = "no_evidence"
            elif st.get("queried"):
                st["state"] = "success"
            else:
                # skipped variants should not mark required as incomplete
                st["state"] = "pending" if not st.get("required") else "incomplete"
        trace.append({
            "step": "complete", "agent": "llm", "completion_state": "degraded" if degraded else "complete",
            "resolver": dict(resolver_status), "providers": {k: dict(v) for k, v in provider_status.items()},
        })

    def _is_error_record(self, record) -> bool:
        return (
            str(getattr(record, "provider_record_id", "")).endswith("_QUERY_ERROR")
            or str((getattr(record, "raw_response", None) or {}).get("record_status") or "").lower() == "error"
            or str((getattr(record, "raw_response", None) or {}).get("evidence_type") or "").lower() == "query_error"
        )

    def _linear_fallback(self, variants: list, trace: list, *, reason: str) -> tuple[list, list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
        from backend.app.services.molecular_evidence import MolecularEvidenceAgent
        trace.append({"step": "llm_fallback_to_deterministic", "reason": reason})
        fallback = MolecularEvidenceAgent(provider_mode="deterministic_fallback", providers=self.providers, resolver=self.resolver)
        recs, fb_trace, fb_status, fb_resolver = fallback.run(variants)
        for entry in fb_trace:
            if entry.get("step") == "plan":
                continue
            trace.append(entry)
        trace.append({"step": "fallback_complete", "reason": reason})
        return recs, trace, fb_status, fb_resolver

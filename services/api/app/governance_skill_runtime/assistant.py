"""Deterministic planning and narration helpers for the Governance assistant."""

from __future__ import annotations

import re
from typing import Any, Literal, cast

from ..governance_skills_schemas import (
    AnswerMode,
    AssistantDispatchRequest,
    DispatchIntent,
    ReadOnlySkillName,
)
from .safety import _contains_sensitive_text, _llm_summary

_RELATION_MODALITY_LABELS = {
    "coRT": "协同转发",
    "coURL": "共链传播",
    "hashSeq": "话题序列",
    "fastRT": "快速转发",
    "tweetSim": "内容相似",
    "fused": "综合关系",
}
_AnswerSkillCall = tuple[ReadOnlySkillName, dict[str, Any], str]


def _deterministic_dispatch_intent(
    message: str,
) -> tuple[DispatchIntent, Literal["confirmed", "rejected", "pending"] | None]:
    normalized = re.sub(r"\s+", " ", message.strip().lower())
    if (
        "?" in normalized
        or "？" in normalized
        or any(
            marker in normalized
            for marker in (
                "如何",
                "怎么",
                "什么",
                "哪些",
                "为什么",
                "为何",
                "是否",
                "能否",
                "可否",
                "需要准备",
                "需要什么",
                "什么条件",
            )
        )
        or re.search(r"\b(?:how|what|why|when|where|which|can|could|should|would)\b", normalized)
    ):
        return "answer", None
    command = normalized.rstrip("。！!").strip()
    decision: Literal["confirmed", "rejected", "pending"] | None = None
    if any(value in normalized for value in ("驳回", "误报", "reject", "false positive")):
        decision = "rejected"
    elif any(value in normalized for value in ("待定", "待复核", "pending")):
        decision = "pending"
    elif any(value in normalized for value in ("确认", "属实", "通过复核", "approve", "confirmed")):
        decision = "confirmed"
    if re.fullmatch(
        r"(?:请)?(?:提交|确认|驳回|设为待定|标记待定)(?:人工)?复核(?:结论)?",
        command,
    ) or re.fullmatch(r"(?:submit|confirm|reject) review(?: decision)?", command):
        return "submit_review", decision
    if re.fullmatch(
        r"(?:请)?(?:生成研判(?:草稿)?|生成报告(?:草稿)?|报告草稿|简报草稿)", command
    ) or re.fullmatch(r"(?:please )?draft report", command):
        return "draft_report", decision
    if re.fullmatch(
        r"(?:请)?(?:打开|进入|查看)(?:人工)?复核(?:工作区|面板)?", command
    ) or re.fullmatch(r"(?:please )?(?:open|enter|view) review(?: panel| workspace)?", command):
        return "open_review", decision
    if re.fullmatch(
        r"(?:请)?(?:开始分析|开始治理|运行分析|执行分析)", command
    ) or re.fullmatch(r"(?:please )?(?:start|run) analysis", command):
        return "start_analysis", decision
    return "answer", decision


def _deterministic_answer_mode(message: str) -> AnswerMode:
    normalized = re.sub(r"\s+", " ", message.strip().lower())
    if any(value in normalized for value in ("知识", "资料", "知识库", "文档", "knowledge")):
        return "knowledge"
    if any(
        value in normalized
        for value in ("适用范围", "方法", "算法", "模型限制", "局限", "global", "method", "scope")
    ):
        return "method_scope"
    if "复核" in normalized and any(
        value in normalized for value in ("下一步", "如何", "怎么", "步骤", "流程", "建议")
    ):
        return "review_guidance"
    if any(
        value in normalized
        for value in ("证据", "核对哪些", "关系和邻域", "关系与邻域", "邻域信息", "evidence")
    ):
        return "evidence_requirements"
    if any(value in normalized for value in ("概括", "概览", "总结", "摘要", "整体", "风险")):
        return "overview"
    return "knowledge"


def _context_items(context: dict[str, Any], key: str, limit: int = 3) -> list[dict[str, Any]]:
    value = context.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value[:limit] if isinstance(item, dict)]


def _display_label(value: Any, fallback: str) -> str:
    label = str(value or fallback).strip()[:64]
    anonymous = re.fullmatch(r"Anonymous account\s+(.+)", label, flags=re.IGNORECASE)
    account = re.fullmatch(r"Account\s+(.+)", label, flags=re.IGNORECASE)
    if anonymous:
        return f"匿名账号 {anonymous.group(1)}"
    if account:
        return f"账号 {account.group(1)}"
    return label


def _percentage(value: Any) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else "待核验"


def _modality_label(value: Any) -> str:
    normalized = str(value).strip()
    return _RELATION_MODALITY_LABELS.get(normalized, normalized[:40])


def _target_label(value: Any) -> str:
    normalized = str(value).strip()[:80]
    russia = re.fullmatch(r"russia:(.+)", normalized, flags=re.IGNORECASE)
    return f"匿名账号 {russia.group(1)}" if russia else normalized


def _risk_band_label(value: Any) -> str:
    return {
        "high": "高风险候选",
        "review": "建议复核",
        "low": "低风险参照",
    }.get(str(value).strip(), "待核验")


def _group_label(value: Any) -> str:
    normalized = str(value or "").strip()[:80]
    match = re.fullmatch(r"group-(\d+)", normalized, flags=re.IGNORECASE)
    return f"协同群组 {int(match.group(1)):02d}" if match else normalized or "未命名群组"


def _case_state_label(value: Any) -> str:
    return {
        "draft": "草稿",
        "active": "研判中",
        "concluded": "已审结",
        "archived": "已归档",
    }.get(str(value).strip(), "待核验")


def _target_type_label(value: Any) -> str:
    return {"node": "账号", "relation": "关系", "group": "群组"}.get(
        str(value).strip(), "对象"
    )


def _decision_label(value: Any) -> str:
    return {"confirmed": "确认", "rejected": "驳回", "pending": "待定"}.get(
        str(value).strip(), "待定"
    )


def _relation_summary(item: dict[str, Any]) -> str:
    node_ids = item.get("nodeIds")
    if isinstance(node_ids, list) and len(node_ids) >= 2:
        endpoints = f"{_target_label(node_ids[0])} ↔ {_target_label(node_ids[1])}"
    else:
        endpoints = str(item.get("id") or item.get("relationId") or "未命名关系")[:80]
    modalities = item.get("modalities")
    modality_text = (
        "、".join(_modality_label(value) for value in modalities[:5])
        if isinstance(modalities, list)
        else ""
    )
    detail = f" · 登记模态 {modality_text}" if modality_text else ""
    priority = item.get("priority")
    if isinstance(priority, (int, float)):
        detail += f" · 排序信号 {_percentage(priority)}"
    return endpoints + detail


def _knowledge_excerpt(item: dict[str, Any]) -> str | None:
    text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
    if not text or _contains_sensitive_text(text):
        return None
    return text[:180]


def _inspection_answer_context(value: dict[str, Any]) -> dict[str, Any]:
    bounded = dict(value)
    relation_counts = value.get("relationCounts")
    if isinstance(relation_counts, dict):
        valid_counts = {
            str(key): count
            for key, count in relation_counts.items()
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0
        }
        bounded["relationCounts"] = valid_counts
        bounded["relationRecordCount"] = sum(valid_counts.values())
        bounded["modalities"] = [key for key, count in valid_counts.items() if count > 0]
    return bounded


def _numeric_facts(value: str) -> set[str]:
    without_ordered_list_markers = re.sub(
        r"(?m)^\s{0,3}\d{1,3}[.)、]\s+", "", value
    )
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", without_ordered_list_markers))


def _case_answer_context(case: Any, selected_target: Any = None) -> dict[str, Any]:
    case_items = tuple(getattr(case, "items", ()))
    items = [
        {
            "targetType": str(getattr(item, "target_type", "unknown")),
            "targetId": str(getattr(item, "target_id", ""))[:300],
            "note": str(getattr(item, "note", ""))[:500],
            "itemHash": str(getattr(item, "item_hash", "")),
        }
        for item in case_items[:10]
    ]
    review_events: list[dict[str, Any]] = []
    for event in tuple(getattr(case, "review_events", ()))[-3:]:
        created_at = getattr(event, "created_at", None)
        event_context: dict[str, Any] = {
            "targetType": str(getattr(event, "target_type", "unknown")),
            "targetId": str(getattr(event, "target_id", ""))[:300],
            "decision": str(getattr(event, "decision", "pending")),
            "reason": str(getattr(event, "reason", ""))[:500],
            "actor": str(getattr(event, "actor", ""))[:100],
            "sequence": getattr(event, "sequence", None),
            "eventHash": str(getattr(event, "event_hash", "")),
        }
        if created_at is not None:
            event_context["createdAt"] = (
                created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
            )[:80]
        review_events.append(event_context)
    current_decisions = getattr(case, "current_decisions", {})
    bounded_decisions = (
        dict(sorted(current_decisions.items())[:50])
        if isinstance(current_decisions, dict)
        else {}
    )
    decision_counts = {
        decision: sum(1 for value in bounded_decisions.values() if value == decision)
        for decision in ("confirmed", "rejected", "pending")
    }
    selected_review: dict[str, Any] | None = None
    if selected_target is not None:
        target_type = str(getattr(selected_target, "target_type", ""))
        target_id = str(getattr(selected_target, "target_id", ""))[:300]
        selected_review = {
            "targetType": target_type,
            "targetId": target_id,
            "decision": bounded_decisions.get(f"{target_type}:{target_id}"),
        }
    payload = {
        "caseId": str(case.case_id),
        "runId": str(case.run_id),
        "title": str(getattr(case, "title", ""))[:300],
        "description": str(getattr(case, "description", ""))[:500],
        "state": str(case.state),
        "items": items,
        "reviewEvents": review_events,
        "currentDecisions": bounded_decisions,
        "reviewProgress": {
            "registeredCount": len(case_items),
            "reviewedCount": len(bounded_decisions),
            "confirmedCount": decision_counts["confirmed"],
            "rejectedCount": decision_counts["rejected"],
            "pendingCount": decision_counts["pending"],
            "latestReviews": list(reversed(review_events)),
            "selectedTarget": selected_review,
        },
        "caseHash": str(case.case_hash),
    }
    safe = _llm_summary(payload)
    return safe if isinstance(safe, dict) else {}


def _review_progress_lines(case_context: dict[str, Any]) -> list[str]:
    progress_value = case_context.get("reviewProgress")
    progress = cast(dict[str, Any], progress_value) if isinstance(progress_value, dict) else {}
    if not progress:
        return []
    lines = [
        "### 人工复核进展",
        "- 已登记 {registered} · 已复核 {reviewed} · 确认 {confirmed} · 驳回 {rejected} · 待定 {pending}".format(
            registered=progress.get("registeredCount", 0),
            reviewed=progress.get("reviewedCount", 0),
            confirmed=progress.get("confirmedCount", 0),
            rejected=progress.get("rejectedCount", 0),
            pending=progress.get("pendingCount", 0),
        ),
    ]
    selected_value = progress.get("selectedTarget")
    selected = cast(dict[str, Any], selected_value) if isinstance(selected_value, dict) else {}
    if selected:
        decision = selected.get("decision")
        decision_text = _decision_label(decision) if decision is not None else "尚未复核"
        lines.append(
            f"- 当前{_target_type_label(selected.get('targetType'))}："
            f"{_target_label(selected.get('targetId') or '待核验')[:24]} · {decision_text}"
        )
    latest_value = progress.get("latestReviews")
    latest = latest_value if isinstance(latest_value, list) else []
    for event in latest[:3]:
        if not isinstance(event, dict):
            continue
        created_at = str(event.get("createdAt") or "").strip()
        timestamp = f" · {created_at[:16]}" if created_at else ""
        lines.append(
            f"- {_decision_label(event.get('decision'))} · "
            f"{_target_label(event.get('targetId') or '待核验')[:20]} · "
            f"{str(event.get('reason') or '未填写理由')[:28]}{timestamp}"
        )
    return lines


def _compact_relation_summary(item: dict[str, Any], *, potential: bool = False) -> str:
    node_ids = item.get("nodeIds")
    if isinstance(node_ids, list) and len(node_ids) >= 2:
        endpoints = (
            f"{_target_label(node_ids[0])[:22]} ↔ {_target_label(node_ids[1])[:22]}"
        )
    else:
        endpoints = str(item.get("id") or item.get("relationId") or "未命名关系")[:46]
    modalities = item.get("modalities")
    modality_text = ""
    if isinstance(modalities, list) and modalities:
        modality_text = " · " + "、".join(_modality_label(value) for value in modalities[:2])[:28]
    priority = item.get("priority")
    priority_text = (
        f" · 复核优先级 {_percentage(priority)}"
        if isinstance(priority, (int, float))
        else ""
    )
    prefix = "非事实边 · " if potential else ""
    return prefix + endpoints + modality_text + priority_text


def _group_relation_types(
    group: dict[str, Any], factual_relations: list[dict[str, Any]]
) -> str:
    modalities = group.get("modalities")
    values = list(modalities[:5]) if isinstance(modalities, list) else []
    group_nodes_value = group.get("nodeIds")
    group_nodes = set(group_nodes_value) if isinstance(group_nodes_value, list) else set()
    if group_nodes:
        for relation in factual_relations:
            relation_nodes = relation.get("nodeIds")
            if not isinstance(relation_nodes, list) or not set(relation_nodes).issubset(group_nodes):
                continue
            relation_modalities = relation.get("modalities")
            if isinstance(relation_modalities, list):
                values.extend(relation_modalities)
    unique = list(dict.fromkeys(str(value) for value in values if str(value).strip()))
    if not unique:
        return "待结合群组内部事实边核对"
    return "、".join(_modality_label(value) for value in unique[:3])[:36]


def _detailed_analysis_fallback(mode: AnswerMode, context: dict[str, Any]) -> str:
    inspection = cast(dict[str, Any], context.get("inspection") or {})
    candidates = [
        item
        for item in cast(list[Any], inspection.get("topCandidates") or [])[:5]
        if isinstance(item, dict) and item.get("riskBand") in {"high", "review"}
    ]
    groups = _context_items(context, "groups", 3)
    factual_relations = _context_items(context, "factualRelations", 3)
    potential_relations = _context_items(context, "potentialRelations", 2)
    case_context = cast(dict[str, Any], context.get("case") or {})
    heading = "全局态势报告" if mode == "analysis_summary" else "群组与关系研判报告"
    lines = [
        f"## {heading}",
        "以下内容来自已绑定模型排序、群组派生结果和关系记录；排序用于安排复核顺序，不代表人工结论。",
    ]
    if case_context:
        lines.extend(["", *_review_progress_lines(case_context)])

    candidate_lines = ["### 重点候选账号"]
    if candidates:
        for index, item in enumerate(candidates, start=1):
            rank = item.get("rank") if isinstance(item.get("rank"), int) else index
            label = _display_label(item.get("label") or item.get("nodeId"), "未命名账号")[:28]
            group = _group_label(item.get("communityId"))[:22]
            candidate_lines.append(
                f"- 原模型排名 {rank} · {label} · {_risk_band_label(item.get('riskBand'))} · {group}"
            )
    else:
        candidate_lines.append("- 当前未取得已校验的高风险或建议复核账号。")

    group_lines = ["### 重点风险群组"]
    if groups:
        for item in groups[:3]:
            priority = item.get("priority")
            priority_text = (
                _percentage(priority) if isinstance(priority, (int, float)) else "待核验"
            )
            group_lines.append(
                f"- {_group_label(item.get('groupId') or item.get('id'))[:22]} · "
                f"成员 {item.get('memberCount', '待核验')} · "
                f"关系类型 {_group_relation_types(item, factual_relations)} · "
                f"复核优先级 {priority_text}"
            )
    else:
        group_lines.append("- 当前未取得已校验的群组派生结果。")

    factual_lines = ["### 重点事实关系"]
    factual_lines.extend(
        f"- {_compact_relation_summary(item)}" for item in factual_relations[:3]
    )
    if not factual_relations:
        factual_lines.append("- 当前未取得已校验的事实关系。")

    potential_lines = ["### 待核验潜在线索"]
    potential_lines.extend(
        f"- {_compact_relation_summary(item, potential=True)}"
        for item in potential_relations[:2]
    )
    if not potential_relations:
        potential_lines.append("- 当前未取得已校验的潜在线索；不得以缺失结果推定不存在关联。")

    ordered_sections = (
        (candidate_lines, group_lines, factual_lines, potential_lines)
        if mode == "analysis_summary"
        else (group_lines, factual_lines, potential_lines, candidate_lines)
    )
    for section in ordered_sections:
        lines.extend(["", *section])
    lines.extend(
        [
            "",
            "### 人工复核建议",
            "优先核对事实关系的两端账号、关系模态、原始权重与证据哈希；潜在线索仅用于扩展核查，并补充发布时间、原帖内容、采集来源和反向证据。",
        ]
    )
    return "\n".join(lines)[:1_500]


def _answer_fallback(mode: AnswerMode, context: dict[str, Any]) -> str:
    inspection = cast(dict[str, Any], context.get("inspection") or {})
    candidates = [
        item
        for item in cast(list[Any], inspection.get("topCandidates") or [])[:5]
        if isinstance(item, dict)
    ]
    factual_relations = _context_items(context, "factualRelations")
    potential_relations = _context_items(context, "potentialRelations")
    evidence = cast(dict[str, Any], context.get("evidence") or {})
    cards = cast(dict[str, Any], context.get("cards") or {})
    knowledge = _context_items(context, "knowledge")
    case_context = cast(dict[str, Any], context.get("case") or {})

    if mode == "overview":
        lines = ["## 图谱基本情况"]
        if inspection:
            relation_counts = cast(dict[str, Any], inspection.get("relationCounts") or {})
            actual_relations = [
                (str(modality), count)
                for modality, count in relation_counts.items()
                if isinstance(count, int) and not isinstance(count, bool) and count > 0
            ]
            relation_record_count = sum(count for _modality, count in actual_relations)
            modality_counts = "、".join(
                f"{_modality_label(modality)}（{modality}）{count} 条"
                for modality, count in actual_relations
            )
            modality_names = "、".join(
                f"{_modality_label(modality)}（{modality}）"
                for modality, _count in actual_relations
            )
            lines.append(
                f"**账号规模** {inspection.get('nodeCount', 0)} 个账号。"
            )
            lines.append(
                f"**事实关系记录** 共 {relation_record_count} 条"
                + (f"：{modality_counts}。" if modality_counts else "。")
            )
            lines.append(
                f"**融合去重关系** {inspection.get('fusedEdgeCount', 0)} 条；"
                "同一账号对跨模态出现时在融合图中只计一条。"
            )
            lines.append(
                "**关系类型** " + (modality_names if modality_names else "未发现已登记关系类型") + "。"
            )
            lines.append(
                f"**连通情况** {inspection.get('componentCount', 0)} 个连通分量，"
                f"{inspection.get('isolateCount', 0)} 个孤立账号。"
            )
        else:
            lines.append("当前未取得图级检查结果，请核对会话中的图谱绑定。")
        return "\n".join(lines)[:700]

    if mode in {"analysis_summary", "coordination_summary"}:
        return _detailed_analysis_fallback(mode, context)

    if mode == "evidence_requirements":
        lines = [
            "## 证据核对要求",
            "当前证据可核对关系两端账号、关系模态、原始权重（rawWeight）与证据哈希；"
            "发布时间、原帖内容和采集来源尚未提供，需另行补充。",
        ]
        node_value = evidence.get("node")
        node = cast(dict[str, Any], node_value) if isinstance(node_value, dict) else {}
        signals_value = evidence.get("structuralSignals")
        signals = (
            cast(dict[str, Any], signals_value) if isinstance(signals_value, dict) else {}
        )
        if node:
            lines.append(
                f"\n**当前账号** {_display_label(node.get('label') or node.get('nodeId'), '未命名账号')} · "
                f"排序信号 {_percentage(node.get('score'))} · 一跳邻居 {signals.get('fusedDegree', 0)} · "
                f"两跳范围 {signals.get('twoHopNodeCount', 0)}"
            )
            neighbor_counts = signals.get("relationNeighborCounts")
            if isinstance(neighbor_counts, dict):
                present = [
                    f"{_modality_label(key)} {value}"
                    for key, value in neighbor_counts.items()
                    if value
                ]
                if present:
                    lines.append("**直接关系计数** " + " · ".join(present[:5]))
            if evidence.get("truncated") is True:
                lines.append("**覆盖提示** 证据子图已截断，不能据此断言完整邻域。")
        else:
            lines.append("\n尚未选择账号；请先在图中选中候选，再读取其直接关系与邻域摘要。")
        if case_context:
            lines.extend(["", *_review_progress_lines(case_context)])
        if factual_relations:
            lines.append("\n**优先核对的事实关系**")
            lines.extend(f"- {_relation_summary(item)}" for item in factual_relations[:2])
        if potential_relations:
            lines.append("\n**待验证潜在线索（不得当作事实）**")
            lines.extend(f"- {_relation_summary(item)}" for item in potential_relations[:2])
        lines.append("\n核对后记录支持证据、反向证据和缺口；证据不足时保持待定。")
        return "\n".join(lines)[: (1_100 if case_context else 700)]

    if mode == "review_guidance":
        lines = [
            "## 人工复核步骤",
            "1. 核对事实边的两端账号、关系模态、rawWeight 与证据哈希。",
            "2. 把群组和相似性结果仅作为扩展核查线索，不作事实认定。",
            "3. 补查当前缺少的发布时间、原帖内容与采集来源，并记录反向证据。",
            "4. 在研判单中填写确认、驳回或待定及理由；提交前不会保存结论。",
        ]
        if candidates:
            first = candidates[0]
            lines.append(
                f"\n建议从 **{_display_label(first.get('label') or first.get('nodeId'), '首位候选')}** 开始，"
                f"其当前排序信号为 {_percentage(first.get('score'))}。"
            )
        if evidence.get("truncated") is True:
            lines.append("当前证据子图已截断，复核时应继续查询原始关系记录。")
        return "\n".join(lines)[:700]

    if mode == "case_draft":
        lines = [
            "## 人工研判草稿",
            "本草稿汇总研判单记录、模型排序和关系线索；各层信息保持分离，不新增人工结论。",
        ]
        title = case_context.get("title") or "当前研判单"
        lines.append(
            f"\n**研判单** {str(title)[:100]} · 状态 {_case_state_label(case_context.get('state'))}"
        )
        description = str(case_context.get("description") or "").strip()
        if description:
            lines.append(f"**登记说明** {description[:100]}")
        case_items = _context_items(case_context, "items", 2)
        lines.append("\n### 已登记研判对象")
        if case_items:
            for item in case_items:
                note = str(item.get("note") or "").strip()
                suffix = f" · {note[:60]}" if note else ""
                lines.append(
                    f"- {_target_type_label(item.get('targetType'))} · "
                    f"{_target_label(item.get('targetId') or '待核验')}{suffix}"
                )
        else:
            lines.append("- 当前研判单尚未登记治理对象。")
        if candidates:
            lines.append("\n### 模型排序发现")
            for item in candidates[:1]:
                label = _display_label(item.get("label") or item.get("nodeId"), "未命名账号")
                lines.append(f"- **{label}** · 排序信号 {_percentage(item.get('score'))}")
        if factual_relations:
            lines.append("\n### 已登记事实关系")
            lines.extend(f"- {_relation_summary(item)}" for item in factual_relations[:1])
        if potential_relations:
            lines.append("\n### 派生潜在线索（非事实边）")
            lines.extend(f"- {_relation_summary(item)}" for item in potential_relations[:1])
        lines.extend(["", *_review_progress_lines(case_context)])
        tail = "待补充：逐项核对直接关系来源与反向证据。本回答未保存草稿或修改研判单。"
        body = "\n".join(lines)
        return f"{body[: 1_499 - len(tail)].rstrip()}\n{tail}"

    model_card = cast(dict[str, Any], cards.get("modelCard") or {})
    dataset_card = cast(dict[str, Any], cards.get("datasetCard") or {})
    input_card = cast(dict[str, Any], cards.get("inputContractCard") or {})
    if mode == "method_scope":
        lines = [
            "## 方法与适用范围",
            "当前图基础模型输出是跨关系模态的排序信号，用于安排研判顺序，不是违法认定、意图证明或处罚依据。",
        ]
        model_id = model_card.get("modelVersionId") or model_card.get("method")
        if model_id:
            lines.append(f"\n**当前模型** {str(model_id)[:120]}")
        dataset_name = dataset_card.get("displayName") or dataset_card.get("datasetId")
        if dataset_name:
            lines.append(
                f"**当前数据** {str(dataset_name)[:80]} · 账号 {dataset_card.get('nodeCount', '待核验')} · "
                f"关系记录 {dataset_card.get('relationRowCount', '待核验')}"
            )
        modalities = input_card.get("modalities") or dataset_card.get("modalities")
        if isinstance(modalities, list):
            lines.append(
                "**支持关系模态** "
                + "、".join(_modality_label(item) for item in modalities[:5])
            )
        if input_card.get("labelsSplitsScoresAccepted") is False:
            lines.append("**输入边界** 上传包不接收外部标签、划分或预置分数。")
        limitations = model_card.get("limitations")
        if isinstance(limitations, list):
            lines.extend(f"- {str(item)[:180]}" for item in limitations[:2])
        lines.append("\n适用性需在图版本、模型版本或关系模态变化后重新核对。")
        return "\n".join(lines)[:1_200]

    lines = [
        "## 知识说明",
        "回答依据已登记模型卡、数据卡、输入合同与本地知识索引；资料只提供解释边界，不会创建运行或保存报告。",
    ]
    if knowledge:
        lines.append("\n### 匹配资料")
        for item in knowledge[:3]:
            label = _display_label(item.get("sourceLabel"), "已登记资料")
            excerpt = _knowledge_excerpt(item)
            lines.append(f"- **{label}**" + (f"：{excerpt}" if excerpt else "：内容因安全边界未发送给叙述模型。"))
    else:
        lines.append("\n当前问题未命中可用知识片段，以下仅依据已登记卡片说明。")
    if model_card:
        model_id = model_card.get("modelVersionId") or model_card.get("method") or "当前发布模型"
        lines.append(f"- 模型：{str(model_id)[:100]}")
    if dataset_card:
        name = dataset_card.get("displayName") or dataset_card.get("datasetId") or "当前绑定数据"
        lines.append(f"- 数据：{str(name)[:100]}")
    return "\n".join(lines)[:1_200]


def _answer_skill_plan(
    request: AssistantDispatchRequest,
    mode: AnswerMode,
    *,
    run_id_override: str | None = None,
    node_id_override: str | None = None,
) -> tuple[_AnswerSkillCall, ...]:
    run_id = run_id_override if run_id_override is not None else request.context.run_id
    inspect_params: dict[str, Any] = {"candidateLimit": 5}
    if run_id is not None:
        inspect_params["runId"] = run_id
    inspect: _AnswerSkillCall = ("inspect_graph", inspect_params, "inspection")
    cards: _AnswerSkillCall = ("get_model_dataset_cards", {}, "cards")
    if mode == "overview":
        return (inspect,)
    if mode in {"analysis_summary", "coordination_summary"}:
        if run_id is None:
            return (inspect,)
        return (
            inspect,
            (
                "discover_coordination_groups",
                {"runId": run_id, "offset": 0, "limit": 3},
                "groups",
            ),
            (
                "rank_coordination_relations",
                {
                    "runId": run_id,
                    "offset": 0,
                    "limit": 3,
                    "relationKind": "factual",
                    "modalities": [],
                },
                "factualRelations",
            ),
            (
                "rank_coordination_relations",
                {
                    "runId": run_id,
                    "offset": 0,
                    "limit": 3,
                    "relationKind": "potential",
                    "modalities": [],
                },
                "potentialRelations",
            ),
        )
    if mode in {"evidence_requirements", "case_draft"}:
        calls: list[_AnswerSkillCall] = [inspect]
        target = request.context.selected_target
        target_node_id = node_id_override
        if (
            mode == "evidence_requirements"
            and target_node_id is None
            and target is not None
            and target.target_type == "node"
        ):
            target_node_id = target.target_id
        if run_id is not None and target_node_id is not None:
            calls.append(
                ("get_evidence_subgraph", {"runId": run_id, "nodeId": target_node_id}, "evidence")
            )
        if run_id is not None and len(calls) < 4:
            if target_node_id is None:
                calls.append(
                    (
                        "discover_coordination_groups",
                        {"runId": run_id, "offset": 0, "limit": 3},
                        "groups",
                    )
                )
            calls.append(
                (
                    "rank_coordination_relations",
                    {
                        "runId": run_id,
                        "offset": 0,
                        "limit": 3,
                        "relationKind": "factual",
                        "modalities": [],
                    },
                    "factualRelations",
                )
            )
            calls.append(
                (
                    "rank_coordination_relations",
                    {
                        "runId": run_id,
                        "offset": 0,
                        "limit": 3,
                        "relationKind": "potential",
                        "modalities": [],
                    },
                    "potentialRelations",
                )
            )
        return tuple(calls)
    if mode == "review_guidance":
        target = request.context.selected_target
        if run_id is not None and target is not None and target.target_type == "node":
            return (
                inspect,
                ("get_evidence_subgraph", {"runId": run_id, "nodeId": target.target_id}, "evidence"),
            )
        return (inspect,)
    if mode == "method_scope":
        return (inspect, cards)
    return (cards,)

__all__ = [
    "_answer_fallback",
    "_answer_skill_plan",
    "_case_answer_context",
    "_deterministic_answer_mode",
    "_deterministic_dispatch_intent",
    "_inspection_answer_context",
    "_numeric_facts",
]

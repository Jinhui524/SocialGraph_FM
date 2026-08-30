import type { GlobalModelModality } from "../types/globalModel";

const GOVERNANCE_MODALITY_LABELS: Readonly<Record<GlobalModelModality, string>> = Object.freeze({
  coRT: "协同转发",
  coURL: "共链传播",
  hashSeq: "话题序列",
  fastRT: "快速转发",
  tweetSim: "内容相似",
  fused: "综合关系",
});

export function governanceModalityLabel(modality: GlobalModelModality): string {
  return GOVERNANCE_MODALITY_LABELS[modality];
}

const SERVING_LIMITATION_LABELS: Readonly<Record<string, string>> = Object.freeze({
  "Scores are analyst-facing risk candidates and never automatic enforcement decisions.": "风险分数仅用于安排人工复核顺序，不作为自动处置结论。",
  "The two-hop view is capped at 300 nodes and 1000 factual edges; relation weights are explanation-only and do not prove coordination or intent.": "两跳证据视图最多展示 300 个节点和 1,000 条事实关系；关系权重仅用于说明已登记连接，不能证明协同行为或主观意图。",
  "Louvain community priority is derived from member risk, not proof of coordination.": "群组复核顺序由成员风险信号派生，不构成成员协同行为的直接证明。",
  "Derived analyst priority over a factual input relation; it is not proof of coordination.": "该优先级从已登记事实关系派生，仅用于安排人工核验，不构成协同行为证明。",
  "Bounded same-community similarity lead; this is not a factual or future edge.": "该线索来自同群组内的有界相似度计算，既不是已登记事实关系，也不代表未来一定形成关系。",
});

export function governanceLimitationLabel(limitation: string | undefined): string {
  const normalized = limitation?.trim();
  if (!normalized) return "证据用于支持人工复核，不构成自动处置结论。";
  return SERVING_LIMITATION_LABELS[normalized] ?? normalized;
}

export function governanceAccountLabel(label: string | null | undefined, fallbackId: string): string {
  const normalized = label?.trim();
  if (!normalized) return fallbackId;
  const anonymousMatch = /^Anonymous account\s+(.+)$/iu.exec(normalized);
  if (anonymousMatch) return `匿名账号 ${anonymousMatch[1]}`;
  const accountMatch = /^Account\s+(.+)$/iu.exec(normalized);
  return accountMatch ? `账号 ${accountMatch[1]}` : normalized;
}

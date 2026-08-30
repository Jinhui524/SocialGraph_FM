import type { AnalysisTask } from "../../types/graph";

export type ExpectedIntentKind = "chat" | "analysis_request";

export interface IntentEvaluationCase {
  readonly id: string;
  readonly input: string;
  readonly expectedKind: ExpectedIntentKind;
  readonly expectedTask?: AnalysisTask;
  readonly expectedTargets?: readonly string[];
  readonly expectedTimeRange?: {
    readonly start?: string;
    readonly end?: string;
  };
  readonly tags: readonly string[];
}

const chatCases: readonly IntentEvaluationCase[] = [
  { id: "chat-01", input: "你好！", expectedKind: "chat", tags: ["chat", "greeting"] },
  { id: "chat-02", input: "你是谁？", expectedKind: "chat", tags: ["chat", "capability"] },
  { id: "chat-03", input: "你能做什么？", expectedKind: "chat", tags: ["chat", "help"] },
  { id: "chat-04", input: "怎么使用？", expectedKind: "chat", tags: ["chat", "help"] },
  { id: "chat-05", input: "谢谢你的说明。", expectedKind: "chat", tags: ["chat", "thanks"] },
  { id: "chat-06", input: "功能介绍。", expectedKind: "chat", tags: ["chat", "capability"] },
  { id: "chat-07", input: "再见，我们下次继续。", expectedKind: "chat", tags: ["chat", "farewell"] },
];

const overviewCases: readonly IntentEvaluationCase[] = [
  { id: "overview-01", input: "生成这张图的概览。", expectedKind: "analysis_request", expectedTask: "overview", tags: ["overview"] },
  { id: "overview-02", input: "总结整体网络结构。", expectedKind: "analysis_request", expectedTask: "overview", tags: ["overview"] },
  { id: "overview-03", input: "统计节点和关系的基本情况。", expectedKind: "analysis_request", expectedTask: "overview", tags: ["overview"] },
  { id: "overview-04", input: "查看当前图谱情况。", expectedKind: "analysis_request", expectedTask: "overview", tags: ["overview"] },
  { id: "overview-05", input: "给我一份网络结构摘要。", expectedKind: "analysis_request", expectedTask: "overview", tags: ["overview"] },
  { id: "overview-06", input: "分析 2024 年的整体网络。", expectedKind: "analysis_request", expectedTask: "overview", expectedTimeRange: { start: "2024", end: "2024" }, tags: ["overview", "year"] },
  { id: "overview-07", input: "概览 2020-2023 年的合作图谱。", expectedKind: "analysis_request", expectedTask: "overview", expectedTimeRange: { start: "2020", end: "2023" }, tags: ["overview", "year-range"] },
  { id: "overview-08", input: "查看“高校协作网”的统计指标。", expectedKind: "analysis_request", expectedTask: "overview", expectedTargets: ["高校协作网"], tags: ["overview", "target"] },
  { id: "overview-09", input: "请概览 @数据集A 的图谱情况。", expectedKind: "analysis_request", expectedTask: "overview", expectedTargets: ["数据集A"], tags: ["overview", "mention"] },
];

const centralityCases: readonly IntentEvaluationCase[] = [
  { id: "centrality-01", input: "计算所有成员的中心性。", expectedKind: "analysis_request", expectedTask: "centrality", tags: ["centrality"] },
  { id: "centrality-02", input: "给出节点度数排名。", expectedKind: "analysis_request", expectedTask: "centrality", tags: ["centrality"] },
  { id: "centrality-03", input: "找出网络里的核心节点。", expectedKind: "analysis_request", expectedTask: "centrality", tags: ["centrality"] },
  { id: "centrality-04", input: "哪些成员的影响力最高？", expectedKind: "analysis_request", expectedTask: "centrality", tags: ["centrality"] },
  { id: "centrality-05", input: "识别关键成员并生成成员排名。", expectedKind: "analysis_request", expectedTask: "centrality", tags: ["centrality"] },
  { id: "centrality-06", input: "比较张三、李四在 2020-2024 年的中心性。", expectedKind: "analysis_request", expectedTask: "centrality", expectedTargets: ["张三", "李四"], expectedTimeRange: { start: "2020", end: "2024" }, tags: ["centrality", "targets", "year-range"] },
  { id: "centrality-07", input: "查看 @王五 的影响力。", expectedKind: "analysis_request", expectedTask: "centrality", expectedTargets: ["王五"], tags: ["centrality", "mention"] },
  { id: "centrality-08", input: "分析“北京实验室”的重要节点。", expectedKind: "analysis_request", expectedTask: "centrality", expectedTargets: ["北京实验室"], tags: ["centrality", "target"] },
  { id: "centrality-09", input: "统计 2022 年关键成员的度数排名。", expectedKind: "analysis_request", expectedTask: "centrality", expectedTimeRange: { start: "2022", end: "2022" }, tags: ["centrality", "year"] },
];

const bridgeCases: readonly IntentEvaluationCase[] = [
  { id: "bridge-01", input: "找出图中的割点。", expectedKind: "analysis_request", expectedTask: "bridge_detection", tags: ["bridge"] },
  { id: "bridge-02", input: "识别关键桥梁。", expectedKind: "analysis_request", expectedTask: "bridge_detection", tags: ["bridge"] },
  { id: "bridge-03", input: "哪些成员是桥接节点？", expectedKind: "analysis_request", expectedTask: "bridge_detection", tags: ["bridge"] },
  { id: "bridge-04", input: "分析网络中的结构洞。", expectedKind: "analysis_request", expectedTask: "bridge_detection", tags: ["bridge"] },
  { id: "bridge-05", input: "检测协作断层和中介节点。", expectedKind: "analysis_request", expectedTask: "bridge_detection", tags: ["bridge"] },
  { id: "bridge-06", input: "查看 @李明 是否是桥接者。", expectedKind: "analysis_request", expectedTask: "bridge_detection", expectedTargets: ["李明"], tags: ["bridge", "mention"] },
  { id: "bridge-07", input: "分析“跨校项目组”的网络断层。", expectedKind: "analysis_request", expectedTask: "bridge_detection", expectedTargets: ["跨校项目组"], tags: ["bridge", "target"] },
  { id: "bridge-08", input: "找出 2019 至 2023 年间的桥接节点。", expectedKind: "analysis_request", expectedTask: "bridge_detection", expectedTimeRange: { start: "2019", end: "2023" }, tags: ["bridge", "year-range"] },
  { id: "bridge-09", input: "检测 2025 年关键桥梁。", expectedKind: "analysis_request", expectedTask: "bridge_detection", expectedTimeRange: { start: "2025", end: "2025" }, tags: ["bridge", "year"] },
];

const communityCases: readonly IntentEvaluationCase[] = [
  { id: "community-01", input: "分析网络社区。", expectedKind: "analysis_request", expectedTask: "community", tags: ["community"] },
  { id: "community-02", input: "进行社群划分。", expectedKind: "analysis_request", expectedTask: "community", tags: ["community"] },
  { id: "community-03", input: "识别图中的群落。", expectedKind: "analysis_request", expectedTask: "community", tags: ["community"] },
  { id: "community-04", input: "看看成员形成了哪些圈层。", expectedKind: "analysis_request", expectedTask: "community", tags: ["community"] },
  { id: "community-05", input: "评估团体结构和社区健康。", expectedKind: "analysis_request", expectedTask: "community", tags: ["community"] },
  { id: "community-06", input: "查看 @学院A 所在的社群。", expectedKind: "analysis_request", expectedTask: "community", expectedTargets: ["学院A"], tags: ["community", "mention"] },
  { id: "community-07", input: "分析“青年学者网络”的社区。", expectedKind: "analysis_request", expectedTask: "community", expectedTargets: ["青年学者网络"], tags: ["community", "target"] },
  { id: "community-08", input: "比较 2020～2024 年的社区结构。", expectedKind: "analysis_request", expectedTask: "community", expectedTimeRange: { start: "2020", end: "2024" }, tags: ["community", "year-range"] },
  { id: "community-09", input: "分析 2023 年网络分区。", expectedKind: "analysis_request", expectedTask: "community", expectedTimeRange: { start: "2023", end: "2023" }, tags: ["community", "year"] },
];

const linkPredictionCases: readonly IntentEvaluationCase[] = [
  { id: "link-01", input: "执行链接预测。", expectedKind: "analysis_request", expectedTask: "link_prediction", tags: ["link-prediction", "gfm"] },
  { id: "link-02", input: "预测还没有出现的潜在关系。", expectedKind: "analysis_request", expectedTask: "link_prediction", tags: ["link-prediction", "gfm"] },
  { id: "link-03", input: "发现团队之间的潜在合作。", expectedKind: "analysis_request", expectedTask: "link_prediction", tags: ["link-prediction", "gfm"] },
  { id: "link-04", input: "推荐可能建立的关系。", expectedKind: "analysis_request", expectedTask: "link_prediction", tags: ["link-prediction", "gfm"] },
  { id: "link-05", input: "帮我寻找新的合作机会。", expectedKind: "analysis_request", expectedTask: "link_prediction", tags: ["link-prediction", "gfm"] },
  { id: "link-06", input: "预测 @张三 的潜在合作。", expectedKind: "analysis_request", expectedTask: "link_prediction", expectedTargets: ["张三"], tags: ["link-prediction", "gfm", "mention"] },
  { id: "link-07", input: "为“实验室A”做关系推荐。", expectedKind: "analysis_request", expectedTask: "link_prediction", expectedTargets: ["实验室A"], tags: ["link-prediction", "gfm", "target"] },
  { id: "link-08", input: "预测 2021-2024 年可能产生的关系。", expectedKind: "analysis_request", expectedTask: "link_prediction", expectedTimeRange: { start: "2021", end: "2024" }, tags: ["link-prediction", "gfm", "year-range"] },
  { id: "link-09", input: "查看 2026 年合作机会。", expectedKind: "analysis_request", expectedTask: "link_prediction", expectedTimeRange: { start: "2026", end: "2026" }, tags: ["link-prediction", "gfm", "year"] },
];

const nodeRoleCases: readonly IntentEvaluationCase[] = [
  { id: "role-01", input: "进行节点角色识别。", expectedKind: "analysis_request", expectedTask: "node_role", tags: ["node-role", "gfm"] },
  { id: "role-02", input: "判断每位成员的角色。", expectedKind: "analysis_request", expectedTask: "node_role", tags: ["node-role", "gfm"] },
  { id: "role-03", input: "完成成员定位。", expectedKind: "analysis_request", expectedTask: "node_role", tags: ["node-role", "gfm"] },
  { id: "role-04", input: "根据网络结构进行节点分类。", expectedKind: "analysis_request", expectedTask: "node_role", tags: ["node-role", "gfm"] },
  { id: "role-05", input: "识别这个网络的核心团队。", expectedKind: "analysis_request", expectedTask: "node_role", tags: ["node-role", "gfm"] },
  { id: "role-06", input: "判断 @赵六 的节点角色。", expectedKind: "analysis_request", expectedTask: "node_role", expectedTargets: ["赵六"], tags: ["node-role", "gfm", "mention"] },
  { id: "role-07", input: "识别“科研秘书”的成员角色。", expectedKind: "analysis_request", expectedTask: "node_role", expectedTargets: ["科研秘书"], tags: ["node-role", "gfm", "target"] },
  { id: "role-08", input: "分析 2020—2022 年的节点角色。", expectedKind: "analysis_request", expectedTask: "node_role", expectedTimeRange: { start: "2020", end: "2022" }, tags: ["node-role", "gfm", "year-range"] },
  { id: "role-09", input: "识别 2024 年成员定位。", expectedKind: "analysis_request", expectedTask: "node_role", expectedTimeRange: { start: "2024", end: "2024" }, tags: ["node-role", "gfm", "year"] },
];

const similarStructureCases: readonly IntentEvaluationCase[] = [
  { id: "similar-01", input: "检索相似结构。", expectedKind: "analysis_request", expectedTask: "similar_structure", tags: ["similar-structure", "gfm"] },
  { id: "similar-02", input: "找出相似案例。", expectedKind: "analysis_request", expectedTask: "similar_structure", tags: ["similar-structure", "gfm"] },
  { id: "similar-03", input: "进行图结构检索。", expectedKind: "analysis_request", expectedTask: "similar_structure", tags: ["similar-structure", "gfm"] },
  { id: "similar-04", input: "有哪些相似图？", expectedKind: "analysis_request", expectedTask: "similar_structure", tags: ["similar-structure", "gfm"] },
  { id: "similar-05", input: "寻找可以对照的类比网络。", expectedKind: "analysis_request", expectedTask: "similar_structure", tags: ["similar-structure", "gfm"] },
  { id: "similar-06", input: "检索与 @项目A 相似的结构。", expectedKind: "analysis_request", expectedTask: "similar_structure", expectedTargets: ["项目A"], tags: ["similar-structure", "gfm", "mention"] },
  { id: "similar-07", input: "寻找与“高校合作网络”相似的案例。", expectedKind: "analysis_request", expectedTask: "similar_structure", expectedTargets: ["高校合作网络"], tags: ["similar-structure", "gfm", "target"] },
  { id: "similar-08", input: "检索 2018 到 2022 年的相似图。", expectedKind: "analysis_request", expectedTask: "similar_structure", expectedTimeRange: { start: "2018", end: "2022" }, tags: ["similar-structure", "gfm", "year-range"] },
  { id: "similar-09", input: "查找 2023 年相似结构。", expectedKind: "analysis_request", expectedTask: "similar_structure", expectedTimeRange: { start: "2023", end: "2023" }, tags: ["similar-structure", "gfm", "year"] },
];

/**
 * Stable, hand-labelled Chinese evaluation set. Keep it independent from model
 * prompts so it can score the local fallback, an HTTP provider, or future GFM
 * task routers without turning production examples into the test answers.
 */
export const CHINESE_INTENT_EVALUATION_CASES = Object.freeze([
  ...chatCases,
  ...overviewCases,
  ...centralityCases,
  ...bridgeCases,
  ...communityCases,
  ...linkPredictionCases,
  ...nodeRoleCases,
  ...similarStructureCases,
]) satisfies readonly IntentEvaluationCase[];

export const SAFE_GRAPH_CONTEXT_FIXTURE = Object.freeze({
  nodeCount: 412,
  edgeCount: 2_893,
  density: 0.028,
  connectedComponents: 3,
  nodeTypes: Object.freeze(["person", "organization", "project"]),
  edgeTypes: Object.freeze(["collaborates_with", "participates_in"]),
  hasWeight: true,
  hasTimestamp: true,
  timeRange: Object.freeze({ start: "2020", end: "2024" }),
});

"""Disabled-by-default public governance task declarations."""

from .public_contracts import CoreTaskManifest

CORE_TASKS = (
    CoreTaskManifest(
        taskId="core.community_health_observation",
        displayName="社区健康观察",
        description="汇总协作、互惠、参与集中度与跨社区连接证据，供人工解释和复核。",
        requiredProfiles=[
            "collaboration.actor-interaction/1.0",
            "collaboration.activity-hetero/1.0",
        ],
        requiresTemporalEdges=True,
        outputKind="community_observation",
        refusalConditions=[
            "未安装并验证支持该 profile 的模型",
            "缺少治理观察窗口所需的时间字段",
            "请求将观察结果直接用于自动处罚",
        ],
    ),
    CoreTaskManifest(
        taskId="core.newcomer_support",
        displayName="桥接者与新成员支持",
        description="识别需要融入支持的成员和潜在桥接关系，输出可解释的人工辅助建议。",
        requiredProfiles=[
            "collaboration.actor-interaction/1.0",
            "collaboration.activity-hetero/1.0",
        ],
        requiresTemporalEdges=True,
        outputKind="member_support",
        refusalConditions=[
            "未安装并验证支持该 profile 的模型",
            "缺少 actor 映射或关系时间",
            "敏感属性未从推理 allowlist 排除",
        ],
    ),
    CoreTaskManifest(
        taskId="core.coordination_review",
        displayName="协同行为复核",
        description="形成需要人工查看的结构或时序协同行为候选，不作违规或恶意定性。",
        requiredProfiles=["collaboration.activity-hetero/1.0"],
        requiresTemporalEdges=True,
        outputKind="review_queue",
        refusalConditions=[
            "未安装并验证支持该 profile 的模型",
            "缺少异构活动关系或可靠时间字段",
            "请求自动封禁、惩罚或公开指认成员",
        ],
    ),
)


def task_payloads() -> list[dict]:
    return [task.model_dump(mode="json", by_alias=True) for task in CORE_TASKS]

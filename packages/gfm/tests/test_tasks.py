from socialgraph_gfm.tasks import CORE_TASKS, task_payloads


def test_core_tasks_are_predeclared_but_disabled_and_human_reviewed():
    expected = {
        "core.community_health_observation",
        "core.newcomer_support",
        "core.coordination_review",
    }
    assert {task.task_id for task in CORE_TASKS} == expected
    assert all(task.enabled is False for task in CORE_TASKS)
    assert all(task.human_review_required is True for task in CORE_TASKS)
    assert all(task.refusal_conditions for task in CORE_TASKS)
    assert {payload["taskId"] for payload in task_payloads()} == expected

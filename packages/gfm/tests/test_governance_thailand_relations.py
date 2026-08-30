from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from socialgraph_gfm.governance import thailand as thailand_module


def _post(
    account: str,
    *,
    second: float = 0,
    repost: str | None = None,
    hashtags: list[str] | None = None,
    urls: list[str] | None = None,
) -> thailand_module._SourcePost:
    timestamp = datetime(2026, 8, 20, tzinfo=UTC) + timedelta(seconds=second)
    return thailand_module._SourcePost.model_validate(
        {
            "postid": f"post-{account}",
            "post_text": f"literal content {account}",
            "post_time": timestamp.isoformat().replace("+00:00", "Z"),
            "accountid": account,
            "is_repost": repost is not None,
            "reposted_accountid": "origin" if repost is not None else None,
            "reposted_postid": repost,
            "hashtags": hashtags or [],
            "urls": urls or [],
            "account_mentions": [],
            "in_reply_to_accountid": None,
            "is_control": True,
        }
    )


def _relation_pairs(
    posts: list[thailand_module._SourcePost],
    accounts: list[str],
    features: np.ndarray | None = None,
) -> dict[str, set[frozenset[str]]]:
    values = features if features is not None else np.zeros((len(accounts), 768), dtype=np.float32)
    relations = thailand_module._derive_relations(posts, accounts, values)
    result = {modality: set() for modality in thailand_module.MODALITIES}
    for relation in relations:
        result[relation.modality].add(frozenset((relation.source, relation.target)))
    return result


def test_cort_requires_the_same_reposted_post_id() -> None:
    posts = [
        _post("a", second=0, repost="shared"),
        _post("b", second=20, repost="shared"),
        _post("c", second=40, repost="different"),
    ]

    pairs = _relation_pairs(posts, ["a", "b", "c"])

    assert pairs["coRT"] == {frozenset(("a", "b"))}


def test_hashseq_matches_ordered_normalized_sequence_and_rejects_reverse_order() -> None:
    posts = [
        _post("a", hashtags=["#One", "Ｔｗｏ"]),
        _post("b", hashtags=["one", "two"]),
        _post("c", hashtags=["two", "one"]),
    ]

    pairs = _relation_pairs(posts, ["a", "b", "c"])

    assert pairs["hashSeq"] == {frozenset(("a", "b"))}


def test_fastrt_includes_ten_second_boundary_and_excludes_greater_than_ten() -> None:
    posts = [
        _post("a", second=0, repost="shared"),
        _post("b", second=10, repost="shared"),
        _post("c", second=11, repost="shared"),
    ]

    pairs = _relation_pairs(posts, ["a", "b", "c"])

    assert pairs["fastRT"] == {
        frozenset(("a", "b")),
        frozenset(("b", "c")),
    }
    assert frozenset(("a", "c")) not in pairs["fastRT"]


def test_courl_matches_only_the_same_normalized_hashed_url() -> None:
    posts = [
        _post("a", urls=[" https://example.invalid/item "]),
        _post("b", urls=["https://example.invalid/item"]),
        _post("c", urls=["https://example.invalid/other"]),
    ]

    pairs = _relation_pairs(posts, ["a", "b", "c"])

    assert pairs["coURL"] == {frozenset(("a", "b"))}


def _unit_vector(*components: tuple[int, float]) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    for index, component in components:
        value[index] = component
    value /= np.linalg.norm(value)
    return value


def test_tweetsim_requires_mutual_top_five_and_threshold_inclusively() -> None:
    accounts = [
        "positive-a",
        "positive-b",
        "below-a",
        "below-b",
        "nonmutual-a",
        "popular",
        "close-0",
        "close-1",
        "close-2",
        "close-3",
        "close-4",
        "close-5",
        "boundary-a",
        "boundary-b",
    ]
    vectors = {
        "positive-a": _unit_vector((20, 1.0)),
        "positive-b": _unit_vector((20, 0.9), (21, 0.4358899)),
        "below-a": _unit_vector((30, 1.0)),
        "below-b": _unit_vector((30, 0.79), (31, 0.613106)),
        "nonmutual-a": _unit_vector((0, 0.81), (1, 0.58643)),
        "popular": _unit_vector((0, 1.0)),
        "boundary-a": _unit_vector((40, 1.0)),
        "boundary-b": _unit_vector((40, 0.8), (41, 0.6)),
    }
    for index in range(6):
        vectors[f"close-{index}"] = _unit_vector((0, 0.95), (2 + index, 0.3122499))
    features = np.vstack([vectors[account] for account in accounts]).astype(np.float32)
    posts = [_post(account) for account in accounts]

    pairs = _relation_pairs(posts, accounts, features)["tweetSim"]

    assert frozenset(("positive-a", "positive-b")) in pairs
    assert frozenset(("boundary-a", "boundary-b")) in pairs
    assert frozenset(("below-a", "below-b")) not in pairs
    assert frozenset(("nonmutual-a", "popular")) not in pairs


def test_group_relation_rejects_a_single_overdense_group_before_pair_expansion() -> None:
    assert hasattr(thailand_module, "GROUP_RELATION_MAX_ACCOUNTS")
    limit = thailand_module.GROUP_RELATION_MAX_ACCOUNTS
    posts = [_post(f"account-{index}", urls=["https://dense.invalid/shared"]) for index in range(limit + 1)]

    with pytest.raises(thailand_module.SourceValidationError, match="coURL group.*dense"):
        _relation_pairs(posts, [post.accountid for post in posts])


def test_group_relation_rejects_total_potential_pairs_and_accepts_the_exact_group_boundary() -> None:
    assert hasattr(thailand_module, "GROUP_RELATION_MAX_ACCOUNTS")
    assert hasattr(thailand_module, "GROUP_RELATION_PAIR_BUDGET")
    limit = thailand_module.GROUP_RELATION_MAX_ACCOUNTS
    boundary_posts = [_post(f"boundary-{index}", urls=["https://dense.invalid/boundary"]) for index in range(limit)]
    boundary_pairs = _relation_pairs(boundary_posts, [post.accountid for post in boundary_posts])
    assert len(boundary_pairs["coURL"]) == limit * (limit - 1) // 2

    posts = [
        _post(f"group-{group}-{index}", urls=[f"https://dense.invalid/group/{group}"])
        for group in range(2)
        for index in range(limit)
    ]
    with pytest.raises(thailand_module.SourceValidationError, match="group relation pair budget"):
        _relation_pairs(posts, [post.accountid for post in posts])


def test_fastrt_rejects_an_adversarial_window_before_expansion_and_accepts_the_boundary() -> None:
    assert hasattr(thailand_module, "FAST_RT_PAIR_BUDGET")
    account_count = 100
    accepted_reposts_per_account = 3
    accepted = [
        _post(f"accepted-{account}", second=float(event) / 100, repost="shared")
        for account in range(account_count)
        for event in range(accepted_reposts_per_account)
    ]
    accepted_accounts = sorted({post.accountid for post in accepted})
    assert _relation_pairs(accepted, accepted_accounts)["fastRT"]

    rejected = [
        _post(f"rejected-{account}", second=float(event) / 100, repost="shared")
        for account in range(account_count)
        for event in range(accepted_reposts_per_account + 1)
    ]
    with pytest.raises(thailand_module.SourceValidationError, match="fastRT pair budget"):
        _relation_pairs(rejected, sorted({post.accountid for post in rejected}))

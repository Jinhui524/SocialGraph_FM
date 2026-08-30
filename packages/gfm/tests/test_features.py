from socialgraph_gfm.features import CategoryVocabulary, NumericStandardizer


def test_numeric_standardizer_preserves_missing_mask_and_zero_fills():
    fitted = NumericStandardizer.fit([1.0, 3.0, None])
    values, missing = fitted.transform([1.0, None, 3.0])
    assert values == [-1.0, 0.0, 1.0]
    assert missing == [False, True, False]


def test_category_vocabulary_is_codepoint_sorted_and_tracks_unknowns():
    fitted = CategoryVocabulary.fit(["中", "a", "é", None])
    assert fitted.categories == ("a", "é", "中")
    values, missing = fitted.transform([None, "unknown", "中"])
    assert values == [0, 1, 4]
    assert missing == [True, False, False]

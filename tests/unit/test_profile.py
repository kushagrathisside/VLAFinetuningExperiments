from __future__ import annotations

from model_training.data import build_dataset_profile


def test_profile_on_clean_version_has_no_splits(settings, catalog, clean_version):
    profile = build_dataset_profile("so100_test", clean_version, settings=settings, catalog=catalog)
    assert profile.episode_count == 8
    assert profile.frame_count == 8 * 20
    assert profile.splits == {}
    assert any("no split assignment" in n for n in profile.notes)


def test_profile_on_training_version_has_splits_and_coverage(settings, catalog, training_version):
    profile = build_dataset_profile("so100_test", training_version, settings=settings, catalog=catalog)
    assert sum(profile.splits.values()) == 8
    assert profile.action["key"] is not None
    assert profile.coverage["action_space_coverage"]["mean_normalized_entropy"] is not None
    assert 0.0 <= profile.coverage["action_space_coverage"]["mean_normalized_entropy"] <= 1.0
    assert profile.coverage["visual_diversity"]["status"] == "unavailable"


def test_profile_to_dict_is_json_shaped(settings, catalog, training_version):
    profile = build_dataset_profile("so100_test", training_version, settings=settings, catalog=catalog)
    d = profile.to_dict()
    import json

    json.dumps(d)  # must not raise -- every field JSON-serialisable
    assert d["dataset_id"] == "so100_test"
    assert d["parent_version"] == "clean/v1"

"""Unit tests for speaker-disjoint split assignment."""

from __future__ import annotations

from synthbanshee.package.splitter import assign_splits

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clip_speaker_map(n_speakers: int, clips_per_speaker: int = 1) -> dict[str, list[str]]:
    """Build a clip→speakers map with no shared speakers across clips."""
    result: dict[str, list[str]] = {}
    for spk_idx in range(n_speakers):
        for clip_idx in range(clips_per_speaker):
            clip_id = f"clip_{spk_idx:03d}_{clip_idx:02d}"
            result[clip_id] = [f"SPK_{spk_idx:03d}"]
    return result


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


class TestAssignSplits:
    def test_empty_map_returns_empty(self):
        assert assign_splits({}) == {}

    def test_all_clips_assigned(self):
        m = _make_clip_speaker_map(10, clips_per_speaker=2)
        splits = assign_splits(m, rng_seed=0)
        assert set(splits.keys()) == set(m.keys())

    def test_only_valid_split_names(self):
        m = _make_clip_speaker_map(20)
        splits = assign_splits(m, rng_seed=7)
        assert set(splits.values()) <= {"train", "val", "test"}

    def test_approximate_fractions(self):
        """With 100 clips, train:val:test ≈ 70:15:15."""
        m = _make_clip_speaker_map(100)
        splits = assign_splits(m, train_frac=0.70, val_frac=0.15, test_frac=0.15, rng_seed=0)
        counts = {"train": 0, "val": 0, "test": 0}
        for s in splits.values():
            counts[s] += 1
        # Allow ±5 clips of slack for rounding
        assert abs(counts["train"] - 70) <= 5
        assert abs(counts["val"] - 15) <= 5
        assert abs(counts["test"] - 15) <= 5

    def test_reproducible_with_same_seed(self):
        m = _make_clip_speaker_map(50)
        s1 = assign_splits(m, rng_seed=42)
        s2 = assign_splits(m, rng_seed=42)
        assert s1 == s2

    def test_different_seeds_produce_different_assignments(self):
        m = _make_clip_speaker_map(50)
        s1 = assign_splits(m, rng_seed=1)
        s2 = assign_splits(m, rng_seed=2)
        # Very unlikely to be identical with 50 clips
        assert s1 != s2


# ---------------------------------------------------------------------------
# Speaker-disjoint constraint
# ---------------------------------------------------------------------------


class TestSpeakerDisjoint:
    def test_shared_speaker_clips_in_same_split(self):
        """Two clips sharing a speaker must land in the same split."""
        m = {
            "clip_A": ["SPK_001", "SPK_002"],
            "clip_B": ["SPK_001", "SPK_003"],  # shares SPK_001 with clip_A
            "clip_C": ["SPK_004"],
            "clip_D": ["SPK_005"],
            "clip_E": ["SPK_006"],
        }
        splits = assign_splits(m, rng_seed=0)
        # clip_A and clip_B share SPK_001 → same split
        assert splits["clip_A"] == splits["clip_B"]

    def test_no_speaker_appears_in_two_splits(self):
        """Build a map where speakers appear in exactly one clip, verify disjoint."""
        m = _make_clip_speaker_map(n_speakers=30, clips_per_speaker=3)
        splits = assign_splits(m, rng_seed=99)
        # Group clips by speaker
        speaker_clips: dict[str, set[str]] = {}
        for clip_id, speaker_ids in m.items():
            for spk in speaker_ids:
                speaker_clips.setdefault(spk, set()).add(clip_id)
        # All clips for a speaker must be in the same split
        for spk, clip_ids in speaker_clips.items():
            clip_splits = {splits[c] for c in clip_ids}
            assert len(clip_splits) == 1, f"Speaker {spk} appears in multiple splits: {clip_splits}"

    def test_transitive_speaker_sharing(self):
        """A — shares SPK_X with — B — shares SPK_Y with — C: all in same split."""
        m = {
            "clip_A": ["SPK_X"],
            "clip_B": ["SPK_X", "SPK_Y"],
            "clip_C": ["SPK_Y"],
            "clip_D": ["SPK_Z"],
            "clip_E": ["SPK_W"],
        }
        splits = assign_splits(m, rng_seed=0)
        assert splits["clip_A"] == splits["clip_B"]
        assert splits["clip_B"] == splits["clip_C"]

    def test_single_clip_assigned(self):
        m = {"only_clip": ["SPEAKER_A"]}
        splits = assign_splits(m, rng_seed=0)
        assert "only_clip" in splits
        assert splits["only_clip"] in {"train", "val", "test"}

    def test_clip_with_no_speakers_assigned(self):
        """Clips with an empty speaker list should still be assigned."""
        m = {"lonely_clip": []}
        splits = assign_splits(m, rng_seed=0)
        assert splits["lonely_clip"] in {"train", "val", "test"}

    def test_large_dataset_disjoint(self):
        """Stress test: 500 clips, 2 speakers each, no speaker shared across clips."""
        m = _make_clip_speaker_map(n_speakers=500)
        splits = assign_splits(m, rng_seed=42)
        assert len(splits) == 500
        assert set(splits.values()) <= {"train", "val", "test"}

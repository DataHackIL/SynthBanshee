"""Speaker-disjoint split assignment for AVDP datasets (milestone 1.4).

The AVDP spec requires that no speaker_id appears in more than one dataset
split (train/val/test). This module enforces that constraint via Union-Find:
clips that share at least one speaker are treated as a single group and
always assigned to the same split.
"""

from __future__ import annotations

import random
from collections import defaultdict


def assign_splits(
    clip_speaker_map: dict[str, list[str]],
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    rng_seed: int = 42,
) -> dict[str, str]:
    """Assign clips to speaker-disjoint train/val/test splits.

    All clips that share at least one speaker are placed in the same split.
    Clip groups are shuffled with ``rng_seed`` before greedy split filling so
    that the assignment is reproducible.

    Args:
        clip_speaker_map: Mapping from clip_id to the list of speaker_ids
            that appear in that clip.
        train_frac: Target fraction of clips to assign to "train".
        val_frac: Target fraction of clips to assign to "val".
        test_frac: Target fraction of clips to assign to "test".
        rng_seed: Random seed for reproducible shuffling.

    Returns:
        Dict mapping clip_id → split name ("train", "val", or "test").
        Every clip in clip_speaker_map appears exactly once in the result.
    """
    if not (0 < train_frac <= 1 and 0 < val_frac <= 1 and 0 < test_frac <= 1):
        raise ValueError(
            "All split fractions must be positive and ≤ 1.0; "
            f"got train={train_frac}, val={val_frac}, test={test_frac}"
        )
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0; got {train_frac + val_frac + test_frac:.6f}"
        )

    all_clips = sorted(clip_speaker_map)
    if not all_clips:
        return {}

    # --- Union-Find over clip IDs ---
    parent: dict[str, str] = {c: c for c in all_clips}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    # Connect clips that share a speaker
    speaker_first_clip: dict[str, str] = {}
    for clip_id, speaker_ids in clip_speaker_map.items():
        for spk in speaker_ids:
            if spk in speaker_first_clip:
                _union(clip_id, speaker_first_clip[spk])
            else:
                speaker_first_clip[spk] = clip_id

    # Collect connected components (groups that must go to the same split)
    components: dict[str, list[str]] = defaultdict(list)
    for clip_id in all_clips:
        components[_find(clip_id)].append(clip_id)

    # Sort groups by their lexicographically smallest clip_id for determinism,
    # then shuffle with the fixed seed.
    groups: list[list[str]] = sorted(components.values(), key=lambda g: min(g))
    random.Random(rng_seed).shuffle(groups)

    total = len(all_clips)
    target_train = round(total * train_frac)
    target_val = round(total * val_frac)

    clip_to_split: dict[str, str] = {}
    assigned_train = assigned_val = 0

    for group in groups:
        if assigned_train < target_train:
            split = "train"
            assigned_train += len(group)
        elif assigned_val < target_val:
            split = "val"
            assigned_val += len(group)
        else:
            split = "test"
        for clip_id in group:
            clip_to_split[clip_id] = split

    return clip_to_split

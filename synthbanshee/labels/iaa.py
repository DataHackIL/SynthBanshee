"""Inter-annotator agreement (IAA) utilities for AVDP (spec.md §6).

Computes Cohen's Kappa per event-category group from pairs of EventLabel
annotation lists.  Supports both standard (nominal) kappa and linear-weighted
kappa for ordinal intensity ratings.

Category groups and spec targets (§6.2)
----------------------------------------
PHYS  — Physical events          target κ ≥ 0.65  min κ ≥ 0.55
VERB  — Verbal aggression        target κ ≥ 0.60  min κ ≥ 0.50
DIST  — Distress signals         target κ ≥ 0.60  min κ ≥ 0.50
ACOU  — Acoustic events          target κ ≥ 0.70  min κ ≥ 0.60
EMOT  — Emotional state          target κ ≥ 0.55  min κ ≥ 0.45
INTENS — Intensity (±1 tol.)     target κ ≥ 0.60  min κ ≥ 0.50

Usage
-----
    from synthbanshee.labels.iaa import run_iaa, IAAReport

    pairs = [(events_annotator_a, events_annotator_b), ...]
    clip_ids = ["clip_001", "clip_002", ...]
    report = run_iaa(pairs, clip_ids)
    print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field

from synthbanshee.labels.schema import EventLabel

# ---------------------------------------------------------------------------
# Spec constants
# ---------------------------------------------------------------------------

# (target_kappa, min_kappa) from spec.md §6.2
_CATEGORY_THRESHOLDS: dict[str, tuple[float, float]] = {
    "PHYS": (0.65, 0.55),
    "VERB": (0.60, 0.50),
    "DIST": (0.60, 0.50),
    "ACOU": (0.70, 0.60),
    "EMOT": (0.55, 0.45),
    "INTENS": (0.60, 0.50),
}

# Minimum fraction of the dataset that must be IAA-reviewed (spec §6.1).
MIN_COVERAGE_FRACTION: float = 0.20

_INTENSITY_MAX: int = 5  # spec: intensity in [1, 5]


# ---------------------------------------------------------------------------
# Kappa computation
# ---------------------------------------------------------------------------


def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Compute Cohen's Kappa for two sequences of nominal integer labels.

    Args:
        labels_a: Label sequence from annotator A.
        labels_b: Label sequence from annotator B (same length as labels_a).

    Returns:
        Cohen's Kappa in [-1, 1].  Returns 0.0 for degenerate cases
        (empty input or zero expected disagreement).

    Raises:
        ValueError: if the two sequences have different lengths.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Label sequences must have equal length; got {len(labels_a)} vs {len(labels_b)}"
        )
    n = len(labels_a)
    if n == 0:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b))
    n_cats = len(categories)
    cat_index = {c: i for i, c in enumerate(categories)}

    # Build confusion matrix
    matrix = [[0] * n_cats for _ in range(n_cats)]
    for a, b in zip(labels_a, labels_b, strict=False):
        matrix[cat_index[a]][cat_index[b]] += 1

    # Observed agreement
    p_o = sum(matrix[i][i] for i in range(n_cats)) / n

    # Expected agreement
    row_sums = [sum(matrix[i]) for i in range(n_cats)]
    col_sums = [sum(matrix[i][j] for i in range(n_cats)) for j in range(n_cats)]
    p_e = sum(row_sums[i] * col_sums[i] for i in range(n_cats)) / (n * n)

    if p_e >= 1.0:
        return 0.0
    return (p_o - p_e) / (1.0 - p_e)


def linear_weighted_kappa(
    labels_a: list[int],
    labels_b: list[int],
    *,
    min_val: int = 1,
    max_val: int = _INTENSITY_MAX,
) -> float:
    """Compute linearly-weighted Cohen's Kappa for ordinal labels.

    Disagreements are penalised proportionally to their distance:
    weight(i, j) = 1 - |i - j| / (max_val - min_val).

    Args:
        labels_a: Ordinal labels from annotator A.
        labels_b: Ordinal labels from annotator B.
        min_val: Minimum possible label value (inclusive).
        max_val: Maximum possible label value (inclusive).

    Returns:
        Weighted kappa in [-1, 1].
        Returns 0.0 for empty input.
        Returns 1.0 when ``min_val == max_val`` (no variance possible) or when
        all raters agree perfectly in the degenerate case where ``p_e >= 1``.

    Raises:
        ValueError: if the sequences have different lengths, or if any label
            falls outside ``[min_val, max_val]``.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Label sequences must have equal length; got {len(labels_a)} vs {len(labels_b)}"
        )
    for val in (*labels_a, *labels_b):
        if val < min_val or val > max_val:
            raise ValueError(f"Label {val!r} is outside the valid range [{min_val}, {max_val}]")
    n = len(labels_a)
    if n == 0:
        return 0.0

    scale = max_val - min_val
    if scale == 0:
        # All values identical — perfect agreement by definition.
        return 1.0

    def weight(i: int, j: int) -> float:
        # Spec §6.2 defines intensity with ±1 tolerance: off-by-one differences
        # count as full agreement; larger gaps are penalised proportionally over
        # the remaining span [2, scale].
        diff = abs(i - j)
        if diff <= 1:
            return 1.0
        remaining_span = scale - 1
        if remaining_span <= 0:
            return 1.0
        return max(0.0, 1.0 - (diff - 1) / remaining_span)

    vals = list(range(min_val, max_val + 1))
    k = len(vals)
    idx = {v: i for i, v in enumerate(vals)}

    # Build confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b, strict=False):
        matrix[idx[a]][idx[b]] += 1

    row_sums = [sum(matrix[i]) for i in range(k)]
    col_sums = [sum(matrix[i][j] for i in range(k)) for j in range(k)]

    # Observed and expected weighted agreement
    p_o = sum(weight(vals[i], vals[j]) * matrix[i][j] for i in range(k) for j in range(k)) / n
    p_e = sum(
        weight(vals[i], vals[j]) * row_sums[i] * col_sums[j] for i in range(k) for j in range(k)
    ) / (n * n)

    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


# ---------------------------------------------------------------------------
# Per-category kappa from annotation pairs
# ---------------------------------------------------------------------------


def _has_category(events: list[EventLabel], prefix: str) -> int:
    """Return 1 if any event's tier1_category starts with *prefix*, else 0."""
    return int(any(e.tier1_category.startswith(prefix) for e in events))


def _intensity_labels(events: list[EventLabel]) -> list[int]:
    """Return intensity values for all events (for weighted kappa)."""
    return [e.intensity for e in events]


def _category_kappa(
    pairs: list[tuple[list[EventLabel], list[EventLabel]]],
    prefix: str,
) -> float:
    """Compute binary presence/absence kappa for *prefix* across all clip pairs."""
    labels_a = [_has_category(a, prefix) for a, _ in pairs]
    labels_b = [_has_category(b, prefix) for _, b in pairs]
    return cohen_kappa(labels_a, labels_b)


def _intensity_kappa(
    pairs: list[tuple[list[EventLabel], list[EventLabel]]],
) -> float:
    """Compute linear-weighted kappa for intensity ratings across all pairs.

    Only clips where *both* annotators labeled at least one event contribute.
    """
    combined_a: list[int] = []
    combined_b: list[int] = []
    for events_a, events_b in pairs:
        ints_a = _intensity_labels(events_a)
        ints_b = _intensity_labels(events_b)
        # Align by position up to the shorter list; only include paired events.
        for ia, ib in zip(ints_a, ints_b, strict=False):
            combined_a.append(ia)
            combined_b.append(ib)
    return linear_weighted_kappa(combined_a, combined_b)


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CategoryKappa:
    """IAA result for one event-category group."""

    category: str
    kappa: float
    n_observations: int
    target_kappa: float
    min_kappa: float

    @property
    def meets_target(self) -> bool:
        return self.kappa >= self.target_kappa

    @property
    def meets_minimum(self) -> bool:
        return self.kappa >= self.min_kappa


@dataclass
class IAAReport:
    """Full IAA report for a dataset sample (spec.md §6)."""

    category_results: list[CategoryKappa]
    n_clips_reviewed: int
    total_clips: int
    disagreement_clip_ids: list[str] = field(default_factory=list)

    # ---------- derived properties ----------

    @property
    def coverage_fraction(self) -> float:
        if self.total_clips == 0:
            return 0.0
        return self.n_clips_reviewed / self.total_clips

    @property
    def meets_coverage(self) -> bool:
        return self.coverage_fraction >= MIN_COVERAGE_FRACTION

    @property
    def all_meet_minimum(self) -> bool:
        return all(r.meets_minimum for r in self.category_results)

    @property
    def passes(self) -> bool:
        return self.meets_coverage and self.all_meet_minimum

    def summary(self) -> str:
        """Return a human-readable multi-line summary."""
        lines: list[str] = [
            f"IAA Report — {self.n_clips_reviewed}/{self.total_clips} clips reviewed"
            f" ({self.coverage_fraction:.1%} coverage;"
            f" minimum required: {MIN_COVERAGE_FRACTION:.0%})",
            "",
        ]
        header = (
            f"{'Category':<10} {'κ':>6}  {'Target':>7}  {'Min':>5}  {'Target?':>8}  {'Min?':>5}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for r in self.category_results:
            t_mark = "✓" if r.meets_target else "✗"
            m_mark = "✓" if r.meets_minimum else "✗"
            lines.append(
                f"{r.category:<10} {r.kappa:>6.3f}  {r.target_kappa:>7.2f}"
                f"  {r.min_kappa:>5.2f}  {t_mark:>8}  {m_mark:>5}"
            )
        lines.append("")
        if self.disagreement_clip_ids:
            lines.append(f"Clips with unresolved disagreement ({len(self.disagreement_clip_ids)}):")
            for cid in self.disagreement_clip_ids:
                lines.append(f"  {cid}")
        status = "PASS" if self.passes else "FAIL"
        lines.append(f"\nOverall: {status}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_CATEGORY_PREFIXES = ["PHYS", "VERB", "DIST", "ACOU", "EMOT"]


def run_iaa(
    annotation_pairs: list[tuple[list[EventLabel], list[EventLabel]]],
    clip_ids: list[str],
    total_clips: int,
) -> IAAReport:
    """Compute an IAA report from paired annotations.

    Args:
        annotation_pairs: Parallel list of (annotator_A_events, annotator_B_events)
            for each dual-annotated clip.
        clip_ids: Clip IDs corresponding to each pair (same order).
        total_clips: Total number of clips in the dataset (for coverage fraction).

    Returns:
        IAAReport with per-category kappa scores and pass/fail status.

    Raises:
        ValueError: if annotation_pairs and clip_ids have different lengths.
    """
    if len(annotation_pairs) != len(clip_ids):
        raise ValueError(
            f"annotation_pairs ({len(annotation_pairs)}) and clip_ids"
            f" ({len(clip_ids)}) must have equal length"
        )

    category_results: list[CategoryKappa] = []

    for prefix in _CATEGORY_PREFIXES:
        target, minimum = _CATEGORY_THRESHOLDS[prefix]
        kappa = _category_kappa(annotation_pairs, prefix)
        n_obs = len(annotation_pairs)
        category_results.append(
            CategoryKappa(
                category=prefix,
                kappa=kappa,
                n_observations=n_obs,
                target_kappa=target,
                min_kappa=minimum,
            )
        )

    # Intensity: linear-weighted kappa across all paired events
    intensity_target, intensity_min = _CATEGORY_THRESHOLDS["INTENS"]
    intensity_kappa = _intensity_kappa(annotation_pairs)
    intensity_obs = sum(min(len(a), len(b)) for a, b in annotation_pairs)
    category_results.append(
        CategoryKappa(
            category="INTENS",
            kappa=intensity_kappa,
            n_observations=intensity_obs,
            target_kappa=intensity_target,
            min_kappa=intensity_min,
        )
    )

    # Disagreement clips: any clip where annotators differ on presence/absence
    # for at least one category.
    disagreement_clip_ids: list[str] = []
    for (events_a, events_b), clip_id in zip(annotation_pairs, clip_ids, strict=False):
        disagrees = any(
            _has_category(events_a, prefix) != _has_category(events_b, prefix)
            for prefix in _CATEGORY_PREFIXES
        )
        if disagrees:
            disagreement_clip_ids.append(clip_id)

    return IAAReport(
        category_results=category_results,
        n_clips_reviewed=len(annotation_pairs),
        total_clips=total_clips,
        disagreement_clip_ids=disagreement_clip_ids,
    )

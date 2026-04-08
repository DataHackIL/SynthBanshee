"""Dataset packaging: assembly, manifests, splits, QA, and clip validation."""

from synthbanshee.package.manifest import ManifestRow, generate_manifest
from synthbanshee.package.qa import DatasetStats, QAReport, run_qa
from synthbanshee.package.splitter import assign_splits
from synthbanshee.package.validator import ValidationResult, validate_clip

__all__ = [
    "DatasetStats",
    "ManifestRow",
    "QAReport",
    "ValidationResult",
    "assign_splits",
    "generate_manifest",
    "run_qa",
    "validate_clip",
]

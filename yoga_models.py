"""Common dataclass models shared across agents and runner.

This module defines lightweight data structures (`PoseInSequence`, `CourseCandidate`)
that were previously located in the legacy `agents/course_finder` package.  They are
now placed at project root so that runtime code no longer depends on the removed
legacy directory.
"""
from __future__ import annotations

import dataclasses
from typing import List


@dataclasses.dataclass
class PoseInSequence:
    """Represents a single pose within a course sequence."""

    pose_name: str
    order: int | None = None
    duration_seconds: int | None = None


@dataclasses.dataclass
class CourseCandidate:
    """Represents a full yoga course candidate with its pose sequence."""

    course_name: str
    description: str
    challenge: int | str | None
    total_duration: str | None
    sequence: List[PoseInSequence]

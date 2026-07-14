"""Dataset-wide metrics for comparing generated and observed code errors.

The functions operate on error labels assigned to code submissions. They do not
select examples or invoke a judge model; error labeling is an upstream step.
"""

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence


ERROR_KEYS = ("errors", "included errors", "included_errors", "error_labels")


def _is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def normalize_error_set(value):
    """Convert common serialized error-label formats to a set of strings."""
    if _is_missing(value):
        return set()

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return set()
        try:
            return normalize_error_set(json.loads(stripped))
        except (json.JSONDecodeError, TypeError):
            return {stripped}

    if isinstance(value, Mapping):
        for key in ERROR_KEYS:
            if key in value:
                return normalize_error_set(value[key])
        return set()

    # Ground-truth annotations are sometimes stored as (category, label).
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, str) for item in value)
    ):
        return {value[-1].strip()} if value[-1].strip() else set()

    if isinstance(value, Iterable):
        errors = set()
        for item in value:
            errors.update(normalize_error_set(item))
        return errors

    return {str(value)}


def normalize_generated_candidates(value):
    """Return one error set per generated candidate for a submission."""
    if _is_missing(value) or isinstance(value, (str, Mapping, tuple)):
        return [normalize_error_set(value)]

    if not isinstance(value, Sequence):
        return [normalize_error_set(value)]
    if not value:
        return [set()]

    # A flat list of labels describes one candidate. A nested list describes k
    # candidates, each with its own error labels.
    if all(isinstance(item, str) for item in value):
        return [normalize_error_set(value)]
    if all(
        isinstance(item, tuple)
        and len(item) == 2
        and all(isinstance(part, str) for part in item)
        for item in value
    ):
        return [normalize_error_set(value)]
    return [normalize_error_set(candidate) for candidate in value]


def error_set_iou(reference_errors, generated_errors, empty_union_score=1.0):
    """Calculate IoU between two error-label sets."""
    reference = normalize_error_set(reference_errors)
    generated = normalize_error_set(generated_errors)
    union = reference | generated
    if not union:
        return float(empty_union_score)
    return len(reference & generated) / len(union)


def submission_error_iou(
    ground_truth_errors,
    generated_errors,
    *,
    empty_union_score=1.0,
):
    """Calculate best-of-k error IoU for every ground-truth submission.

    ``generated_errors[i]`` may contain one error-label set or a nested list of
    error-label sets for multiple generations. The reported score for that
    submission is the maximum IoU across its generated candidates.
    """
    if len(ground_truth_errors) != len(generated_errors):
        raise ValueError(
            "ground_truth_errors and generated_errors must have equal length"
        )

    per_submission = []
    for reference, candidates_value in zip(ground_truth_errors, generated_errors):
        candidates = normalize_generated_candidates(candidates_value)
        candidate_scores = [
            error_set_iou(reference, candidate, empty_union_score)
            for candidate in candidates
        ]
        per_submission.append(max(candidate_scores))

    mean_iou = (
        sum(per_submission) / len(per_submission) if per_submission else 0.0
    )
    return {
        "submission_error_iou": mean_iou,
        "per_submission_iou": per_submission,
        "num_submissions": len(per_submission),
    }


def problem_error_coverage(
    problem_ids,
    ground_truth_errors,
    generated_errors,
    *,
    problem_error_universe=None,
):
    """Measure unique-error coverage across all generations for each problem.

    If ``problem_error_universe`` is provided, it maps each problem to the full
    set of known ground-truth errors. Otherwise the universe is derived by
    unioning the errors in the supplied ground-truth submissions.
    """
    size = len(problem_ids)
    if len(ground_truth_errors) != size or len(generated_errors) != size:
        raise ValueError("problem_ids and both error sequences must have equal length")

    reference_by_problem = defaultdict(set)
    generated_by_problem = defaultdict(set)
    for problem, reference, generated_value in zip(
        problem_ids, ground_truth_errors, generated_errors
    ):
        reference_by_problem[problem].update(normalize_error_set(reference))
        for candidate in normalize_generated_candidates(generated_value):
            generated_by_problem[problem].update(candidate)

    if problem_error_universe is not None:
        for problem in set(problem_ids):
            reference_by_problem[problem] = normalize_error_set(
                problem_error_universe.get(problem, set())
            )

    per_problem = {}
    total_covered = 0
    total_reference = 0
    for problem in dict.fromkeys(problem_ids):
        reference = reference_by_problem[problem]
        generated = generated_by_problem[problem]
        covered = reference & generated
        missed = reference - generated
        unexpected = generated - reference
        coverage = len(covered) / len(reference) if reference else 1.0

        per_problem[problem] = {
            "coverage": coverage,
            "num_reference_errors": len(reference),
            "num_covered_errors": len(covered),
            "reference_errors": sorted(reference),
            "covered_errors": sorted(covered),
            "missed_errors": sorted(missed),
            "unexpected_errors": sorted(unexpected),
        }
        total_covered += len(covered)
        total_reference += len(reference)

    macro_coverage = (
        sum(item["coverage"] for item in per_problem.values()) / len(per_problem)
        if per_problem
        else 0.0
    )
    micro_coverage = (
        total_covered / total_reference if total_reference else 1.0
    )
    return {
        "problem_error_coverage_macro": macro_coverage,
        "problem_error_coverage_micro": micro_coverage,
        "per_problem": per_problem,
        "num_problems": len(per_problem),
    }


def evaluate_error_metrics(
    problem_ids,
    ground_truth_errors,
    generated_errors,
    *,
    problem_error_universe=None,
    empty_union_score=1.0,
):
    """Evaluate submission IoU and problem-level coverage together."""
    return {
        **submission_error_iou(
            ground_truth_errors,
            generated_errors,
            empty_union_score=empty_union_score,
        ),
        **problem_error_coverage(
            problem_ids,
            ground_truth_errors,
            generated_errors,
            problem_error_universe=problem_error_universe,
        ),
    }

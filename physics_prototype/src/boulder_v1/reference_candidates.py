"""Three fixed scratch reference strategies, not route search or native success."""
from dataclasses import dataclass, replace
import math
from numbers import Real

from .transfer_feasibility import ReferencePolicy, TransferAssessment, assess_hand_transfer
from .whole_body_motion import WholeBodyMotion


REFERENCE_CANDIDATES = (
    ("default", ReferencePolicy()),
    ("neutral_yaw", ReferencePolicy(yaw_fraction=0.)),
    ("conservative", ReferencePolicy(rise_fraction=.18, yaw_fraction=0., waist_pitch_fraction=0.)),
)


@dataclass(frozen=True)
class CandidateSelection:
    """Retain every candidate's evidence, including local failures.

    Assessment motions retain their fixed 4 s timing. ``motion`` alone carries
    the execution timing override, which is not a dynamics certificate. Pass the
    EXACT preceding final_reference to the stock executor, never an assessment's
    scratch source_reference. Selection neither executes nor adopts any pose.
    """

    feasible: bool
    selected_index: int | None
    selected_name: str | None
    motion: WholeBodyMotion | None
    assessments: tuple[tuple[str, TransferAssessment], ...]
    classification: str
    reason: str

    def __post_init__(self):
        object.__setattr__(self, "assessments", tuple((label, result) for label, result in self.assessments))


def choose_hand_reference(model, data, scene, profile, reference, manager, request, *, prepare_s=4.):
    """Assess all three policies on the same unchanged live session, in order.

    Feasibility means bounded sampled geometry/admission/support only. Neither
    first-admitted selection nor a longer execution preparation proves physical
    convergence. Exhaustion rejects only these three strategies, not all motions.
    Stock assessment owns source/request validation and its existing error types;
    unexpected callback errors propagate rather than becoming search failures.
    """
    dt = float(model.opt.timestep)
    if (isinstance(prepare_s, bool) or not isinstance(prepare_s, Real)
            or not math.isfinite(prepare_s) or prepare_s <= 0
            or not math.isfinite(dt) or dt <= 0
            or not math.isfinite(intervals := prepare_s / dt)
            or abs(intervals - round(intervals)) > 1e-8):
        raise ValueError("prepare_s requires positive finite integral native intervals")
    assessments = tuple(
        (label, assess_hand_transfer(model, data, scene, profile, reference, manager, request, policy=policy))
        for label, policy in REFERENCE_CANDIDATES
    )
    for index, (label, assessment) in enumerate(assessments):
        if assessment.feasible:
            return CandidateSelection(
                True, index, label, replace(assessment.motion, prepare_s=float(prepare_s)), assessments,
                assessment.classification,
                "First admitted bounded candidate; geometry/static estimates only. Execution timing "
                "is separate from the fixed 4 s assessment; dynamics, capture and readiness untested.",
            )
    return CandidateSelection(
        False, None, None, None, assessments, "BOUNDED_CANDIDATE_SET_EXHAUSTED",
        "All three fixed local reference strategies failed: "
        + "; ".join(f"{label}: {result.classification} ({result.reason})" for label, result in assessments)
        + ". This is not a global transfer infeasibility proof; no native execution attempted.",
    )

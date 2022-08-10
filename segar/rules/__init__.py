__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
from .fields import lorentz_law
from .collisions import (
    overlap_time,
    object_collision,
    overlaps_wall,
    overlap_time_wall,
    fix_overlap_wall,
    fix_overlap_objects,
)
from .transitions import (
    Differential,
    SetFactor,
    Aggregate,
    DidNotMatch,
    DidNotPass,
    Transition,
    TransitionFunction,
    conditional_transition,
)
from .prior import Prior
from .relations import (
    Relation,
    overlaps,
    Or,
    And,
    IsEqual,
    IsOn,
    Contains,
    colliding,
)
from .rules import Rule, inspect_signature, match_pattern
from .special import (
    move,
    stop_condition,
    kill_condition,
    apply_burn,
    apply_friction,
    consume,
    accelerate
)

__all__ = (
    "overlap_time",
    "object_collision",
    "overlaps_wall",
    "overlap_time_wall",
    "fix_overlap_wall",
    "fix_overlap_objects",
    "lorentz_law",
    "Prior",
    "Relation",
    "overlaps",
    "Or",
    "And",
    "IsEqual",
    "IsOn",
    "Contains",
    "colliding",
    "Differential",
    "SetFactor",
    "Aggregate",
    "DidNotMatch",
    "DidNotPass",
    "Transition",
    "TransitionFunction",
    "conditional_transition",
    "Rule",
    "inspect_signature",
    "match_pattern",
    "move",
    "stop_condition",
    "kill_condition",
    "apply_burn",
    "apply_friction",
    "consume",
    "accelerate"
)

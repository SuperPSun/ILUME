"""Retired ObjectEncoder API boundaries; historical runs require historical Git."""
from .model import Ownership
OBJECT_ENCODER_OWNER = Ownership("ENCODER_STAGE2")
def retired_object_encoder(*args, **kwargs):
    raise ValueError("ObjectEncoder retired; use the historical Git revision")
load_object_phase1_source = validate_encoder_source = retired_object_encoder
build_object_phase1_model = validate_initial_object_embeddings = retired_object_encoder
ObjectPhase1Model = ObjectPhase1Representations = retired_object_encoder

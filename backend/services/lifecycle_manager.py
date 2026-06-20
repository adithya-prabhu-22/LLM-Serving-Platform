from backend.services.registry_service import get_model, update_model_status

ALLOWED_TRANSITIONS = {
    "REGISTERED": {"LOADING", "FAILED"},
    "LOADING": {"READY", "FAILED"},
    "READY": {"REGISTERED"},
    "FAILED": {"REGISTERED"},
}

def get_status(model_id: str) -> str:
    model = get_model(model_id)
    return model["status"]

def can_transition(current_status: str, new_status: str) -> bool:
    if current_status not in ALLOWED_TRANSITIONS:
        raise ValueError(f"Unknown status: {current_status}")
    return new_status in ALLOWED_TRANSITIONS[current_status]

def transition_to(model_id: str, new_status: str):
    current_status = get_status(model_id)
    if not can_transition(current_status, new_status):
        raise ValueError(f"Invalid transition: {current_status} -> {new_status}")
    update_model_status(model_id, new_status)
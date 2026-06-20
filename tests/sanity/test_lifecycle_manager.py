import pytest
from backend.services.registry_service import register_model, delete_model
from backend.services.lifecycle_manager import get_status, can_transition, transition_to

def test_can_transition():
    assert can_transition("REGISTERED", "LOADING")
    assert can_transition("REGISTERED", "FAILED")
    assert can_transition("LOADING", "READY")
    assert can_transition("LOADING", "FAILED")
    assert not can_transition("REGISTERED", "READY")
    assert not can_transition("READY", "LOADING")
    print("✓ State transition validation")

def test_valid_lifecycle():
    model_id = "lifecycle_test"
    register_model(
        model_id=model_id,
        name="Medical GPT",
        architecture="GPT",
        config_path="config.json",
        weights_path="model.safetensors",
    )
    assert get_status(model_id) == "REGISTERED"
    transition_to(model_id, "LOADING")
    assert get_status(model_id) == "LOADING"
    transition_to(model_id, "READY")
    assert get_status(model_id) == "READY"
    delete_model(model_id)
    print("✓ Valid lifecycle")

def test_invalid_transition():
    model_id = "invalid_transition_test"
    register_model(
        model_id=model_id,
        name="Test GPT",
        architecture="GPT",
        config_path="config.json",
        weights_path="model.safetensors",
    )
    with pytest.raises(ValueError):
        transition_to(model_id, "READY")
    delete_model(model_id)
    print("✓ Invalid transition validation")

def run_all_tests():
    print("\nRunning Lifecycle Manager tests...\n")
    test_can_transition()
    test_valid_lifecycle()
    test_invalid_transition()
    print("\n✓ All Lifecycle Manager tests passed")

if __name__ == "__main__":
    run_all_tests()
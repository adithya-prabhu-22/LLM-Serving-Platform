import pytest
from backend.services.registry_service import register_model, get_model, list_models, update_model_status, delete_model

def test_registry_service():
    model_id = "test_model"
    register_model(
        model_id=model_id,
        name="Test GPT",
        architecture="GPT",
        config_path="config.json",
        weights_path="model.safetensors",
    )
    model = get_model(model_id)
    assert model["model_id"] == model_id
    assert model["name"] == "Test GPT"
    models = list_models()
    assert len(models) > 0
    update_model_status(model_id, "READY")
    model = get_model(model_id)
    assert model["status"] == "READY"
    delete_model(model_id)
    with pytest.raises(ValueError):
        get_model(model_id)
    print("✓ Registry CRUD operations")

def test_duplicate_registration():
    model_id = "duplicate_model"
    register_model(
        model_id=model_id,
        name="Test GPT",
        architecture="GPT",
        config_path="config.json",
        weights_path="model.safetensors",
    )
    with pytest.raises(ValueError):
        register_model(
            model_id=model_id,
            name="Test GPT",
            architecture="GPT",
            config_path="config.json",
            weights_path="model.safetensors",
        )
    delete_model(model_id)
    print("✓ Duplicate registration validation")

def test_invalid_status():
    model_id = "status_test"
    register_model(
        model_id=model_id,
        name="Test GPT",
        architecture="GPT",
        config_path="config.json",
        weights_path="model.safetensors",
    )
    with pytest.raises(ValueError):
        update_model_status(model_id, "INVALID_STATUS")
    delete_model(model_id)
    print("✓ Status validation")

def run_all_tests():
    print("\nRunning Registry Service tests...\n")
    test_registry_service()
    test_duplicate_registration()
    test_invalid_status()
    print("\n✓ All Registry Service tests passed")

if __name__ == "__main__":
    run_all_tests()
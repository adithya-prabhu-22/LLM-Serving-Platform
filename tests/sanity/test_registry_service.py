from backend.services.registry_service import (
    register_model,
    get_model,
    list_models,
    update_model_status,
    delete_model,
)


def test_registry_service():

    model_id = "test_model"

    register_model(
    model_id=model_id,
    name="Test GPT",
    architecture="GPT",
    config_path="config.json",
    weights_path="model.safetensors",
    tokenizer_backend="huggingface",
    tokenizer_path="tokenizer.json",
    )

    model = get_model(
        model_id
    )

    assert model["model_id"] == model_id

    assert model["name"] == (
        "Test GPT"
    )

    models = list_models()

    assert len(
        models
    ) > 0

    update_model_status(
        model_id,
        "READY",
    )

    model = get_model(
        model_id
    )

    assert (
        model["status"]
        == "READY"
    )

    delete_model(
        model_id
    )

    try:

        get_model(
            model_id
        )

        assert False

    except ValueError:

        assert True
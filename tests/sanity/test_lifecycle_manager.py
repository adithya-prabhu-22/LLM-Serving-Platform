import pytest

from backend.services.registry_service import (
    register_model,
    delete_model,
)

from backend.services.lifecycle_manager import (
    get_status,
    can_transition,
    transition_to,
)


def test_can_transition():

    assert can_transition(
        "UPLOADED",
        "VALIDATING",
    )

    assert can_transition(
        "VALIDATING",
        "LOADING",
    )

    assert can_transition(
        "LOADING",
        "READY",
    )

    assert not can_transition(
        "UPLOADED",
        "READY",
    )

    assert not can_transition(
        "READY",
        "UPLOADED",
    )


def test_valid_lifecycle():

    model_id = (
        "lifecycle_test"
    )

    register_model(
        model_id=model_id,
        name="Medical GPT",
        architecture="GPT",
        config_path="config.json",
        weights_path="model.safetensors",
        tokenizer_path="tokenizer.json",
    )

    assert (
        get_status(
            model_id
        )
        == "UPLOADED"
    )

    transition_to(
        model_id,
        "VALIDATING",
    )

    assert (
        get_status(
            model_id
        )
        == "VALIDATING"
    )

    transition_to(
        model_id,
        "LOADING",
    )

    assert (
        get_status(
            model_id
        )
        == "LOADING"
    )

    transition_to(
        model_id,
        "READY",
    )

    assert (
        get_status(
            model_id
        )
        == "READY"
    )

    delete_model(
        model_id
    )


def test_invalid_transition():

    model_id = (
        "invalid_transition_test"
    )

    register_model(
        model_id=model_id,
        name="Medical GPT",
        architecture="GPT",
        config_path="config.json",
        weights_path="model.safetensors",
        tokenizer_path="tokenizer.json",
    )

    with pytest.raises(
        ValueError
    ):

        transition_to(
            model_id,
            "READY",
        )

    delete_model(
        model_id
    )
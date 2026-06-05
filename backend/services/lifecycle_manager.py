from backend.services.registry_service import (
    get_model,
    update_model_status,
)


ALLOWED_TRANSITIONS = {
    "UPLOADED": {
        "VALIDATING",
        "FAILED",
    },
    "VALIDATING": {
        "LOADING",
        "FAILED",
    },
    "LOADING": {
        "READY",
        "FAILED",
    },
    "READY": set(),
    "FAILED": set(),
}


def get_status(
    model_id: str,
) -> str:

    model = get_model(
        model_id
    )

    return model[
        "status"
    ]


def can_transition(
    current_status: str,
    new_status: str,
) -> bool:

    return (
        new_status
        in ALLOWED_TRANSITIONS[
            current_status
        ]
    )


def transition_to(
    model_id: str,
    new_status: str,
):

    current_status = (
        get_status(
            model_id
        )
    )

    if not can_transition(
        current_status,
        new_status,
    ):
        raise ValueError(
            f"Invalid transition: "
            f"{current_status} -> "
            f"{new_status}"
        )

    update_model_status(
        model_id,
        new_status,
    )
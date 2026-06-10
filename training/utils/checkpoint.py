import json
from pathlib import Path


def save_config(
    config,
    save_path: str,
):

    save_path = Path(
        save_path
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        save_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config.__dict__,
            file,
            indent=4,
        )
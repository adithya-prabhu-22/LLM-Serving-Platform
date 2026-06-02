from pathlib import Path


PROJECT_DIRS = [
    "core/models",
    "core/config",

    "backend/api",
    "backend/services",
    "backend/database",
    "backend/schemas",

    "frontend/templates",
    "frontend/static/css",
    "frontend/static/js",

    "storage/uploads",
    "storage/deployed_models",
    "storage/logs",

    "infrastructure/docker",
    "infrastructure/kubernetes",
    "infrastructure/monitoring/prometheus",
    "infrastructure/monitoring/grafana/dashboards",

    "tests/sanity",

    "docs",

    "requirements",
]


PROJECT_FILES = [
    "core/__init__.py",

    "core/models/__init__.py",
    "core/models/activations.py",
    "core/models/attention.py",
    "core/models/embeddings.py",
    "core/models/feedforward.py",
    "core/models/normalization.py",
    "core/models/transformer_block.py",
    "core/models/gpt.py",

    "core/config/__init__.py",
    "core/config/gpt_config.py",

    "backend/__init__.py",
    "backend/main.py",

    "backend/api/__init__.py",
    "backend/api/upload.py",
    "backend/api/models.py",
    "backend/api/chat.py",

    "backend/services/__init__.py",
    "backend/services/model_loader.py",
    "backend/services/generator.py",
    "backend/services/validator.py",

    "backend/database/__init__.py",
    "backend/database/db.py",
    "backend/database/models.py",

    "backend/schemas/__init__.py",
    "backend/schemas/upload_schema.py",
    "backend/schemas/chat_schema.py",

    "frontend/templates/index.html",
    "frontend/templates/upload.html",
    "frontend/templates/chat.html",

    "frontend/static/css/style.css",
    "frontend/static/js/app.js",

    "storage/uploads/.gitkeep",
    "storage/deployed_models/.gitkeep",
    "storage/logs/.gitkeep",

    "infrastructure/docker/Dockerfile",
    "infrastructure/docker/docker-compose.yml",

    "infrastructure/kubernetes/deployment.yaml",
    "infrastructure/kubernetes/service.yaml",
    "infrastructure/kubernetes/hpa.yaml",

    "infrastructure/monitoring/prometheus/prometheus.yml",

    "docs/architecture.md",
    "docs/api.md",
    "docs/deployment.md",

    "tests/sanity/test_activations.py",
    "tests/sanity/test_attention.py",
    "tests/sanity/test_embeddings.py",
    "tests/sanity/test_feedforward.py",
    "tests/sanity/test_normalization.py",
    "tests/sanity/test_transformer_block.py",
    "tests/sanity/test_gpt.py",
    "tests/sanity/run_all.py",
    "tests/sanity/test_gpt_config.py",

    "requirements/base.txt",
    "requirements/dev.txt",
    "requirements/serving.txt",

    ".gitignore",
    "README.md",
    "pyproject.toml",
]


def create_project_structure():
    root = Path.cwd()

    for directory in PROJECT_DIRS:
        (root / directory).mkdir(
            parents=True,
            exist_ok=True,
        )

    for file_path in PROJECT_FILES:
        path = root / file_path

        if not path.exists():
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            path.touch()

    print("LLM Serving Platform structure created successfully.")


if __name__ == "__main__":
    create_project_structure()
# LLM Serving Platform

A lightweight end-to-end platform for training, managing, and serving decoder-only Large Language Models (LLMs).

This project focuses on providing a simple and extensible infrastructure for:

* Training GPT-style language models
* Managing model artifacts
* Exporting models using SafeTensors
* Monitoring training and inference
* Deploying models for inference

---

## Features

### Model Architecture

* Decoder-only Transformer
* Multi-Head Causal Self Attention
* Rotary Positional Embeddings (RoPE)
* Flash Attention support
* KV Cache support
* Ring KV Cache support
* Configurable Feed Forward Network
* Configurable activation functions
* Layer Normalization
* SafeTensor export

### Training

* GPT-2 tokenizer
* Text chunking dataset pipeline
* Dynamic batch collation
* Cross-Entropy loss
* Validation loss tracking
* Perplexity evaluation
* Gradient clipping
* Gradient accumulation
* Automatic checkpointing
* Resume training support
* Best model tracking
* Automatic mixed precision (AMP)
* Cosine learning rate scheduling
* SafeTensor model exports

### Infrastructure

* Docker support
* Prometheus monitoring
* Grafana dashboards
* Jenkins automation
* Future Kubernetes support

---

## Repository Structure

```text
project-root/
│
├── core/
│   ├── cache/
│   ├── config/
│   ├── models/
│   └── generation/
│
├── training/
│   ├── datasets/
│   ├── trainer/
│   ├── utils/
│   └── train.py
│
├── backend/
│   ├── api/
│   ├── database/
│   ├── schemas/
│   └── services/
│
├── frontend/
│   ├── static/
│   └── templates/
│
├── storage/
│   ├── uploads/
│   ├── deployed_models/
│   └── logs/
│
├── infrastructure/
│   ├── docker/
│   ├── monitoring/
│   │   ├── prometheus/
│   │   └── grafana/
│   └── kubernetes_future/
│
├── tests/
├── docs/
└── requirements/
```

---

## Training Pipeline

The training pipeline consists of:

1. Dataset loading
2. Tokenization
3. Sequence chunking
4. Data collation
5. Forward pass
6. Loss computation
7. Backpropagation
8. Validation evaluation
9. Checkpoint creation
10. SafeTensor export

---

## Training Configuration

Example configuration:

```json
{
  "model": {
    "block_size": 1024,
    "d_model": 768,
    "num_heads": 12,
    "num_layers": 12,
    "dropout": 0.1,
    "activation": "gelu",
    "use_flash_attention": true,
    "cache_type": "ring"
  },

  "training": {
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "epochs": 10,
    "eval_interval": 500,
    "save_interval": 1000,
    "max_grad_norm": 1.0
  },

  "paths": {
    "train_dir": "./data/train",
    "val_dir": "./data/val",
    "checkpoint_dir": "./checkpoints",
    "output_dir": "./outputs"
  }
}
```

---

## Running Training

```bash
python training/train.py \
    --config configs/train_config.json
```

---

## Checkpointing

The platform automatically saves:

```text
checkpoint_step_1000.pt
checkpoint_step_2000.pt
...
```

Each checkpoint contains:

* Model state
* Optimizer state
* Scheduler state
* AMP scaler state
* Current epoch
* Current step
* Best validation loss

Training can resume automatically from the latest checkpoint.

---

## Model Export

Models are exported as:

```text
model.safetensors
best_model.safetensors
```

Configuration is also exported:

```text
config.json
```

This enables reproducible inference deployments.

---

## Monitoring

Planned monitoring stack:

### Prometheus

Tracks:

* Training throughput
* GPU utilization
* Memory usage
* Request latency
* Model metrics

### Grafana

Provides dashboards for:

* Training monitoring
* Inference monitoring
* Resource utilization
* Model health

---

## Deployment Architecture (V1)

```text
                 ┌──────────────────┐
                 │ User Interface   │
                 └─────────┬────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ API Server       │
                 └─────────┬────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ Model Service    │
                 └─────────┬────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ GPT Model        │
                 └──────────────────┘
```

---

## Roadmap

### Current

* GPT architecture
* Training pipeline
* Checkpointing
* SafeTensor export
* Validation tracking
* AMP support
* Gradient accumulation

### Planned

* Inference API
* Model Registry
* Model Versioning
* Jenkins automation
* Docker deployment
* Prometheus metrics
* Grafana dashboards
* Multi-model serving
* Kubernetes deployment
* Distributed training

---

## License

This project is intended for educational, research, and personal deployment purposes.

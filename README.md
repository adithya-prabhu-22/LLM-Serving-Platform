# LLM Serving Platform

A lightweight, end-to-end platform for training, managing, and serving decoder-only Large Language Models (LLMs). Built with simplicity and extensibility in mind, this platform allows you to train GPT-style models from scratch, monitor training, export weights, and deploy them for inference with minimal friction.

---

## Table of Contents

- [Features](#features)
  - [Model Architecture](#model-architecture)
  - [Training Pipeline](#training-pipeline)
  - [Serving and Infrastructure](#serving-and-infrastructure)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Training Configuration](#training-configuration)
- [Checkpointing and S3 Integration](#checkpointing-and-s3-integration)
- [Observability and Metrics](#observability-and-metrics)
- [Model Behavior and Limitations](#model-behavior-and-limitations)
- [Deployment Architecture](#deployment-architecture)
- [Trained Models](#trained-models)
- [Requirements](#requirements)
- [Roadmap](#roadmap)
- [License](#license)
- [Contact](#contact)

---

## Features

### Model Architecture

- Decoder-only Transformer
- Multi-Head Causal Self-Attention
- Rotary Positional Embeddings (RoPE)
- Flash Attention support
- KV Cache and Ring KV Cache
- Configurable activation functions (GELU, ReLU, SiLU, Tanh)
- Layer Normalization
- SafeTensor export

### Training Pipeline

- GPT-2 tokenizer integration
- Streaming dataset builder (supports Wikipedia, Healix-Shot, and others)
- Non-overlapping sequence chunking (configurable)
- Cross-Entropy loss with `ignore_index`
- Validation loss and Perplexity tracking
- Gradient accumulation and clipping
- Automatic Mixed Precision (AMP)
- Cosine annealing learning rate scheduler
- Automatic checkpointing and resumption
- Best model tracking
- S3 checkpoint upload — local checkpoints are uploaded to S3 and deleted locally to save disk space

### Serving and Infrastructure

- FastAPI backend with dynamic model loading from S3
- Web-based UI (HTML + CSS + JS)
- Docker deployment
- AWS S3 integration for model storage
- Model registry (local + S3)
- LangSmith tracing for end-to-end observability
- Latency percentile tracking (P50, P95, P99)
- Token throughput and streaming metrics

---

## Repository Structure

```text
LLM-Serving-Platform/
│
├── core/                     # Core model architecture
│   ├── cache/                # KV Cache implementations
│   ├── config/                # GPTConfig
│   └── models/                # Attention, Embeddings, FFN, RoPE, etc.
│
├── training/                 # Training pipeline
│   ├── dataset_builder/      # Build chunks from raw datasets
│   ├── datasets/              # StreamingDataset, chunk loader
│   ├── trainer/                # Training loop, evaluator, loss
│   └── utils/                  # Checkpointing, Safetensor export, tokenizer
│
├── backend/                  # FastAPI serving platform
│   ├── api/                    # Routes, schemas
│   ├── services/               # Model loader, registry, inference engine
│   └── database/               # SQLite model registry
│
├── frontend/                 # Web UI
│   ├── static/                 # CSS, JS
│   └── templates/              # HTML pages
│
├── infrastructure/           # Deployment
│   └── docker/                 # Dockerfile
│
├── storage/                  # Local storage (models, logs, checkpoints)
├── tests/                     # Sanity tests
├── requirements/              # Python dependencies
└── docs/                       # Documentation
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/adithya-prabhu-22/LLM-Serving-Platform.git
cd LLM-Serving-Platform
```

### 2. Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --no-cache-dir -r requirements/base.txt -r requirements/serving.txt
```

### 3. Train a Model (Example)

```bash
python -m training.train_streaming --config training/configs/gpt_150m_fast.json
```

### 4. Deploy the Serving Platform

```bash
docker build -t llm-backend -f infrastructure/docker/backend/Dockerfile .
docker run -d --name llm-backend -p 8000:8000 --env-file .env llm-backend
```

### 5. Open the UI

Visit `http://<your-ec2-ip>:8000` in your browser. Select your model and start generating text.

---

## Training Configuration

Example configuration for a 150M model with fast training (`gpt_150m_fast.json`):

```json
{
  "model": {
    "block_size": 1024,
    "d_model": 768,
    "num_heads": 12,
    "num_layers": 12,
    "dropout": 0.1,
    "ff_dim": 3072,
    "activation": "gelu",
    "qkv_bias": false,
    "use_flash_attention": false,
    "cache_type": "ring"
  },
  "training": {
    "batch_size": 2,
    "gradient_accumulation_steps": 16,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "epochs": 1,
    "eval_interval": 500,
    "save_interval": 1000000,
    "max_grad_norm": 1.0,
    "num_workers": 4,
    "seed": 42
  },
  "paths": {
    "checkpoint_dir": "storage/checkpoints_streaming",
    "output_dir": "storage/deployed_models/gpt_150m_fast"
  },
  "dataset": {
    "manifest": "storage/dataset_build/combined_150m_manifest.json"
  }
}
```

---

## Checkpointing and S3 Integration

Checkpoints are saved locally and automatically uploaded to S3 to prevent disk space exhaustion during long training runs.

- Local checkpoint saved → uploaded to S3 → deleted locally
- Resumable training from S3 checkpoints
- Centralized model storage for disaster recovery

---

## Observability and Metrics

The platform is instrumented with **LangSmith** for end-to-end tracing of every inference request. Metrics are automatically captured and visualized through the LangSmith dashboard.

### Tracked Metrics

| Metric | Description |
|---|---|
| First Token Latency (P50 / P95) | Time to produce the first generated token |
| End-to-End Latency (P50 / P95 / P99) | Total wall-clock time for a complete generation |
| Token Throughput | Tokens generated per second |
| Streaming Adoption | Percentage of requests using the streaming endpoint |
| Error Rate | Percentage of failed requests |
| Token Usage | Total prompt + generated tokens per request |
| Trace Count | Total inference requests tracked |

### Traced Runs

Each inference request generates a hierarchical trace:

- `LLM-Generate-Stream` (parent)
  - `Tokenizer-Encode` (child)
  - `Tokenizer-Decode` (child)

### Sample Live Metrics

| Metric | Value |
|---|---|
| First Token Latency (P50) | 0.08s |
| First Token Latency (P95) | 0.11s |
| End-to-End Latency (P50) | 8.29s |
| End-to-End Latency (P95) | 8.42s |
| End-to-End Latency (P99) | 8.46s |
| Streaming Adoption | 75% |

All metrics are viewable in real-time at [smith.langchain.com](https://smith.langchain.com) under the `medical-llm-platform` project.

---

## Model Behavior and Limitations

This model was trained as a decoder-only language model with a document-completion objective. Its behavior is strongly shaped by that training objective, and understanding its strengths and limitations is essential for correct usage.

### What Works Well

| Prompt Style | Example Prompt | Output Quality |
|---|---|---|
| Clinical case reports | "A 62-year-old male presented to the emergency department with..." | High – generates coherent patient narratives with vitals, labs, and clinical reasoning |
| Research abstract continuation | "In this randomized controlled trial, we evaluated the efficacy..." | High – produces structured abstracts with statistical notation (RR, 95% CI) and conclusions |
| Treatment protocol continuation | "The recommended treatment approach for patients with newly diagnosed type 2 diabetes includes..." | High – generates guideline-style text with drug names and monitoring plans |
| Mechanism of action | "The mechanism of action of metformin involves..." | Medium-High – describes biochemical pathways, though can drift |
| Drug trial results | "In this phase III trial, patients with... were treated with pembrolizumab..." | High – produces NCT registration numbers and endpoint discussions |

### What Does Not Work Well

| Prompt Style | Example Prompt | Observed Behavior |
|---|---|---|
| Factual question answering | "What is dengue?" | Generates off-topic text unrelated to the question |
| Definition requests | "Define chemotherapy" | Drifts into unrelated medical content |
| Knowledge retrieval | "Explain the causes of malaria" | Produces plausible-sounding but factually incorrect content |

### Why This Happens

- **Training Objective Mismatch** — The model was trained to continue documents, not to answer questions. It has no explicit instruction-following or question-answering training.
- **Model Capacity** — At 123M parameters, the model cannot store and retrieve factual knowledge reliably. Factual Q&A typically requires 1B+ parameters or retrieval augmentation.
- **Repetition Collapse** — After ~150–200 tokens, small models tend to loop on high-frequency phrases from training data (e.g., repeating acronyms or clinical boilerplate).
- **Hallucination** — The model generates statistically plausible text without factual grounding. It will confidently produce incorrect medical information.

### Best Practices for Usage

- Use continuation-style prompts (provide document context for the model to extend)
- Cap generation at 150–200 tokens to avoid repetition collapse
- Use lower temperature (0.7) for more coherent output
- Do not rely on the model for factual information – treat output as synthetic text, not medical advice

### Roadmap to Address Limitations

| Limitation | Planned Fix |
|---|---|
| Factual inaccuracy | Add Retrieval-Augmented Generation (RAG) over a curated medical knowledge base |
| No instruction following | Fine-tune on instruction datasets (e.g., medical QA pairs) |
| Limited context retention | Scale to 1B+ parameters with longer context (2048–4096 tokens) |
| Repetition collapse | Add repetition penalty and nucleus (top-p) sampling at inference time |

---

## Deployment Architecture

The platform follows a clean separation of concerns with a scalable, cloud-native architecture.

### User-Facing Layer

- Browser-based web interface
- REST API for inference requests

### Backend Layer

- FastAPI server handling model loading, inference, and registry management
- Dynamic model loading from S3

### Observability Layer

- LangSmith for end-to-end tracing
- Latency percentile tracking (P50 / P95 / P99)
- Token throughput and error rate monitoring

### Storage Layer

- AWS S3 for model weights, configurations, and training checkpoints

---

## Trained Models

| Model | Parameters | Dataset | Tokens | Final Loss | Context | Notes |
|---|---|---|---|---|---|---|
| gpt-150m-fast-v1 | 123.5M | General + Medical | 2.5B | 2.99 | 1024 | Strong on document continuation; weak on factual Q&A |

---

## Requirements

- Python 3.9+
- PyTorch 2.3+
- Docker
- AWS CLI (optional, for S3 integration)
- LangSmith API key (for tracing)

---

## Roadmap

### Current Features

- [x] GPT architecture with RoPE
- [x] Streaming training pipeline
- [x] Checkpointing with S3 sync
- [x] SafeTensor export
- [x] FastAPI serving platform
- [x] LangSmith tracing with P50 / P95 / P99 metrics
- [x] Docker deployment

### Planned Features

- [ ] Retrieval-Augmented Generation (RAG) for factual grounding
- [ ] Instruction fine-tuning on medical QA datasets
- [ ] Scaling to 1B+ parameters with extended context
- [ ] Repetition penalty and nucleus sampling at inference
- [ ] Multi-GPU training (DDP/FSDP)
- [ ] Distributed inference
- [ ] Quantization (GPTQ, AWQ)
- [ ] LoRA fine-tuning support
- [ ] Kubernetes deployment
- [ ] CI/CD via GitHub Actions

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

**Adithya Prabhu**
GitHub: [adithya-prabhu-22](https://github.com/adithya-prabhu-22)
Project Link: [LLM-Serving-Platform](https://github.com/adithya-prabhu-22/LLM-Serving-Platform)
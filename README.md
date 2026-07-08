# SATD: A Dual-Track Evidence-Driven Framework for Training-Free Video Anomaly Detection

Welcome to the official repository for **SATD: A Dual-Track Evidence-Driven Framework for Training-Free Video Anomaly Detection**.

This repository provides a novel pipeline that eliminates the need for expensive optical flow computations and mitigates VLM hallucinations through a "Hypothesis-Evidence-Judgement" reasoning loop.

## 🚀 Architecture Overview

Our framework consists of four primary stages driven by Large Vision-Language Models (VLMs) and Open-Vocabulary probes:

1. **Stage 1: Memory-Augmented Event Segmenter (`mem_event_segmenter.py`)**
   Replaces traditional dense optical flow (RAFT) with a high-speed Memory-Bank Novelty Score, computing temporal graphs to slice long videos into context-aware candidate events at an end-to-end pre-processing speed of ~450 FPS (~82x speedup).
2. **Stage 2: VLM Hypothesis Generation (`vlm_reasoner.py`)**
   The VLM acts as an initial investigator, proposing double-track hypotheses (Normal vs. Abnormal) and generating specific lists of entities, objects, and actions to look for.
3. **Stage 3: Evidence Retriever (`evidence_retriever.py` & `yolo_detector.py`)**
   Visual probes (YOLO-World & CLIP) dynamically ingest the VLM's generated vocabulary and search the physical frames. They output probabilities indicating the physical existence of the hypothesized evidence.
4. **Stage 4: VLM Judge (`vlm_reasoner.py`)**
   The VLM reviews its initial hypothesis alongside the hard physical probabilities returned by Stage 3. It then makes a final, grounded determination and assigns a continuous anomaly score [0.0, 1.0].

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/xxxx/SATD-VAD.git
cd SATD-VAD

# Create a virtual environment
conda create -n satd python=3.10
conda activate satd

# Install dependencies
pip install -r requirements.txt
```

## 🛠️ Quick Start

**1. Set up your API Keys**
This project requires access to a Vision-Language Model API (e.g., Aliyun Qwen-VL-Max).
```bash
# Windows (PowerShell)
$env:DASHSCOPE_API_KEY="your-api-key-here"

# Linux/Mac
export DASHSCOPE_API_KEY="your-api-key-here"
```

**2. Download Weights**
Download the `yolov8s-world.pt` weights and place them in the project root.

**3. Run the Demo**
You can run the zero-shot single video inference demo:
```bash
python demo.py
```

## 📊 Benchmarks
We have extensively evaluated our framework on standard benchmarks in a strictly **Zero-Shot** setting.

| Dataset | Metric | Score |
| :--- | :--- | :---: |
| **UCF-Crime** | AUC | 89.16% |
| **XD-Violence** | AP | 72.01%* |

*(Note: XD-Violence score reflects vision-only subset evaluation)*

## 📄 Citation
If you find this code useful in your research, please consider citing our paper (Coming Soon).

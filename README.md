# Code for "MAR: Metacognitive Agentic Reasoning for Multimodal Fake News Detection"(KDD 2026)



## Project Structure

```
 MAR
 ├── README.md                           # Project documentation
 ├── MAR_framework/                      # Core analysis module
 │   ├── main.py                         # Main program entry point
 │   ├── main_eval.py                    # Evaluation module
 │   ├── NewsAnalyzer.py                 # Core news analyzer class
 │   ├── external_knowledge.py           # External knowledge retrieval module
 │   └── prompt_template/                # Prompt templates
 │       ├── news_analyzer/              # News analyzer prompts
 │       │   ├── system_prompts.py
 │       │   └── user_prompts.py
 │       └── external_knowledge/         # External knowledge retrieval prompts
 │           ├── system_prompts.py
 │           └── user_prompts.py
 ├── data_processing/                    # Data preprocessing
 │   ├── processing_mmfakebench.py       # MMFakeBench dataset processing
 │   ├── processing_dgm4.py              # DGM4 dataset processing
 │   ├── MMFakeBench/                    # Processed MMFakeBench data
 │   │   ├── train_set.json
 │   │   └── test_set.json
 │   └── DGM4/                           # Processed DGM4 data
 │       ├── train_final.json
 │       └── test_final.json
 ├── dataset/                            # Original dataset storage
 │   ├── MMFakeBench/
 │   └── DGM4/
 ├── dataset_final/                      # Processed data output directory
 │   ├── MMFakeBench/
 │   │   └── Qwen3-VL-32B-Instruct-FP8/
 │   └── DGM4/
 │       └── Qwen3-VL-32B-Instruct-FP8/
 └── reverse_image/                      # Reverse image search module
     ├── reverse_image_search.py         # Reverse image search main program
     ├── serper_lens.py                  # Serper search engine client
     ├── download_inverse_annotations_dirs.py  # Download inverse annotations
     └── utils.py                        # Utility functions
```

## Requirements

### Python Dependencies

```bash
 pip install torch langchain langchain-openai langchain-huggingface
 pip install loguru tqdm
 pip install scikit-learn numpy
 pip install httpx requests
 pip install -U ddgs
```

### Model Requirements

- **Multi-modal Language Model**: Qwen3-VL-32B-Instruct-FP8(path:`~/model_path/Qwen3-VL-32B-Instruct-FP8`)
- **Embedding Model**: BGE-base-en-v1.5 for knowledge retrieval (path: `~/model_path/bge-base-en-v1.5/`)

## Datasets

### Supported Datasets

The project supports the following two datasets:

1. **MMFakeBench**: [liuxuannan/MMFakeBench: [ICLR 2025\] MMFakeBench: A Mixed-Source Multimodal Misinformation Detection Benchmark for LVLMs](https://github.com/liuxuannan/MMFakeBench)
2. **DGM4**: [rshaojimmy/MultiModal-DeepFake: [TPAMI 2024 & CVPR 2023\] PyTorch code for DGM4: Detecting and Grounding Multi-Modal Media Manipulation and beyond](https://github.com/rshaojimmy/MultiModal-DeepFake)

### Dataset Preparation

1. Download the original datasets to the `dataset/` directory
2. Run the data preprocessing scripts:

```bash
 # Process MMFakeBench
 cd data_processing
 python processing_mmfakebench.py
 
 # Process DGM4
 python processing_dgm4.py
```

1. Add reverse image search annotations for MMFakeBench and DGM4 dataset:

```bash
 cd reverse_image
 # Configure SERPER_API_KEYS and IMGBB_API_KEYS
 python reverse_image_search.py
```

In the reverse image search section, we referred to the work of [S-Abdelnabi/OoC-multi-modal-fc: Code for our CVPR'22 paper: Open-Domain, Content-based, Multi-modal Fact-checking of Out-of-Context Images via Online Resources](https://github.com/S-Abdelnabi/OoC-multi-modal-fc)

## Usage

**Before all the operations, you need to ensure that the paths of the model and the data are correct.**

### 1. Start vLLM Service

First, you should start the local deployment of the vllm service.

### 2. Proxy

Based on our experience, if your web search is not working properly, you can refer to [deedy5/ddgs: A metasearch library that aggregates results from diverse web search services](https://github.com/deedy5/ddgs) to configure the proxy.

### 3. Run Detection

```bash
 cd MAR_framework
 
 # Run with default parameters (processes MMFakeBench test set)
 python main.py
```

### 4. Processing DGM4 Dataset

To process the DGM4 dataset, uncomment the corresponding configuration in [`main.py`](MAR_framework/main.py), then you need to modify the path of the corresponding DGM4 dataset.

```python
 # input_file_map = {
 #     "train": '../data_processing/DGM4/train_final.json',
 #     "test": '../data_processing/DGM4/test_final.json'
 # }
```

## Evaluation

Run the evaluation script to check model performance:

```bash
 cd MAR_framework
 python main_eval.py
```

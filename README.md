# 🚀 data-tagger

[中文](./README_CN.md)

<p align="center">
  <a href="https://pypi.org/project/data-tagger/"><img alt="Python version" src="https://img.shields.io/badge/python-3.11+-blue.svg"></a>
  <a href="https://github.com/Tendo33/data-tagger/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Tendo33/data-tagger"></a>
  <a href="https://github.com/Tendo33/data-tagger/pulls"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
  <a href="https://github.com/Tendo33/data-tagger/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Tendo33/data-tagger?style=social"></a>
</p>

<h3 align="center">
  An Efficient, Flexible, Multi-task Batch Data Labeling Tool
</h3>

<p align="center">
  <b>data-tagger</b> provides an all-in-one solution for large-scale dataset labeling, including quality assessment, difficulty evaluation, intent classification, safety detection, reward scoring, language identification, and embedding generation. It seamlessly supports both local VLLM models and remote API inference, with flexible configuration and easy extensibility.
</p>

---

## Table of Contents
- [🚀 data-tagger](#-data-tagger)
  - [Table of Contents](#table-of-contents)
  - [🌟 Features](#-features)
  - [� Installation](#-installation)
  - [🚀 Quick Start](#-quick-start)
    - [Local VLLM Inference](#local-vllm-inference)
    - [Remote API Inference](#remote-api-inference)
  - [⚙️ Configuration](#️-configuration)
  - [🧩 Task Types \& Data Fields](#-task-types--data-fields)
    - [Supported Task Types](#supported-task-types)
    - [Output Data Fields](#output-data-fields)
  - [🛠️ Data Formatting Tool](#️-data-formatting-tool)
  - [❓ FAQ](#-faq)
  - [📂 Directory Structure](#-directory-structure)
  - [🤝 Contributing \& Support](#-contributing--support)

---

## 🌟 Features

- **Multi-task Support**: Built-in support for various labeling tasks, such as QUALITY, DIFFICULTY, CLASSIFICATION, SAFETY, REWARD, LANGUAGE, and EMBEDDING.
- **Dual Inference Modes**: Freely choose between local **VLLM** model inference or remote **API** service, balancing performance and cost.
- **Efficient Data Processing**: Includes data formatting tools for easy data cleaning and format conversion.
- **Flexible Configuration**: Customize task type, model, batch size, input/output fields, etc., via CLI or config files.
- **Embedding Storage**: Supports storing generated embeddings to local **Faiss** or distributed **Milvus**.
- **Easy to Extend**: Modular design makes it easy to add new labeling task types.

---

## 📦 Installation

- **Requirements**: **Python >= 3.11**
- It is recommended to use a virtual environment to avoid dependency conflicts.

```bash
# 1. Create and activate a virtual environment
cd data-tagger

uv sync
```

**Main dependencies** (see `pyproject.toml`):
- `vllm`: Local large model inference
- `lingua-language-detector`: Language identification
- `loguru`: Advanced logging
- `pydantic-settings`: Environment and config management
- `json-repair`: Repairing non-standard JSON output

---

## 🚀 Quick Start

We provide ready-to-use test scripts, or you can run commands manually.

### Local VLLM Inference

Example for local model classification task:

```bash
# Run the test script directly
bash scripts/vllm/classification_test.sh
```

Or run manually:

```bash
python -m datatagger.tagger.unified_tagger_vllm \
  --vllm_model_path <your local model path> \
  --tag_mission CLASSIFICATION \
  --input_file data/test_data/sample_data_for_vllm_tagger.jsonl \
  --output_file data/test_output/classification_vllm_output.jsonl \
  --prompt_field instruction \
  --batch_size 5 \
  --device 0
```

### Remote API Inference

Example for running multiple labeling tasks via API:

```bash
# Run the aggregate script directly
bash scripts/api/run_all_taggers_api.sh
```

Or run a single task manually (e.g., quality assessment):

```bash
python -m datatagger.tagger.unified_tagger_api \
  --api_model_name <your API model name> \
  --api_url <your API url> \
  --api_key <your API key> \
  --tag_mission QUALITY \
  --input_file data/test_data/sample_data_for_api_tagger.jsonl \
  --output_file data/test_output/quality_api_output.jsonl
```

---

## ⚙️ Configuration

`data-tagger` supports rich CLI parameters to control task behavior.

<details>
<summary><b>📚 Click to expand/collapse main parameter descriptions</b></summary>

| Parameter | Description |
|---|---|
| `--tag_mission` | **Required.** Task type, e.g., `QUALITY`, `DIFFICULTY`, `CLASSIFICATION`, etc. |
| `--input_file` / `--output_file` | **Required.** Input and output file paths. |
| `--prompt_field` / `--output_field` | Field names for prompt and response in the input file. |
| `--batch_size` | Batch size, default is `5`. |
| `--device` | **VLLM mode.** GPU device ID. |
| `--vllm_model_path` | **VLLM mode.** Local model path. |
| `--api_model_name` / `--api_url` / `--api_key` | **API mode.** API service parameters. |
| `--faiss_store_embeddings` / `--milvus_store_embeddings` | **EMBEDDING task.** Whether to store embeddings to Faiss or Milvus. |
| `...` | More parameters can be found in the `settings` directory and script comments. |

</details>

---

## 🧩 Task Types & Data Fields

### Supported Task Types

| Task Type | Description |
|---|---|
| **QUALITY** | Dialogue quality assessment. Scores (1-5) and briefly analyzes the overall quality of the input. |
| **DIFFICULTY** | Difficulty assessment. Analyzes the difficulty of understanding/solving the input, outputs a float 0-5. |
| **CLASSIFICATION** | Intent classification. Categorizes the main intent of the input, outputs primary and secondary tags. |
| **SAFETY** | Safety detection. Determines if the content involves violence, pornography, privacy, etc., outputs a safety label. |
| **REWARD** | Reward scoring. Quantitatively scores the reward value of the content (0-5). |
| **LANGUAGE** | Language identification. Identifies the main language type of the input. |
| **EMBEDDING** | Embedding generation. Converts the input into a vector for downstream ML tasks. |

### Output Data Fields

The processed `JSONL` file will add the following fields.

<details>
<summary><b>📄 Click to expand/collapse detailed field descriptions</b></summary>

| Field Name | Description | Example/Range |
|---|---|---|
| `id` | Auto-generated unique identifier | `"a1b2c3d4"` |
| `system`, `conversations`, `instruction`, `output` | Original data fields | ... |
| `prompt_field`, `output_field` | Prompt and output field names used in this task | `"instruction"`, `"output"` |
| `prompt_field_length`, `output_field_length` | Character length of prompt and output fields | `20`, `100` |
| **`difficulty`** | **[Difficulty]** Difficulty score, float 0-5 | `2.5` |
| **`input_quality`**, **`response_quality`** | **[Quality]** Input/output quality score, float 1-5 | `4.2`, `4.5` |
| **`input_quality_explanation`**, **`response_quality_explanation`** | **[Quality]** Brief explanation for quality score | `"Input is clear and detailed..."` |
| **`task_category`**, **`other_task_category`** | **[Classification]** Main and secondary task categories | `"Coding & Debugging"`, `["Information seeking"]` |
| **`language`** | **[Language]** Main language type | `"zh"`, `"en"` |
| **`safety`** | **[Safety]** Safety label | `"Safe"` |
| **`instruct_reward`** | **[Reward]** Reward score, float 0-5 | `3.8` |
| `min_neighbor_distance` | **[Embedding]** Minimum neighbor distance for similarity analysis | `0.12` |
| `repeat_count` | Repeat count for deduplication analysis | `1` |

</details>

<details>
<summary><b>🏷️ Click to expand/collapse all possible values for `task_category` and `safety`</b></summary>

  - **`task_category` possible values**:
    `Information seeking`, `Reasoning`, `Planning`, `Editing`, `Coding & Debugging`, `Math`, `Role playing`, `Data analysis`, `Creative writing`, `Advice seeking`, `Translation`, `Brainstorming`, `Others`

  - **`safety` possible values**:
    `Violent Crimes`, `Non-Violent Crimes`, `Sex-Related Crimes`, `Child Sexual Exploitation`, `Defamation`, `Specialized Advice`, `Privacy`, `Intellectual Property`, `Indiscriminate Weapons`, `Hate`, `Suicide & Self-Harm`, `Sexual Content`, `Elections`, `Code Interpreter Abuse`, `Safe`

</details>

---

## 🛠️ Data Formatting Tool

A convenient script is provided for batch formatting, cleaning, and standardizing large datasets.

```bash
python -m datatagger.formatter.data_formatter \
  --input_file <your raw data file> \
  --output_file <formatted output file> \
  --save_as jsonl
```

---

## ❓ FAQ

1.  **What input formats are supported?**

    > Supports `JSON` and `JSONL`. For large-scale data, it is strongly recommended to use `JSONL` (one JSON object per line).

2.  **How to customize model or API?**

    > **VLLM**: Specify the local model path via `--vllm_model_path`.
    > **API**: Configure via `--api_model_name`, `--api_url`, `--api_key`.

3.  **What embedding storage is supported?**

    > Currently supports local **Faiss** and distributed **Milvus** for storing and querying embeddings.

---

## 📂 Directory Structure

```text
datatagger/         # Main program module
  tagger/           # Labeling task implementations
  formatter/        # Data formatting tools
  settings/         # Config & parameter definitions
  utils/            # Utility functions
scripts/            # Common task scripts
data/               # Example data & outputs
```

---

## 🤝 Contributing & Support

- Please check the source code or contact the maintainer for questions.
- Contributions (issues/PRs) are welcome!
- [中文版 README 请见 README_CN.md](./README_CN.md)

---


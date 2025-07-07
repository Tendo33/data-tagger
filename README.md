# 🚀 data-tagger

> **Efficient, Flexible, Multi-task Batch Data Labeling Tool**

[中文](./README_CN.md)

---

<p align="center">
  <b>data-tagger</b> is an efficient tool for large-scale dataset quality assessment, difficulty evaluation, classification, safety detection, reward scoring, language identification, and embedding generation. Supports both local VLLM and remote API inference, with flexible configuration and easy integration.
</p>

---

## 🌟 Features

- **Multi-task Batch Labeling**: Supports QUALITY, DIFFICULTY, CLASSIFICATION, SAFETY, REWARD, LANGUAGE, EMBEDDING, etc.
- **Local & API Inference**: Choose between local VLLM model or remote API.
- **Efficient Data Formatting**: Built-in tools for data cleaning and format conversion.
- **Flexible Configuration**: Customize tasks, models, batch size, field names, etc.
- **Embedding Storage**: Supports local Faiss or Milvus.
- **Easy to Extend**: Modular design for new task types.

---

## 📦 Installation & Dependencies

- **Python >= 3.11**
- Recommended: use a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or install via pyproject.toml
```

**Main dependencies** (see `pyproject.toml`):
- vllm
- lingua-language-detector
- loguru
- pydantic-settings
- json-repair

---

## 🚀 Quick Start

### Local VLLM Inference

```bash
bash scripts/classification_test.sh
```

Or run manually:

```bash
python -m datatagger.tagger.unified_tagger_vllm \
  --vllm_model_path <local model path> \
  --tag_mission CLASSIFICATION \
  --input_file <input file> \
  --output_file <output file> \
  --prompt_field instruction \
  --output_field output \
  --batch_size 5 \
  --device 0
```

### Remote API Inference

```bash
bash scripts/run_all_taggers_api.sh
```

Or run manually:

```bash
python -m datatagger.tagger.unified_tagger_api \
  --api_model_name <API model name> \
  --api_url <API url> \
  --api_key <API key> \
  --tag_mission QUALITY \
  --input_file <input file> \
  --output_file <output file>
```

---

## ⚙️ Configuration (Main Parameters)

| Parameter                        | Description                                                        |
|-----------------------------------|--------------------------------------------------------------------|
| `--tag_mission`                   | Task type (QUALITY, DIFFICULTY, CLASSIFICATION, etc.)              |
| `--input_file` / `--output_file`  | Input/output file path                                             |
| `--prompt_field` / `--output_field` | Prompt/response field name in input file                         |
| `--batch_size`                    | Batch size                                                         |
| `--device`                        | GPU device id (local inference)                                    |
| `--vllm_model_path`               | Local model path (VLLM)                                            |
| `--api_model_name` / `--api_url` / `--api_key` | API mode parameters                                 |
| `--faiss_store_embeddings` / `--milvus_store_embeddings` | Whether to store embedding                |
| ...                               | See settings files and script comments for more details            |

---

## 🛠️ Data Formatting Tool

Batch format conversion, cleaning, and standardization for large datasets:

```bash
python -m datatagger.formatter.data_formatter \
  --input_file <raw data file> \
  --output_file <formatted file> \
  --save_as jsonl
```

---

## 📚 Scripts & Usage Examples

- `scripts/classification_test.sh`: Local classification task example
- `scripts/run_all_taggers_api.sh`: API multi-task batch processing example
- `scripts/api/`, `scripts/vllm/`: Individual task example scripts

---

## ❓ FAQ

1. **What input formats are supported?**
   - Supports JSON/JSONL, JSONL (one sample per line) is recommended.
2. **How to customize model or API?**
   - Modify config files or script arguments.
3. **How to extend new tasks?**
   - Refer to implementations in `datatagger/tagger/`, inherit `BaseUnifiedTagger`.
4. **What embedding storage is supported?**
   - Supports local Faiss and Milvus.

---

## 🧩 Task Types & Data Field Description

### Task Types

| Task Type        | Description                                                                                   |
|------------------|---------------------------------------------------------------------------------------------|
| QUALITY          | Quality assessment. Scores and briefly analyzes the overall quality of the input.            |
| DIFFICULTY       | Difficulty assessment. Analyzes the complexity of understanding/solving the input, outputs a float 0-5. |
| CLASSIFICATION   | Classification task. Categorizes the main intent/task of the input, outputs primary and secondary tags. |
| SAFETY           | Safety detection. Determines if the content involves violence, pornography, privacy, etc., outputs a safety label. |
| REWARD           | Reward scoring. Quantitatively scores the reward value of the content.                       |
| LANGUAGE         | Language identification. Identifies the main language type of the input.                     |
| EMBEDDING        | Embedding generation. Converts the input into a vector for downstream ML tasks.              |

### Data Field Description

| Field Name                   | Description                                                                 | Example/Range                      |
|------------------------------|-----------------------------------------------------------------------------|------------------------------------|
| id                           | Unique identifier (auto-generated)                                          | "a1b2c3d4"                         |
| system                       | System prompt (optional)                                                    | "You are an expert..."             |
| conversations                | List of conversation turns, with role (human/gpt) and text                  | [{"from": "human", "value": ...}]  |
| instruction                  | Main user instruction/question                                              | "Write a sorting algorithm"        |
| output                       | Main AI output/answer                                                       | "Here is a sorting algorithm..."   |
| prompt_field                 | Name of the prompt field (e.g., instruction)                                | "instruction"                      |
| output_field                 | Name of the output field (e.g., output)                                     | "output"                           |
| prompt_field_length          | Character length of the prompt field                                        | 20                                 |
| output_field_length          | Character length of the output field                                        | 100                                |
| intent                       | User intent analysis result (see prompt_utils.py)                           | "The user wants to ..."            |
| knowledge                    | Knowledge required to solve the task                                        | "Requires knowledge of ..."         |
| difficulty                   | Difficulty score, float 0-5, higher means harder                            | 2.5                                |
| input_quality                | Input quality score, float 1-5, higher is better                            | 4.2                                |
| response_quality             | Output quality score, float 1-5, higher is better                           | 4.5                                |
| input_quality_explanation    | Brief explanation for input quality score                                   | "Input is clear and detailed..."    |
| response_quality_explanation | Brief explanation for output quality score                                  | "Answer is accurate and clear..."   |
| task_category                | Main task category (see ALLOWED_TASK_CATEGORIES)                            | "Coding & Debugging"               |
| other_task_category          | Other related task categories (list)                                        | ["Information seeking"]             |
| language                     | Main language type (e.g., "zh", "en")                                     | "zh"                               |
| safety                       | Safety label (see ALLOWED_SAFETY_LABELS)                                    | "Safe"                             |
| instruct_reward              | Reward score, float 0-5                                                     | 3.8                                |
| task_category_generator      | Task category generator (optional, source of classification)                | "prompt_utils"                     |
| min_neighbor_distance        | Minimum neighbor distance (for embedding similarity analysis)                | 0.12                               |
| repeat_count                 | Repeat count (e.g., for deduplication analysis)                             | 1                                  |
| min_similar_instruction      | Most similar instruction (if any)                                           | "Write a sorting algorithm"        |

#### task_category possible values (see ALLOWED_TASK_CATEGORIES):
Information seeking, Reasoning, Planning, Editing, Coding & Debugging, Math, Role playing, Data analysis, Creative writing, Advice seeking, Translation, Brainstorming, Others

#### safety possible values (see ALLOWED_SAFETY_LABELS):
Violent Crimes, Non-Violent Crimes, Sex-Related Crimes, Child Sexual Exploitation, Defamation, Specialized Advice, Privacy, Intellectual Property, Indiscriminate Weapons, Hate, Suicide & Self-Harm, Sexual Content, Elections, Code Interpreter Abuse, Safe

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


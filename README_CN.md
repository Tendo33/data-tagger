# 🚀 data-tagger

[English](./README.md)

<p align="center">
  <a href="https://pypi.org/project/data-tagger/"><img alt="Python version" src="https://img.shields.io/badge/python-3.11+-blue.svg"></a>
  <a href="https://github.com/Tendo33/data-tagger/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Tendo33/data-tagger"></a>
  <a href="https://github.com/Tendo33/data-tagger/pulls"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
  <a href="https://github.com/Tendo33/data-tagger/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Tendo33/data-tagger?style=social"></a>
</p>

<h3 align="center">
  一个高效、灵活、多任务的批量数据标注工具
</h3>

<p align="center">
  <b>data-tagger</b> 提供一站式大规模数据集标注解决方案，包括质量评估、难度定级、意图分类、安全检测、奖励打分、语种识别和向量生成。支持本地 VLLM 模型和远程 API 推理，配置灵活，易于扩展。
</p>

---

## 目录
- [🚀 data-tagger](#-data-tagger)
  - [目录](#目录)
  - [🌟 主要特性](#-主要特性)
  - [📦 安装指南](#-安装指南)
  - [🚀 快速开始](#-快速开始)
    - [本地 VLLM 推理](#本地-vllm-推理)
    - [远程 API 推理](#远程-api-推理)
  - [⚙️ 配置说明](#️-配置说明)
  - [🧩 任务类型与数据字段](#-任务类型与数据字段)
    - [支持的任务类型](#支持的任务类型)
    - [输出数据字段](#输出数据字段)
      - [`task_category` 可能值：](#task_category-可能值)
      - [`safety` 可能值：](#safety-可能值)
  - [🛠️ 数据格式化工具](#️-数据格式化工具)

---

## 🌟 主要特性

- **多任务支持**：内置多种标注任务，如 QUALITY、DIFFICULTY、CLASSIFICATION、SAFETY、REWARD、LANGUAGE、EMBEDDING。
- **双推理模式**：可选择本地 VLLM 模型或远程 API 服务，兼顾性能与成本。
- **高效数据处理**：内置数据格式化工具，便于数据清洗和格式转换。
- **灵活配置**：可通过 CLI 或配置文件自定义任务类型、模型、批量大小、输入输出字段等。
- **向量存储**：支持将生成的嵌入向量存储到本地 Faiss 或分布式 Milvus。
- **易于扩展**：模块化设计，便于添加新任务类型。

---

## 📦 安装指南

- **环境要求**：**Python >= 3.11**
- **uv**

```bash
# 1. 创建并激活虚拟环境
cd data-tagger

uv sync
```

---

## 🚀 快速开始

我们提供了开箱即用的测试脚本，也可手动运行命令。

### 本地 VLLM 推理

推荐模型：
- 通用任务：[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- 嵌入任务：[Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- 奖励任务：[Skywork-Reward-V2-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B)
- 安全任务：[Llama-Guard-3-8B](https://huggingface.co/Skywork/Llama-Guard-3-8B)

本地模型分类任务示例：

```bash
# 直接运行测试脚本
bash examples/vllm/run_all_taggers_vllm.sh
```

或手动运行：

```bash
python -m datatagger.tagger.unified_tagger_vllm \
  --vllm_model_path <你的本地模型路径> \
  --tag_mission CLASSIFICATION \
  --input_file data/alpaca_zh_demo.json \
  --output_file data/alpaca_zh_demo_classification_vllm_output.jsonl \
  --prompt_field instruction \
  --batch_size 5 \
  --device 0
```

### 远程 API 推理

API 多任务标注示例：

```bash
# 1. 拷贝示例环境变量文件
mv .env.example .env

# 2. 编辑环境变量文件

# 3. 运行测试脚本
bash scripts/api/run_all_taggers_api.sh
```

或手动运行单个任务（如质量评估）：

```bash
python -m datatagger.tagger.unified_tagger_api \
  --api_model_name <你的API模型名> \
  --api_url <你的API地址> \
  --api_key <你的API密钥> \
  --tag_mission QUALITY \
  --input_file data/test_data/sample_data_for_api_tagger.jsonl \
  --output_file data/test_output/quality_api_output.jsonl
```

---

## ⚙️ 配置说明

`data-tagger` 支持丰富的命令行参数来控制任务行为。

```bash
# VLLM 模式
python -m datatagger.tagger.unified_tagger_vllm --help

# API 模式
python -m datatagger.tagger.unified_tagger_api --help
```

| 参数 | 描述 |
|---|---|
| `--tag_mission` | **必填。** 任务类型，如 QUALITY、DIFFICULTY、CLASSIFICATION 等。 |
| `--input_file` / `--output_file` | **必填。** 输入和输出文件路径。 |
| `--prompt_field` / `--output_field` | 输入文件中 prompt 和 response 字段名。 |
| `--batch_size` | 批量大小，默认 5。 |
| `--device` | **VLLM 模式。** GPU 设备 ID。 |
| `--vllm_model_path` | **VLLM 模式。** 本地模型路径。 |
| `--api_model_name` / `--api_url` / `--api_key` | **API 模式。** API 服务参数。 |
| `--faiss_store_embeddings` / `--milvus_store_embeddings` | **EMBEDDING 任务。** 是否存储到 Faiss 或 Milvus。 |
| `...` | 更多参数见 settings 目录和脚本注释。 |

---

## 🧩 任务类型与数据字段

### 支持的任务类型

| 任务类型 | 描述 |
|---|---|
| **QUALITY** | 对话质量评估。对输入整体质量打分（1-5分）并简要分析。 |
| **DIFFICULTY** | 难度评估。分析理解/解决输入的难度，输出 0-5 浮点数。 |
| **CLASSIFICATION** | 意图分类。分类输入主意图，输出主/次标签。 |
| **SAFETY** | 安全检测。判断内容是否涉及暴力、色情、隐私等，输出安全标签。 |
| **REWARD** | 奖励打分。对内容奖励价值量化打分（0-5分）。 |
| **LANGUAGE** | 语种识别。识别输入主要语言类型。 |
| **EMBEDDING** | 向量生成。将输入转为向量用于下游任务。 |

### 输出数据字段

处理后的 `JSONL` 文件会增加以下字段：

| 字段名 | 描述 | 示例/范围 |
|---|---|---|
| `id` | 自动生成唯一标识符 | `"a1b2c3d4"` |
| `system`, `conversations`, `instruction`, `output` | 原始数据字段 | ... |
| `prompt_field`, `output_field` | 本次任务用的 prompt 和 output 字段名 | `"instruction"`, `"output"` |
| `prompt_field_length`, `output_field_length` | prompt/output 字段字符长度 | `20`, `100` |
| **`difficulty`** | **[难度]** 难度分数，0-5 浮点数 | `2.5` |
| **`input_quality`**, **`response_quality`** | **[质量]** 输入/输出质量分数，1-5 浮点数 | `4.2`, `4.5` |
| **`input_quality_explanation`**, **`response_quality_explanation`** | **[质量]** 质量分数简要解释 | `"输入清晰，细节充分..."` |
| **`task_category`**, **`other_task_category`** | **[分类]** 主/次任务类别 | `"Coding & Debugging"`, `["Information seeking"]` |
| **`language`** | **[语种]** 主要语言类型 | `"zh"`, `"en"` |
| **`safety`** | **[安全]** 安全标签 | `"Safe"` |
| **`instruct_reward`** | **[奖励]** 奖励分数，0-5 浮点数 | `3.8` |
| `min_neighbor_distance` | **[向量]** 最小邻居距离 | `0.12` |
| `repeat_count` | 重复次数 | `1` |

#### `task_category` 可能值：
`Information seeking`, `Reasoning`, `Planning`, `Editing`, `Coding & Debugging`, `Math`, `Role playing`, `Data analysis`, `Creative writing`, `Advice seeking`, `Translation`, `Brainstorming`, `Others`

#### `safety` 可能值：
`Violent Crimes`, `Non-Violent Crimes`, `Sex-Related Crimes`, `Child Sexual Exploitation`, `Defamation`, `Specialized Advice`, `Privacy`, `Intellectual Property`, `Indiscriminate Weapons`, `Hate`, `Suicide & Self-Harm`, `Sexual Content`, `Elections`, `Code Interpreter Abuse`, `Safe`

---

## 🛠️ 数据格式化工具

我们提供了便捷脚本用于批量格式化、清洗和标准化大型数据集。

```bash
python -m datatagger.formatter.data_formatter \
  --input_file <你的原始数据文件> \
  --output_file <格式化输出文件> \
  --save_as jsonl
```

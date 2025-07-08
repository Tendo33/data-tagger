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
  <b>data-tagger</b> 旨在为大规模数据集提供一站式的标注解决方案，包括质量评估、难度定级、意图分类、安全检测、奖励模型打分、语种识别和向量嵌入生成。它无缝支持本地 VLLM 模型和远程 API 推理，配置灵活，易于扩展。
</p>

---

## 目录
- [🚀 data-tagger](#-data-tagger)
  - [目录](#目录)
  - [🌟 主要特性](#-主要特性)
  - [💡 工作流](#-工作流)
  - [📦 安装指南](#-安装指南)
  - [🚀 快速开始](#-快速开始)
    - [本地 VLLM 推理](#本地-vllm-推理)
    - [远程 API 推理](#远程-api-推理)
  - [⚙️ 配置说明](#️-配置说明)
  - [🧩 任务类型与数据字段](#-任务类型与数据字段)
    - [支持的任务类型](#支持的任务类型)
    - [输出数据字段](#输出数据字段)
  - [🛠️ 数据格式化工具](#️-数据格式化工具)
  - [❓ FAQ](#-faq)

---

## 🌟 主要特性

- **多任务支持**: 内置多种标注任务，如质量 (`QUALITY`)、难度 (`DIFFICULTY`)、分类 (`CLASSIFICATION`)、安全 (`SAFETY`)、奖励 (`REWARD`)、语种 (`LANGUAGE`) 和向量 (`EMBEDDING`)。
- **双模式推理**: 可自由选择使用本地部署的 **VLLM** 模型进行推理，或调用远程 **API** 服务，兼顾性能与成本。
- **高效数据处理**: 提供数据格式化工具，轻松完成数据清洗和格式转换。
- **灵活配置**: 通过命令行参数或配置文件，可轻松定制任务类型、模型、批处理大小、输入输出字段等。
- **向量存储**: 支持将生成的 Embedding 存储到本地 **Faiss** 或分布式 **Milvus**。
- **易于扩展**: 采用模块化设计，可以方便地添加新的标注任务类型。

---

## 💡 工作流

`data-tagger` 的核心工作流程非常简单和清晰：

```mermaid
graph LR
    A[原始数据 (JSON/JSONL)] --> B(data-tagger);
    B --> C{选择推理模式};
    C -- 本地 --> D[⚡ VLLM 高速推理];
    C -- 远程 --> E[☁️ API 服务];
    D --> F[标注后数据 (JSONL)];
    E --> F;
```

-----

## 📦 安装指南

  - **环境要求**: **Python \>= 3.11**
  - 推荐使用虚拟环境以避免包版本冲突。

<!-- end list -->

```bash
# 1. 创建并激活虚拟环境
cd data-tagger

uv sync
```

**主要依赖项** (详情请见 `pyproject.toml`):

  - `vllm`: 用于本地大模型推理。
  - `lingua-language-detector`: 用于语种识别。
  - `loguru`: 提供更强大的日志记录。
  - `pydantic-settings`: 用于环境和配置管理。
  - `json-repair`: 用于修复不规范的 JSON 输出。

-----

## 🚀 快速开始

我们提供了开箱即用的测试脚本，您也可以手动执行命令。

### 本地 VLLM 推理

使用本地模型进行分类任务的示例：

```bash
# 直接运行测试脚本
bash scripts/vllm/classification_test.sh
```

或者手动运行：

```bash
python -m datatagger.tagger.unified_tagger_vllm \
  --vllm_model_path <你的本地模型路径> \
  --tag_mission CLASSIFICATION \
  --input_file data/test_data/sample_data_for_vllm_tagger.jsonl \
  --output_file data/test_output/classification_vllm_output.jsonl \
  --prompt_field instruction \
  --batch_size 5 \
  --device 0
```

### 远程 API 推理

使用 API 同时执行多个标注任务的示例：

```bash
# 直接运行聚合脚本
bash scripts/api/run_all_taggers_api.sh
```

或者手动运行单个任务（以质量评估为例）：

```bash
python -m datatagger.tagger.unified_tagger_api \
  --api_model_name <你的API模型名> \
  --api_url <你的API地址> \
  --api_key <你的API密钥> \
  --tag_mission QUALITY \
  --input_file data/test_data/sample_data_for_api_tagger.jsonl \
  --output_file data/test_output/quality_api_output.jsonl
```

-----

## ⚙️ 配置说明

`data-tagger` 支持丰富的命令行参数来控制任务行为。

\<details\>
\<summary\>\<b\>📚 点击展开/折叠主要参数说明\</b\>\</summary\>

| 参数 | 描述 |
|---|---|
| `--tag_mission` | **必须**。指定任务类型，如 `QUALITY`, `DIFFICULTY`, `CLASSIFICATION` 等。 |
| `--input_file` / `--output_file` | **必须**。输入和输出文件的路径。 |
| `--prompt_field` / `--output_field` | 输入文件中作为 prompt 和 response 的字段名。 |
| `--batch_size` | 处理数据的批次大小，默认为 `5`。 |
| `--device` | **VLLM 模式**。指定使用的 GPU 设备 ID。 |
| `--vllm_model_path` | **VLLM 模式**。本地模型的路径。 |
| `--api_model_name` / `--api_url` / `--api_key` | **API 模式**。API 服务的相关参数。 |
| `--faiss_store_embeddings` / `--milvus_store_embeddings` | **EMBEDDING 任务**。是否将向量存储到 Faiss 或 Milvus。 |
| `...` | 更多参数请参考 `settings` 目录中的定义和脚本注释。 |

\</details\>

-----

## 🧩 任务类型与数据字段

### 支持的任务类型

| 任务类型 | 描述 |
|---|---|
| **QUALITY** | 对话质量评估。对输入内容的整体质量进行打分（1-5分）并给出简要分析。 |
| **DIFFICULTY** | 难度评估。分析理解或解决输入内容所需的难度，输出一个 0-5 的浮点数。 |
| **CLASSIFICATION** | 意图分类。对输入内容的主要意图进行分类，输出主要和次要标签。 |
| **SAFETY** | 安全检测。判断内容是否涉及暴力、色情、隐私等，输出安全标签。 |
| **REWARD** | 奖励模型打分。对内容的奖励价值进行量化打分（0-5分）。 |
| **LANGUAGE** | 语种识别。识别输入内容的主要语言类型。 |
| **EMBEDDING** | 向量生成。将输入内容转换为向量，用于下游机器学习任务。 |

### 输出数据字段

处理完成后的 `JSONL` 文件会增加以下字段。

\<details\>
\<summary\>\<b\>📄 点击展开/折叠详细字段说明\</b\>\</summary\>

| 字段名 | 描述 | 示例/范围 |
|---|---|---|
| `id` | 自动生成的唯一标识符 | `"a1b2c3d4"` |
| `system`, `conversations`, `instruction`, `output` | 原始数据字段 | ... |
| `prompt_field`, `output_field` | 本次任务使用的 prompt 和 output 字段名 | `"instruction"`, `"output"` |
| `prompt_field_length`, `output_field_length` | prompt 和 output 字段的字符长度 | `20`, `100` |
| **`difficulty`** | **[难度]** 难度分数，0-5 的浮点数 | `2.5` |
| **`input_quality`**, **`response_quality`** | **[质量]** 输入/输出的质量分数，1-5 的浮点数 | `4.2`, `4.5` |
| **`input_quality_explanation`**, **`response_quality_explanation`** | **[质量]** 对质量分数的简要解释 | `"输入清晰，细节充分..."` |
| **`task_category`**, **`other_task_category`** | **[分类]** 主要和次要任务类别 | `"Coding & Debugging"`, `["Information seeking"]` |
| **`language`** | **[语种]** 主要语言类型 | `"zh"`, `"en"` |
| **`safety`** | **[安全]** 安全标签 | `"Safe"` |
| **`instruct_reward`** | **[奖励]** 奖励模型分数，0-5 的浮点数 | `3.8` |
| `min_neighbor_distance` | **[向量]** 最小邻居距离，用于相似性分析 | `0.12` |
| `repeat_count` | 重复次数，用于去重分析 | `1` |

\</details\>

\<details\>
\<summary\>\<b\>🏷️ 点击展开/折叠 `task_category` 和 `safety` 的所有可能值\</b\>\</summary\>

  - **`task_category` 可能值**:
    `Information seeking`, `Reasoning`, `Planning`, `Editing`, `Coding & Debugging`, `Math`, `Role playing`, `Data analysis`, `Creative writing`, `Advice seeking`, `Translation`, `Brainstorming`, `Others`

  - **`safety` 可能值**:
    `Violent Crimes`, `Non-Violent Crimes`, `Sex-Related Crimes`, `Child Sexual Exploitation`, `Defamation`, `Specialized Advice`, `Privacy`, `Intellectual Property`, `Indiscriminate Weapons`, `Hate`, `Suicide & Self-Harm`, `Sexual Content`, `Elections`, `Code Interpreter Abuse`, `Safe`

\</details\>

-----

## 🛠️ 数据格式化工具

我们提供了一个便捷的脚本，用于批量格式化、清洗和标准化大型数据集。

```bash
python -m datatagger.formatter.data_formatter \
  --input_file <你的原始数据文件> \
  --output_file <格式化后的输出文件> \
  --save_as jsonl
```

-----

## ❓ FAQ

1.  **支持哪些输入格式？**

    > 支持 `JSON` 和 `JSONL`。为了处理大规模数据，强烈推荐使用 `JSONL` 格式（每行一个 JSON 对象）。

2.  **如何自定义模型或 API？**

    > **VLLM**: 通过 `--vllm_model_path` 指定本地模型路径。
    > **API**: 通过 `--api_model_name`, `--api_url`, `--api_key` 参数进行配置。

3.  **支持哪些向量数据库？**

    > 目前支持本地的 **Faiss** 和分布式的 **Milvus** 用于存储和查询 Embedding。




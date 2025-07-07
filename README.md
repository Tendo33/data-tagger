# data-tagger

## 项目简介

data-tagger 是一个支持多种打标签任务的高效批量数据处理工具，适用于大规模数据集的质量评估、难度评估、分类、安全性检测、奖励评分、语言识别及嵌入向量生成。支持本地 VLLM 推理和远程 API 推理两种模式，配置灵活，易于集成。

---

## 主要功能

- **多任务批量打标签**：支持 QUALITY（质量）、DIFFICULTY（难度）、CLASSIFICATION（分类）、SAFETY（安全性）、REWARD（奖励）、LANGUAGE（语言识别）、EMBEDDING（嵌入向量）等任务。
- **本地与API推理**：可选择本地 VLLM 模型或远程 API 进行推理。
- **高效数据格式化**：内置数据格式化工具，支持多种输入输出格式。
- **灵活配置**：通过配置文件或命令行参数自定义任务、模型、批量大小、字段名等。
- **嵌入向量存储**：支持将 embedding 存入本地 Faiss 或 Milvus。

---

## 安装与依赖

- Python >= 3.11
- 推荐使用虚拟环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # 或根据 pyproject.toml 安装
```

- 主要依赖（详见 pyproject.toml）：
  - vllm
  - lingua-language-detector
  - loguru
  - pydantic-settings
  - json-repair

---

## 快速开始

### 1. 本地 VLLM 推理示例

以分类任务为例：

```bash
bash scripts/classification_test.sh
```

或手动运行：

```bash
python -m datatagger.tagger.unified_tagger_vllm \
  --vllm_model_path <本地模型路径> \
  --tag_mission CLASSIFICATION \
  --input_file <输入文件> \
  --output_file <输出文件> \
  --prompt_field instruction \
  --output_field output \
  --batch_size 5 \
  --device 0
```

### 2. 远程 API 推理示例

以 run_all_taggers_api.sh 为例，支持多任务串行处理：

```bash
bash scripts/run_all_taggers_api.sh
```

或手动运行：

```bash
python -m datatagger.tagger.unified_tagger_api \
  --api_model_name <API模型名> \
  --api_url <API地址> \
  --api_key <API密钥> \
  --tag_mission QUALITY \
  --input_file <输入文件> \
  --output_file <输出文件>
```

---

## 配置说明（主要参数）

- `--tag_mission`：任务类型（QUALITY、DIFFICULTY、CLASSIFICATION、SAFETY、REWARD、LANGUAGE、EMBEDDING）
- `--input_file` / `--output_file`：输入/输出文件路径
- `--prompt_field` / `--output_field`：输入文件中 prompt/response 字段名
- `--batch_size`：批处理大小
- `--device`：GPU 设备号（本地推理）
- `--vllm_model_path`：本地模型路径（VLLM）
- `--api_model_name` / `--api_url` / `--api_key`：API 模式相关参数
- `--faiss_store_embeddings` / `--milvus_store_embeddings`：是否存储 embedding
- 详细参数见各 settings 文件和脚本注释

---

## 数据格式化工具

支持大规模数据集的批量格式转换、清洗、标准化。

```bash
python -m datatagger.formatter.data_formatter \
  --input_file <原始数据文件> \
  --output_file <格式化后文件> \
  --save_as jsonl
```

---

## 脚本与用法示例

- `scripts/classification_test.sh`：本地分类任务示例
- `scripts/run_all_taggers_api.sh`：API 多任务批量处理示例
- `scripts/api/`、`scripts/vllm/`：各任务单独示例脚本

---

## FAQ

1. **支持哪些输入格式？**
   - 支持 JSON/JSONL，推荐每行为一个样本的 JSONL。
2. **如何自定义模型或API？**
   - 修改配置文件或脚本参数即可。
3. **如何扩展新任务？**
   - 参考 datatagger/tagger/ 目录下的实现，继承 BaseUnifiedTagger。
4. **embedding 支持哪些存储？**
   - 支持本地 Faiss 和 Milvus。

---

## 目录结构简述

- `datatagger/`：主程序模块
  - `tagger/`：各类打标签任务实现
  - `formatter/`：数据格式化工具
  - `settings/`：配置与参数定义
  - `utils/`：通用工具函数
- `scripts/`：常用任务脚本
- `data/`：示例数据与输出

---

如有问题请查阅源码或联系维护者。

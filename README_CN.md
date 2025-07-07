# 🚀 data-tagger

> **高效、灵活的多任务批量数据打标签工具**

---

<p align="center">
  <b>data-tagger</b> 是一款高效支持大规模数据集质量评估、难度评估、分类、安全性检测、奖励评分、语言识别及嵌入向量生成的工具。支持本地 VLLM 推理和远程 API 推理，配置灵活，易于集成。
</p>

---

## 🌟 主要特性

- **多任务批量打标签**：支持 QUALITY、DIFFICULTY、CLASSIFICATION、SAFETY、REWARD、LANGUAGE、EMBEDDING 等任务类型
- **本地与 API 推理**：可选本地 VLLM 模型或远程 API
- **高效数据格式化**：内置多格式数据清洗与转换工具
- **灵活配置**：支持配置文件与命令行参数，任务/模型/批量/字段名等均可自定义
- **嵌入向量存储**：支持本地 Faiss 或 Milvus
- **易于扩展**：模块化设计，便于新增任务类型

---

## 📦 安装与依赖

- **Python >= 3.11**
- 推荐使用虚拟环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # 或根据 pyproject.toml 安装
```

**主要依赖**（详见 `pyproject.toml`）：
- vllm
- lingua-language-detector
- loguru
- pydantic-settings
- json-repair

---

## 🚀 快速开始

### 本地 VLLM 推理

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

### 远程 API 推理

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

## ⚙️ 配置说明（主要参数）

| 参数 | 说明 |
|------|------|
| `--tag_mission` | 任务类型（QUALITY、DIFFICULTY、CLASSIFICATION 等） |
| `--input_file` / `--output_file` | 输入/输出文件路径 |
| `--prompt_field` / `--output_field` | 输入文件中 prompt/response 字段名 |
| `--batch_size` | 批处理大小 |
| `--device` | GPU 设备号（本地推理） |
| `--vllm_model_path` | 本地模型路径（VLLM） |
| `--api_model_name` / `--api_url` / `--api_key` | API 模式相关参数 |
| `--faiss_store_embeddings` / `--milvus_store_embeddings` | 是否存储 embedding |
| ... | 更多参数详见各 settings 文件和脚本注释 |

---

## 🛠️ 数据格式化工具

支持大规模数据集的批量格式转换、清洗、标准化：

```bash
python -m datatagger.formatter.data_formatter \
  --input_file <原始数据文件> \
  --output_file <格式化后文件> \
  --save_as jsonl
```

---

## 📚 脚本与用法示例

- `scripts/classification_test.sh`：本地分类任务示例
- `scripts/run_all_taggers_api.sh`：API 多任务批量处理示例
- `scripts/api/`、`scripts/vllm/`：各任务单独示例脚本

---

## ❓ FAQ

1. **支持哪些输入格式？**
   - 支持 JSON/JSONL，推荐每行为一个样本的 JSONL。
2. **如何自定义模型或API？**
   - 修改配置文件或脚本参数即可。
3. **如何扩展新任务？**
   - 参考 `datatagger/tagger/` 目录下的实现，继承 `BaseUnifiedTagger`。
4. **embedding 支持哪些存储？**
   - 支持本地 Faiss 和 Milvus。

---

## 🧩 任务类型与数据字段说明

### 任务类型说明

| 任务类型         | 作用说明                                                                                   |
|------------------|------------------------------------------------------------------------------------------|
| QUALITY          | 质量评估。对输入内容的整体质量进行打分和简要分析。                                         |
| DIFFICULTY       | 难度评估。分析输入内容的理解/解答难度，输出 0-5 浮点分值。                                 |
| CLASSIFICATION   | 分类任务。对输入内容进行主意图/主任务分类，输出主标签和次标签。                            |
| SAFETY           | 安全性检测。判断内容是否涉及暴力、色情、隐私等敏感类别，输出安全标签。                    |
| REWARD           | 奖励评分。对内容的奖励价值进行定量打分。                                                   |
| LANGUAGE         | 语言识别。识别输入内容的主要语言类型。                                                    |
| EMBEDDING        | 嵌入向量生成。将输入内容转为向量，便于后续检索、聚类等机器学习任务。                      |

### 数据字段说明

| 字段名                        | 说明                                                                                   | 取值示例/范围                      |
|-------------------------------|--------------------------------------------------------------------------------------|------------------------------------|
| id                            | 唯一标识符（自动生成）                                                                | "a1b2c3d4"                         |
| system                        | 系统提示词（可选）                                                                    | "You are an expert..."             |
| conversations                 | 对话内容列表，含角色（human/gpt）和文本                                               | [{"from": "human", "value": ...}]  |
| instruction                   | 用户输入的主指令/问题                                                                 | "请帮我写一个排序算法"              |
| output                        | AI 的主要输出/回答                                                                   | "以下是排序算法..."                 |
| prompt_field                  | 指定的 prompt 字段名（如 instruction）                                                | "instruction"                      |
| output_field                  | 指定的 output 字段名（如 output）                                                     | "output"                           |
| prompt_field_length           | prompt 字段的字符长度                                                                 | 20                                 |
| output_field_length           | output 字段的字符长度                                                                 | 100                                |
| intent                        | 用户意图分析结果（JSON 字符串，见 prompt_utils.py）                                   | "The user wants to ..."            |
| knowledge                     | 解决该任务所需的知识点描述                                                            | "Requires knowledge of ..."         |
| difficulty                    | 难度评分，0-5 浮点数，越高越难                                                        | 2.5                                |
| input_quality                 | 输入质量评分，1-5 浮点数，越高越好                                                    | 4.2                                |
| response_quality              | 输出质量评分，1-5 浮点数，越高越好                                                    | 4.5                                |
| input_quality_explanation     | 输入质量评分简要说明                                                                  | "输入清晰，细节充分..."             |
| response_quality_explanation  | 输出质量评分简要说明                                                                  | "回答准确，结构清晰..."             |
| task_category                 | 主任务分类（枚举，见 ALLOWED_TASK_CATEGORIES）                                        | "Coding & Debugging"               |
| other_task_category           | 其他相关任务分类（列表）                                                              | ["Information seeking"]             |
| language                      | 主要语言类型（如 "zh", "en"）                                                        | "zh"                               |
| safety                        | 安全标签（枚举，见 ALLOWED_SAFETY_LABELS）                                            | "Safe"                             |
| instruct_reward               | 奖励分数，0-5 浮点数                                                                  | 3.8                                |
| task_category_generator       | 任务分类生成器（可选，记录分类来源）                                                  | "prompt_utils"                     |
| min_neighbor_distance         | 最近邻距离（用于 embedding 相似性分析）                                               | 0.12                               |
| repeat_count                  | 重复次数（如用于去重分析）                                                            | 1                                  |
| min_similar_instruction       | 最相似的 instruction（如有）                                                          | "请帮我写一个排序算法"              |

#### task_category 可选值（见 ALLOWED_TASK_CATEGORIES）：
Information seeking, Reasoning, Planning, Editing, Coding & Debugging, Math, Role playing, Data analysis, Creative writing, Advice seeking, Translation, Brainstorming, Others

#### safety 可选值（见 ALLOWED_SAFETY_LABELS）：
Violent Crimes, Non-Violent Crimes, Sex-Related Crimes, Child Sexual Exploitation, Defamation, Specialized Advice, Privacy, Intellectual Property, Indiscriminate Weapons, Hate, Suicide & Self-Harm, Sexual Content, Elections, Code Interpreter Abuse, Safe

---

## 📂 目录结构

```text
datatagger/         # 主程序模块
  tagger/           # 各类打标签任务实现
  formatter/        # 数据格式化工具
  settings/         # 配置与参数定义
  utils/            # 通用工具函数
scripts/            # 常用任务脚本
data/               # 示例数据与输出
```

---

## 🤝 贡献与支持

- 如有问题请查阅源码或联系维护者。
- 欢迎 issue/PR 贡献改进！
- [English version README see README.md](./README.md)

---
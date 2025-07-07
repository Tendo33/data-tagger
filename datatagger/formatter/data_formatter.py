import json
import os
import sys
import uuid
from collections import OrderedDict
from typing import Any, Dict, Generator, List, Optional

from tqdm import tqdm

from datatagger.settings.base_formatter_setting import BaseFormatterSettings

ALLOWED_TASK_CATEGORIES = [
    "Information seeking",
    "Reasoning",
    "Planning",
    "Editing",
    "Coding & Debugging",
    "Math",
    "Role playing",
    "Data analysis",
    "Creative writing",
    "Advice seeking",
    "Translation",
    "Brainstorming",
    "Others",
]

ALLOWED_SAFETY_LABELS = [
    "Violent Crimes",
    "Non-Violent Crimes",
    "Sex-Related Crimes",
    "Child Sexual Exploitation",
    "Defamation",
    "Specialized Advice",
    "Privacy",
    "Intellectual Property",
    "Indiscriminate Weapons",
    "Hate",
    "Suicide & Self-Harm",
    "Sexual Content",
    "Elections",
    "Code Interpreter Abuse",
    "Safe",
]


class UnifiedDataFormatter:
    """
    一个统一、高效的数据格式化工具，通过批处理支持大型数据集。
    与 BaseFormatterSettings 完全集成，支持命令行配置。
    """

    DEFAULT_BATCH_SIZE = 1000

    def __init__(self, settings: BaseFormatterSettings):
        self.settings = settings
        self._total_processed = 0

    def run(self):
        """执行格式化的主流程：计数 -> 迭代处理 -> 分批保存 -> 完成。"""
        print("🚀 Starting data formatting...")
        print(f"  - Input: {self.settings.input_file}")
        print(f"  - Output: {self.settings.output_file}")
        print(f"  - Internal batch size: {self.DEFAULT_BATCH_SIZE}")

        is_json_format = self.settings.output_file.lower().endswith(".json")
        output_path = self.settings.output_file
        temp_path = f"{output_path}.tmp" if is_json_format else output_path

        self._prepare_output_files(output_path, temp_path)

        print("  - Counting total entries...")
        total_entries = self._count_entries()
        print(f"  - Found {total_entries} entries to process.")

        batch = []
        data_iterator = self._iter_data()

        with tqdm(total=total_entries, desc="Formatting entries") as pbar:
            for entry_data in data_iterator:
                processed_entry = self.process_entry(entry_data)
                if processed_entry:
                    batch.append(processed_entry)

                if len(batch) >= self.DEFAULT_BATCH_SIZE:
                    self._write_batch(batch, temp_path)
                    pbar.update(len(batch))
                    batch = []
                else:
                    pbar.update(1)

        if batch:
            self._write_batch(batch, temp_path)
            pbar.update(len(batch))

        if is_json_format:
            print("  - Finalizing JSON file...")
            self._convert_jsonl_to_json(temp_path, output_path)
            os.remove(temp_path)

        print(
            f"\n✅ Successfully converted {self._total_processed} entries to '{output_path}'."
        )

    def _prepare_output_files(self, output_path: str, temp_path: str):
        """清理可能存在的旧输出文件。"""
        for path in {output_path, temp_path}:
            if os.path.exists(path):
                print(f"  - Removing existing file: {path}")
                os.remove(path)

    def _count_entries(self) -> int:
        """快速遍历文件以计算总行数/条目数。"""
        try:
            # Handle reading from stdin
            if self.settings.input_file == "/dev/stdin":
                print(
                    "  - Reading from stdin, cannot pre-count. Progress bar will not show total."
                )
                return 0  # Cannot determine total for a stream

            with open(self.settings.input_file, "r", encoding="utf-8") as f:
                if self.settings.input_file.endswith(".jsonl"):
                    return sum(1 for line in f if line.strip())
                else:
                    return len(json.load(f))
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error counting entries in file: {e}", file=sys.stderr)
            sys.exit(1)

    def _iter_data(self) -> Generator[Dict[str, Any], None, None]:
        """创建一个数据生成器，逐行/逐条读取文件以节省内存。"""
        try:
            # Handle reading from stdin
            if self.settings.input_file == "/dev/stdin":
                f = sys.stdin
            else:
                f = open(self.settings.input_file, "r", encoding="utf-8")

            if not self.settings.input_file.endswith(".jsonl"):
                data = json.load(f)
                for entry in data:
                    yield entry
            else:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading data file: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            if "f" in locals() and f is not sys.stdin:
                f.close()

    def _write_batch(self, batch: List[OrderedDict], path: str):
        """以追加模式将一个批次的数据写入文件（始终为JSONL格式）。"""
        try:
            with open(path, "a", encoding="utf-8") as f:
                for item in batch:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            self._total_processed += len(batch)
        except IOError as e:
            print(f"Error writing batch to file: {e}", file=sys.stderr)
            sys.exit(1)

    def _convert_jsonl_to_json(self, jsonl_path: str, json_path: str):
        """将JSONL临时文件转换为一个合法的JSON数组文件，并格式化输出。"""
        try:
            with (
                open(jsonl_path, "r", encoding="utf-8") as f_in,
                open(json_path, "w", encoding="utf-8") as f_out,
            ):
                items = [json.loads(line) for line in f_in if line.strip()]
                json.dump(items, f_out, ensure_ascii=False, indent=4)
        except (IOError, FileNotFoundError) as e:
            print(f"Error converting JSONL to JSON: {e}", file=sys.stderr)
            sys.exit(1)

    def process_entry(self, entry: Dict[str, Any]) -> Optional[OrderedDict]:
        """处理单个数据条目，将其转换为目标格式。"""
        cleaned_entry = self._clean_entry(entry)

        if not self._can_build_conversation(cleaned_entry):
            tqdm.write("Warning: Skipping an entry due to missing conversation fields.")
            return None

        od = OrderedDict()

        # 1. 核心身份信息 (使用 UUID)
        new_id = str(uuid.uuid4())[:8]
        od["id"] = new_id

        # 2. 对话内容
        od["system"] = cleaned_entry.get("system")
        od["conversations"] = self._build_conversations(cleaned_entry, self.settings)
        # prompt 字段校验逻辑
        prompt_value = cleaned_entry.get(
            self.settings.prompt_field,
            od["conversations"][0]["value"] if od["conversations"] else None,
        )
        if od["conversations"]:
            first_conv = od["conversations"][0]
            if first_conv.get("from") != "human":
                print(
                    f"[Warning] Entry id={od['id']} prompt_field未正确填充，conversations第一项不是human，prompt字段将设为None。"
                )
                prompt_value = None
        od[self.settings.prompt_field] = prompt_value
        # output 字段校验逻辑
        output_value = cleaned_entry.get(
            self.settings.output_field,
            od["conversations"][-1]["value"] if od["conversations"] else None,
        )
        # conversations 最后一项不是 gpt，则 output 设为 None 并警告
        if od["conversations"]:
            last_conv = od["conversations"][-1]
            if last_conv.get("from") != "gpt":
                print(
                    f"[Warning] Entry id={od['id']} output_field未正确填充，conversations最后一项不是gpt，output字段将设为None。"
                )
                output_value = None
        od[self.settings.output_field] = output_value
        od[f"{self.settings.prompt_field}_length"] = len(od[self.settings.prompt_field])
        od[f"{self.settings.output_field}_length"] = len(od[self.settings.output_field])

        # 3. 元数据和评估字段（新版适配）
        od["intent"] = cleaned_entry.get("intent")
        od["knowledge"] = cleaned_entry.get("knowledge")
        od["difficulty"] = self._parse_score(cleaned_entry.get("difficulty"), 1, 5)
        od["input_quality"] = self._parse_score(
            cleaned_entry.get("input_quality"), 0, 5
        )
        od["response_quality"] = self._parse_score(
            cleaned_entry.get("response_quality"), 0, 5
        )
        od["input_quality_explanation"] = cleaned_entry.get("input_quality_explanation")
        od["response_quality_explanation"] = cleaned_entry.get(
            "response_quality_explanation"
        )
        od["task_category"] = (
            cleaned_entry.get("task_category")
            if cleaned_entry.get("task_category") in ALLOWED_TASK_CATEGORIES
            else "Others"
        )
        od["other_task_category"] = cleaned_entry.get("other_task_category", [])
        od["language"] = cleaned_entry.get("language")

        # 4. 其他机器学习相关字段
        od["safety"] = (
            cleaned_entry.get("safety")
            if cleaned_entry.get("safety") in ALLOWED_SAFETY_LABELS
            else "Safe"
        )
        od["instruct_reward"] = self._parse_score(
            cleaned_entry.get("instruct_reward"), 0, 5
        )
        od["task_category_generator"] = cleaned_entry.get("task_category_generator")
        od["min_neighbor_distance"] = self._parse_score(
            cleaned_entry.get("min_neighbor_distance"), 0, float("inf")
        )
        od["repeat_count"] = self._parse_score(
            cleaned_entry.get("repeat_count"), 0, float("inf"), as_int=True
        )
        od["min_similar_instruction"] = cleaned_entry.get("min_similar_instruction")

        # 移除所有值为 None 的字段，保持输出整洁
        return OrderedDict((k, v) for k, v in od.items() if v is not None)

    def _clean_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """对字典中的每个值应用清洗逻辑。"""
        return {k: self._clean_value(v) for k, v in entry.items()}

    @staticmethod
    def _clean_value(val: Any) -> Any:
        """将各种形式的"空"值标准化为 None。"""
        if val is None or val in ("", [], "N/A", "null"):
            return None
        if isinstance(val, str) and not val.strip():
            return None
        return val

    @staticmethod
    def _validate_field(value: Optional[str], allowed_values: set) -> Optional[str]:
        """验证字段值是否在允许的集合内，否则返回 None。"""
        return value if value in allowed_values else None

    @staticmethod
    def _build_conversations(
        entry: Dict[str, Any], settings: BaseFormatterSettings
    ) -> List[Dict[str, str]]:
        """根据条目中的不同字段，灵活地构建 conversation 列表。"""
        if entry.get("conversations"):
            return entry["conversations"]
        conversations = []
        # 1. history 模式
        if entry.get("history"):
            for q, a in entry["history"]:
                if q is not None and str(q).strip():
                    conversations.append({"from": "human", "value": str(q).strip()})
                if a is not None and str(a).strip():
                    conversations.append({"from": "gpt", "value": str(a).strip()})
        # 2. instruction + input + output/chosen 模式
        instruction = entry.get(settings.prompt_field)
        input_text = entry.get("input")
        if instruction is not None and str(instruction).strip():
            if input_text is not None and str(input_text).strip():
                human_value = f"{instruction}\n{input_text}".strip()
            else:
                human_value = str(instruction).strip()
            conversations.append({"from": "human", "value": human_value})
        output = entry.get("chosen") or entry.get(settings.output_field)
        if output is not None and str(output).strip():
            conversations.append({"from": "gpt", "value": str(output).strip()})
        return conversations

    @staticmethod
    def _can_build_conversation(entry: Dict[str, Any]) -> bool:
        """检查条目是否包含足够的信息来构建一个有效的对话。"""
        has_sharegpt = bool(entry.get("conversations"))
        has_alpaca = entry.get("instruction") and (
            entry.get("output") or entry.get("chosen")
        )
        has_history = bool(entry.get("history"))
        return has_sharegpt or has_alpaca or has_history

    @staticmethod
    def _parse_score(value, min_value=0, max_value=5, as_int=False):
        """将分数字段转为float，限定在min_value-max_value之间，as_int为True时转为int，异常或超出范围返回None"""
        try:
            if value is None:
                return None
            v = float(value)
            if min_value <= v <= max_value:
                return int(v) if as_int else v
            return None
        except Exception:
            return None


if __name__ == "__main__":
    try:
        settings = BaseFormatterSettings()
        formatter = UnifiedDataFormatter(settings)
        formatter.run()
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

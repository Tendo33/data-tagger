import json
import os
import shutil
import uuid


class CheckpointManager:
    def __init__(self, data_file: str, state_file: str):
        self.data_file = data_file
        self.state_file = state_file

    def save(self, dataset: list, current_index: int):
        tmp_data_file = self.data_file + ".tmp"
        tmp_state_file = self.state_file + ".tmp"
        # 只保存已处理部分
        with open(tmp_data_file, "w", encoding="utf-8") as f:
            json.dump(dataset[:current_index], f, ensure_ascii=False, indent=2)
        with open(tmp_state_file, "w", encoding="utf-8") as f:
            json.dump({"current_index": current_index}, f)
        shutil.move(tmp_data_file, self.data_file)
        shutil.move(tmp_state_file, self.state_file)

    def load(self):
        if not (os.path.exists(self.data_file) and os.path.exists(self.state_file)):
            return None
        with open(self.state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        current_index = state.get("current_index", 0)
        with open(self.data_file, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
        return processed_data, current_index

    def cleanup(self):
        for f in [self.data_file, self.state_file]:
            if os.path.exists(f):
                os.remove(f)


# File I/O utilities
def load_jsonl_to_list(jsonl_file_path):
    data_list = []
    with open(jsonl_file_path, "r") as file:
        for line in file:
            json_obj = json.loads(line)
            data_list.append(json_obj)
    return data_list


# Load dataset
def load_dataset_from_file(filename):
    # if the file is json
    if filename.endswith(".json"):
        with open(filename, "r") as file:
            return json.load(file)
    elif filename.endswith(".jsonl"):
        return load_jsonl_to_list(filename)
    else:
        raise ValueError("Invalid file format. Please provide a .json or .jsonl file.")


# Save dataset
def save_dataset(data: list, file_path: str, ext: str = ".jsonl"):
    if ext == ".jsonl":
        with open(file_path, "w", encoding="utf-8") as file:
            for obj in data:
                file.write(json.dumps(obj, ensure_ascii=False) + "\n")
    elif ext == ".json":
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    else:
        raise ValueError("Invalid file format. Please provide a .json or .jsonl file.")


# UUID
def generate_uuid(name):
    namespace = uuid.NAMESPACE_DNS
    return str(uuid.uuid5(namespace, name))


# 定义释放GPU显存的函数
def release_gpu_memory(llm):
    """
    释放GPU显存，防止内存泄漏
    """
    import gc

    import torch
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor.driver_worker
    del llm.llm_engine.model_executor
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print(f"cuda memory: {torch.cuda.memory_allocated() // 1024 // 1024}MB")

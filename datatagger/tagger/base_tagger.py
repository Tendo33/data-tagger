import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from lingua import LanguageDetectorBuilder

from datatagger.settings.base_tagger_setting import BaseTaggerSettings, TagMission
from datatagger.tagger.tag_missions import TagMissionProcessor
from datatagger.utils.file_utils import CheckpointManager, save_dataset
from datatagger.utils.logger import setup_logger


class BaseUnifiedTagger:
    def __init__(self, settings: BaseTaggerSettings, is_api: bool = False) -> None:
        self.settings = settings
        self.mission = settings.tag_mission
        if is_api:
            assert self.mission not in [TagMission.SAFETY, TagMission.REWARD], (
                "API mode does not support safety and reward tasks"
            )
        self.mission_processor = TagMissionProcessor(self.mission, settings)
        self.tag_mission = self.mission_processor.get_name()
        self.batch_size = settings.batch_size
        self.checkpoint_every = settings.checkpoint_every
        self.debug = settings.debug
        self.dimension = settings.dimension
        self.prompt_field = settings.prompt_field
        self.output_field = settings.output_field
        self.milvus_store_embeddings = settings.milvus_store_embeddings
        self.faiss_store_embeddings = settings.faiss_store_embeddings
        self.faiss_index_file = settings.faiss_index_file
        self.faiss_meta_file = settings.faiss_meta_file
        self.logger = setup_logger(
            project_name=self.tag_mission, console_log_level=self.settings.log_level
        )
        if self.milvus_store_embeddings:
            from datatagger.utils.milvus_utils import MilvusClient

            self.logger.info(
                f"Initializing Milvus client with host: {settings.milvus_host}, port: {settings.milvus_port}, collection: {settings.milvus_collection}, dim: {self.dimension}"
            )
            self.milvus_client = MilvusClient(
                host=settings.milvus_host,
                port=settings.milvus_port,
                collection_name=settings.milvus_collection,
                dim=self.dimension,
            )
        if self.faiss_store_embeddings:
            from datatagger.utils.faiss_utils import FaissClient

            self.logger.info(
                f"Initializing Faiss client with index_file: {self.faiss_index_file}, meta_file: {self.faiss_meta_file}, dim: {self.dimension}"
            )
            self.faiss_client = FaissClient(
                index_file=self.faiss_index_file,
                meta_file=self.faiss_meta_file,
                dim=self.dimension,
            )
        if settings.input_file:
            _, self.checkpoint_data_file, self.checkpoint_state_file = (
                self.get_output_files(
                    settings=settings,
                    tag_mission=self.tag_mission,
                    input_file=settings.input_file,
                )
            )
        else:
            self.checkpoint_data_file = None
            self.checkpoint_state_file = None
        if self.mission == TagMission.LANGUAGE:
            self.logger.info("Building language detector from all languages")
            self.detector = LanguageDetectorBuilder.from_all_languages().build()
            self.logger.info("Language detector built successfully")

    @staticmethod
    def get_output_files(
        settings: BaseTaggerSettings, tag_mission: TagMission, input_file: str
    ) -> Tuple[str, str, str]:
        """
        Automatically determine output and checkpoint file suffixes based on settings.output_file (if provided),
        ensuring checkpoint_data_file matches output_file's suffix, and checkpoint_state_file is always .json.
        """
        import os

        if settings.output_file:
            output_file = settings.output_file
            ext = os.path.splitext(output_file)[1]
        else:
            base_name = input_file[: input_file.rfind(".")]
            ext = ".jsonl"
            output_file = f"{base_name}_{tag_mission}{ext}"
        base_ckpt = os.path.splitext(output_file)[0]
        checkpoint_data_file = f"{base_ckpt}_checkpoint{ext}"
        checkpoint_state_file = f"{base_ckpt}_checkpoint_state.json"
        return output_file, checkpoint_data_file, checkpoint_state_file

    @staticmethod
    def load_checkpoint_state(checkpoint_state_file: str) -> Optional[Dict[str, Any]]:
        if os.path.exists(checkpoint_state_file):
            with open(checkpoint_state_file, "r") as f:
                return json.load(f)
        return None

    @staticmethod
    def save_checkpoint_state(
        checkpoint_state_file: str, current_index: int, total_items: int
    ) -> None:
        state = {
            "current_index": current_index,
            "total_items": total_items,
            "timestamp": str(datetime.datetime.now()),
        }
        with open(checkpoint_state_file, "w") as f:
            json.dump(state, f)

    @staticmethod
    def cleanup_checkpoint(
        checkpoint_state_file: str, checkpoint_data_file: str
    ) -> None:
        if os.path.exists(checkpoint_state_file):
            os.remove(checkpoint_state_file)
        if os.path.exists(checkpoint_data_file):
            os.remove(checkpoint_data_file)

    @staticmethod
    def process_batch_with_language_detection(
        detector,
        logger,
        batch_indices: List[int],
        dataset: List[Dict[str, Any]],
        prompt_field: str,
    ) -> None:
        logger.info(
            f"Processing batch with language detection for indices: {batch_indices}"
        )
        for idx in batch_indices:
            item = dataset[idx]
            if item[prompt_field] != "":
                try:
                    logger.debug(
                        f"Detecting language for index {idx}: {item[prompt_field][:100]}..."
                    )
                    detected_language = detector.detect_language_of(
                        item[prompt_field]
                    ).iso_code_639_1.name
                    item["language"] = detected_language
                    logger.debug(
                        f"Successfully detected language for index {idx}: {detected_language}"
                    )
                except Exception as e:
                    logger.error(f"Failed to detect language for index {idx}: {str(e)}")
                    item["language"] = None
            else:
                logger.debug(
                    f"Empty prompt field for index {idx}, setting language to None"
                )
                item["language"] = None

    def generate_and_update_with_checkpoint(
        self,
        dataset: List[Dict[str, Any]],
        output_file: str,
        checkpoint_data_file: str,
        checkpoint_state_file: str,
        process_batch_fn,
        batch_size: int = None,
        checkpoint_every: int = None,
        logger=None,
        postprocess_fn=None,
    ):
        """
        General checkpoint-resume main loop, for subclass use.
        process_batch_fn(batch_indices, dataset) is the batch processing function.
        postprocess_fn(dataset) is optional, for post-processing before final save.
        """
        if not batch_size:
            batch_size = self.batch_size
        if not checkpoint_every:
            checkpoint_every = self.checkpoint_every
        if not logger:
            logger = self.logger
        checkpoint_manager = CheckpointManager(
            checkpoint_data_file, checkpoint_state_file
        )
        loaded = checkpoint_manager.load()
        if loaded:
            processed_data, last_checkpoint_idx = loaded
            logger.info(f"Checkpoint found. Resuming from index {last_checkpoint_idx}.")
            dataset[:last_checkpoint_idx] = processed_data
        else:
            last_checkpoint_idx = 0
        num_batches = (
            len(dataset) - last_checkpoint_idx + batch_size - 1
        ) // batch_size
        try:
            for i in range(num_batches):
                start_idx = i * batch_size + last_checkpoint_idx
                end_idx = min((i + 1) * batch_size + last_checkpoint_idx, len(dataset))
                batch_indices = list(range(start_idx, end_idx))
                process_batch_fn(batch_indices, dataset)
                if (i + 1) % checkpoint_every == 0:
                    checkpoint_manager.save(dataset, end_idx)
                    logger.info(f"Checkpoint saved at index {end_idx}.")

            # Save final result before completion
            if postprocess_fn is not None:
                postprocess_fn(dataset)

            ext = os.path.splitext(output_file)[1].lower()
            save_dataset(data=dataset, file_path=output_file, ext=ext)

            checkpoint_manager.cleanup()
            logger.info("Processing completed. Checkpoint cleaned up.")
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            checkpoint_manager.save(dataset, end_idx)
            raise

    def get_neighbor_info(
        self,
        embedding: list,
        backend: str = "faiss",
        distance_threshold: float = 0.1,
        top_k: int = 5,
        meta: str = None,
    ) -> Tuple[float, int, str]:
        """
        计算embedding的最近邻距离、重复计数和最相似instruction。
        :param embedding: 查询向量
        :param backend: 'faiss'或'milvus'
        :param distance_threshold: 距离阈值
        :param top_k: 检索top_k个邻居
        :param meta: 当前样本的meta（用于排除自身）
        :return: (min_neighbor_distance, repeat_count, min_similar_instruction)
        """
        min_distance = None
        repeat_count = 0
        min_similar_instruction = None
        if backend == "milvus":
            results = self.milvus_client.search(embedding, top_k=top_k + 1)
            hits = results[0] if hasattr(results[0], "__iter__") else results
            for hit in hits:
                score = getattr(hit, "distance", getattr(hit, "score", None))
                hit_meta = (
                    hit.entity.get("meta", None)
                    if hasattr(hit, "entity")
                    else hit.get("meta", None)
                )
                if meta is not None and hit_meta == meta:
                    continue
                if min_distance is None or score < min_distance:
                    min_distance = score
                    min_similar_instruction = hit_meta  # 假设meta存了prompt内容
                if score < distance_threshold:
                    repeat_count += 1
        elif backend == "faiss":
            results = self.faiss_client.search(embedding, top_k=top_k + 1)
            for hit_meta, score in results:
                if meta is not None and hit_meta == meta:
                    continue
                if min_distance is None or score < min_distance:
                    min_distance = score
                    min_similar_instruction = hit_meta
                if score < distance_threshold:
                    repeat_count += 1
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        return (
            round(min_distance, 4) if min_distance is not None else None,
            repeat_count,
            min_similar_instruction,
        )

    def update_similarity_fields(self, dataset, backend=None, field=None):
        # backend: 'milvus' 或 'faiss'，自动判断
        # field: 'prompt' 或 'output'，默认只处理 prompt
        if backend is None:
            backend = (
                "milvus" if getattr(self, "milvus_store_embeddings", False) else "faiss"
            )
        if field is None:
            field = getattr(self, "prompt_field", "prompt")
        for item in dataset:
            meta = str(item.get(field, ""))
            if backend == "milvus" and hasattr(self, "milvus_client"):
                results = self.milvus_client.search_by_meta(meta, top_k=1)
                if (
                    results
                    and hasattr(results[0], "outputs")
                    and hasattr(results[0].outputs, "embedding")
                ):
                    embedding = results[0].outputs.embedding
                else:
                    continue
            elif backend == "faiss" and hasattr(self, "faiss_client"):
                embedding = self.faiss_client.get_embedding_by_meta(meta)
                if embedding is None:
                    continue
            else:
                continue
            min_dist, repeat_count, min_similar_instruction = self.get_neighbor_info(
                embedding=embedding, backend=backend, meta=meta
            )
            item["min_neighbor_distance"] = (
                round(min_dist, 4) if min_dist is not None else None
            )
            item["repeat_count"] = repeat_count
            item["min_similar_instruction"] = min_similar_instruction

from typing import Any, Dict, List, Optional, Tuple

from lingua import LanguageDetectorBuilder
from transformers import AutoTokenizer
from vllm import LLM, PoolingParams, SamplingParams

from datatagger.settings.base_tagger_setting import TagMission
from datatagger.settings.tagger_settings_vllm import TaggerSettingsVLLM
from datatagger.tagger.base_tagger import BaseUnifiedTagger
from datatagger.utils.file_utils import load_dataset_from_file


class UnifiedTaggerVLLM(BaseUnifiedTagger):
    def __init__(self, settings: TaggerSettingsVLLM) -> None:
        super().__init__(settings, is_api=False)
        self.settings = settings
        self.vllm_model_path = settings.vllm_model_path
        self.dtype = settings.dtype
        self.quantization = settings.quantization
        self.kv_cache_dtype = settings.kv_cache_dtype
        self.max_model_len = settings.max_model_len
        self.tensor_parallel_size = settings.tensor_parallel_size
        self.gpu_memory_utilization = settings.gpu_memory_utilization
        # Initialize language detector if mission is LANGUAGE
        if self.mission == TagMission.LANGUAGE:
            self.logger.info("Building language detector from all languages")
            self.detector = LanguageDetectorBuilder.from_all_languages().build()
            self.logger.info("Language detector built successfully")

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

        # Dynamically import and initialize milvus_client (if needed)
        self.faiss_client = getattr(self, "faiss_client", None)
        self.milvus_client = None
        if getattr(self, "milvus_store_embeddings", False) or getattr(
            settings, "milvus_store_embeddings", False
        ):
            try:
                from datatagger.utils.milvus_utils import MilvusClient

                self.milvus_client = MilvusClient()
            except ImportError:
                self.logger.error(
                    "Milvus related dependencies are not installed, please install them or disable milvus_store_embeddings."
                )
                raise

    def get_llm(self) -> Tuple[Optional[LLM], Optional[Any], Optional[Any]]:
        if self.mission == TagMission.LANGUAGE:
            return None, None, None
        if self.mission == TagMission.REWARD:
            self.logger.info(
                f"Loading reward model from {self.settings.vllm_model_path}"
            )
            rm_llm = LLM(
                model=self.settings.vllm_model_path,
                task="classify",
                override_pooler_config={"softmax": False},
                dtype=self.settings.dtype,
                quantization=self.settings.quantization
                if self.settings.quantization != "None"
                else None,
                kv_cache_dtype=self.settings.kv_cache_dtype,
                max_model_len=self.settings.max_model_len,
                tensor_parallel_size=self.settings.tensor_parallel_size,
                gpu_memory_utilization=self.settings.gpu_memory_utilization,
                trust_remote_code=True,
                enable_prefix_caching=True,
                enforce_eager=True,
            )
            rm_tokenizer = AutoTokenizer.from_pretrained(self.settings.vllm_model_path)
            return rm_llm, None, rm_tokenizer
        if self.mission == TagMission.SAFETY:
            self.logger.info("Loading vllm model for SAFETY task...")
            llm = LLM(
                model=self.vllm_model_path,
                dtype=self.settings.dtype,
                quantization=self.settings.quantization
                if self.settings.quantization != "None"
                else None,
                kv_cache_dtype=self.settings.kv_cache_dtype,
                max_model_len=self.settings.max_model_len,
                tensor_parallel_size=self.settings.tensor_parallel_size,
                gpu_memory_utilization=self.settings.gpu_memory_utilization,
                trust_remote_code=True,
                task="generate",
                enable_prefix_caching=True,
                enforce_eager=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.vllm_model_path,
                use_fast=True,
                trust_remote_code=True,
            )
            params = SamplingParams(
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                repetition_penalty=self.settings.repetition_penalty,
                stop=["}"],
                include_stop_str_in_output=True,
            )
            return llm, params, tokenizer
        if self.mission == TagMission.EMBEDDING:
            self.logger.info("Loading vllm model for EMBEDDING task...")
            llm = LLM(
                model=self.vllm_model_path,
                dtype=self.settings.dtype,
                quantization=self.settings.quantization
                if self.settings.quantization != "None"
                else None,
                kv_cache_dtype=self.settings.kv_cache_dtype,
                max_model_len=self.settings.max_model_len,
                tensor_parallel_size=self.settings.tensor_parallel_size,
                gpu_memory_utilization=self.settings.gpu_memory_utilization,
                trust_remote_code=True,
                task="embedding",
                enable_prefix_caching=True,
                enforce_eager=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.vllm_model_path,
                use_fast=True,
                trust_remote_code=True,
            )
            params = PoolingParams(dimensions=self.dimension)
            return llm, params, tokenizer
        self.logger.info("Loading vllm model for general task...")
        llm = LLM(
            model=self.vllm_model_path,
            dtype=self.settings.dtype,
            quantization=self.settings.quantization
            if self.settings.quantization != "None"
            else None,
            kv_cache_dtype=self.settings.kv_cache_dtype,
            max_model_len=self.settings.max_model_len,
            tensor_parallel_size=self.settings.tensor_parallel_size,
            gpu_memory_utilization=self.settings.gpu_memory_utilization,
            trust_remote_code=True,
            task="generate",
            enable_prefix_caching=True,
            enforce_eager=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.vllm_model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        params = SamplingParams(
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            repetition_penalty=self.settings.repetition_penalty,
            stop=["}"],
            include_stop_str_in_output=True,
        )
        return llm, params, tokenizer

    def process_batch(
        self,
        batch_indices: List[int],
        dataset: List[Dict[str, Any]],
        llm: Optional[LLM] = None,
        params: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.logger.info(
            f"Processing batch for indices: {batch_indices} with mission: {self.mission}"
        )

        if self.mission == TagMission.EMBEDDING:
            prompt_texts = [dataset[idx][self.prompt_field] for idx in batch_indices]
            prompt_embeddings = llm.embed(prompt_texts)
            prompt_emb_list = []
            prompt_metas = []
            for output, idx in zip(prompt_embeddings, batch_indices):
                embedding = output.outputs.embedding
                prompt_emb_list.append(embedding)
                prompt_metas.append(str(dataset[idx].get(self.prompt_field, "")))
            # Directly insert into faiss or milvus
            if self.milvus_store_embeddings and self.milvus_client:
                self.logger.info(
                    f"Inserting {len(prompt_emb_list)} prompt embeddings to Milvus..."
                )
                self.milvus_client.insert_embeddings(prompt_emb_list, prompt_metas)
            if self.faiss_store_embeddings and self.faiss_client:
                self.logger.info(
                    f"Inserting {len(prompt_emb_list)} prompt embeddings to Faiss..."
                )
                self.faiss_client.insert_embeddings(prompt_emb_list, prompt_metas)
            return
        prompts = []
        for idx in batch_indices:
            item = dataset[idx]
            if self.mission == TagMission.SAFETY:
                chat = [
                    {"role": "user", "content": item[self.prompt_field]},
                    {"role": "assistant", "content": item[self.output_field]},
                ]
                template = tokenizer.apply_chat_template(chat, tokenize=False)
            else:
                messages = [
                    {
                        "role": "user",
                        "content": self.mission_processor.get_prompt(
                            item[self.prompt_field],
                            item.get(self.output_field)
                            if self.mission == TagMission.QUALITY
                            else None,
                        ),
                    },
                    {"role": "assistant", "content": "{"},
                ]
                template = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.settings.enable_thinking,
                )
            prompts.append(template)
        outputs = llm.generate(prompts, params)
        for output, idx in zip(outputs, batch_indices):
            response = output.outputs[0].text
            if (
                self.mission != TagMission.SAFETY
                and self.mission != TagMission.EMBEDDING
            ):
                response = "{" + response
            self.mission_processor.process_response(response, dataset[idx])

    def process_batch_with_reward_model(
        self,
        batch_indices: List[int],
        dataset: List[Dict[str, Any]],
        llm: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.logger.info(
            f"Processing batch with reward model for indices: {batch_indices}"
        )
        for idx in batch_indices:
            try:
                item = dataset[idx]
                # TODO Single question multiple answers, create rm, DPO dataset
                chat = [
                    {"role": "user", "content": item[self.prompt_field]},
                    {"role": "assistant", "content": item[self.output_field]},
                ]
                prompt = tokenizer.apply_chat_template(chat, tokenize=False)
                encoded = tokenizer(prompt)
                input_ids = encoded["input_ids"]
                output = llm.encode(prompt_token_ids=input_ids)
                score = output[0].outputs.data
                score_int = score.item()
                dataset[idx]["instruct_reward"] = score_int
                self.logger.debug(
                    f"Successfully processed reward for index {idx}: {score_int}"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to process item: {dataset[idx]} with error: {str(e)}"
                )
                dataset[idx]["instruct_reward"] = None

    def generate_and_update(
        self, dataset: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        output_file, checkpoint_data_file, checkpoint_state_file = (
            self.get_output_files(
                settings=self.settings,
                tag_mission=self.tag_mission,
                input_file=self.settings.input_file,
            )
        )
        if dataset is None:
            dataset = load_dataset_from_file(self.settings.input_file)
            if self.debug:
                self.logger.warning(
                    "Debug mode enabled. Only processing the first 100 samples."
                )
                dataset = dataset[:100]
        llm, params, tokenizer = self.get_llm()

        def process_batch_fn(batch_indices, dataset):
            if self.mission == TagMission.LANGUAGE:
                self.process_batch_with_language_detection(
                    detector=self.detector,
                    logger=self.logger,
                    batch_indices=batch_indices,
                    dataset=dataset,
                    prompt_field=self.prompt_field,
                )
            elif self.mission == TagMission.REWARD:
                self.process_batch_with_reward_model(
                    batch_indices=batch_indices,
                    dataset=dataset,
                    llm=llm,
                    tokenizer=tokenizer,
                )
            else:
                self.process_batch(
                    batch_indices=batch_indices,
                    dataset=dataset,
                    llm=llm,
                    params=params,
                    tokenizer=tokenizer,
                )

        def postprocess_fn(dataset):
            if self.mission == TagMission.EMBEDDING:
                self.update_similarity_fields(dataset=dataset, field=self.prompt_field)

        self.generate_and_update_with_checkpoint(
            dataset=dataset,
            output_file=output_file,
            checkpoint_data_file=checkpoint_data_file,
            checkpoint_state_file=checkpoint_state_file,
            process_batch_fn=process_batch_fn,
            batch_size=self.batch_size,
            checkpoint_every=self.checkpoint_every,
            logger=self.logger,
            postprocess_fn=postprocess_fn,
        )

    @staticmethod
    def get_settings() -> TaggerSettingsVLLM:
        return TaggerSettingsVLLM()


if __name__ == "__main__":
    settings = TaggerSettingsVLLM()
    tagger = UnifiedTaggerVLLM(settings)
    tagger.generate_and_update()

from typing import Any, Dict, List, Optional

from datatagger.settings.base_tagger_setting import TagMission
from datatagger.settings.tagger_settings_api import TaggerSettingsAPI
from datatagger.tagger.base_tagger import BaseUnifiedTagger
from datatagger.utils.api_utils import (
    get_completion_with_retry,
    get_embedding_with_retry,
)
from datatagger.utils.file_utils import load_dataset_from_file


class UnifiedTaggerAPI(BaseUnifiedTagger):
    def __init__(self, settings: TaggerSettingsAPI) -> None:
        super().__init__(settings, is_api=True)
        self.settings = settings
        self.api_model_name = settings.api_model_name
        self.api_base_url = settings.api_url.rstrip("/")
        self.api_key = settings.api_key
        self.api_params = {
            "model": self.api_model_name,
            "max_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
            "repetition_penalty": self.settings.repetition_penalty,
            "stop": ["}"],
        }

        # 是否开启思考
        if self.settings.enable_thinking:
            self.api_params["chat_template_kwargs"] = {"enable_thinking": True}
        else:
            self.api_params["chat_template_kwargs"] = {"enable_thinking": False}
        self.api_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 动态导入和初始化 milvus_client（如需）
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
                    "未安装 milvus 相关依赖包，请先安装或关闭 milvus_store_embeddings 配置。"
                )
                raise

        # 删除 EmbeddingStore 初始化
        # self.embedding_store = EmbeddingStore(
        #     faiss_client=self.faiss_client,
        #     milvus_client=self.milvus_client,
        # )

    def get_api_url(self, endpoint: str) -> str:
        base_url = self.api_base_url.rstrip("/")
        endpoint_map = {
            "chat/completions": "/v1/chat/completions",
            "embeddings": "/v1/embeddings",
        }
        endpoint_path = endpoint_map.get(endpoint, f"/v1/{endpoint.lstrip('/')}")
        return f"{base_url}{endpoint_path}"

    def process_batch_with_api(
        self, batch_indices: List[int], dataset: List[Dict[str, Any]]
    ) -> None:
        self.logger.info(
            f"Processing batch with API for indices: {batch_indices} for mission: {self.mission}"
        )
        if self.mission == TagMission.EMBEDDING:
            import concurrent.futures

            prompt_texts = [dataset[idx][self.prompt_field] for idx in batch_indices]
            api_url = self.get_api_url("embeddings")
            prompt_embeddings = [None] * len(prompt_texts)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_idx = {
                    executor.submit(
                        get_embedding_with_retry,
                        text,
                        api_url,
                        self.api_headers,
                        self.api_model_name,
                    ): i
                    for i, text in enumerate(prompt_texts)
                }
                for future in concurrent.futures.as_completed(future_to_idx):
                    i = future_to_idx[future]
                    try:
                        embedding = future.result()
                        if embedding is not None:
                            prompt_embeddings[i] = embedding
                        else:
                            self.logger.error(
                                f"Invalid embedding response for prompt: {prompt_texts[i]}"
                            )
                    except Exception as e:
                        self.logger.error(
                            f"Exception in prompt embedding for index {i}: {e}"
                        )
            if (self.faiss_store_embeddings and self.faiss_client) or (
                self.milvus_store_embeddings and self.milvus_client
            ):
                prompt_metas = [
                    str(dataset[idx].get(self.prompt_field, ""))
                    for idx in batch_indices
                ]
                if self.milvus_store_embeddings and self.milvus_client:
                    self.logger.info(
                        f"Inserting {len(prompt_embeddings)} prompt embeddings to Milvus..."
                    )
                    self.milvus_client.insert_embeddings(
                        prompt_embeddings, prompt_metas
                    )
                if self.faiss_store_embeddings and self.faiss_client:
                    self.logger.info(
                        f"Inserting {len(prompt_embeddings)} prompt embeddings to Faiss..."
                    )
                    self.faiss_client.insert_embeddings(prompt_embeddings, prompt_metas)
                return
        # 多线程获取API响应
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_item = {
                executor.submit(
                    get_completion_with_retry,
                    [
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
                    ],
                    self.api_params,
                    self.get_api_url("chat/completions"),
                    self.api_headers,
                ): (idx, item)
                for idx, item in enumerate(dataset)
                if idx in batch_indices
            }
            for future in concurrent.futures.as_completed(future_to_item):
                idx, item = future_to_item[future]
                try:
                    api_response = future.result()
                    api_response = "{" + api_response + "}"
                    self.mission_processor.process_response(api_response, dataset[idx])
                except Exception as e:
                    self.logger.error(
                        f"Error processing API response for index {idx}: {e}"
                    )

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

        def process_batch_fn(batch_indices, dataset):
            if self.mission == TagMission.LANGUAGE:
                self.process_batch_with_language_detection(
                    detector=self.detector,
                    logger=self.logger,
                    dataset=dataset,
                    prompt_field=self.prompt_field,
                    batch_indices=batch_indices,
                )
            else:
                self.process_batch_with_api(batch_indices, dataset)

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
    def get_settings() -> TaggerSettingsAPI:
        return TaggerSettingsAPI()


if __name__ == "__main__":
    settings = TaggerSettingsAPI()
    tagger = UnifiedTaggerAPI(settings)
    tagger.generate_and_update()

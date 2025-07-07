from typing import Any, Dict, List, Optional

from datatagger.settings.base_tagger_setting import TagMission
from datatagger.settings.tagger_settings_api import TaggerSettingsAPI
from datatagger.tagger.base_tagger import BaseUnifiedTagger
from datatagger.utils.api_utils import (
    get_completion_with_retry,
    get_embedding_with_retry,
)
from datatagger.utils.file_utils import load_dataset_from_file
from datatagger.utils.embedding_store import EmbeddingStore


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

        # 初始化EmbeddingStore
        self.embedding_store = EmbeddingStore(
            faiss_client=getattr(self, "faiss_client", None),
            milvus_client=getattr(self, "milvus_client", None),
        )

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
            # output_texts = [dataset[idx][self.output_field] for idx in batch_indices]  # 回答不再embed
            api_url = self.get_api_url("embeddings")
            # 多线程获取prompt_embeddings
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
            # 多线程获取output_embeddings
            # output_embeddings = [None] * len(output_texts)
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     future_to_idx = {
            #         executor.submit(
            #             get_embedding_with_retry,
            #             text,
            #             api_url,
            #             self.api_headers,
            #             self.api_model_name,
            #             self.dimension,
            #         ): i
            #         for i, text in enumerate(output_texts)
            #     }
            #     for future in concurrent.futures.as_completed(future_to_idx):
            #         i = future_to_idx[future]
            #         try:
            #             embedding = future.result()
            #             if embedding is not None:
            #                 output_embeddings[i] = embedding
            #             else:
            #                 self.logger.error(
            #                     f"Invalid embedding response for output: {output_texts[i]}"
            #                 )
            #         except Exception as e:
            #             self.logger.error(
            #                 f"Exception in output embedding for index {i}: {e}"
            #             )
            # faiss写入
            if self.faiss_store_embeddings or self.milvus_store_embeddings:
                prompt_metas = [
                    str(dataset[idx].get(self.prompt_field, ""))
                    for idx in batch_indices
                ]
                # output_metas = [
                #     str(dataset[idx].get(self.output_field, ""))
                #     for idx in batch_indices
                # ]
                self.logger.info(
                    f"Inserting {len(prompt_embeddings)} prompt embeddings to Milvus/Faiss via EmbeddingStore..."
                )
                self.embedding_store.insert(
                    prompt_embeddings,
                    prompt_metas,
                    use_faiss=self.faiss_store_embeddings,
                    use_milvus=self.milvus_store_embeddings,
                )
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

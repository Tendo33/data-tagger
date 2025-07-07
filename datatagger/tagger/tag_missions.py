from typing import Any, Dict, List

import json_repair

from datatagger.settings.base_tagger_setting import BaseTaggerSettings, TagMission
from datatagger.utils.prompt_utils import (
    combined_quality_rating,
    input_classification,
    input_difficulty_rating,
    input_quality_rating,
)


class TagMissionProcessor:
    SAFETY_LABEL_MAPPING = {
        "S1": "Violent Crimes",
        "S2": "Non-Violent Crimes",
        "S3": "Sex-Related Crimes",
        "S4": "Child Sexual Exploitation",
        "S5": "Defamation",
        "S6": "Specialized Advice",
        "S7": "Privacy",
        "S8": "Intellectual Property",
        "S9": "Indiscriminate Weapons",
        "S10": "Hate",
        "S11": "Suicide & Self-Harm",
        "S12": "Sexual Content",
        "S13": "Elections",
        "S14": "Code Interpreter Abuse",
        "safe": "Safe",
    }

    def __init__(self, mission: TagMission, settings: BaseTaggerSettings):
        self.mission = mission
        self.settings = settings

    def get_prompt(self, input_text: str, response_text: str = None) -> str:
        """Get the prompt for the given mission."""
        if self.mission == TagMission.QUALITY:
            if response_text:
                return combined_quality_rating(query=input_text, response=response_text)
            return input_quality_rating(input=input_text)
        elif self.mission == TagMission.DIFFICULTY:
            return input_difficulty_rating(input=input_text)
        elif self.mission == TagMission.CLASSIFICATION:
            return input_classification(input=input_text)
        elif self.mission == TagMission.SAFETY:
            # Safety mission uses a different format
            return input_text
        else:
            raise ValueError(f"Unsupported mission: {self.mission}")

    def process_response(self, response_text: str, item: Dict[str, Any]) -> None:
        """Process the response and update the item."""
        try:
            if self.mission == TagMission.SAFETY:
                item["safety"] = self.SAFETY_LABEL_MAPPING.get(
                    response_text.strip(), None
                )
                return
            elif self.mission == TagMission.EMBEDDING:
                # For embedding mission, we don't need to process response
                # as embeddings are already added in process_batch
                return

            response_json = json_repair.loads(response_text)

            if self.mission == TagMission.QUALITY:
                item["input_quality"] = response_json.get("input_quality", None)
                item["response_quality"] = response_json.get("response_quality", None)
                item["input_quality_explanation"] = response_json.get(
                    "input_quality_explanation", None
                )
                item["response_quality_explanation"] = response_json.get(
                    "response_quality_explanation", None
                )
            elif self.mission == TagMission.DIFFICULTY:
                item["intent"] = response_json.get("intent", None)
                item["knowledge"] = response_json.get("knowledge", None)
                item["difficulty"] = response_json.get("difficulty", None)
            elif self.mission == TagMission.CLASSIFICATION:
                item["task_category"] = response_json.get("primary_tag", None)
                item["other_task_category"] = response_json.get("other_tags", None)
            elif self.mission == TagMission.REWARD:
                item["instruct_reward"] = response_json.get("score", None)
            elif self.mission == TagMission.LANGUAGE:
                item["language"] = response_json.get("language", None)
        except Exception as e:
            self._handle_error(item, e)

    def _handle_error(self, item: Dict[str, Any], error: Exception) -> None:
        """Handle errors by setting default values."""
        if self.mission == TagMission.QUALITY:
            item["input_quality"] = None
            item["response_quality"] = None
            item["input_quality_explanation"] = None
            item["response_quality_explanation"] = None
        elif self.mission == TagMission.DIFFICULTY:
            item["intent"] = None
            item["knowledge"] = None
            item["difficulty"] = None
        elif self.mission == TagMission.CLASSIFICATION:
            item["task_category"] = None
            item["other_task_category"] = None
        elif self.mission == TagMission.SAFETY:
            item["safety"] = None
        elif self.mission == TagMission.REWARD:
            item["instruct_reward"] = None
        elif self.mission == TagMission.LANGUAGE:
            item["language"] = None
        elif self.mission == TagMission.EMBEDDING:
            # item[f"{self.prompt_field}_embedding"] = None
            # item[f"{self.output_field}_embedding"] = None
            pass

    def get_output_fields(self) -> List[str]:
        """Get the output fields for the mission."""
        if self.mission == TagMission.QUALITY:
            return [
                "input_quality",
                "response_quality",
                "input_quality_explanation",
                "response_quality_explanation",
            ]
        elif self.mission == TagMission.DIFFICULTY:
            return ["intent", "knowledge", "difficulty"]
        elif self.mission == TagMission.CLASSIFICATION:
            return ["task_category", "other_task_category"]
        elif self.mission == TagMission.SAFETY:
            return ["safety"]
        elif self.mission == TagMission.REWARD:
            return ["instruct_reward"]
        elif self.mission == TagMission.LANGUAGE:
            return ["language"]
        elif self.mission == TagMission.EMBEDDING:
            # return [f"{self.prompt_field}_embedding", f"{self.output_field}_embedding"]
            pass
        else:
            raise ValueError(f"Unsupported mission: {self.mission}")

    def get_name(self) -> str:
        """Get the name of the mission."""
        return self.mission.name.lower()

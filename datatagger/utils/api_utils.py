from time import sleep
from typing import Any, Dict, List

import requests


# Function to make a single API request with exponential back-off
def get_completion_with_retry(
    message: List[Dict[str, Any]],
    api_params: Dict[str, Any],
    api_endpoint: str,
    api_headers: Dict[str, Any],
    max_retries: int = 5,
):
    payload = api_params.copy()
    payload["messages"] = message

    for attempt in range(max_retries):
        try:
            response = requests.post(api_endpoint, json=payload, headers=api_headers)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            sleep(2**attempt)  # Exponential back-off

    print("All retry attempts failed.")
    return None


def get_embedding_with_retry(
    text: str,
    api_endpoint: str,
    api_headers: Dict[str, Any],
    api_model_name: str,
    max_retries: int = 5,
    # dimension: int = 1024,
):
    payload = {
        "input": text,
        "model": api_model_name,
        # "dimensions": dimension,
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(api_endpoint, json=payload, headers=api_headers)
            response.raise_for_status()
            data = response.json()
            if (
                "data" in data
                and len(data["data"]) > 0
                and "embedding" in data["data"][0]
            ):
                return data["data"][0]["embedding"]
            else:
                print(f"Invalid embedding response: {data}")
                return None
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} (embedding) failed: {str(e)}")
            sleep(2**attempt)
    print("All retry attempts for embedding failed.")
    return None

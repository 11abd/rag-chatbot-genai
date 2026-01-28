import requests
import time
from utils.logger import logger

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

def generate_answer(prompt: str) -> str:
    logger.info("Sending prompt to Ollama")


    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 150,
            "temperature": 0.2
        }
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    logger.info("Received response from Ollama")

    return response.json().get("response", "").strip()



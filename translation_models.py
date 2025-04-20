# translation_models.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils.hub import cached_file
from transformers.utils import EntryNotFoundError

MODEL_NAME = "facebook/nllb-200-distilled-600M"

def is_model_downloaded(model_name: str) -> bool:
    try:
        # Check for presence of model config file in cache
        cached_file(model_name, "config.json")
        return True
    except EntryNotFoundError:
        return False

def install():
    if is_model_downloaded(MODEL_NAME):
        print(f"✔ Model already downloaded: {MODEL_NAME}")
    else:
        print(f"⬇ Downloading model: {MODEL_NAME}")
        AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        AutoTokenizer.from_pretrained(MODEL_NAME)
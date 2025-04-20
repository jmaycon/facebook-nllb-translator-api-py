import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

import syntok.segmenter as segmenter
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM

import translation_models
from tokenizer_pool import TokenizerPool

# Install Models
translation_models.install()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


# Devices
CPU = torch.device("cpu")
CUDA = torch.device("cuda") if torch.cuda.is_available() else None

# Language codes
lang_codes = {
    "de": "deu_Latn",
    "en": "eng_Latn"
}

# Tokenizer pool (shared for CPU and GPU)
tokenizer_pool = TokenizerPool("facebook/nllb-200-distilled-600M")

# Load models
models = {
    "cpu": {
        "model": AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(CPU)
    }
}
if CUDA:
    models["gpu"] = {
        "model": AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(CUDA)
    }

# Warm-up
logger.info("Warming up models...")
for device_key in models:
    tokenizer = tokenizer_pool.acquire()
    model = models[device_key]["model"]
    for direction in ["de-en", "en-de"]:
        try:
            input_text = "Hallo Welt" if direction == "de-en" else "Hello world"
            src_lang = lang_codes[direction.split("-")[0]]
            tgt_lang = lang_codes[direction.split("-")[1]]
            tokenizer.src_lang = src_lang
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
            bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
            model.generate(**inputs, forced_bos_token_id=bos_token_id)
            logger.info(f"Warm-up completed for {direction} on {device_key}")
        except Exception as e:
            logger.warning(f"Warm-up failed for {direction} on {device_key}: {e}")
    tokenizer_pool.release(tokenizer)

logger.info("Warm-up done.")


# Input model
class TranslationRequest(BaseModel):
    text: str
    direction: str  # "de-en" or "en-de"


# Sentence splitting
def split_into_sentences(text: str):
    sentences = []
    for paragraph in segmenter.process(text):
        for sentence in paragraph:
            sentences.append("".join(token.value for token in sentence))
    return sentences


# Translation logic
def translate_sentences(sentences, tokenizer_pool, model, device, source_lang, target_lang):
    translations = []
    for sentence in sentences:
        logger.info(f"Processing sentence: {sentence}")
        tokenizer = tokenizer_pool.acquire()
        try:
            tokenizer.src_lang = source_lang
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang)
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            translations.append(decoded)
        finally:
            tokenizer_pool.release(tokenizer)
    return translations


def parallel_translate(text, tokenizer_pool, model, device, source_lang, target_lang):
    sentences = split_into_sentences(text)
    logger.info(f"Will process {len(sentences)} sentence(s) in parallel.")

    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for i, sentence in enumerate(sentences):
            logger.info(f"Submitting translation task {i+1}/{len(sentences)}")
            futures.append(
                executor.submit(
                    translate_sentences,
                    [sentence],
                    tokenizer_pool,
                    model,
                    device,
                    source_lang,
                    target_lang
                )
            )
        results = [future.result()[0] for future in futures]

    return " ".join(results)



# Translation wrapper
def perform_translation(device_key: str, request: TranslationRequest):
    device = CPU if device_key == "cpu" else CUDA
    if device is None:
        raise HTTPException(status_code=503, detail=f"{device_key.upper()} not available on this system.")
    if request.direction not in ["de-en", "en-de"]:
        raise HTTPException(status_code=400, detail="Invalid direction (use 'de-en' or 'en-de')")

    try:
        model = models[device_key]["model"]
        source_lang, target_lang = request.direction.split("-")
        return {
            "translation": parallel_translate(
                request.text, tokenizer_pool, model, device,
                lang_codes[source_lang], lang_codes[target_lang]
            )
        }
    except Exception as e:
        logger.error(f"{device_key.upper()} translation failed:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


# Endpoints
@app.post("/translate-cpu")
def translate_cpu(request: TranslationRequest):
    return perform_translation("cpu", request)


@app.post("/translate-gpu")
def translate_gpu(request: TranslationRequest):
    return perform_translation("gpu", request)

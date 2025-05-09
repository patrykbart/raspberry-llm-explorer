import ollama
import logging
from schema.model import MODEL_NAME, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Stwórz model (raz, przy starcie)
try:
    ollama.create(model=MODEL_NAME, from_="llava", system=SYSTEM_PROMPT)
    logger.info("Custom model created successfully.")
except Exception as e:
    logger.warning(f"Model create skipped or failed: {e}")

# Preload modelu
try:
    ollama.generate(model=MODEL_NAME, prompt="", images=[], stream=False)
    logger.info("Model preloaded successfully.")
except Exception as e:
    logger.warning(f"Model preload skipped or failed: {e}")

# Funkcja do generowania odpowiedzi AI
def ollama_generate(**kwargs):
    return ollama.generate(**kwargs)
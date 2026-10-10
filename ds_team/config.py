"""Project paths and LLM configuration shared by every agent."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# load the api keys
load_dotenv()

# path for the base/project directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

# SQLite file that stores conversation checkpoints (the team's memory)
DB_PATH = BASE_DIR / "memory.db"

# default dataset the team works on
DEFAULT_DATASET = DATA_DIR / "car_details.csv"

# "provider:model", e.g. "openai:gpt-5-mini" or "groq:llama-3.3-70b-versatile"
MODEL = os.getenv("MODEL", "openai:gpt-5-mini")


def get_model() -> BaseChatModel:
    """Create the chat model configured in .env (MODEL=provider:model)."""
    return init_chat_model(MODEL)

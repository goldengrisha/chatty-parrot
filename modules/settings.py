import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Settings:
    @staticmethod
    def get_ai_key(alias: str) -> str:
        alias_map = {"GPT-3.5": "OPENAI_API_3_5_KEY", "GPT-4": "OPENAI_API_4_KEY"}
        return os.getenv(alias_map.get(alias, ""), "")

    @staticmethod
    def get_tg_token() -> str:
        return os.getenv("TELEGRAM_BOT_TOKEN", "")

    @staticmethod
    def get_chunk_size() -> int:
        return int(os.getenv("CHUNK_SIZE", 500))

    @staticmethod
    def get_chunk_overlap() -> int:
        return int(os.getenv("CHUNK_OVERLAP", 50))

    @staticmethod
    def get_search_precision() -> float:
        return float(os.getenv("SEARCH_PRECISION", 0.45))

import diskcache
from pathlib import Path

cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)

# Persistent cache (hard disk par save)
cache = diskcache.Cache(str(cache_dir))

def get_cached_response(query: str, patient_id: int) -> str | None:
    """Cache se jawab lao"""
    key = f"{patient_id}:{query.lower().strip()}"
    return cache.get(key)

def set_cached_response(query: str, patient_id: int, response: str):
    """Jawab cache mein store karo"""
    key = f"{patient_id}:{query.lower().strip()}"
    cache.set(key, response, expire=3600)  # 1 ghanta expire

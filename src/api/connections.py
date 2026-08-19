import os

# Loads external service config from environment variables.
# Extend this as you add real services (cloud storage, DB, notifications).

STORAGE_API_URL = os.getenv("STORAGE_API_URL", "")
STORAGE_API_KEY = os.getenv("STORAGE_API_KEY", "")


def get_storage_headers():
    return {"Authorization": f"Bearer {STORAGE_API_KEY}"}

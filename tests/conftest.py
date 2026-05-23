import os

import pytest


os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("GROQ_API_KEY", "gsk-test-key")


@pytest.fixture
def settings():
    from config.settings import Settings

    return Settings(_env_file=None, GROQ_API_KEY="gsk-test-key")

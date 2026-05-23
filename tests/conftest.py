import os

import pytest


os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("XAI_API_KEY", "xai-test-key")


@pytest.fixture
def settings():
    from config.settings import Settings

    return Settings(XAI_API_KEY="xai-test-key")

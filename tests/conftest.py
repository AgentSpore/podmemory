import os

os.environ.setdefault("GROQ_API_KEY", "test-key")

import pytest
from fastapi.testclient import TestClient

from podmemory.main import app


@pytest.fixture()
def client():
    return TestClient(app)

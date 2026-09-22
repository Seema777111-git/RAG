"""Keeps .env.example honest: every setting is documented, and the file itself parses."""

import re
from pathlib import Path

from backend.app.core.config import PROJECT_ROOT, Settings


def _example_keys() -> set[str]:
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    return set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", text, flags=re.MULTILINE))


def test_every_setting_is_documented_in_env_example():
    fields = {name.upper() for name in Settings.model_fields}
    assert fields - _example_keys() == set(), "settings missing from .env.example"
    assert _example_keys() - fields == set(), "unknown keys in .env.example"


def test_env_example_loads_cleanly():
    s = Settings(_env_file=PROJECT_ROOT / ".env.example")
    assert s.llm_model and s.chunk_size >= 200
    assert s.data_dir == Path(PROJECT_ROOT / "data")
    assert not s.llm_configured  # shipped without a key
    assert s.llm_provider == "openai" and s.llm_model == "gpt-4o" and s.extraction_model == "gpt-4o-mini"


def test_cors_origins_parse():
    s = Settings(_env_file=None, cors_origins="http://a.test, http://b.test")
    assert s.cors_origin_list == ["http://a.test", "http://b.test"]


def test_blank_values_are_not_swallowed_by_inline_comments():
    """`KEY=   # comment` is parsed by dotenv as the value '# comment'. No setting may end up that way."""
    s = Settings(_env_file=PROJECT_ROOT / ".env.example")
    for name, value in s.model_dump().items():
        text = value.get_secret_value() if hasattr(value, "get_secret_value") else value
        assert not (isinstance(text, str) and text.lstrip().startswith("#")), f"{name} was parsed as a comment"
    assert s.openai_base_url == "" and s.openai_api_key.get_secret_value() == ""
    assert (PROJECT_ROOT / ".env").read_text() == (PROJECT_ROOT / ".env.example").read_text()

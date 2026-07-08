from local_automata_core.config import DEFAULT_MODEL, Config


def test_defaults():
    c = Config.load()
    assert c.model == DEFAULT_MODEL
    assert c.base_url == "http://localhost:8799/v1"
    assert c.tool_mode == "prompt"  # 既定は自作方式（過程を生で見やすい）
    assert c.verbose is True  # 既定で過程を全部表示
    assert c.enable_thinking is False
    assert c.allow_install is False
    assert c.stream is True


def test_verbose_override():
    assert Config.load({"verbose": False}).verbose is False
    assert Config.load({"tool_mode": "native"}).tool_mode == "native"


def test_file_overrides_defaults():
    c = Config.load(
        {
            "model": "m",
            "base_url": "http://x:1/v1",
            "temperature": 0.7,
            "max_tokens": 8000,
            "enable_thinking": True,
            "stream": False,
            "tool_mode": "prompt",
            "allow_install": True,
        }
    )
    assert c.model == "m"
    assert c.base_url == "http://x:1/v1"
    assert c.temperature == 0.7
    assert c.max_tokens == 8000
    assert c.enable_thinking is True
    assert c.stream is False
    assert c.tool_mode == "prompt"
    assert c.allow_install is True


def test_unknown_keys_ignored():
    # backend など Config に無いキーは無視される
    c = Config.load({"backend": "x", "parallel": 4})
    assert not hasattr(c, "backend")
    assert c.model == DEFAULT_MODEL

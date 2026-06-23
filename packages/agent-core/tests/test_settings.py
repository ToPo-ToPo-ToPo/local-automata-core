import pytest

from agent_core.settings import find_config_path, load_agent_config


def write(tmp_path, text):
    p = tmp_path / "agent.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_none_path_gives_defaults():
    ac = load_agent_config(None)
    assert ac.tools is None and ac.runtime == {} and ac.system_prompt is None


def test_tools_and_runtime(tmp_path):
    # model を指定したら base_url も必須（モデル→ポートの取り違え防止ルール）。
    p = write(
        tmp_path,
        'tools = ["read_file"]\nmodel = "m"\n'
        'base_url = "http://localhost:8124/v1"\ntemperature = 0.5\n',
    )
    ac = load_agent_config(p)
    assert ac.tools == ["read_file"]
    assert ac.runtime["model"] == "m" and ac.runtime["temperature"] == 0.5


def test_forward_tool_images_in_runtime(tmp_path):
    # forward_tool_images は runtime に入り、Config まで伝わる
    # （MCPツールが返す画像をVLMへ転送するスイッチ。vision 利用時に true）。
    from agent_core.config import Config

    p = write(
        tmp_path,
        'model = "m"\nbase_url = "http://localhost:8124/v1"\n'
        'forward_tool_images = true\n',
    )
    ac = load_agent_config(p)
    assert ac.runtime["forward_tool_images"] is True
    assert Config.load(ac.runtime).forward_tool_images is True


def test_profiles_default_and_select(tmp_path):
    p = write(
        tmp_path,
        'default_profile = "a"\n'
        '[profiles.a]\nmodel = "ma"\nbase_url = "http://localhost:8124/v1"\ntools = ["read_file"]\n'
        '[profiles.b]\nmodel = "mb"\nbase_url = "http://localhost:8125/v1"\ntools = []\n',
    )
    assert load_agent_config(p).runtime["model"] == "ma"
    assert load_agent_config(p, profile="b").runtime["model"] == "mb"
    assert load_agent_config(p, profile="b").tools == []


def test_model_requires_base_url(tmp_path):
    # model を指定したら base_url（ポート）も必須。省くとエラーで止める。
    p = write(tmp_path, 'model = "custom-model"\n')
    with pytest.raises(ValueError):
        load_agent_config(p)


def test_default_model_needs_no_port(tmp_path):
    # model も base_url も省けば既定（Qwen3.6 @ ゲートウェイ :8799）で通る。
    p = write(tmp_path, 'tools = ["read_file"]\n')
    ac = load_agent_config(p)
    assert "model" not in ac.runtime and "base_url" not in ac.runtime


def test_web_port_in_runtime(tmp_path):
    p = write(tmp_path, "web_port = 8771\n")
    assert load_agent_config(p).runtime["web_port"] == 8771


def test_web_port_out_of_range(tmp_path):
    p = write(tmp_path, "web_port = 70000\n")
    with pytest.raises(ValueError):
        load_agent_config(p)


def test_profile_must_be_selected(tmp_path):
    p = write(tmp_path, "[profiles.a]\ntools = []\n[profiles.b]\ntools = []\n")
    with pytest.raises(ValueError):
        load_agent_config(p)  # default_profile 無し → 選択必須


def test_unknown_profile(tmp_path):
    p = write(tmp_path, '[profiles.a]\ntools = []\n')
    with pytest.raises(ValueError):
        load_agent_config(p, profile="zzz")


def test_system_prompt_deprecated(tmp_path):
    p = write(tmp_path, 'system_prompt = "x"\n')
    with pytest.raises(ValueError):
        load_agent_config(p)


def test_instructions_file(tmp_path):
    (tmp_path / "P.md").write_text("ペルソナ本文", encoding="utf-8")
    p = write(tmp_path, 'instructions_file = "P.md"\n')
    assert "ペルソナ本文" in load_agent_config(p).system_prompt


def test_memory(tmp_path):
    p = write(tmp_path, "[memory]\nenabled = true\npath = 'm.json'\n")
    ac = load_agent_config(p)
    assert ac.memory.enabled and ac.memory.path == "m.json"


def test_voice_default_is_none(tmp_path):
    ac = load_agent_config(write(tmp_path, "tools = []\n"))
    assert ac.voice is None


def test_voice_enabled_defaults(tmp_path):
    ac = load_agent_config(write(tmp_path, "tools = []\n[voice]\nenabled = true\n"))
    assert ac.voice.enabled
    assert ac.voice.model == "mlx-community/whisper-large-v3-mlx"
    assert ac.voice.language is None
    assert ac.voice.correct is False  # 既定は校正なし


def test_voice_full(tmp_path):
    ac = load_agent_config(
        write(tmp_path, '[voice]\nenabled = true\nmodel = "m"\nlanguage = "ja"\ncorrect = true\n')
    )
    assert ac.voice.enabled and ac.voice.model == "m" and ac.voice.language == "ja"
    assert ac.voice.correct is True


def test_voice_correct_must_be_bool(tmp_path):
    with pytest.raises(ValueError, match="correct"):
        load_agent_config(write(tmp_path, '[voice]\ncorrect = "yes"\n'))


def test_voice_validation(tmp_path):
    with pytest.raises(ValueError):
        load_agent_config(write(tmp_path, '[voice]\nenabled = "yes"\n'))
    with pytest.raises(ValueError):
        load_agent_config(write(tmp_path, "[voice]\nmodel = 1\n"))
    with pytest.raises(ValueError):
        load_agent_config(write(tmp_path, "[voice]\nlanguage = 2\n"))


def test_runtime_validation(tmp_path):
    with pytest.raises(ValueError):
        load_agent_config(write(tmp_path, 'temperature = "hot"\n'))
    with pytest.raises(ValueError):
        load_agent_config(write(tmp_path, 'tool_mode = "x"\n'))
    with pytest.raises(ValueError, match="image_context_mode"):
        load_agent_config(write(tmp_path, 'image_context_mode = "smart"\n'))
    # 正しい値は通る
    ac = load_agent_config(write(tmp_path, 'image_context_mode = "on_demand"\n'))
    assert ac.runtime["image_context_mode"] == "on_demand"


def test_find_config_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert find_config_path() is None
    (tmp_path / "agent.toml").write_text("tools = []\n")
    assert find_config_path() is not None
    assert find_config_path("/x/y.toml").name == "y.toml"

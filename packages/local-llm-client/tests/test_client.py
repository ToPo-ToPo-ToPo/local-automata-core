"""LLMClient / connect / 整形ヘルパのテスト（openai をモック、実サーバーには繋がない）。"""
from __future__ import annotations

import base64

import pytest

from local_llm_client import ServerNotRunningError
from local_llm_client import client as client_mod
from local_llm_client.client import (
    LLMClient,
    build_user_content,
    connect,
    thinking_extra_body,
    to_image_url,
)


# --- マルチモーダル content 構築 -------------------------------------------
def test_build_user_content_text_only():
    assert build_user_content("hello") == "hello"


def test_build_user_content_with_image_passthrough_url():
    content = build_user_content("見て", images=["https://example.com/a.png"])
    assert content[0] == {"type": "text", "text": "見て"}
    assert content[1]["image_url"]["url"] == "https://example.com/a.png"


def test_to_image_url_local_file_becomes_data_uri(tmp_path):
    p = tmp_path / "pix.png"
    p.write_bytes(b"\x89PNG\r\n")
    url = to_image_url(str(p))
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\x89PNG\r\n"


def test_to_image_url_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        to_image_url("/no/such/file.png")


# --- respond（openai クライアントをフェイクに差し替え） --------------------
class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeStreamChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeStreamChunk:
    def __init__(self, content):
        self.choices = [_FakeStreamChoice(content)]


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([_FakeStreamChunk("こん"), _FakeStreamChunk("にちは")])
        return _FakeResp("done")


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    def __init__(self, *a, **k):
        self.init_kwargs = k
        self.chat = _FakeChat()


@pytest.fixture
def fake_openai(monkeypatch):
    monkeypatch.setattr(client_mod, "OpenAI", _FakeOpenAI)


def test_respond_non_stream_returns_text(fake_openai):
    llm = LLMClient(model="m")
    assert llm.respond("hi") == "done"
    sent = llm.openai.chat.completions.calls[0]
    assert sent["messages"][-1] == {"role": "user", "content": "hi"}


def test_respond_includes_system_prompt(fake_openai):
    llm = LLMClient(model="m")
    llm.respond("hi", system_prompt="be brief")
    msgs = llm.openai.chat.completions.calls[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "be brief"}


def test_respond_stream_yields_pieces(fake_openai):
    llm = LLMClient(model="m")
    assert list(llm.respond("hi", stream=True)) == ["こん", "にちは"]


def test_respond_passes_images(fake_openai):
    llm = LLMClient(model="m")
    llm.respond("見て", images=["https://example.com/a.png"])
    content = llm.openai.chat.completions.calls[0]["messages"][-1]["content"]
    assert isinstance(content, list) and content[1]["type"] == "image_url"


def test_max_tokens_forwarded(fake_openai):
    LLMClient(model="m", max_tokens=128).respond("hi")
    assert LLMClient(model="m", max_tokens=128).max_tokens == 128


def test_openai_client_accessible(fake_openai):
    # 土台の openai クライアントに直接アクセスできる（高度操作用）。
    llm = LLMClient(model="m", base_url="http://127.0.0.1:8080/v1")
    assert llm.openai.init_kwargs["base_url"] == "http://127.0.0.1:8080/v1"


def test_timeout_passed_to_openai(fake_openai):
    # timeout を渡したときだけ openai クライアントへ伝える（None なら既定に任せる）。
    assert "timeout" not in LLMClient(model="m").openai.init_kwargs
    assert LLMClient(model="m", timeout=42.0).openai.init_kwargs["timeout"] == 42.0


# --- connect（起動中ゲートウェイに繋ぐだけ。自動起動しない） ----------------
def test_connect_returns_client_when_gateway_ready(fake_openai, monkeypatch):
    monkeypatch.setattr(client_mod, "is_ready", lambda url, *a, **k: True)
    llm = connect(model="m", base_url="http://127.0.0.1:8799/v1")
    assert isinstance(llm, LLMClient)
    assert llm.base_url == "http://127.0.0.1:8799/v1"


def test_connect_raises_when_gateway_down(monkeypatch):
    # 未起動なら自前で立てず、親切なエラーを投げる。
    monkeypatch.setattr(client_mod, "is_ready", lambda url, *a, **k: False)
    with pytest.raises(ServerNotRunningError):
        connect(model="m", base_url="http://127.0.0.1:8799/v1")


# --- thinking_extra_body（バックエンド protocol ヘルパ） --------------------
def test_thinking_extra_body_emits_both_forms():
    on = thinking_extra_body(True)
    assert on["enable_thinking"] is True                       # mlx-vlm 形式
    assert on["chat_template_kwargs"]["enable_thinking"] is True  # mlx_lm/llama 形式
    off = thinking_extra_body(False)
    assert off["enable_thinking"] is False
    assert off["chat_template_kwargs"]["enable_thinking"] is False


# --- chat()（tool-calling）---------------------------------------------------
class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _StreamChoice:
    def __init__(self, delta):
        self.choices = [type("C", (), {"delta": delta})()]


def _fake_stream(pieces):
    for p in pieces:
        yield _StreamChoice(_Delta(content=p))


def test_chat_prompt_mode_parses_tool_calls(monkeypatch):
    import local_llm_client.client as c

    class FakeChat:
        def create(self, **kw):
            # prompt-mode はストリーム。tool ブロックを含む本文を返す。
            return _fake_stream(['実行します\n', '```tool\n{"name":"run_command",',
                                 '"arguments":{"command":"ls"}}\n```'])
    class FakeOpenAI:
        def __init__(self, *a, **k): self.chat = type("X", (), {"completions": FakeChat()})()
    monkeypatch.setattr(c, "OpenAI", FakeOpenAI)

    llm = c.LLMClient(model="m", tool_mode="prompt", stream=True)
    out = []
    res = llm.chat(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "ls して"}],
        tools=[{"function": {"name": "run_command", "description": "", "parameters": {}}}],
        on_text=out.append,
    )
    assert res.tool_calls and res.tool_calls[0].function.name == "run_command"
    assert "ls" in res.tool_calls[0].function.arguments
    assert "".join(out).strip().startswith("実行します")  # 生テキストを on_text に流す

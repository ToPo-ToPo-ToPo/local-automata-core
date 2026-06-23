import pytest

from agent_core.stt import DEFAULT_STT_MODEL, Transcriber


def test_transcriber_defaults():
    t = Transcriber()
    assert t.model == DEFAULT_STT_MODEL and t.language is None
    t2 = Transcriber(model="", language="")
    assert t2.model == DEFAULT_STT_MODEL and t2.language is None  # 空文字は既定/自動判定
    assert Transcriber(language="ja")._kwargs()["language"] == "ja"


def test_transcribe_requires_mlx_whisper():
    # mlx-whisper 未導入なら、ファイル/PCM どちらも親切な RuntimeError にする。
    import importlib.util

    if importlib.util.find_spec("mlx_whisper") is not None:
        pytest.skip("mlx_whisper が導入済みのため未導入パスは検証しない")
    with pytest.raises(RuntimeError, match="mlx-whisper"):
        Transcriber().transcribe_file("x.wav")
    with pytest.raises(RuntimeError, match="mlx-whisper"):
        Transcriber().transcribe_pcm(b"\x00\x00\x00\x00")


def test_correct_transcript():
    from types import SimpleNamespace
    from agent_core.stt import correct_transcript

    class LLM:
        def chat(self, messages, tools, on_text=lambda t: None):
            # system + user の2メッセージが渡る
            assert messages[0]["role"] == "system" and "校正" in messages[0]["content"]
            return SimpleNamespace(content="その関数は正しく動作します。")

    assert "関数" in correct_transcript(LLM(), "その換数は正しく道作します")
    assert correct_transcript(LLM(), "   ") == ""  # 空はそのまま

    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError("down")

    assert correct_transcript(Boom(), "元の文") == "元の文"  # 失敗時は素の文字起こし

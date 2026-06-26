"""音声→テキスト（STT）。ローカルの mlx-whisper を使う。

このライブラリは「テキストを食べる」エージェントのコアには手を入れず、入力層で
だけ音声をテキストへ変換する。変換結果は通常のテキスト入力として Agent.run() に
渡るので、コアは音声を一切意識しない。外部ライブラリから使う場合も同じで、UI で
集めた音声を Transcriber でテキスト化し、その文字列を Agent.run(text) に渡せばよい。

- transcribe_file(path): 任意の音声ファイル（wav/mp3/m4a/webm …）をそのまま渡せる
  （mlx-whisper が ffmpeg でデコード。要 ffmpeg）。外部ライブラリ向けの入口。
- transcribe_pcm(pcm): 16kHz モノラルの float32 PCM を直接渡す（ffmpeg 不要。Web GUI は
  ブラウザ側で PCM に整えてからこれを使う）。

mlx-whisper / numpy は Apple Silicon の既定依存（`uv sync` で入る）。import は遅延させ、
他OS（mlx 系が入らない）でも本体の import を妨げない。
"""
from __future__ import annotations

# Whisper が期待するサンプリングレート（mlx-whisper の既定と一致）
SAMPLE_RATE = 16000

# 文字起こし校正のシステム指示（LLM に渡す。意味を変えず最小限だけ直させる）。
_CORRECT_INSTRUCTION = (
    "あなたは音声認識(ASR)の出力を校正するアシスタントです。"
    "渡された文を、文脈から自然になるよう最小限だけ修正してください。"
    "特に日本語の同音異義語・漢字の誤変換を直します。意味は変えず、語を勝手に"
    "足したり削ったりしないこと。句読点は読みやすく整える程度にとどめます。"
    "解説や前置きは一切付けず、修正後の本文だけを出力してください。"
)


def correct_transcript(llm, text: str, language: str | None = None) -> str:
    """文字起こし結果を LLM で校正して返す（同音異義語・漢字の誤変換を文脈で直す）。

    llm は `chat(messages, tools)` を持つ LLM クライアント。失敗時や空入力は素のまま返す。
    """
    text = (text or "").strip()
    if not text:
        return text
    try:
        result = llm.chat(
            [
                {"role": "system", "content": _CORRECT_INSTRUCTION},
                {"role": "user", "content": text},
            ],
            [],
        )
    except Exception:  # noqa: BLE001 - 校正失敗時は素の文字起こしを使う
        return text
    out = (getattr(result, "content", "") or "").strip()
    return out or text

# voice 既定モデル。非量子化の full large-v3 が最高精度（多言語＝日本語含むで turbo より上）。
# 速度/省メモリを優先するなら agent.toml で whisper-large-v3-turbo / -8bit などに変更可。
DEFAULT_STT_MODEL = "mlx-community/whisper-large-v3-mlx"


class Transcriber:
    """mlx-whisper による文字起こし。モデルは初回利用時に遅延ロードする。"""

    def __init__(self, model: str = DEFAULT_STT_MODEL, language: str | None = None) -> None:
        self.model = model or DEFAULT_STT_MODEL
        # 空文字は「自動判定」とみなす
        self.language = language or None

    def _require_mlx(self):
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as exc:  # 親切なエラーにして呼び出し側へ返す
            raise RuntimeError(
                "音声認識には mlx-whisper が必要です（既定の依存。Apple Silicon で "
                "`uv sync` すると入ります）。見つからない場合は `uv add mlx-whisper`"
                "（mlx-whisper は Apple Silicon 専用）。"
            ) from exc
        return mlx_whisper

    def _kwargs(self) -> dict:
        kwargs: dict = {"path_or_hf_repo": self.model}
        if self.language:
            kwargs["language"] = self.language
        return kwargs

    def transcribe_file(self, path: str) -> str:
        """任意の音声ファイル（wav/mp3/m4a/webm/mp4 …）を文字起こしして返す。

        mlx-whisper が内部で ffmpeg を使ってデコード・リサンプルする（要 ffmpeg）。
        外部ライブラリは録音をファイルに保存してこれを呼ぶのが最も簡単。
        """
        mlx_whisper = self._require_mlx()
        result = mlx_whisper.transcribe(str(path), **self._kwargs())
        return (result.get("text") or "").strip()

    def transcribe_pcm(self, pcm: "list[float] | bytes") -> str:
        """16kHz モノラルの float32 PCM（bytes か配列）を文字起こしして返す。"""
        mlx_whisper = self._require_mlx()
        import numpy as np

        if isinstance(pcm, (bytes, bytearray, memoryview)):
            audio = np.frombuffer(bytes(pcm), dtype=np.float32)
        else:
            audio = np.asarray(pcm, dtype=np.float32)
        if audio.size == 0:
            return ""
        result = mlx_whisper.transcribe(audio, **self._kwargs())
        return (result.get("text") or "").strip()

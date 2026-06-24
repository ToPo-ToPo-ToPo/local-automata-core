"""トークン数の概算。

コンテキスト圧縮の基準をトークンで測るためのカウンタ。

- まずモデルの実トークナイザ（transformers, ローカルキャッシュのみ）で正確に数える。
  mlx などでモデルを取得済みなら tokenizer ファイルもキャッシュにあるため、その場合は
  モデルと同じトークン数になる（ネットワークアクセスはしない: local_files_only=True）。
- トークナイザが無い/読めない場合は、日本語（CJK）を考慮した概算にフォールバックする
  （CJK ≒ 1 トークン/字、その他 ≒ 4 字/トークン）。
"""
from __future__ import annotations

import re

# 漢字・ひらがな・カタカナ・全角記号など（概算で 1 文字 ≒ 1 トークンとみなす範囲）
_CJK = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿＀-￯]")


def _estimate(text: str) -> int:
    """トークナイザが使えないときの概算。"""
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


class TokenCounter:
    """モデルのトークナイザ（あれば）でトークン数を数える。無ければ概算する。"""

    def __init__(self, model: str) -> None:
        self._model = model
        self._tok = None
        self._tried = False
        self._cache: dict[str, int] = {}

    def _load(self) -> None:
        if self._tried:
            return
        self._tried = True
        try:
            from transformers import AutoTokenizer  # type: ignore

            # local_files_only=True: 取得済み（キャッシュ）のみ参照しネットワークは使わない
            self._tok = AutoTokenizer.from_pretrained(
                self._model, local_files_only=True
            )
        except Exception:  # noqa: BLE001 - 失敗時は概算にフォールバック
            self._tok = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        self._load()
        if self._tok is not None:
            try:
                n = len(self._tok.encode(text))
            except Exception:  # noqa: BLE001 - 以降は概算に切り替える
                self._tok = None
                n = _estimate(text)
        else:
            n = _estimate(text)
        if len(self._cache) < 4000:
            self._cache[text] = n
        return n

    @property
    def exact(self) -> bool:
        """実トークナイザで数えているか（概算でないか）。"""
        self._load()
        return self._tok is not None

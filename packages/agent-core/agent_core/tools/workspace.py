from __future__ import annotations

import datetime
import os
import re
from pathlib import Path


def session_timestamp() -> str:
    """セッションフォルダ名に使う日時（例: 20260527-143052）。"""
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def slugify(text: str, max_len: int = 40) -> str:
    """タスク文をフォルダ名に使える短いスラッグへ変換する（不可なら空文字）。"""
    s = text.strip().splitlines()[0] if text.strip() else ""
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", s)  # ファイル名に使えない文字を除去
    s = re.sub(r"\s+", "-", s).strip("-.")
    return s[:max_len].strip("-.")


def _reserve_dir(base: Path, name: str) -> Path:
    """base 配下に name でディレクトリを作って返す。

    同名が既にあれば name-2, name-3 ... と衝突しない名前にする。mkdir は
    アトミックなので、同じ秒に複数のエージェントが起動しても混ざらない。
    """
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidate = base / (name if n == 1 else f"{name}-{n}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            n += 1


def _free_name(base: Path, name: str) -> str:
    """base 配下で未使用のフォルダ名を返す（衝突時は name-2, name-3 ...）。"""
    n = 1
    while True:
        candidate = name if n == 1 else f"{name}-{n}"
        if not (base / candidate).exists():
            return candidate
        n += 1


class Workspace:
    """ツール群が参照する作業ディレクトリ。

    起動時は日時のみのフォルダで作成し、最初のタスクが分かった時点で
    name_from_task() して名前を補完する。root を更新するだけなので、参照して
    いるツール（resolver / run_command の cwd）は組み直さなくてよい。
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    @classmethod
    def new_session(cls, base: str | Path) -> "Workspace":
        """base/<日時>/ を衝突しないよう作成して返す（セッションごとの分離用）。

        同じ秒に複数起動しても <日時>-2, <日時>-3 ... と別フォルダになる。
        """
        return cls(_reserve_dir(Path(base), session_timestamp()))

    def renew(self, base: str | Path | None = None) -> Path:
        """新しいセッションフォルダを作って root を切り替える（チャットのリセット用）。

        base 省略時は現在のフォルダと同じ親（既定の workspace/）の下に作る。
        root を差し替えるだけなので、参照しているツールは組み直さなくてよい。
        """
        parent = Path(base) if base is not None else self.root.parent
        self.root = _reserve_dir(parent, session_timestamp())
        return self.root

    def ensure(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def rename(self, new_name: str) -> Path:
        """同じ親ディレクトリの下で、末尾のフォルダ名を new_name に変更する。"""
        target = (self.root.parent / new_name).resolve()
        if target != self.root:
            os.rename(self.root, target)
            self.root = target
        return self.root

    def name_from_task(self, text: str, title: str | None = None) -> bool:
        """最初のタスクからフォルダ名を補完する。

        title（LLM 等で作った短いタイトル）があればそれを優先し、無効なら text
        から作る。改名したら True、スラッグが空で改名しなかったら False を返す。
        """
        slug = slugify(title) if title else ""
        if not slug:
            slug = slugify(text)
        if not slug:
            return False
        self.rename(_free_name(self.root.parent, f"{self.root.name}_{slug}"))
        return True

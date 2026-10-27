from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .base import Tool
from .workspace import Workspace

FILESYSTEM_TOOL_NAMES = ("read_file", "write_file", "edit_file", "list_dir", "grep", "glob")

# Python フォールバック検索で枝刈りするディレクトリ（rg が無い環境向け）。
# rg があるときは .gitignore を尊重するのでこの集合は使わない。
_IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "dist", "build", ".next", ".cache", ".idea", ".gradle", "target",
}


def _walk_files(base: Path, follow_links: bool = False, only: frozenset[str] | None = None):
    """base 配下のファイルを列挙する（無視ディレクトリ・隠しは枝刈り）。

    only を渡すと base 直下は only の名前だけを歩く（読むだけの場所で見せる名前を絞る）。
    """
    if base.is_file():
        yield base
        return
    for root, dirs, names in os.walk(base, followlinks=follow_links):
        at_top = Path(root) == base
        dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS and not d.startswith(".")
                   and (only is None or not at_top or d in only)]
        for n in names:
            if n.startswith("."):
                continue
            if only is not None and at_top and n not in only:
                continue
            yield Path(root) / n


def _grep_with_rg(
    rg: str, pattern: str, rel_base: str, glob: str | None,
    ignore_case: bool, context: int, max_results: int, root: Path,
) -> str:
    """ripgrep で検索する（.gitignore を尊重し高速・大規模リポジトリ向け）。"""
    cmd = [rg, "--line-number", "--no-heading", "--color=never"]
    if ignore_case:
        cmd.append("--ignore-case")
    if context > 0:
        cmd += ["--context", str(context)]
    if glob:
        cmd += ["--glob", glob]
    cmd += ["--", pattern, rel_base]
    try:
        proc = subprocess.run(
            cmd, cwd=str(root), capture_output=True, text=True, timeout=30
        )
    except subprocess.TimeoutExpired:
        return "Error: 検索がタイムアウトしました（30s）"
    if proc.returncode not in (0, 1):  # 1 = 一致なし、>1 = エラー
        err = proc.stderr.strip()
        return f"Error: rg 失敗: {err}" if err else "(一致なし)"
    lines = [ln[:300] for ln in proc.stdout.splitlines()]
    if not lines:
        return "(一致なし)"
    if len(lines) > max_results:
        lines = lines[:max_results] + [f"…（打ち切り: {max_results} 行）"]
    return "\n".join(lines)


def _grep_python(
    base: Path, pattern: str, glob: str | None,
    ignore_case: bool, context: int, max_results: int, root: Path,
    prefix: str | None = None, only: frozenset[str] | None = None,
) -> list[str]:
    """純Python検索（無視ディレクトリは枝刈り）。一致行のリストを返す（上限で打ち切り）。

    prefix を渡すと読むだけの場所の検索: 結果は `<prefix>/<root からの相対>` で示し、リンクもたどる。
    only は base 直下で歩く名前の絞り込み（読むだけの場所の見せる名前）。
    """
    regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    hits: list[str] = []
    for f in _walk_files(base, follow_links=prefix is not None, only=only):
        if not f.is_file():
            continue
        rel = f.relative_to(root)
        shown = f"{prefix}/{rel.as_posix()}" if prefix else rel.as_posix()
        if glob and not (
            fnmatch.fnmatch(f.name, glob) or fnmatch.fnmatch(rel.as_posix(), glob)
        ):
            continue
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue  # バイナリ・読めないものは飛ばす
        for i, line in enumerate(lines):
            if not regex.search(line):
                continue
            if context > 0:
                lo, hi = max(0, i - context), min(len(lines), i + context + 1)
                for j in range(lo, hi):
                    sep = ":" if j == i else "-"
                    hits.append(f"{shown}:{j + 1}{sep} {lines[j].strip()[:200]}")
            else:
                hits.append(f"{shown}:{i + 1}: {line.strip()[:200]}")
            if len(hits) >= max_results:
                return hits
    return hits


def _glob_regex(pattern: str) -> re.Pattern:
    """グロブを正規表現へ（`**/` は 0 個以上のフォルダ、`*` と `?` は `/` をまたがない）。

    読むだけの場所を作業フォルダの中の `apps/…` として照合するのに使う（Path.glob は実在する
    フォルダしか歩けないため）。Python 3.11 でも同じ規則で動くよう自前で変換する。
    """
    i, out = 0, []
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[" and "]" in pattern[i + 1:]:
            j = pattern.index("]", i + 1)
            body = pattern[i + 1:j]
            out.append("[" + ("^" + body[1:] if body.startswith("!") else body) + "]")
            i = j + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


@dataclass(frozen=True)
class Mount:
    """読むだけの場所。path の中身が作業フォルダの中の `<名前>/` として見える。

    only が None でなければ、直下は only の名前だけが「存在する」。それ以外の直下の名前は
    一覧に出ず、読む・探すも隠しファイルと同じ扱いで差し戻す（見せるアプリを絞る用途）。
    """

    path: Path
    only: frozenset[str] | None = None

    def allows(self, child: str) -> bool:
        return self.only is None or child in self.only


#: read_roots の 1 要素。文字列/Path は「フォルダ名で見せる」。タプルは (名前, 場所[, 見せる直下の名前])。
MountSpec = "str | Path | tuple[str, str | Path] | tuple[str, str | Path, Sequence[str] | None]"


def _mounts_of(read_roots) -> dict[str, Mount]:
    """読むだけの場所を「作業フォルダの中の名前 → Mount」にする。

    - 文字列・Path: 名前はフォルダ名。例: read_roots=[".../agents/x/apps"] なら `apps/` として見える
    - タプル (名前, 場所): 名前を明示する
    - タプル (名前, 場所, only): さらに直下で見せる名前を only に絞る（None なら全部）
    """
    mounts: dict[str, Mount] = {}
    for r in read_roots or []:
        if isinstance(r, (tuple, list)):
            if len(r) not in (2, 3) or not isinstance(r[0], str):
                raise ValueError(f"read_roots のタプルは (名前, 場所[, 見せる名前の列]) です: {r!r}")
            name = r[0]
            path = Path(os.path.abspath(os.path.expanduser(str(r[1]))))
            only = r[2] if len(r) == 3 else None
            only_set = None if only is None else frozenset(str(x) for x in only)
        else:
            path = Path(os.path.abspath(os.path.expanduser(str(r))))
            name = path.name
            only_set = None
        if not name or name in mounts or name.startswith(".") or "/" in name or "\\" in name:
            raise ValueError(f"read_roots の名前（フォルダ名）が空・重複・隠しになっている: {r!r}")
        mounts[name] = Mount(path, only_set)
    return mounts


def _hidden(parts) -> bool:
    """隠し（. 始まり）や生成物のフォルダ（.venv・node_modules・build など）を含むか。"""
    return any(p.startswith(".") or p in _IGNORE_DIRS for p in parts)


def _make_resolver(ws: Workspace, mounts: dict[str, Mount] | None = None):
    """ws.root 配下に閉じ込めたパス解決関数を作る（workspace 外へのアクセスを拒否）。

    root は実行時に ws.root を読むので、セッションフォルダの改名にも追従する。

    mounts は読むだけの場所（名前 → Mount）。`apps/x.py` のように名前で始まる相対パスは
    その場所の中を指し、読む道具（read=True）だけが通る。書く・直すは「読むだけ」で断る。
    判定は字面で行う（`..` は先に畳む）。場所の中のリンク（視界のアプリへのリンク）はたどってよい
    — 場所を決めるのは設定を書いた人で、読むだけなので外へ書き出す経路にはならない。
    隠しファイルと生成物のフォルダ、そして only で見せないと決めた直下の名前は読ませない。
    """
    mounts = mounts or {}

    def resolve(path: str, read: bool = False) -> Path:
        root = ws.root
        candidate = Path(path)
        if candidate.is_absolute():
            raise ValueError(
                f"絶対パスは使えません。workspace 相対で指定してください: {path}"
            )
        parts = Path(os.path.normpath(path)).parts
        if parts and parts[0] in mounts:
            if not read:
                raise ValueError(
                    f"{parts[0]}/ は読むだけの場所です。書く・直すは workspace の中で行ってください: {path}"
                )
            if _hidden(parts[1:]):
                raise ValueError(
                    f"隠しファイルと生成物のフォルダ（.git・.venv・node_modules・build など）は読めません: {path}"
                )
            mount = mounts[parts[0]]
            if len(parts) > 1 and not mount.allows(parts[1]):
                raise ValueError(f"path not found: {path}")
            return mount.path.joinpath(*parts[1:])
        target = (root / candidate).resolve()
        if target != root and root not in target.parents:
            raise ValueError(
                f"workspace ({root}) の外へのアクセスは禁止されています: {path}"
            )
        return target

    return resolve


def build_filesystem_tools(
    ws: Workspace, read_roots: list | None = None
) -> list[Tool]:
    """ws（workspace）に閉じ込めたファイル/検索ツールを返す。

    read_roots を渡すと、その場所が作業フォルダの中の `<名前>/` として読むだけで見える
    （例: `apps/cad-tool/src/x.py`）。要素は文字列/Path（名前はフォルダ名）か、
    (名前, 場所[, 見せる直下の名前の列]) のタプル（MountSpec）。一覧・検索・グロブも
    作業フォルダ全体を見るときはそこを含める。道具も指し方も増えない（読む・探すの届く範囲が
    広がるだけ）。書く・直すは workspace の中だけ。
    """
    mounts = _mounts_of(read_roots)
    resolve = _make_resolver(ws, mounts)

    def mount_of(p: Path) -> tuple[str, Mount] | None:
        for name, m in mounts.items():
            if p == m.path or m.path in p.parents:
                return name, m
        return None

    def read_file(path: str, offset: int = 0, limit: int | None = None) -> str:
        p = resolve(path, read=True)
        if not p.exists():
            return f"Error: file not found: {path}"
        if p.is_dir():
            return f"Error: {path} is a directory"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        if offset or limit is not None:
            end = offset + limit if limit is not None else len(lines)
            lines = lines[offset:end]
        numbered = "\n".join(f"{offset + i + 1}\t{line}" for i, line in enumerate(lines))
        # 長すぎる場合の上限・切り詰めは中央（Agent._execute）で一律に行う。先頭だけ残す
        # 旧方式と違い、中央は先頭と末尾の両方を残す（offset/limit で続きも読める）。
        return numbered or "(empty file)"

    def write_file(path: str, content: str) -> str:
        p = resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"

    def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
        p = resolve(path)
        if not p.exists():
            return f"Error: file not found: {path}"
        text = p.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            return f"Error: old string not found in {path}"
        if count > 1 and not replace_all:
            return (
                f"Error: old string is not unique ({count} matches). "
                "Add more surrounding context, or set replace_all=true."
            )
        text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        p.write_text(text, encoding="utf-8")
        return f"Edited {path} ({count if replace_all else 1} replacement(s))"

    def list_dir(path: str = ".") -> str:
        p = resolve(path, read=True)
        if not p.exists():
            return f"Error: path not found: {path}"
        if p.is_file():
            return path
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        area = mount_of(p)
        if area is not None:   # 読むだけの場所では隠し・生成物を見せない
            entries = [e for e in entries if not _hidden([e.name])]
            if p == area[1].path:   # 場所の直下は、見せると決めた名前だけ
                entries = [e for e in entries if area[1].allows(e.name)]
        listing = [f"{e.name}/" if e.is_dir() else e.name for e in entries]
        if p == ws.root:   # 作業フォルダの直下には、読むだけの場所（apps/ など）も並べる
            listing = [f"{name}/" for name in mounts if f"{name}/" not in listing] + listing
        return "\n".join(listing) or "(empty directory)"

    def grep(
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        ignore_case: bool = False,
        context: int = 0,
        max_results: int = 100,
    ) -> str:
        base = resolve(path, read=True)
        if not base.exists():
            return f"Error: path not found: {path}"
        try:
            re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regex: {exc}"
        ctx = max(0, int(context or 0))
        limit = max(1, int(max_results or 100))

        def text(hits: list[str]) -> str:
            if not hits:
                return "(一致なし)"
            if len(hits) >= limit:
                return "\n".join(hits[:limit] + [f"…（打ち切り: {limit} 件）"])
            return "\n".join(hits)

        area = mount_of(base)
        if area is not None:   # 読むだけの場所の中（apps/… と示す。リンクもたどる）
            name, m = area
            only = m.only if base == m.path else None
            return text(_grep_python(base, pattern, glob, ignore_case, ctx, limit, m.path,
                                     prefix=name, only=only))
        root = ws.root
        rel_base = "." if base == root else base.relative_to(root).as_posix()
        # ripgrep があれば優先（高速・.gitignore 尊重）。無ければ純Pythonで代替。
        rg = shutil.which("rg")
        out = (_grep_with_rg(rg, pattern, rel_base, glob, ignore_case, ctx, limit, root) if rg
               else text(_grep_python(base, pattern, glob, ignore_case, ctx, limit, root)))
        if base != root or not mounts or out.startswith("Error") or "…（打ち切り" in out:
            return out
        # 作業フォルダ全体を探すときは、読むだけの場所（apps/ など）も探す
        hits = [] if out == "(一致なし)" else out.splitlines()
        for name, m in mounts.items():
            if len(hits) >= limit:
                break
            hits += _grep_python(m.path, pattern, glob, ignore_case, ctx, limit - len(hits),
                                 m.path, prefix=name, only=m.only)
        return text(hits)

    def glob(pattern: str, max_results: int = 200) -> str:
        if pattern.startswith("/") or ".." in pattern:
            return "Error: パターンに絶対パスや '..' は使えません"
        root = ws.root
        found = [str(m.relative_to(root)) + ("/" if m.is_dir() else "")
                 for m in sorted(root.glob(pattern))]
        # 読むだけの場所（apps/ など）も作業フォルダの中として照合する。リンクもたどり、隠し・生成物は除く
        regex = _glob_regex(pattern)
        head = pattern.split("/", 1)[0]
        for name, m in mounts.items():
            if len(found) >= max_results:
                break
            if not any(c in head for c in "*?[") and head != name:
                continue   # 先頭が別の名前（例 src/*.py）なら、この場所は歩かない
            if regex.match(name):
                found.append(f"{name}/")
            for dirpath, dirs, files in os.walk(m.path, followlinks=True):
                at_top = Path(dirpath) == m.path
                dirs[:] = sorted(d for d in dirs if not _hidden([d]) and (not at_top or m.allows(d)))
                if at_top:
                    files = [f for f in files if m.allows(f)]
                rel_dir = Path(dirpath).relative_to(m.path).as_posix()
                prefix = name if rel_dir == "." else f"{name}/{rel_dir}"
                found += [f"{prefix}/{d}/" for d in dirs if regex.match(f"{prefix}/{d}")]
                found += [f"{prefix}/{f}" for f in sorted(files)
                          if not f.startswith(".") and regex.match(f"{prefix}/{f}")]
                if len(found) >= max_results:
                    break
        return "\n".join(found[:max_results]) or "(一致なし)"

    return [
        Tool(
            name="read_file",
            description="ファイルの内容を行番号付きで読む。offset/limit で範囲指定可能。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "読むファイルのパス"},
                    "offset": {"type": "integer", "description": "開始行（0始まり）"},
                    "limit": {"type": "integer", "description": "読む行数"},
                },
                "required": ["path"],
            },
            func=read_file,
        ),
        Tool(
            name="write_file",
            description=(
                "ファイルを新規作成または完全に上書きする。親ディレクトリは自動作成。"
                "既存ファイルの一部を修正するときはこれを使わず edit_file を使うこと"
                "（全文再生成は遅く、変更不要な箇所まで書き換えてしまう）。"
                "write_file は新規作成か全面刷新のときだけ使う。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "書き込むファイルのパス"},
                    "content": {"type": "string", "description": "ファイルの全内容"},
                },
                "required": ["path", "content"],
            },
            func=write_file,
        ),
        Tool(
            name="edit_file",
            description=(
                "既存ファイルの一部を修正するための推奨ツール。ファイル内の old を new に"
                "置換する（ファイル全体は再生成しない）。old は変更箇所を一意に特定できる"
                "最小限の文脈だけを含め、変更しない行まで old/new に入れないこと。"
                "old が一意でないときは周辺の文脈を少し足すか replace_all=true を使う。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "編集するファイルのパス"},
                    "old": {"type": "string", "description": "置換対象の文字列"},
                    "new": {"type": "string", "description": "置換後の文字列"},
                    "replace_all": {"type": "boolean", "description": "全一致を置換するか"},
                },
                "required": ["path", "old", "new"],
            },
            func=edit_file,
        ),
        Tool(
            name="list_dir",
            description="ディレクトリの中身を一覧する。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "一覧するディレクトリ（既定: workspace 直下）"},
                },
            },
            func=list_dir,
        ),
        Tool(
            name="grep",
            description=(
                "正規表現でファイル内を再帰検索し、一致行を path:行番号 付きで返す。"
                "ripgrep があれば高速・.gitignore 尊重で検索する（無ければ純Pythonで代替）。"
                "大規模コードベースの検索に向く。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "検索する正規表現"},
                    "path": {"type": "string", "description": "検索対象（既定: workspace 直下）"},
                    "glob": {
                        "type": "string",
                        "description": "対象ファイルを絞るグロブ（例: '*.py'、'**/*.ts'）",
                    },
                    "ignore_case": {"type": "boolean", "description": "大文字小文字を区別しない"},
                    "context": {
                        "type": "integer",
                        "description": "一致行の前後に表示する文脈行数（既定0）",
                    },
                    "max_results": {"type": "integer", "description": "最大出力行数（既定100）"},
                },
                "required": ["pattern"],
            },
            func=grep,
        ),
        Tool(
            name="glob",
            description="グロブパターンでファイル/ディレクトリを探す（例: '**/*.py'）。",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "グロブパターン"},
                    "max_results": {"type": "integer", "description": "最大件数（既定200）"},
                },
                "required": ["pattern"],
            },
            func=glob,
        ),
    ]

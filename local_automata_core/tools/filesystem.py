from __future__ import annotations

import fnmatch
import glob as _glob
import os
import re
import shutil
import subprocess
from pathlib import Path

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


def _walk_files(base: Path, follow_links: bool = False):
    """base 配下のファイルを列挙する（無視ディレクトリ・隠しは枝刈り）。"""
    if base.is_file():
        yield base
        return
    for root, dirs, names in os.walk(base, followlinks=follow_links):
        dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS and not d.startswith(".")]
        for n in names:
            if not n.startswith("."):
                yield Path(root) / n


def _grep_with_rg(
    rg: str, pattern: str, rel_base: str, glob: str | None,
    ignore_case: bool, context: int, max_results: int, root: Path,
    follow_links: bool = False,
) -> str:
    """ripgrep で検索する（.gitignore を尊重し高速・大規模リポジトリ向け）。"""
    cmd = [rg, "--line-number", "--no-heading", "--color=never"]
    if follow_links:
        cmd.append("--follow")
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
    absolute: bool = False,
) -> str:
    """rg が無い環境向けの純Python検索（無視ディレクトリは枝刈り）。

    absolute=True（読むだけの場所の検索）は結果を絶対パスで示し、リンクもたどる。
    """
    regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    hits: list[str] = []
    for f in _walk_files(base, follow_links=absolute):
        if not f.is_file():
            continue
        rel = f.relative_to(root)
        shown = f if absolute else rel
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
                hits.append(f"…（打ち切り: {max_results} 件）")
                return "\n".join(hits)
    return "\n".join(hits) or "(一致なし)"


def _hidden_below(root: Path, target: Path) -> bool:
    """root から target までに、隠し（. 始まり）や無視ディレクトリ（.venv・node_modules など）があるか。"""
    return any(p.startswith(".") or p in _IGNORE_DIRS for p in target.relative_to(root).parts)


def _read_root_of(read_roots: list[Path], target: Path) -> Path | None:
    """target を含む読むだけの場所（無ければ None）。"""
    for r in read_roots:
        if target == r or r in target.parents:
            return r
    return None


def _make_resolver(ws: Workspace, read_roots: list[Path] | None = None):
    """ws.root 配下に閉じ込めたパス解決関数を作る（workspace 外へのアクセスを拒否）。

    root は実行時に ws.root を読むので、セッションフォルダの改名にも追従する。

    read_roots は「読むだけ」なら指してよい workspace 外の場所（絶対パス）。読む道具
    （read=True）だけが、その下を絶対パスで指せる。書く・直す道具は従来どおり workspace の中だけ。
    判定は字面で行う（`..` は先に畳む）。場所の中のリンク（視界のアプリへのリンクなど）は
    たどってよい — 場所を決めるのは設定を書いた人で、読むだけなので外へ書き出す経路にはならない。
    隠しファイル・無視ディレクトリ（.git・.venv・node_modules・build など）は読ませない。
    """
    roots = list(read_roots or [])

    def resolve(path: str, read: bool = False) -> Path:
        root = ws.root
        candidate = Path(path)
        if candidate.is_absolute():
            target = Path(os.path.normpath(candidate))
            area = _read_root_of(roots, target)
            if area is not None and not read:
                raise ValueError(
                    f"{area} は読むだけの場所です。書く・直すは workspace の中で行ってください: {path}"
                )
            if area is not None:
                if _hidden_below(area, target):
                    raise ValueError(
                        f"隠しファイルと生成物のフォルダ（.git・.venv・node_modules・build など）は読めません: {path}"
                    )
                return target
            hint = f"（読むだけなら {', '.join(map(str, roots))} の下も絶対パスで指せる）" if roots else ""
            raise ValueError(
                f"絶対パスは使えません。workspace 相対で指定してください{hint}: {path}"
            )
        target = (root / candidate).resolve()
        if target != root and root not in target.parents:
            raise ValueError(
                f"workspace ({root}) の外へのアクセスは禁止されています: {path}"
            )
        return target

    return resolve


def build_filesystem_tools(
    ws: Workspace, read_roots: list[str | Path] | None = None
) -> list[Tool]:
    """ws（workspace）に閉じ込めたファイル/検索ツールを返す。

    read_roots を渡すと、読む道具（read_file・list_dir・grep・glob）だけがその下も
    絶対パスで指せる（アプリのコードや文書を調べる用途。書く・直すは workspace の中だけ）。
    """
    roots = [Path(os.path.abspath(os.path.expanduser(str(r)))) for r in (read_roots or [])]
    resolve = _make_resolver(ws, roots)
    # 読める場所の案内は、場所があるときだけ説明に足す（無ければ説明は従来と同じ）
    note = (f" 作業フォルダの外でも、読むだけなら次の場所の下を絶対パスで指せる: {', '.join(map(str, roots))}"
            if roots else "")

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
        if _read_root_of(roots, p) is not None:   # 読むだけの場所では隠し・生成物を見せない
            entries = [e for e in entries if not e.name.startswith(".") and e.name not in _IGNORE_DIRS]
        listing = "\n".join(f"{e.name}/" if e.is_dir() else e.name for e in entries)
        return listing or "(empty directory)"

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
        area = _read_root_of(roots, base)
        rg = shutil.which("rg")
        if area is not None:
            # 読むだけの場所: 結果は絶対パスで示し(そのまま read_file に渡せる)、アプリへのリンクもたどる
            if rg:
                return _grep_with_rg(rg, pattern, str(base), glob, ignore_case, ctx, limit, area,
                                     follow_links=True)
            return _grep_python(base, pattern, glob, ignore_case, ctx, limit, area, absolute=True)
        root = ws.root
        rel_base = "." if base == root else base.relative_to(root).as_posix()
        # ripgrep があれば優先（高速・.gitignore 尊重）。無ければ純Pythonで代替。
        if rg:
            return _grep_with_rg(rg, pattern, rel_base, glob, ignore_case, ctx, limit, root)
        return _grep_python(base, pattern, glob, ignore_case, ctx, limit, root)

    def glob(pattern: str, max_results: int = 200) -> str:
        if pattern.startswith("/"):
            # 読むだけの場所の下のパターン。アプリへのリンクもたどり、隠し・生成物は除く
            area = _read_root_of(roots, Path(os.path.normpath(pattern.split("*")[0] or "/")))
            if area is None or ".." in pattern:
                hint = f"（読むだけなら {', '.join(map(str, roots))} の下は指せる）" if roots else ""
                return f"Error: パターンに絶対パスや '..' は使えません{hint}"
            found = [Path(m) for m in sorted(_glob.glob(pattern, recursive=True))]
            found = [m for m in found if _read_root_of(roots, m) is area and not _hidden_below(area, m)]
            shown = [str(m) + ("/" if m.is_dir() else "") for m in found[:max_results]]
            return "\n".join(shown) or "(一致なし)"
        if ".." in pattern:
            return "Error: パターンに絶対パスや '..' は使えません"
        root = ws.root
        matches = sorted(root.glob(pattern))
        rels = [
            str(m.relative_to(root)) + ("/" if m.is_dir() else "")
            for m in matches[:max_results]
        ]
        return "\n".join(rels) or "(一致なし)"

    return [
        Tool(
            name="read_file",
            description="ファイルの内容を行番号付きで読む。offset/limit で範囲指定可能。" + note,
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
            description="ディレクトリの中身を一覧する。" + note,
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
            ) + note,
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
            description="グロブパターンでファイル/ディレクトリを探す（例: '**/*.py'）。" + note,
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

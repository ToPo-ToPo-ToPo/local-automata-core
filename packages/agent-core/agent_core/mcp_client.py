"""agent.toml の MCP サーバー設定からツールを取り込むためのブリッジ。

MCP(Model Context Protocol)サーバーへ接続し、提供されるツールを local-automata の
Tool として登録できるようにする。MCP SDK は非同期なので、バックグラウンドの
イベントループで接続を保持し、同期の func から呼び出す。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import tomllib
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import AsyncExitStack
from typing import Any, Callable

from .constants import project_cache_dir
from .tools.base import Tool

Logger = Callable[[str, str], None]

# AIOS のようなフォルダを走査して MCP サーバーを見つけるときの規約。
# フォルダ内のエントリスクリプト候補（マニフェストが無いときの自動起動対象）。
_ENTRY_SCRIPTS = ("server.py", "main.py", "__main__.py")
# 走査対象外のディレクトリ名（補助ファイル置き場・キャッシュなど）。
_SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules"}

# ツール結果に画像が含まれるとき、一時ファイルに保存してこのマーカーで参照を渡す。
# agent 側がこのマーカーを画像メッセージ(image_url)に変換してモデルへ渡す。
IMAGE_MARKER = "[[MCP_IMAGE:{path}]]"
_IMAGE_MARKER_RE = re.compile(r"\[\[MCP_IMAGE:(.+?)\]\]")


def _noop(event: str, detail: str) -> None:  # pragma: no cover
    pass


def resolve_call_timeout(raw: float | None) -> float | None:
    """runtime の mcp_call_timeout を実効値へ変換する。

    0 / 負 / None は「無制限」を表す None に、正の数はその秒数になる。
    呼び出し側の既定（例: 120 秒）は raw に渡してから解決する。
    """
    if raw is None or raw <= 0:
        return None
    return raw


def _save_image_block(block: Any) -> str | None:
    """画像ブロック(ImageContent)を一時ファイルに保存しパスを返す。"""
    data = getattr(block, "data", None)
    if not data:
        return None
    mime = getattr(block, "mimeType", "") or "image/png"
    ext = ".jpg" if "jpeg" in mime or "jpg" in mime else ".png"
    # 一時ファイルもプロジェクト内（./.local-automata/tmp/）に置く（/tmp 等の外部に出さない）。
    tmp_dir = os.path.join(project_cache_dir(), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="mcp_img_", suffix=ext, dir=tmp_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(base64.b64decode(data))
    except Exception:
        return None
    return path


def _result_to_text(result: Any) -> str:
    """CallToolResult をテキスト化する。画像ブロックは一時ファイルに保存し、
    IMAGE_MARKER で参照を埋め込む(agent 側が画像メッセージへ変換する)。"""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
            continue
        if getattr(block, "type", None) == "image":
            path = _save_image_block(block)
            if path:
                parts.append(IMAGE_MARKER.format(path=path))
                continue
        parts.append(str(block))
    out = "\n".join(parts)
    if getattr(result, "isError", False):
        return f"Error: {out}".strip()
    return out or "(出力なし)"


def split_image_markers(text: str) -> tuple[str, list[str]]:
    """ツール結果テキストから画像マーカーを抜き出し、(表示用テキスト, 画像パス列) を返す。"""
    paths = _IMAGE_MARKER_RE.findall(text or "")
    if not paths:
        return text, []
    cleaned = _IMAGE_MARKER_RE.sub("(画像を添付)", text).strip()
    return cleaned, paths


def _safe_name(server: str, tool: str) -> str:
    """function calling で許容される名前に整える（server_tool 形式）。"""
    name = f"{server}_{tool}"
    name = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    return name[:64]


# --- ディレクトリ走査によるサーバーディスカバリ（AIOS フォルダ方式） ---------


def _desc_from_script(path: str) -> str | None:
    """スクリプト先頭から説明を拾う。`# description:` 行か module docstring の1行目。"""
    try:
        with open(path, encoding="utf-8") as f:
            head = f.read(2048)
    except OSError:
        return None
    m = re.search(r"^#\s*description:\s*(.+)$", head, re.MULTILINE | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r'^\s*[rubf]*"""(.*?)"""', head, re.DOTALL)
    if m:
        for line in m.group(1).strip().splitlines():
            if line.strip():
                return line.strip()
    return None


def _server_from_manifest(name: str, folder: str, manifest: str) -> tuple[str, dict] | None:
    """フォルダ内の mcp.toml からサーバー定義を作る。command か url が必須。"""
    try:
        with open(manifest, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    if not isinstance(data, dict) or (not data.get("command") and not data.get("url")):
        return None
    cfg: dict = {}
    for key in ("command", "args", "env", "url", "description", "timeout"):
        if key in data:
            cfg[key] = data[key]
    # cwd は既定でフォルダ。相対指定はフォルダ基準で解決する。
    cwd = data.get("cwd")
    if cwd:
        cfg["cwd"] = cwd if os.path.isabs(cwd) else os.path.join(folder, cwd)
    elif not cfg.get("url"):
        cfg["cwd"] = folder
    cfg.setdefault("description", f"(AIOS: {name})")
    return name, cfg


def _server_from_script(name: str, cwd: str, script: str) -> tuple[str, dict]:
    """単体スクリプトを現在の Python で起動するサーバー定義を作る。"""
    return name, {
        "command": sys.executable,
        "args": [script],
        "cwd": cwd,
        "description": _desc_from_script(script) or f"(AIOS: {name})",
    }


def _server_from_folder(name: str, folder: str) -> tuple[str, dict] | None:
    """フォルダ1個からサーバー定義を作る。mcp.toml を優先し、無ければエントリスクリプト。"""
    manifest = os.path.join(folder, "mcp.toml")
    if os.path.isfile(manifest):
        return _server_from_manifest(name, folder, manifest)
    for entry in (*_ENTRY_SCRIPTS, f"{name}.py"):
        path = os.path.join(folder, entry)
        if os.path.isfile(path):
            return _server_from_script(name, folder, path)
    return None


def _server_sig_hash(cfg: dict) -> str:
    """サーバー定義からスキーマキャッシュ用のキーを作る。

    起動方法（command/args/cwd か url）が変われば別キー。ローカルスクリプトは
    mtime も混ぜるので、コードを編集するとキャッシュが自然に無効化される。
    """
    if cfg.get("url"):
        sig = "url:" + str(cfg["url"])
    else:
        parts = [
            str(cfg.get("command", "")),
            repr(cfg.get("args", [])),
            str(cfg.get("cwd", "")),
        ]
        for a in cfg.get("args", []):
            if isinstance(a, str) and os.path.isfile(a):
                try:
                    parts.append(f"{a}:{os.path.getmtime(a)}")
                except OSError:
                    pass
        sig = "|".join(parts)
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()


def discover_servers(dirs: list[str]) -> dict[str, dict]:
    """ディレクトリ群を走査し、見つけた MCP サーバー定義（名前→cfg）を返す。

    各ディレクトリ直下について:
      - サブフォルダ `<name>/` … `mcp.toml` があればそれ、無ければ
        `server.py`/`main.py`/`<name>.py` を現在の Python で起動。
      - 単体ファイル `<name>.py` … 現在の Python で起動（_ 始まりは無視）。
    名前は basename。先に見つかった定義を優先する（重複名は無視）。
    """
    out: dict[str, dict] = {}
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for entry in sorted(os.listdir(d)):
            path = os.path.join(d, entry)
            if os.path.isdir(path):
                if entry in _SKIP_DIRS or entry.startswith("."):
                    continue
                found = _server_from_folder(entry, path)
            elif entry.endswith(".py") and not entry.startswith("_"):
                found = _server_from_script(entry[:-3], d, path)
            else:
                continue
            if found is not None and found[0] not in out:
                out[found[0]] = found[1]
    return out


class MCPManager:
    """MCP サーバー群への接続を管理し、ツールを Tool として提供する。

    servers の各エントリ:
        {"command": ..., "args": [...], "env": {...}, "cwd": ...}  # stdio
        {"url": "http://..."}                                      # Streamable HTTP
    任意で各サーバーに "timeout"（秒。0/負で無制限）を指定でき、全体既定の
    call_timeout を上書きする。

    call_timeout: ツール呼び出しの既定タイムアウト秒数。None で無制限に待つ。
    """

    def __init__(
        self,
        servers: dict[str, dict],
        call_timeout: float | None = 120.0,
        dirs: list[str] | None = None,
        cache_path: str | None = None,
    ) -> None:
        self._servers = servers or {}
        # AIOS フォルダ方式: ここを走査して見つけたサーバーもカタログに加える。
        # 走査は catalog()/activate() のたびに行うので、起動後に置いたツールも拾える。
        self._dirs = list(dirs or [])
        # auto モードで全ツールのスキーマを保存するキャッシュ（起動なしで一覧提示するため）。
        self._cache_path = cache_path
        self._call_timeout = call_timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stack: AsyncExitStack | None = None
        self._tools: list[Tool] = []
        self._log: Logger = _noop
        # 遅延起動（方式B）用: サーバー単位の接続を個別に管理する。
        self._active: set[str] = set()
        self._active_stacks: dict[str, AsyncExitStack] = {}
        self._active_tools: dict[str, list[Tool]] = {}

    def _ensure_loop(self) -> None:
        """バックグラウンドのイベントループ／スレッドを（未起動なら）立ち上げる。"""
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _all_servers(self) -> dict[str, dict]:
        """設定（agent.toml）＋ディレクトリ走査で見つかったサーバーを統合する。

        名前が衝突したら agent.toml 側を優先する。走査は毎回行うため、起動後に
        AIOS フォルダへ置いたツールも次の catalog()/activate() で見える。
        """
        merged = discover_servers(self._dirs) if self._dirs else {}
        merged.update(self._servers)
        return merged

    def start(self, log: Logger = _noop) -> None:
        """全サーバーへ即時接続する（従来動作）。"""
        if not self._servers and not self._dirs:
            return
        self._log = log
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(self._connect_all(log), self._loop)
        future.result()  # 接続とツール一覧取得まで待つ

    def start_lazy(self, log: Logger = _noop) -> None:
        """イベントループだけ起動し、接続は activate() 時まで遅延する（方式B）。"""
        if not self._servers and not self._dirs:
            return
        self._log = log
        self._ensure_loop()

    def tools(self) -> list[Tool]:
        return list(self._tools)

    def catalog(self) -> list[dict]:
        """利用可能サーバーの一覧（名前・説明・起動状態・トランスポート・由来）を返す。"""
        out: list[dict] = []
        for name, cfg in self._all_servers().items():
            out.append(
                {
                    "name": name,
                    "description": cfg.get("description") or "",
                    "active": name in self._active,
                    "transport": "http" if cfg.get("url") else "stdio",
                    "source": "config" if name in self._servers else "dir",
                }
            )
        return out

    def activate(self, name: str, log: Logger | None = None) -> list[Tool]:
        """サーバー1台に接続（stdio なら起動）し、その配下ツールを返す。

        既に起動済みなら取り込み済みツールを返す（冪等）。未知の名前は KeyError。
        接続失敗時は例外を送出する（呼び出し側が握ってメッセージ化する）。
        """
        log = log or self._log
        if name in self._active:
            return list(self._active_tools.get(name, []))
        servers = self._all_servers()
        if name not in servers:
            raise KeyError(name)
        self._ensure_loop()
        cfg = servers[name]
        stack = AsyncExitStack()

        async def _do() -> list[Tool]:
            try:
                return await self._connect_one(name, cfg, stack, log)
            except BaseException:
                await stack.aclose()
                raise

        tools = asyncio.run_coroutine_threadsafe(_do(), self._loop).result()
        self._active_stacks[name] = stack
        self._active.add(name)
        self._active_tools[name] = tools
        self._tools.extend(tools)
        return tools

    def deactivate(self, name: str) -> list[str] | None:
        """起動済みサーバーを停止し、取り除いたツール名のリストを返す。

        起動していなければ None。close() と違い1台だけ畳む。
        """
        stack = self._active_stacks.pop(name, None)
        if stack is None or self._loop is None:
            return None
        try:
            asyncio.run_coroutine_threadsafe(stack.aclose(), self._loop).result(
                timeout=10
            )
        except Exception:  # noqa: BLE001 - 切断失敗でも内部状態は片付ける
            pass
        self._active.discard(name)
        removed = self._active_tools.pop(name, [])
        removed_names = {t.name for t in removed}
        self._tools = [t for t in self._tools if t.name not in removed_names]
        return list(removed_names)

    # --- 方式A2: 全ツールを最初から提示し、プロセスは初回呼び出し時に起動する ----

    def harvest(
        self, name: str, cfg: dict, log: Logger = _noop, timeout: float = 30.0
    ) -> list[dict]:
        """サーバーへ一瞬だけ接続してツールのスキーマ（名前・説明・引数）を取得する。

        プロセスは取得後すぐ切断する。auto モードで「起動せずに一覧を提示」するため、
        ここで得たスキーマをキャッシュしてプロキシツールを作る。
        """
        self._ensure_loop()
        stack = AsyncExitStack()

        async def _do() -> list[dict]:
            try:
                tools = await self._connect_one(name, cfg, stack, log)
                return [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                    for t in tools
                ]
            finally:
                await stack.aclose()

        return asyncio.run_coroutine_threadsafe(_do(), self._loop).result(timeout=timeout)

    def lazy_tools(self, log: Logger = _noop, harvest_timeout: float = 30.0) -> list[Tool]:
        """全サーバーの全ツールを「プロキシツール」として返す（auto モード）。

        スキーマはキャッシュ優先で取得し（無ければ一度だけ接続して取得・保存）、
        プロセスは起動しない。プロキシは呼ばれた瞬間に該当サーバーを起動して転送する。
        """
        self._log = log
        cache = self._load_cache()
        dirty = False
        proxies: list[Tool] = []
        for name, cfg in self._all_servers().items():
            key = _server_sig_hash(cfg)
            entry = cache.get(key)
            if entry is None:
                try:
                    metas = self.harvest(name, cfg, log, harvest_timeout)
                except Exception as exc:  # noqa: BLE001 - 取得失敗でも他は続行
                    log("mcp", f"{name}: ツール一覧の取得に失敗しました（{exc}）")
                    continue
                cache[key] = {"name": name, "tools": metas}
                dirty = True
                log("mcp", f"{name}: {len(metas)} 個のツールをスキャン（キャッシュ保存）")
            else:
                metas = entry.get("tools", [])
            for m in metas:
                proxies.append(
                    self._make_proxy(
                        name,
                        m["name"],
                        m.get("description") or "",
                        m.get("parameters") or {"type": "object", "properties": {}},
                    )
                )
        if dirty:
            self._save_cache(cache)
        return proxies

    def _make_proxy(
        self, server: str, safe_name: str, description: str, parameters: dict
    ) -> Tool:
        def func(**kwargs: Any) -> str:
            return self._call_proxy(server, safe_name, kwargs)

        return Tool(
            name=safe_name,
            description=description,
            parameters=parameters,
            func=func,
        )

    def _bound_tool(self, server: str, safe_name: str) -> Tool:
        """サーバーを（未起動なら起動して）接続し、該当ツールの実体を返す。"""
        for tool in self.activate(server):
            if tool.name == safe_name:
                return tool
        raise RuntimeError(f"tool '{safe_name}' not found on server '{server}'")

    def _call_proxy(self, server: str, safe_name: str, kwargs: dict) -> str:
        """プロキシ呼び出し: 起動確認→転送。プロセス断は一度だけ再起動して再試行。"""
        try:
            return self._bound_tool(server, safe_name).func(**kwargs)
        except KeyboardInterrupt:
            raise
        except KeyError:
            return f"Error: MCP サーバー '{server}' が見つかりません。"
        except Exception:  # noqa: BLE001 - プロセス断の可能性。畳んで張り直す
            self.deactivate(server)
            try:
                return self._bound_tool(server, safe_name).func(**kwargs)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - 再試行も失敗ならメッセージ化
                return (
                    f"Error: MCP ツール '{safe_name}'（{server}）の実行に失敗しました（{exc}）。"
                )

    def _load_cache(self) -> dict:
        if not self._cache_path or not os.path.isfile(self._cache_path):
            return {}
        try:
            with open(self._cache_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_cache(self, cache: dict) -> None:
        if not self._cache_path:
            return
        try:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    async def _connect_one(
        self, name: str, cfg: dict, stack: AsyncExitStack, log: Logger
    ) -> list[Tool]:
        """サーバー1台へ接続し、Tool 列を返す（stack に後始末を積む）。"""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if cfg.get("url"):
            from mcp.client.streamable_http import streamable_http_client

            # Streamable HTTP は (read, write, get_session_id) を返す。
            # セッションID取得用の第3要素はここでは使わない。
            read, write, _ = await stack.enter_async_context(
                streamable_http_client(cfg["url"])
            )
        else:
            params = StdioServerParameters(
                command=cfg["command"],
                args=list(cfg.get("args", [])),
                env=cfg.get("env"),
                cwd=cfg.get("cwd"),
            )
            read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
        tools = [self._wrap(name, cfg, session, tool) for tool in listed.tools]
        log("mcp", f"{name}: {len(listed.tools)} 個のツールを取り込み")
        return tools

    async def _connect_all(self, log: Logger) -> None:
        self._stack = AsyncExitStack()
        for name, cfg in self._all_servers().items():
            try:
                tools = await self._connect_one(name, cfg, self._stack, log)
                self._tools.extend(tools)
                self._active.add(name)
                self._active_tools[name] = tools
            except Exception as exc:  # noqa: BLE001 - 1台の失敗で全体を止めない
                log("mcp", f"{name}: 接続に失敗しました（{exc}）")

    def _wrap(self, server: str, cfg: dict, session: Any, tool: Any) -> Tool:
        tool_name = tool.name
        # サーバー個別の timeout が全体既定を上書きする。0/負で無制限（None）。
        eff = cfg.get("timeout")
        eff = self._call_timeout if eff is None else (None if eff <= 0 else eff)

        def func(**kwargs: Any) -> str:
            holder: dict[str, Any] = {}

            async def _call() -> Any:
                async def on_progress(
                    progress: float, total: float | None, message: str | None
                ) -> None:
                    if message:
                        self._log("mcp", f"{tool_name}: {message}")

                # call_tool が使う request id を控える。読み取りから send_request が
                # この値を読むまで await を挟まないため、確実に一致する。
                holder["rid"] = getattr(session, "_request_id", None)
                # SDK 側の read_timeout は張らない。タイムアウトの権威は外側の
                # future.result(timeout=eff) に一本化する（両方張ると同値レースになり、
                # かつ SDK 側タイムアウト経路はサーバーへ cancelled を送らない）。
                return await session.call_tool(
                    tool_name,
                    kwargs,
                    progress_callback=on_progress,
                )

            future = asyncio.run_coroutine_threadsafe(_call(), self._loop)
            try:
                return _result_to_text(future.result(timeout=eff))
            except FuturesTimeoutError:
                self._cancel(session, holder.get("rid"))  # サーバーへ cancelled 通知
                future.cancel()
                return f"Error: MCP tool '{tool_name}' timed out after {eff}s"
            except KeyboardInterrupt:
                self._cancel(session, holder.get("rid"))
                future.cancel()
                raise

        return Tool(
            name=_safe_name(server, tool_name),
            description=tool.description or f"MCP tool {tool_name} ({server})",
            parameters=tool.inputSchema or {"type": "object", "properties": {}},
            func=func,
        )

    def _cancel(self, session: Any, rid: Any) -> None:
        """実行中のリクエストをサーバーへキャンセル通知する（notifications/cancelled）。

        mcp SDK はクライアント側 call_tool のキャンセルでは cancelled を自動送出しない
        （1.27 時点で実機確認）。重いサーバーのワーカーを解放させるため、call_tool が使う
        request id を指定して明示的に送る。rid 不明（疑似セッション等）なら何もしない。
        """
        if rid is None or self._loop is None:
            return
        from mcp.types import (
            CancelledNotification,
            CancelledNotificationParams,
            ClientNotification,
        )

        note = ClientNotification(
            CancelledNotification(
                method="notifications/cancelled",
                params=CancelledNotificationParams(
                    requestId=rid, reason="client cancelled"
                ),
            )
        )
        try:
            asyncio.run_coroutine_threadsafe(
                session.send_notification(note), self._loop
            ).result(timeout=5)
        except Exception:  # noqa: BLE001 - 通知失敗で呼び出し側を妨げない
            pass

    async def _drain(self) -> None:
        """ループ停止前に未完了タスク（キャンセル中の呼び出し等）を片付ける。"""
        current = asyncio.current_task()
        pending = [t for t in asyncio.all_tasks() if t is not current]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def close(self) -> None:
        if self._loop is None:
            return
        # 即時接続(start)の共有スタックと、遅延起動(activate)のサーバー別スタックを畳む。
        stacks = list(self._active_stacks.values())
        if self._stack is not None:
            stacks.append(self._stack)
        for stack in stacks:
            try:
                asyncio.run_coroutine_threadsafe(
                    stack.aclose(), self._loop
                ).result(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        try:
            asyncio.run_coroutine_threadsafe(self._drain(), self._loop).result(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None


def build_mcp_meta_tools(
    manager: "MCPManager",
    registry: Any,
    log: Logger = _noop,
) -> list[Tool]:
    """「必要なときに探して起動する」ためのメタツールを作る（方式B）。

    - list_mcp_servers : 設定済みサーバーの一覧・説明・起動状態を返す（アプリ一覧）。
    - activate_mcp_server : 指定サーバーを起動し、配下ツールを registry に動的登録する。
    - deactivate_mcp_server : 使い終わったサーバーを停止しツールを片付ける。

    registry はエージェントが毎ステップ参照する ToolRegistry。ここへ register/
    unregister することで、起動したツールが次ターンからモデルに見えるようになる。
    """

    def list_servers() -> str:
        catalog = manager.catalog()
        if not catalog:
            return "利用可能な MCP サーバーはありません。"
        lines = ["利用可能な MCP サーバー（必要なものを activate_mcp_server で起動）:"]
        for c in catalog:
            status = "起動中" if c["active"] else "停止中"
            desc = c["description"] or "(説明なし)"
            origin = "AIOS" if c.get("source") == "dir" else "設定"
            lines.append(
                f"- {c['name']} [{status}] ({c['transport']}/{origin}): {desc}"
            )
        return "\n".join(lines)

    def activate_server(server: str) -> str:
        try:
            tools = manager.activate(server, log)
        except KeyError:
            names = ", ".join(c["name"] for c in manager.catalog()) or "(なし)"
            return (
                f"Error: MCP サーバー '{server}' は設定にありません。"
                f" 利用可能: {names}"
            )
        except Exception as exc:  # noqa: BLE001 - 起動失敗はモデルへ返す
            return f"Error: MCP サーバー '{server}' の起動に失敗しました（{exc}）。"
        for tool in tools:
            registry.register(tool)
        if not tools:
            return f"'{server}' を起動しましたが、提供ツールはありませんでした。"
        listing = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        return (
            f"MCP サーバー '{server}' を起動しました。次のツールが使用可能です:\n{listing}"
        )

    def deactivate_server(server: str) -> str:
        removed = manager.deactivate(server)
        if removed is None:
            return f"'{server}' は起動していません。"
        for name in removed:
            unregister = getattr(registry, "unregister", None)
            if unregister is not None:
                unregister(name)
        return f"MCP サーバー '{server}' を停止し、{len(removed)} 個のツールを片付けました。"

    return [
        Tool(
            name="list_mcp_servers",
            description=(
                "利用可能な MCP サーバー（外部ツール群）の一覧・説明・起動状態を返す。"
                "必要な機能が今あるツールに見当たらないとき、まずこれを呼んで"
                "使えるサーバーを探す。"
            ),
            parameters={"type": "object", "properties": {}},
            func=lambda: list_servers(),
        ),
        Tool(
            name="activate_mcp_server",
            description=(
                "指定した MCP サーバーを起動し、その配下のツールを使用可能にする。"
                "list_mcp_servers で見つけたサーバーを、実際にツールを使う前に起動する。"
                "起動後、返り値に列挙されたツール名をそのまま呼び出せる。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "起動する MCP サーバー名（list_mcp_servers の名前）",
                    }
                },
                "required": ["server"],
            },
            func=lambda server: activate_server(server),
        ),
        Tool(
            name="deactivate_mcp_server",
            description=(
                "使い終わった MCP サーバーを停止し、その配下ツールを片付ける（任意）。"
                "リソースを解放したいときに使う。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "停止する MCP サーバー名",
                    }
                },
                "required": ["server"],
            },
            func=lambda server: deactivate_server(server),
        ),
    ]

"""agent.toml の MCP サーバー設定からツールを取り込むためのブリッジ。

MCP(Model Context Protocol)サーバーへ接続し、提供されるツールを利用側の
Tool として登録できるようにする。MCP SDK は非同期なので、バックグラウンドの
イベントループで接続を保持し、同期の func から呼び出す。
"""
from __future__ import annotations

import asyncio
import base64
import os
import re
import tempfile
import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import AsyncExitStack
from typing import Any, Callable

from .constants import project_cache_dir
from .tools.base import Tool

Logger = Callable[[str, str], None]

# ツールの「出力先ディレクトリ」を表す引数名。これらを持つツールには、呼び出し側が
# 値を渡していなければ MCPManager の workspace（利用側の作業ディレクトリ）を注入する。
_WORKSPACE_PARAMS = ("workspace", "output_dir", "out_dir", "save_dir")

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


def _purge_old_tmp_images(tmp_dir: str, max_age_sec: float = 24 * 3600.0) -> None:
    """tmp 内の古い mcp_img_* を消す（保持を有界にする）。

    画像は「作るだけ」で削除機構が無いと会話のたびに無限に溜まる。会話中の参照は
    直近だけなので、一定より古いものは安全に消せる（失敗は無視）。
    """
    import time
    cutoff = time.time() - max_age_sec
    try:
        for name in os.listdir(tmp_dir):
            if not name.startswith("mcp_img_"):
                continue
            p = os.path.join(tmp_dir, name)
            try:
                if os.path.getmtime(p) < cutoff:
                    os.unlink(p)
            except OSError:
                pass
    except OSError:
        pass


def _save_image_block(block: Any) -> str | None:
    """画像ブロック(ImageContent)を一時ファイルに保存しパスを返す。"""
    data = getattr(block, "data", None)
    if not data:
        return None
    mime = getattr(block, "mimeType", "") or "image/png"
    ext = ".jpg" if "jpeg" in mime or "jpg" in mime else ".png"
    # 一時ファイルもプロジェクト内のキャッシュ配下に置く（/tmp 等の外部に出さない）。
    tmp_dir = os.path.join(project_cache_dir(), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    _purge_old_tmp_images(tmp_dir)   # 1日より古い一時画像は掃除（無限成長の防止）
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
        workspace: str | None = None,
    ) -> None:
        self._servers = servers or {}
        # ツールが出力先パラメータ（workspace/output_dir 等）を持つとき、未指定なら
        # ここを注入する。OS ゲートウェイ越しでも引数はアプリへそのまま転送されるため、
        # ここでの注入が有効（絶対パスを渡す）。
        self._workspace = workspace
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
        """接続対象サーバー（明示設定のみ）。発見は外部（OS 層）が担う。"""
        return self._servers

    def start(self, log: Logger = _noop) -> None:
        """全サーバーへ即時接続する。"""
        if not self._servers:
            return
        self._log = log
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(self._connect_all(log), self._loop)
        future.result()  # 接続とツール一覧取得まで待つ

    def start_lazy(self, log: Logger = _noop) -> None:
        """イベントループだけ起動し、接続は activate() 時まで遅延する。"""
        if not self._servers:
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

        # サーバーが実行中に送る logging 通知（ctx.log / 進捗テキスト）を拾い、log へ流す。
        # 長時間ツール（例: 自然言語→CAD）の途中経過を CLI/GUI に出すために使う。
        async def _on_log(params: Any) -> None:
            try:
                logger = getattr(params, "logger", None)
                data = getattr(params, "data", "")
                text = data if isinstance(data, str) else str(data)
                detail = f"{name}[{logger}]: {text}" if logger else f"{name}: {text}"
                self._log("mcp", detail)
            except Exception:  # noqa: BLE001 - 表示の失敗で本処理を止めない
                pass

        session = await stack.enter_async_context(
            ClientSession(read, write, logging_callback=_on_log)
        )
        await session.initialize()
        # サーバーへ logging の最小レベルを通知する（対応サーバーだけ。未対応は無視）。
        try:
            await session.set_logging_level("info")
        except Exception:  # noqa: BLE001 - logging 非対応サーバーでも接続は続ける
            pass
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
        # このツールが持つ「出力先」引数（未指定なら workspace を注入する対象）。
        schema = getattr(tool, "inputSchema", None) or {}
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        ws_params = [p for p in _WORKSPACE_PARAMS if p in props]

        def func(**kwargs: Any) -> str:
            # 出力先が未指定なら利用側の作業ディレクトリを注入する（生成物を
            # 外部側でなく呼び出し側に書き出させる）。呼び出し側が明示した値は尊重する。
            if self._workspace:
                for p in ws_params:
                    if not kwargs.get(p):
                        kwargs[p] = self._workspace
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
            origin = "自動発見" if c.get("source") == "dir" else "設定"
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

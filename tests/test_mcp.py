import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_automata_core.mcp_client import (
    MCPManager,
    _result_to_text,
    _safe_name,
    build_mcp_meta_tools,
    discover_servers,
    resolve_call_timeout,
)
from local_automata_core.settings import load_agent_config

_SERVER = str(Path(__file__).with_name("_mcp_server.py"))


def write(tmp_path, text):
    p = tmp_path / "agent.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_parse_mcp_servers(tmp_path):
    p = write(
        tmp_path,
        "[mcp.servers.fs]\n"
        'command = "npx"\n'
        'args = ["-y", "@modelcontextprotocol/server-filesystem", "/data"]\n'
        "[mcp.servers.remote]\n"
        'url = "http://localhost:3000/mcp"\n',
    )
    ac = load_agent_config(p)
    assert ac.mcp_servers["fs"]["command"] == "npx"
    assert ac.mcp_servers["fs"]["args"][0] == "-y"
    assert ac.mcp_servers["remote"]["url"].endswith("/mcp")


def test_parse_mcp_requires_command_or_url(tmp_path):
    with pytest.raises(ValueError):
        load_agent_config(write(tmp_path, "[mcp.servers.bad]\nfoo = 1\n"))


def test_no_mcp_is_empty(tmp_path):
    assert load_agent_config(write(tmp_path, "tools = []\n")).mcp_servers == {}


def test_manager_no_servers_is_noop():
    m = MCPManager({})
    m.start()  # 何もしない
    assert m.tools() == []
    m.close()  # 例外なし


def test_safe_name():
    assert _safe_name("fs", "read_file") == "fs_read_file"
    assert _safe_name("a b", "x/y") == "a_b_x_y"  # 不正文字は _ に


def test_result_to_text():
    text_block = SimpleNamespace(text="hello", type="text")
    ok = SimpleNamespace(content=[text_block], isError=False)
    assert _result_to_text(ok) == "hello"
    err = SimpleNamespace(content=[SimpleNamespace(text="boom", type="text")], isError=True)
    assert _result_to_text(err).startswith("Error:")
    empty = SimpleNamespace(content=[], isError=False)
    assert _result_to_text(empty) == "(出力なし)"



def test_stdio_server_end_to_end():
    """実際の MCP サーバー（stdio サブプロセス）に接続してツールを呼ぶ。"""
    pytest.importorskip("mcp")
    servers = {"demo": {"command": sys.executable, "args": [_SERVER]}}
    m = MCPManager(servers)
    m.start()
    try:
        names = {t.name: t for t in m.tools()}
        assert "demo_add" in names
        assert "demo_greet" in names
        assert names["demo_add"].func(a=2, b=3) == "5"
        assert names["demo_greet"].func(name="Mina") == "Hello, Mina!"
    finally:
        m.close()


def test_logging_notification_reaches_log():
    """サーバーの logging 通知（ctx.log）が log("mcp", ...) に届く（途中経過の配信）。"""
    pytest.importorskip("mcp")
    logs: list[tuple[str, str]] = []
    m = MCPManager({"demo": {"command": sys.executable, "args": [_SERVER]}})
    m.start()
    m._log = lambda event, detail: logs.append((event, detail))
    try:
        names = {t.name: t for t in m.tools()}
        assert names["demo_log_then_return"].func() == "ok"
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not any(
            "progress-42" in d for _, d in logs
        ):
            time.sleep(0.05)
        # logger 名つきで mcp イベントとして届く
        assert any(e == "mcp" and "progress-42" in d for e, d in logs), logs
        assert any("demo-stream" in d for _, d in logs), logs
    finally:
        m.close()


# --- Streamable HTTP（url 指定のリモート接続）-------------------------------

_HTTP_SERVER = str(Path(__file__).parent.parent / "scripts" / "_verify_streamable_server.py")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http_ready(url: str, proc: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (code {proc.returncode})")
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except urllib.error.HTTPError:
            return  # 4xx でも HTTP 応答が返れば起動済み
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    raise TimeoutError(f"server not ready within {timeout}s at {url}")


def test_streamable_http_end_to_end():
    """url 指定で Streamable HTTP のリモートサーバーに接続してツールを呼ぶ。"""
    pytest.importorskip("mcp.client.streamable_http")
    host, port = "127.0.0.1", _free_port()
    url = f"http://{host}:{port}/mcp"
    proc = subprocess.Popen([sys.executable, _HTTP_SERVER, host, str(port)])
    try:
        _wait_http_ready(url, proc)
        m = MCPManager({"remote": {"url": url}})
        m.start()
        try:
            names = {t.name: t for t in m.tools()}
            assert "remote_add" in names
            assert "remote_greet" in names
            assert names["remote_add"].func(a=2, b=3) == "5"
            assert names["remote_greet"].func(name="Mina") == "Hello, Mina!"
        finally:
            m.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# --- 設定検証（cwd / timeout / mcp_call_timeout） ---------------------------


def test_parse_mcp_cwd_and_timeout(tmp_path):
    """cwd / timeout が server dict にそのまま届く（criterion 1/3 の前段）。"""
    ac = load_agent_config(
        write(
            tmp_path,
            "[mcp.servers.s]\n"
            'command = "x"\n'
            'cwd = "/abs/path"\n'
            "timeout = 0\n",
        )
    )
    assert ac.mcp_servers["s"]["cwd"] == "/abs/path"
    assert ac.mcp_servers["s"]["timeout"] == 0


def test_parse_mcp_cwd_must_be_str(tmp_path):
    with pytest.raises(ValueError):
        load_agent_config(
            write(tmp_path, '[mcp.servers.s]\ncommand = "x"\ncwd = 123\n')
        )


def test_parse_mcp_timeout_must_be_number(tmp_path):
    with pytest.raises(ValueError):
        load_agent_config(
            write(tmp_path, '[mcp.servers.s]\ncommand = "x"\ntimeout = "slow"\n')
        )


def test_parse_mcp_call_timeout_runtime(tmp_path):
    assert load_agent_config(
        write(tmp_path, "mcp_call_timeout = 0\n")
    ).runtime["mcp_call_timeout"] == 0


def test_parse_mcp_call_timeout_must_be_number(tmp_path):
    with pytest.raises(ValueError):
        load_agent_config(write(tmp_path, 'mcp_call_timeout = "x"\n'))


# --- resolve_call_timeout（0/負/None → 無制限） -----------------------------


def test_resolve_call_timeout():
    assert resolve_call_timeout(120.0) == 120.0
    assert resolve_call_timeout(30) == 30
    assert resolve_call_timeout(0) is None
    assert resolve_call_timeout(-5) is None
    assert resolve_call_timeout(None) is None


def test_default_call_timeout_is_120():
    """後方互換: 未指定なら従来どおり既定 120 秒。"""
    assert MCPManager({"demo": {}})._call_timeout == 120.0


# --- 呼び出しタイムアウト / 進捗 / キャンセル ------------------------------


class _FakeSession:
    """call_tool を模した非同期セッション。スリープ・進捗・キャンセルを観測する。"""

    def __init__(self, sleep: float = 0.1, message: str | None = "working"):
        self.sleep = sleep
        self.message = message
        self.cancelled = False
        self.read_timeout = "unset"
        self._request_id = 7  # call_tool が使う想定の request id
        self.cancel_notes: list = []

    async def call_tool(
        self, name, arguments=None, read_timeout_seconds=None, progress_callback=None
    ):
        self.read_timeout = read_timeout_seconds
        try:
            if progress_callback and self.message:
                await progress_callback(0.0, 1.0, self.message)
            await asyncio.sleep(self.sleep)
            return SimpleNamespace(
                content=[SimpleNamespace(text="done")], isError=False
            )
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def send_notification(self, note):
        self.cancel_notes.append(note)


def _running_manager(call_timeout):
    """サーバー接続を伴わず、バックグラウンドのイベントループだけ起動した Manager。"""
    m = MCPManager({}, call_timeout=call_timeout)
    m._loop = asyncio.new_event_loop()
    m._thread = threading.Thread(target=m._loop.run_forever, daemon=True)
    m._thread.start()
    return m


def _fake_tool(name="slow"):
    return SimpleNamespace(name=name, description=None, inputSchema=None)


def test_call_timeout_unlimited_waits():
    """call_timeout=None（無制限）なら read_timeout=None で待ち切る。"""
    m = _running_manager(None)
    sess = _FakeSession(sleep=0.2)
    try:
        tool = m._wrap("demo", {}, sess, _fake_tool())
        assert tool.func() == "done"
        assert sess.read_timeout is None
    finally:
        m.close()


def test_call_timeout_positive_times_out_and_cancels():
    """正のタイムアウト → 期限切れ。明示の cancelled 通知＋コルーチンのキャンセル。"""
    m = _running_manager(0.3)
    sess = _FakeSession(sleep=5)
    try:
        out = m._wrap("demo", {}, sess, _fake_tool()).func()
        assert "timed out after 0.3s" in out
        # 有限タイムアウトでも SDK へ read_timeout は張らない（外側 future が権威）。
        assert sess.read_timeout is None
        # call_tool の request id を指定した notifications/cancelled が送られる。
        assert len(sess.cancel_notes) == 1
        note = sess.cancel_notes[0]
        assert note.root.method == "notifications/cancelled"
        assert note.root.params.requestId == 7
        # コルーチンもキャンセルされ、サーバー側へ伝播する（少し待つ）。
        for _ in range(50):
            if sess.cancelled:
                break
            time.sleep(0.02)
        assert sess.cancelled is True
    finally:
        m.close()


def test_server_timeout_overrides_global():
    """[mcp.servers.<name>].timeout が全体既定を上書きする。"""
    # 全体は無制限でも、サーバー個別 timeout=0.3 が効いて期限切れになる。
    m = _running_manager(None)
    sess = _FakeSession(sleep=5)
    try:
        out = m._wrap("demo", {"timeout": 0.3}, sess, _fake_tool()).func()
        assert "timed out after 0.3s" in out
    finally:
        m.close()

    # 逆に全体 0.3 でも、サーバー個別 timeout=0（無制限）なら待ち切る。
    m2 = _running_manager(0.3)
    sess2 = _FakeSession(sleep=0.5)
    try:
        tool = m2._wrap("demo", {"timeout": 0}, sess2, _fake_tool())
        assert tool.func() == "done"
        assert sess2.read_timeout is None
    finally:
        m2.close()


def test_progress_callback_logs_message():
    """progress_callback の message がログ（mcp イベント）に出る。"""
    logs: list[tuple[str, str]] = []
    m = _running_manager(None)
    m._log = lambda event, detail: logs.append((event, detail))
    sess = _FakeSession(sleep=0.05, message="solving...")
    try:
        tool = m._wrap("demo", {}, sess, _fake_tool("opt"))
        assert tool.func() == "done"
        assert ("mcp", "opt: solving...") in logs
    finally:
        m.close()


def test_keyboard_interrupt_cancels_and_reraises(monkeypatch):
    """Ctrl-C（KeyboardInterrupt）で future.cancel() を呼んで再送出する。"""
    import local_automata_core.mcp_client as mc

    cancelled = {"v": False}

    class _StubFuture:
        def result(self, timeout=None):
            raise KeyboardInterrupt

        def cancel(self):
            cancelled["v"] = True

    def _fake_submit(coro, loop):
        coro.close()  # 「awaited されていない」警告を避ける
        return _StubFuture()

    monkeypatch.setattr(mc.asyncio, "run_coroutine_threadsafe", _fake_submit)
    m = MCPManager({"demo": {}}, call_timeout=1.0)
    m._loop = object()  # run_coroutine_threadsafe を差し替えたので未使用
    tool = m._wrap("demo", {}, object(), _fake_tool())
    with pytest.raises(KeyboardInterrupt):
        tool.func()
    assert cancelled["v"] is True


def test_stdio_cwd_passed(tmp_path):
    """cwd 指定が StdioServerParameters に渡る（サブプロセスの cwd で検証）。"""
    pytest.importorskip("mcp")
    servers = {
        "demo": {"command": sys.executable, "args": [_SERVER], "cwd": str(tmp_path)}
    }
    m = MCPManager(servers)
    m.start()
    try:
        names = {t.name: t for t in m.tools()}
        assert "demo_whereami" in names
        got = names["demo_whereami"].func()
        assert os.path.realpath(got) == os.path.realpath(str(tmp_path))
    finally:
        m.close()


# --- 方式B: 遅延起動（探索＋起動） ------------------------------------------


def test_parse_mcp_description(tmp_path):
    ac = load_agent_config(
        write(
            tmp_path,
            "[mcp.servers.s]\n"
            'command = "x"\n'
            'description = "ファイル操作ツール群"\n',
        )
    )
    assert ac.mcp_servers["s"]["description"] == "ファイル操作ツール群"


def test_parse_mcp_description_must_be_str(tmp_path):
    with pytest.raises(ValueError):
        load_agent_config(
            write(tmp_path, '[mcp.servers.s]\ncommand = "x"\ndescription = 1\n')
        )


def test_parse_mcp_mode_runtime(tmp_path):
    assert (
        load_agent_config(write(tmp_path, 'mcp_mode = "on_demand"\n')).runtime[
            "mcp_mode"
        ]
        == "on_demand"
    )


def test_parse_mcp_mode_invalid(tmp_path):
    with pytest.raises(ValueError):
        load_agent_config(write(tmp_path, 'mcp_mode = "nope"\n'))


def test_lazy_does_not_connect_until_activate():
    """start_lazy では接続せず、tools() は空。catalog は停止中として並ぶ。"""
    pytest.importorskip("mcp")
    servers = {
        "demo": {
            "command": sys.executable,
            "args": [_SERVER],
            "description": "計算と挨拶",
        }
    }
    m = MCPManager(servers)
    m.start_lazy()
    try:
        assert m.tools() == []
        cat = m.catalog()
        assert cat == [
            {
                "name": "demo",
                "description": "計算と挨拶",
                "active": False,
                "transport": "stdio",
                "source": "config",
            }
        ]
    finally:
        m.close()


def test_activate_and_deactivate_roundtrip():
    """activate で接続・ツール取得、deactivate で片付け。冪等性も確認。"""
    pytest.importorskip("mcp")
    servers = {"demo": {"command": sys.executable, "args": [_SERVER]}}
    m = MCPManager(servers)
    m.start_lazy()
    try:
        tools = m.activate("demo")
        names = {t.name: t for t in tools}
        assert "demo_add" in names
        assert names["demo_add"].func(a=2, b=3) == "5"
        # tools() / catalog に反映される
        assert "demo_add" in {t.name for t in m.tools()}
        assert m.catalog()[0]["active"] is True
        # 冪等: 再 activate は同じツール集合、tools() は重複しない
        again = m.activate("demo")
        assert {t.name for t in again} == set(names)
        assert len([t for t in m.tools() if t.name == "demo_add"]) == 1
        # deactivate で取り除く
        removed = m.deactivate("demo")
        assert "demo_add" in set(removed)
        assert "demo_add" not in {t.name for t in m.tools()}
        assert m.catalog()[0]["active"] is False
        assert m.deactivate("demo") is None  # 二重停止は None
    finally:
        m.close()


def test_activate_unknown_server_raises():
    m = MCPManager({"demo": {"command": "x"}})
    m.start_lazy()
    try:
        with pytest.raises(KeyError):
            m.activate("nope")
    finally:
        m.close()


def test_meta_tools_register_into_registry():
    """list/activate/deactivate メタツールが registry を動的に増減させる。"""
    pytest.importorskip("mcp")
    from local_automata_core.tools.base import ToolRegistry

    servers = {
        "demo": {
            "command": sys.executable,
            "args": [_SERVER],
            "description": "計算と挨拶",
        }
    }
    m = MCPManager(servers)
    m.start_lazy()
    registry = ToolRegistry()
    try:
        meta = {t.name: t for t in build_mcp_meta_tools(m, registry)}
        assert set(meta) == {
            "list_mcp_servers",
            "activate_mcp_server",
            "deactivate_mcp_server",
        }
        # 一覧に説明と停止中が出る
        listing = meta["list_mcp_servers"].func()
        assert "demo" in listing and "計算と挨拶" in listing
        # 起動するとツールが registry に入る
        out = meta["activate_mcp_server"].func(server="demo")
        assert "demo_add" in out
        assert registry.get("demo_add") is not None
        assert registry.get("demo_add").func(a=4, b=5) == "9"
        # 停止すると registry から消える
        meta["deactivate_mcp_server"].func(server="demo")
        assert registry.get("demo_add") is None
    finally:
        m.close()


# --- ディレクトリ走査によるディスカバリ（AIOS フォルダ方式） ------------------


def test_discover_single_script(tmp_path):
    """直下の .py を現在の Python で起動するサーバーとして発見する。"""
    aios = tmp_path / "AIOS"
    aios.mkdir()
    (aios / "calc.py").write_text(
        "# description: 電卓ツール\nprint('hi')\n", encoding="utf-8"
    )
    (aios / "_helper.py").write_text("x = 1\n", encoding="utf-8")  # _ 始まりは無視
    found = discover_servers([str(aios)])
    assert set(found) == {"calc"}
    assert found["calc"]["command"] == sys.executable
    assert found["calc"]["args"] == [str(aios / "calc.py")]
    assert found["calc"]["cwd"] == str(aios)
    assert found["calc"]["description"] == "電卓ツール"


def test_discover_folder_with_entry_script(tmp_path):
    """マニフェストの無いフォルダは server.py を docstring 付きで発見する。"""
    aios = tmp_path / "AIOS"
    (aios / "weather").mkdir(parents=True)
    (aios / "weather" / "server.py").write_text(
        '"""天気ツール\n\n詳細...\n"""\nprint("x")\n', encoding="utf-8"
    )
    found = discover_servers([str(aios)])
    assert set(found) == {"weather"}
    assert found["weather"]["args"] == [str(aios / "weather" / "server.py")]
    assert found["weather"]["cwd"] == str(aios / "weather")
    assert found["weather"]["description"] == "天気ツール"


def test_discover_folder_with_manifest(tmp_path):
    """.mcp.json があればそれを優先。キー＝フォルダ名一致を採用。

    cwd 既定はフォルダ、相対 cwd はフォルダ基準。`aios` など未知キーは無視する。
    """
    aios = tmp_path / "AIOS"
    (aios / "opt").mkdir(parents=True)
    (aios / "opt" / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "opt": {  # ← キー＝フォルダ名
                        "command": "julia",
                        "args": ["main.jl"],
                        "description": "最適化",
                        "cwd": "src",
                        "aios": {"category": "tool"},  # 未知キー → 無視される
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    found = discover_servers([str(aios)])
    assert found["opt"]["command"] == "julia"
    assert found["opt"]["description"] == "最適化"
    assert found["opt"]["cwd"] == str(aios / "opt" / "src")
    assert "aios" not in found["opt"]  # 起動定義には混ざらない


def test_discover_json_single_entry_key_mismatch(tmp_path):
    """エントリが 1 つだけならキー名がフォルダ名と違っても採用（名前はフォルダ名）。"""
    aios = tmp_path / "AIOS"
    (aios / "weather").mkdir(parents=True)
    (aios / "weather" / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"anything": {"url": "http://x/mcp"}}}),
        encoding="utf-8",
    )
    found = discover_servers([str(aios)])
    assert set(found) == {"weather"}  # サーバ名はフォルダ名
    assert found["weather"]["url"] == "http://x/mcp"


def test_discover_json_multi_entry_without_match_is_skipped(tmp_path):
    """複数エントリでフォルダ名一致が無ければ曖昧なので採用しない。"""
    aios = tmp_path / "AIOS"
    (aios / "multi").mkdir(parents=True)
    (aios / "multi" / ".mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"a": {"command": "x"}, "b": {"command": "y"}}}
        ),
        encoding="utf-8",
    )
    assert discover_servers([str(aios)]) == {}


def test_discover_skips_missing_and_invalid(tmp_path):
    aios = tmp_path / "AIOS"
    (aios / "empty").mkdir(parents=True)  # スクリプトもマニフェストも無い → 無視
    (aios / "bad").mkdir()
    (aios / "bad" / ".mcp.json").write_text(  # command/url 無し
        json.dumps({"mcpServers": {"bad": {"description": "no launch"}}}),
        encoding="utf-8",
    )
    (aios / "broken").mkdir()
    (aios / "broken" / ".mcp.json").write_text("{ not json", encoding="utf-8")
    assert discover_servers([str(aios)]) == {}
    assert discover_servers([str(tmp_path / "missing")]) == {}  # 無いディレクトリ


def test_dir_servers_merge_into_catalog_and_activate():
    """走査で見つかったサーバーが catalog に出て、activate で起動・呼び出しできる。"""
    pytest.importorskip("mcp")
    # 実 stdio サーバー（_mcp_server.py）を単体スクリプトとして置いた一時 AIOS を作る。
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / "demo.py"
        dst.write_text(Path(_SERVER).read_text(encoding="utf-8"), encoding="utf-8")
        m = MCPManager({}, dirs=[d])
        m.start_lazy()
        try:
            cat = {c["name"]: c for c in m.catalog()}
            assert "demo" in cat and cat["demo"]["source"] == "dir"
            tools = {t.name: t for t in m.activate("demo")}
            assert tools["demo_add"].func(a=2, b=3) == "5"
            assert "demo_add" in {t.name for t in m.tools()}
        finally:
            m.close()


def test_config_wins_over_dir_on_name_clash(tmp_path):
    aios = tmp_path / "AIOS"
    aios.mkdir()
    (aios / "demo.py").write_text("print('x')\n", encoding="utf-8")  # dir 側 demo
    m = MCPManager(
        {"demo": {"command": "x", "description": "設定側"}}, dirs=[str(aios)]
    )
    try:
        cat = {c["name"]: c for c in m.catalog()}
        assert cat["demo"]["source"] == "config"
        assert cat["demo"]["description"] == "設定側"
    finally:
        m.close()


def test_lazy_tools_proxy_auto_start_and_cache(tmp_path):
    """auto モード: 全ツールを最初から proxy として提示し、呼んだ瞬間に自動起動。"""
    pytest.importorskip("mcp")
    aios = tmp_path / "AIOS"
    aios.mkdir()
    (aios / "demo.py").write_text(
        "# description: デモ\n" + Path(_SERVER).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    cache = tmp_path / "cache.json"
    m = MCPManager({}, dirs=[str(aios)], cache_path=str(cache))
    m.start_lazy()
    try:
        proxies = {t.name: t for t in m.lazy_tools()}
        assert "demo_add" in proxies and "demo_greet" in proxies
        # まだプロセスは起動していない
        assert all(not c["active"] for c in m.catalog())
        # proxy を呼ぶと自動起動して結果が返る
        assert proxies["demo_add"].func(a=2, b=3) == "5"
        assert any(c["active"] for c in m.catalog())
        # スキャン結果がキャッシュに保存されている
        import json

        data = json.loads(cache.read_text(encoding="utf-8"))
        names = {t["name"] for e in data.values() for t in e["tools"]}
        assert "demo_add" in names
    finally:
        m.close()


def test_lazy_tools_uses_cache_without_harvest(tmp_path):
    """キャッシュがあれば harvest（接続）せずに proxy を構築できる。"""
    cache = tmp_path / "cache.json"
    cfg = {"command": "does-not-exist", "args": []}
    key = __import__("local_automata_core.mcp_client", fromlist=["_server_sig_hash"])._server_sig_hash(cfg)
    cache.write_text(
        '{"%s": {"name": "ghost", "tools": [{"name": "ghost_do", "description": "d", "parameters": {"type": "object", "properties": {}}}]}}'
        % key,
        encoding="utf-8",
    )
    m = MCPManager({"ghost": cfg}, cache_path=str(cache))
    m.start_lazy()
    try:
        # 起動不能なコマンドでも、キャッシュ由来なので harvest されず proxy ができる
        proxies = {t.name: t for t in m.lazy_tools()}
        assert "ghost_do" in proxies
    finally:
        m.close()


def test_parse_mcp_dir_runtime(tmp_path):
    assert (
        load_agent_config(write(tmp_path, 'mcp_dir = "AIOS"\n')).runtime["mcp_dir"]
        == "AIOS"
    )


def test_meta_tool_activate_unknown_returns_error():
    m = MCPManager({"demo": {"command": "x"}})
    m.start_lazy()
    from local_automata_core.tools.base import ToolRegistry

    try:
        meta = {t.name: t for t in build_mcp_meta_tools(m, ToolRegistry())}
        out = meta["activate_mcp_server"].func(server="ghost")
        assert out.startswith("Error:") and "ghost" in out
    finally:
        m.close()

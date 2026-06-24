import os

import pytest

from local_automata_core.tools import Workspace, build_registry


def reg_map(tmp_path, names=None, allow_install=False):
    r = build_registry(names, str(tmp_path / "ws"), allow_install)
    return {t.name: t for t in r}


def test_unknown_tool(tmp_path):
    with pytest.raises(ValueError):
        build_registry(["nope"], str(tmp_path))


def test_memory_name_hint(tmp_path):
    with pytest.raises(ValueError, match="memory"):
        build_registry(["read_file", "remember"], str(tmp_path))


def test_workspace_created(tmp_path):
    build_registry(["read_file"], str(tmp_path / "ws"))
    assert (tmp_path / "ws").is_dir()


def test_default_workdir_creates_workspace(tmp_path, monkeypatch):
    # workdir 未指定なら既定で workspace/ を作る（カレントに散らからない）
    monkeypatch.chdir(tmp_path)
    build_registry(["read_file"])
    assert (tmp_path / "workspace").is_dir()


def test_read_write_edit_list(tmp_path):
    t = reg_map(tmp_path)
    assert "Wrote" in t["write_file"].func(path="a.txt", content="hi\nthere\n")
    assert (tmp_path / "ws" / "a.txt").read_text() == "hi\nthere\n"
    assert "1\thi" in t["read_file"].func(path="a.txt")
    t["edit_file"].func(path="a.txt", old="there", new="world")
    assert "world" in (tmp_path / "ws" / "a.txt").read_text()
    assert "a.txt" in t["list_dir"].func()


def test_jail_blocks_escape(tmp_path):
    t = reg_map(tmp_path)
    for bad in ["../escape.txt", "/etc/passwd", "../../x"]:
        with pytest.raises(ValueError):
            t["write_file"].func(path=bad, content="x")
    assert not (tmp_path / "escape.txt").exists()


def test_jail_blocks_symlink(tmp_path):
    t = reg_map(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "ws").mkdir(exist_ok=True)
    os.symlink(str(outside), str(tmp_path / "ws" / "link"))
    with pytest.raises(ValueError):
        t["write_file"].func(path="link/evil.txt", content="x")
    assert not (outside / "evil.txt").exists()


def test_grep_and_glob(tmp_path):
    t = reg_map(tmp_path, ["read_file", "write_file", "grep", "glob"])
    t["write_file"].func(path="a.py", content="def foo():\n    pass\n")
    t["write_file"].func(path="sub/b.py", content="def bar():\n    pass\n")
    out = t["grep"].func(pattern=r"def ")
    assert "a.py:1" in out and "sub/b.py:1" in out
    assert "a.py" in t["glob"].func(pattern="**/*.py")
    assert "Error" in t["glob"].func(pattern="../*")
    assert "invalid regex" in t["grep"].func(pattern="([")


def test_grep_options_and_ignores(tmp_path):
    t = reg_map(tmp_path, ["write_file", "grep"])
    t["write_file"].func(path="src/auth.py", content="def Login(user):\n    TOKEN = 'x'\n    return TOKEN\n")
    t["write_file"].func(path="src/util.js", content="const TOKEN = 1;\n")
    # 無視ディレクトリ（rg なし時の枝刈り対象）
    t["write_file"].func(path="node_modules/junk.py", content="TOKEN noise\n")
    t["write_file"].func(path=".git/x.py", content="TOKEN in git\n")

    # glob でファイル種別を絞る
    py_only = t["grep"].func(pattern="TOKEN", glob="*.py")
    assert "src/auth.py" in py_only and "src/util.js" not in py_only
    # 無視ディレクトリは出てこない
    assert "node_modules" not in py_only and ".git" not in py_only
    # 大文字小文字無視
    assert "src/auth.py:1" in t["grep"].func(pattern="login", ignore_case=True)
    # 文脈行（前後）
    ctx = t["grep"].func(pattern="TOKEN = 'x'", context=1)
    assert "auth.py:1-" in ctx and "auth.py:2:" in ctx


def test_install_guard(tmp_path):
    blocked = reg_map(tmp_path, ["run_command"], allow_install=False)["run_command"]
    assert "[ブロック]" in blocked.func(command="uv pip install scipy")
    assert "hello" in blocked.func(command="echo hello")  # 通常コマンドは通る
    allowed = reg_map(tmp_path, ["run_command"], allow_install=True)["run_command"]
    assert "[ブロック]" not in allowed.func(command="echo ok")


def test_run_command_cwd(tmp_path):
    t = reg_map(tmp_path, ["run_command"], allow_install=True)
    out = t["run_command"].func(command="pwd")
    assert str((tmp_path / "ws").resolve()) in out


def test_run_command_forces_matplotlib_agg(tmp_path):
    # 図が別ウィンドウで開かないよう、run_command は MPLBACKEND=Agg を強制する。
    t = reg_map(tmp_path, ["run_command"], allow_install=True)
    out = t["run_command"].func(command='python -c "import os; print(os.environ.get(\'MPLBACKEND\'))"')
    assert "Agg" in out


def test_truncate_middle_keeps_head_and_tail():
    from local_automata_core.tools.base import truncate_middle

    short = "abc"
    assert truncate_middle(short, 100) == short  # 上限以内は素通し

    text = "H" * 1000 + "M" * 5000 + "T" * 1000  # 末尾 T が肝心（=エラー想定）
    out = truncate_middle(text, 500)
    assert out.startswith("H")  # 先頭の文脈が残る
    assert out.rstrip().endswith("T")  # 末尾（エラー）が残る — 先頭だけ切り詰めとの違い
    assert "省略" in out and "全 7,000 文字" in out  # 省略マーカー
    assert len(out) < len(text)


def test_run_command_returns_raw_output_tail(tmp_path):
    # 切り詰めは中央（Agent._execute）の役目。ツール自体は末尾まで生の出力を返す
    # （中央が先頭＋末尾を残して切り詰める前提）。
    t = reg_map(tmp_path, ["run_command"], allow_install=True)
    script = (
        'python -c "import sys; print(\'X\'*200000); '
        'sys.stderr.write(\'FATAL_TAIL_MARKER\')"'
    )
    out = t["run_command"].func(command=script)
    assert "FATAL_TAIL_MARKER" in out  # 末尾（stderr）も含めて返る
    assert "省略" not in out  # ツール段階では切り詰めない


def test_workspace_new_session_and_name(tmp_path):
    base = tmp_path / "workspace"
    ws = Workspace.new_session(base)
    assert ws.root.parent == base.resolve()
    assert ws.root.is_dir()
    ts = ws.root.name
    assert ws.name_from_task("放物運動シミュ") is True
    assert ws.root.name == f"{ts}_放物運動シミュ"
    # スラッグ化で空になる入力では改名しない
    ws2 = Workspace.new_session(base)
    ts2 = ws2.root.name
    assert ws2.name_from_task("***") is False
    assert ws2.root.name == ts2


def test_new_session_unique_same_timestamp(tmp_path, monkeypatch):
    # 同じ秒に複数起動しても別フォルダになり、出力が混ざらない
    monkeypatch.setattr(
        "local_automata_core.tools.workspace.session_timestamp", lambda: "20260101-000000"
    )
    a = Workspace.new_session(tmp_path)
    b = Workspace.new_session(tmp_path)
    assert a.root != b.root
    assert a.root.is_dir() and b.root.is_dir()
    assert b.root.name == "20260101-000000-2"


def test_name_from_task_uses_title(tmp_path):
    ws = Workspace(tmp_path / "20260101-000000")
    ws.ensure()
    # 長いタスク文でも、与えたタイトルを優先して短い名前にする
    ws.name_from_task("放物運動の軌道を計算して可視化する詳細なスクリプトを書いて", title="放物運動")
    assert ws.root.name == "20260101-000000_放物運動"


def test_name_from_task_falls_back_to_text_when_title_empty(tmp_path):
    ws = Workspace(tmp_path / "20260101-000000")
    ws.ensure()
    ws.name_from_task("斜方投射", title="***")  # タイトルが無効ならタスク文から
    assert ws.root.name == "20260101-000000_斜方投射"


def test_name_from_task_avoids_collision(tmp_path):
    (tmp_path / "20260101-000000_task").mkdir()  # 既に同名フォルダがある状況
    ws = Workspace(tmp_path / "20260101-000000")
    ws.ensure()
    ws.name_from_task("task")
    assert ws.root.name == "20260101-000000_task-2"


def test_workspace_rename_tracks_root(tmp_path):
    # 改名後も同じツールが新しいフォルダを指す（組み直し不要）
    ws = Workspace(tmp_path / "20260101-000000")
    ws.ensure()
    r = build_registry(["write_file", "read_file", "run_command"], ws, allow_install=True)
    t = {tool.name: tool for tool in r}
    ws.rename("20260101-000000_my-task")
    assert (tmp_path / "20260101-000000_my-task").is_dir()
    assert not (tmp_path / "20260101-000000").exists()
    t["write_file"].func(path="a.txt", content="hi")
    assert (tmp_path / "20260101-000000_my-task" / "a.txt").read_text() == "hi"
    assert str(tmp_path / "20260101-000000_my-task") in t["run_command"].func(command="pwd")


def test_dispatch_agent_explores_and_summarizes(tmp_path):
    # サブエージェント探索: 子が grep して結論だけを返す（書き込みツールは持たない）。
    from types import SimpleNamespace
    from local_automata_core.config import Config
    from local_automata_core.tools.workspace import Workspace
    from local_automata_core.tools.subagent import build_dispatch_agent_tool, DISPATCH_AGENT_TOOL
    from conftest import msg, tool_call

    ws = Workspace(str(tmp_path / "ws"))
    ws.ensure()
    (ws.root / "app").mkdir(parents=True, exist_ok=True)
    (ws.root / "app" / "auth.py").write_text("def authenticate(u):\n    return True\n", encoding="utf-8")

    class ChildLLM:
        def __init__(self):
            self.n = 0
            self.saw_tools = None

        def chat(self, messages, tools, on_text=lambda t: None):
            self.n += 1
            self.saw_tools = {s["function"]["name"] for s in tools}
            if self.n == 1:
                return msg(content="探索", tool_calls=[tool_call("grep", {"pattern": "authenticate"})])
            return msg(content="認証は app/auth.py:1 です。")

    llm = ChildLLM()
    statuses = []
    host = SimpleNamespace(on_status=lambda s: statuses.append(s))
    tool = build_dispatch_agent_tool(llm, Config(), ws, host)
    assert tool.name == DISPATCH_AGENT_TOOL
    result = tool.func(task="認証処理はどこ？")
    assert "app/auth.py:1" in result
    # 子は読み取り専用ツールのみ（書き込み・実行は無い）
    assert "write_file" not in llm.saw_tools and "run_command" not in llm.saw_tools
    assert "grep" in llm.saw_tools
    # 探索中の状況は親インジケータへ転送される
    assert any(s for s in statuses)


def test_dispatch_agent_registered_when_read_tools_present(tmp_path):
    # build_agent 経由で read 系ツールがあると dispatch_agent が自動登録される。
    from local_automata_core import compose_agent as build_agent
    from local_automata_core.config import Config
    from local_automata_core.settings import AgentConfig

    agent = build_agent(
        Config(), AgentConfig(tools=["read_file", "grep"]), str(tmp_path / "ws"), planning=False
    )
    assert agent.tools.get("dispatch_agent") is not None


def test_view_image_queues_image_and_flushes(tmp_path):
    # view_image は workspace 内の画像をモデルへの視覚入力として待避し、
    # _flush_pending で次ターンのユーザー発話（image_url 付き）に流し込む。
    import base64
    from local_automata_core import Agent
    from local_automata_core.config import Config
    from local_automata_core.tools.workspace import Workspace
    from local_automata_core.tools.vision import build_view_image_tool, VIEW_IMAGE_TOOL

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    ws = Workspace(str(tmp_path / "ws"))
    ws.ensure()
    (ws.root / "plot.png").write_bytes(png)

    class FakeLLM:
        def chat(self, *a, **k):
            raise RuntimeError("not used")

    agent = Agent(FakeLLM(), build_registry([], ws), Config())
    tool = build_view_image_tool(ws, agent)
    assert tool.name == VIEW_IMAGE_TOOL

    out = tool.func(path="plot.png")
    assert "plot.png" in out
    assert len(agent._pending_user_parts) == 1
    agent._flush_pending()
    last = agent.messages[-1]
    assert last["role"] == "user"
    kinds = [p["type"] for p in last["content"]]
    assert "image_url" in kinds
    assert agent._pending_user_parts == []  # 流し込み後は空


def test_view_image_jail_and_missing(tmp_path):
    from local_automata_core import Agent
    from local_automata_core.config import Config
    from local_automata_core.tools.workspace import Workspace
    from local_automata_core.tools.vision import build_view_image_tool

    ws = Workspace(str(tmp_path / "ws"))
    ws.ensure()

    class FakeLLM:
        def chat(self, *a, **k):
            raise RuntimeError("not used")

    agent = Agent(FakeLLM(), build_registry([], ws), Config())
    tool = build_view_image_tool(ws, agent)
    assert "Error" in tool.func(path="../escape.png")
    assert "Error" in tool.func(path="/etc/hosts")
    assert "Error" in tool.func(path="missing.png")
    assert agent._pending_user_parts == []  # 失敗時は何も積まれない


def test_read_pdf_pages_queues_images(tmp_path):
    # read_pdf_pages は workspace 内の PDF をページ画像として待避キューへ積む。
    import pymupdf
    from local_automata_core import Agent
    from local_automata_core.config import Config
    from local_automata_core.tools.workspace import Workspace
    from local_automata_core.tools.vision import build_read_pdf_pages_tool, READ_PDF_PAGES_TOOL

    ws = Workspace(str(tmp_path / "ws"))
    ws.ensure()
    doc = pymupdf.open()
    for i in range(3):
        doc.new_page().insert_text((72, 72), f"Page {i + 1}")
    doc.save(str(ws.root / "report.pdf"))
    doc.close()

    class FakeLLM:
        def chat(self, *a, **k):
            raise RuntimeError("not used")

    agent = Agent(FakeLLM(), build_registry([], ws), Config())
    tool = build_read_pdf_pages_tool(ws, agent)
    assert tool.name == READ_PDF_PAGES_TOOL

    out = tool.func(path="report.pdf", pages="1,3")
    assert "report.pdf" in out and "1, 3" in out
    assert len(agent._pending_user_parts) == 2
    assert all(p["type"] == "image_url" for p in agent._pending_user_parts)
    agent._flush_pending()
    last = agent.messages[-1]
    assert last["role"] == "user"
    assert "image_url" in [p["type"] for p in last["content"]]
    assert agent._pending_user_parts == []


def test_read_pdf_pages_jail_and_errors(tmp_path):
    from local_automata_core import Agent
    from local_automata_core.config import Config
    from local_automata_core.tools.workspace import Workspace
    from local_automata_core.tools.vision import build_read_pdf_pages_tool

    ws = Workspace(str(tmp_path / "ws"))
    ws.ensure()
    (ws.root / "notes.txt").write_text("hi", encoding="utf-8")

    class FakeLLM:
        def chat(self, *a, **k):
            raise RuntimeError("not used")

    agent = Agent(FakeLLM(), build_registry([], ws), Config())
    tool = build_read_pdf_pages_tool(ws, agent)
    assert "Error" in tool.func()  # path 未指定
    assert "Error" in tool.func(path="../escape.pdf")
    assert "Error" in tool.func(path="/etc/hosts")
    assert "Error" in tool.func(path="missing.pdf")
    assert "Error" in tool.func(path="notes.txt")  # PDF でない
    assert agent._pending_user_parts == []  # 失敗時は何も積まれない


def test_read_pdf_pages_registered_via_attach(tmp_path):
    from local_automata_core import Agent
    from local_automata_core.config import Config
    from local_automata_core.tools import attach_agentic_tools, Workspace

    class FakeLLM:
        def chat(self, *a, **k):
            raise RuntimeError("not used")

    ws = Workspace(str(tmp_path / "ws"))
    ws.ensure()
    agent = Agent(FakeLLM(), build_registry(["read_file"], ws), Config())
    assert agent.tools.get("read_pdf_pages") is None
    attach_agentic_tools(agent, ws)
    assert agent.tools.get("read_pdf_pages") is not None


def test_view_image_and_dispatch_registered_via_build_agent(tmp_path):
    from local_automata_core import compose_agent as build_agent
    from local_automata_core.config import Config
    from local_automata_core.settings import AgentConfig

    agent = build_agent(
        Config(), AgentConfig(tools=["read_file"]), str(tmp_path / "ws"), planning=False
    )
    assert agent.tools.get("view_image") is not None
    assert agent.tools.get("dispatch_agent") is not None


def test_attach_agentic_tools_registers_both(tmp_path):
    # ライブラリから手で組んだ Agent に view_image / dispatch_agent を1行で付けられる。
    from local_automata_core import Agent
    from local_automata_core.config import Config
    from local_automata_core.tools import attach_agentic_tools, Workspace

    class FakeLLM:
        def chat(self, *a, **k):
            raise RuntimeError("not used")

    ws = Workspace(str(tmp_path / "ws"))
    ws.ensure()
    agent = Agent(FakeLLM(), build_registry(["read_file", "grep"], ws), Config())
    assert agent.tools.get("view_image") is None and agent.tools.get("dispatch_agent") is None
    attach_agentic_tools(agent, ws)
    assert agent.tools.get("view_image") is not None
    assert agent.tools.get("dispatch_agent") is not None

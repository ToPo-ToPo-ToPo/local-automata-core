import httpx
import pytest
from openai import APIConnectionError

from agent_core import Agent
from agent_core.config import Config
from agent_core.tools import Tool, ToolRegistry, build_registry
from conftest import FakeLLM, msg, tool_call


def make_agent(tmp_path, responses, tools=None, **cfg):
    reg = tools if tools is not None else build_registry(
        ["read_file", "write_file", "list_dir", "edit_file", "run_command"],
        str(tmp_path / "ws"),
    )
    out = []
    agent = Agent(FakeLLM(responses), reg, Config(**cfg), on_text=out.append)
    return agent, out


def test_final_answer(tmp_path):
    agent, out = make_agent(tmp_path, [msg(content="完了です")])
    assert agent.run("やって") == "完了です"
    assert "完了です" in "".join(out)


def test_execute_truncates_large_tool_output(tmp_path):
    # 中央集約: どのツールの結果も _execute で一律に中央省略カットされる。
    # （run_command/read_file だけでなく、自作ツール・MCP・subagent も対象）
    reg = ToolRegistry()
    big = "H" * 200_000 + "FATAL_TAIL_MARKER"
    reg.register(Tool(name="bigtool", description="d", parameters={"type": "object", "properties": {}},
                      func=lambda: big))
    agent = Agent(FakeLLM([]), reg, Config(tool_output_max_chars=10_000))
    out = agent._execute(tool_call("bigtool", {}))
    assert out.startswith("H")              # 先頭の文脈が残る
    assert out.endswith("FATAL_TAIL_MARKER")  # 末尾（エラー想定）が残る
    assert "省略" in out                     # 中央省略マーカー
    assert len(out) < 11_000                 # 上限（+マーカー）に収まる


def test_execute_respects_tool_output_limit_disabled(tmp_path):
    # tool_output_max_chars <= 0 で切り詰め無効（全文を返す）。
    reg = ToolRegistry()
    big = "Z" * 50_000
    reg.register(Tool(name="bigtool", description="d", parameters={"type": "object", "properties": {}},
                      func=lambda: big))
    agent = Agent(FakeLLM([]), reg, Config(tool_output_max_chars=0))
    out = agent._execute(tool_call("bigtool", {}))
    assert out == big  # 無効化時は素通し


def test_tool_call_executes(tmp_path):
    responses = [
        msg(content="書きます", tool_calls=[tool_call("write_file", {"path": "x.txt", "content": "hi"})]),
        msg(content="できました"),
    ]
    agent, out = make_agent(tmp_path, responses)
    ans = agent.run("ファイル作って")
    joined = "".join(out)
    assert (tmp_path / "ws" / "x.txt").read_text() == "hi"
    # 既定は verbose: 呼び出しは全引数を表示する
    assert "● write_file(" in joined and '"path": "x.txt"' in joined
    assert ans == "できました"


def test_blank_line_before_generation(tmp_path):
    # 各生成の直前に空行を1行入れて、直前の出力と分離する
    responses = [
        msg(content="書きます", tool_calls=[tool_call("list_dir", {})]),
        msg(content="完了"),
    ]
    agent, out = make_agent(tmp_path, responses)
    agent.run("やって")
    assert out[0] == "\n"  # 最初の生成前
    # ツール結果（⎿ 行）の後にも空行を挟んでから次の生成が始まる
    joined = "".join(out)
    assert "\n\n完了" in joined


def test_empty_response(tmp_path):
    agent, out = make_agent(tmp_path, [msg(content="")])
    agent.run("...")
    assert "(empty response)" in "".join(out)


def test_max_steps(tmp_path):
    # 毎ターン tool_call を返し続ける → 上限で停止
    calls = [msg(content="", tool_calls=[tool_call("list_dir", {})]) for _ in range(5)]
    agent, out = make_agent(tmp_path, calls, max_steps=3)
    ans = agent.run("loop")
    assert "stopped" in ans


def test_api_error_graceful(tmp_path):
    class FailLLM:
        def chat(self, *a, **k):
            raise APIConnectionError(request=httpx.Request("POST", "http://x/v1"))

    reg = build_registry([], str(tmp_path / "ws"))
    out = []
    agent = Agent(FailLLM(), reg, Config(), on_text=out.append)
    ans = agent.run("hi")
    assert "Could not connect" in ans


def test_streaming_timeout_graceful(tmp_path):
    # ストリーミング中に飛んでくる生の httpx.ReadTimeout を握りつぶさず親切に通知
    class TimeoutLLM:
        def chat(self, *a, **k):
            raise httpx.ReadTimeout("timed out")

    reg = build_registry([], str(tmp_path / "ws"))
    out = []
    agent = Agent(TimeoutLLM(), reg, Config(), on_text=out.append)
    ans = agent.run("hi")
    assert "timed out" in ans


def test_context_compaction(tmp_path):
    # 大きな本文を返し続け、上限超過で要約が走ることを確認
    class BigLLM:
        def __init__(self):
            self.summarize = 0

        def chat(self, messages, tools, on_text=lambda t: None):
            if "会話履歴を要約" in messages[0]["content"]:
                self.summarize += 1
                return msg(content="要約")
            return msg(content="X" * 3000)

    reg = build_registry([], str(tmp_path / "ws"))
    agent = Agent(BigLLM(), reg, Config(max_context_tokens=1000))
    for i in range(6):
        agent.run(f"q{i}")
    assert agent.llm.summarize >= 1
    assert sum(1 for m in agent.messages if m["role"] == "system") == 1


def test_compaction_emits_status(tmp_path):
    # 圧縮中は必ず on_status("コンテキストを圧縮中…") が通知される（GUI 表示の保証）。
    class BigLLM:
        def chat(self, messages, tools, on_text=lambda t: None):
            if "会話履歴を要約" in messages[0]["content"]:
                return msg(content="要約")
            return msg(content="X" * 3000)

    statuses = []
    reg = build_registry([], str(tmp_path / "ws"))
    agent = Agent(
        BigLLM(), reg, Config(max_context_tokens=1000),
        on_status=lambda s: statuses.append(s),
    )
    for i in range(6):
        agent.run(f"q{i}")
    assert any("Compressing" in s for s in statuses)


def test_compaction_runs_mid_task_between_tool_steps(tmp_path):
    # タスク途中（ツール往復の各ステップ間）でも上限超過なら圧縮が走り、状況が通知される。
    class ToolThenDoneLLM:
        def __init__(self):
            self.calls = 0
            self.summarize = 0

        def chat(self, messages, tools, on_text=lambda t: None):
            if "会話履歴を要約" in messages[0]["content"]:
                self.summarize += 1
                return msg(content="要約")
            self.calls += 1
            if self.calls == 1:
                # 1回目: 大きな出力を伴うツール呼び出し（履歴を上限超過まで膨らませる）
                return msg(
                    content="X" * 10000,
                    tool_calls=[tool_call("run_command", {"command": "echo hi"})],
                )
            return msg(content="完了")  # 2回目（次ステップ）で終了

    statuses = []
    reg = build_registry(["run_command"], str(tmp_path / "ws"), allow_install=True)
    agent = Agent(
        ToolThenDoneLLM(), reg, Config(max_context_tokens=1000),
        on_status=lambda s: statuses.append(s),
    )
    agent.run("やって")  # 1ターン内で 2回 _chat（ステップ間）→ 2回目の直前で圧縮
    assert agent.llm.summarize >= 1
    assert any("Compressing" in s for s in statuses)


def test_compaction_single_turn_keeps_user_query(tmp_path):
    # 単一の依頼に対しツール往復が続く（user 発話が先頭1件のみ）対話で圧縮が走っても、
    # 圧縮後のメッセージに user 発話が必ず1件は残ること。残らないと Qwen 等のチャット
    # テンプレートが「No user query found in messages」で 500 になる退行が起きる。
    seen_after_compact = []

    class ToolLoopLLM:
        def __init__(self):
            self.calls = 0
            self.summarize = 0

        def chat(self, messages, tools, on_text=lambda t: None):
            if "会話履歴を要約" in messages[0]["content"]:
                self.summarize += 1
                return msg(content="要約")
            self.calls += 1
            # 圧縮が走った後（要約が system に入った後）に渡されたメッセージを記録
            if self.summarize >= 1:
                seen_after_compact.append([m["role"] for m in messages])
            if self.calls <= 6:
                # 毎ステップ大きな出力＋ツール呼び出しで履歴を上限超過まで膨らませる
                return msg(
                    content="X" * 8000,
                    tool_calls=[tool_call("run_command", {"command": "echo hi"})],
                )
            return msg(content="完了")

    reg = build_registry(["run_command"], str(tmp_path / "ws"), allow_install=True)
    agent = Agent(ToolLoopLLM(), reg, Config(max_context_tokens=1000))
    agent.run("一度だけ依頼する")  # 以後 user 発話は無く、ツール往復だけが続く
    assert agent.llm.summarize >= 1
    # 圧縮後の履歴に user 発話が残っている（[system] だけになっていない）
    assert any(m["role"] == "user" for m in agent.messages)
    # 圧縮後に LLM へ渡した各メッセージ列にも user が含まれる（テンプレ要件）
    assert seen_after_compact, "圧縮後の chat 呼び出しが観測されていない"
    assert all("user" in roles for roles in seen_after_compact)


def test_custom_tool(tmp_path):
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="echo",
            description="echo",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            func=lambda x: f"got {x}",
        )
    )
    responses = [
        msg(content="呼ぶ", tool_calls=[tool_call("echo", {"x": "hello"})]),
        msg(content="done"),
    ]
    out = []
    agent = Agent(FakeLLM(responses), reg, Config(), on_text=out.append)
    agent.run("use echo")
    assert "got hello" in "".join(out)


def test_verbose_shows_full_call_and_result(tmp_path):
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="echo",
            description="echo",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            },
            func=lambda a, b: "line1\nline2\nline3",
        )
    )
    responses = [
        msg(content="呼ぶ", tool_calls=[tool_call("echo", {"a": "x", "b": "y"})]),
        msg(content="done"),
    ]
    out = []
    agent = Agent(FakeLLM(responses), reg, Config(verbose=True), on_text=out.append)
    agent.run("go")
    joined = "".join(out)
    # 全引数が出る（要約せず JSON で表示）
    assert '"a": "x"' in joined and '"b": "y"' in joined
    # 結果が全行表示される（1行要約ではない）
    assert "line1" in joined and "line2" in joined and "line3" in joined


def test_non_verbose_truncates_result(tmp_path):
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="echo",
            description="echo",
            parameters={"type": "object", "properties": {"a": {"type": "string"}}},
            func=lambda a: "line1\nline2\nline3",
        )
    )
    responses = [
        msg(content="呼ぶ", tool_calls=[tool_call("echo", {"a": "x"})]),
        msg(content="done"),
    ]
    out = []
    agent = Agent(FakeLLM(responses), reg, Config(verbose=False), on_text=out.append)
    agent.run("go")
    joined = "".join(out)
    assert "line1" in joined and "line2" not in joined  # 1行目だけ


def test_custom_system_prompt(tmp_path):
    reg = build_registry([], str(tmp_path / "ws"))
    agent = Agent(FakeLLM([msg(content="hi")]), reg, Config(), system_prompt="カスタム指示")
    assert agent.messages[0]["content"] == "カスタム指示"


def test_reset_clears_history(tmp_path):
    reg = build_registry([], str(tmp_path / "ws"))
    agent = Agent(FakeLLM([msg(content="a"), msg(content="b")]), reg, Config(), system_prompt="S")
    agent.run("q1")
    assert len(agent.messages) > 1
    agent.reset()
    assert agent.messages == [{"role": "system", "content": "S"}]
    assert agent._summary is None
    # reset 後も続けて使える
    assert agent.run("q2") == "b"


def _asst(cid, name, args):
    import json
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        ],
    }


def _toolmsg(cid, content):
    return {"role": "tool", "tool_call_id": cid, "content": content}


def test_prune_history_elides_stale_and_deleted(tmp_path):
    import json
    from agent_core.agent import _ELIDED_DELETED, _ELIDED_STALE

    reg = build_registry([], str(tmp_path / "ws"))
    agent = Agent(FakeLLM([]), reg, Config(), system_prompt="S")
    agent.messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "task"},
        _asst("c1", "read_file", {"path": "a.py"}),
        _toolmsg("c1", "1\told\n2\tDELETED\n"),                       # 後で読み直す→省略
        _asst("c2", "edit_file", {"path": "a.py", "old": "DELETED", "new": "NEW"}),
        _toolmsg("c2", "Edited a.py (1 replacement(s))"),
        _asst("c3", "write_file", {"path": "b.py", "content": "v1"}),  # 後で上書き→省略
        _toolmsg("c3", "Wrote 2 bytes to b.py"),
        _asst("c4", "write_file", {"path": "b.py", "content": "v2"}),  # 最新→保持
        _toolmsg("c4", "Wrote 2 bytes to b.py"),
        _asst("c5", "read_file", {"path": "a.py"}),
        _toolmsg("c5", "1\tcurrent\n2\tNEW\n"),                        # 最新→保持
    ]
    agent._prune_history()

    # 古い read 全文は省略、最新は保持
    assert agent.messages[3]["content"] == _ELIDED_STALE
    assert "current" in agent.messages[11]["content"]
    # edit の old（削除コード）は常に省略、new も後続の全文読みで省略
    e = json.loads(agent.messages[4]["tool_calls"][0]["function"]["arguments"])
    assert e["old"] == _ELIDED_DELETED and e["new"] == _ELIDED_STALE
    # 上書きされた古い write は省略、最新は保持
    w3 = json.loads(agent.messages[6]["tool_calls"][0]["function"]["arguments"])
    w4 = json.loads(agent.messages[8]["tool_calls"][0]["function"]["arguments"])
    assert w3["content"] == _ELIDED_STALE and w4["content"] == "v2"


def test_prune_history_is_idempotent_and_keeps_single_read(tmp_path):
    import copy

    reg = build_registry([], str(tmp_path / "ws"))
    agent = Agent(FakeLLM([]), reg, Config(), system_prompt="S")
    agent.messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "u"},
        _asst("d1", "read_file", {"path": "only.py"}),
        _toolmsg("d1", "full content KEEP"),
    ]
    agent._prune_history()
    assert "KEEP" in agent.messages[3]["content"]  # 一度きりの読みは保持
    snapshot = copy.deepcopy(agent.messages)
    agent._prune_history()
    assert agent.messages == snapshot  # べき等


def _img(url):
    return {"type": "image_url", "image_url": {"url": url}}


def _umsg(text, *urls):
    return {"role": "user", "content": [{"type": "text", "text": text}, *[_img(u) for u in urls]]}


def _img_count(m):
    c = m.get("content")
    return sum(1 for p in c if isinstance(p, dict) and p.get("type") == "image_url") if isinstance(c, list) else 0


def test_prune_old_images_keeps_recent(tmp_path):
    from agent_core.agent import _ELIDED_IMAGE

    reg = build_registry([], str(tmp_path / "ws"))
    agent = Agent(FakeLLM([]), reg, Config(max_context_images=2), system_prompt="S")
    agent.messages = [
        {"role": "system", "content": "S"},
        _umsg("画像1を見て", "data:img1"),
        {"role": "assistant", "content": "画像1は猫です"},
        _umsg("画像2を見て", "data:img2"),
        {"role": "assistant", "content": "犬です"},
        _umsg("動画フレーム", "data:f1", "data:f2", "data:f3"),  # 最新（動画フレーム束）
    ]
    agent._prune_history()
    # 最古の画像は外れ、説明テキストとプレースホルダは残る
    assert _img_count(agent.messages[1]) == 0
    assert agent.messages[2]["content"] == "画像1は猫です"  # モデルの説明は保持
    assert any(
        isinstance(p, dict) and _ELIDED_IMAGE in p.get("text", "")
        for p in agent.messages[1]["content"]
    )
    # 直近2つの画像メッセージは丸ごと残る（動画の3フレームも保持）
    assert _img_count(agent.messages[3]) == 1
    assert _img_count(agent.messages[5]) == 3


def test_prune_old_images_disabled_and_zero(tmp_path):
    reg = build_registry([], str(tmp_path / "ws"))
    # -1 で無効（全保持）
    a = Agent(FakeLLM([]), reg, Config(max_context_images=-1), system_prompt="S")
    a.messages = [{"role": "system", "content": "S"}, _umsg("x", "data:a"), _umsg("y", "data:b")]
    a._prune_history()
    assert _img_count(a.messages[1]) == 1 and _img_count(a.messages[2]) == 1
    # 0 で常に外す
    b = Agent(FakeLLM([]), reg, Config(max_context_images=0), system_prompt="S")
    b.messages = [{"role": "system", "content": "S"}, _umsg("x", "data:a"), _umsg("y", "data:b")]
    b._prune_history()
    assert _img_count(b.messages[1]) == 0 and _img_count(b.messages[2]) == 0


def test_image_on_demand_mode_registers_and_refetches(tmp_path):
    # on_demand: 古い画像に id を振りラベル付きで省略、view_image(id) で呼び戻せる。
    from agent_core.tools.workspace import Workspace
    from agent_core.tools.vision import build_view_image_tool

    ws = Workspace(str(tmp_path / "ws"))
    ws.ensure()
    agent = Agent(
        FakeLLM([]), build_registry([], ws),
        Config(max_context_images=1, image_context_mode="on_demand"),
        system_prompt="S",
    )
    agent.messages = [
        {"role": "system", "content": "S"},
        _umsg("コンター図Aを確認", "data:imgA"),
        {"role": "assistant", "content": "収束しています"},
        _umsg("最新の図", "data:imgB"),
    ]
    agent._prune_history()
    # 古い画像は id 付きプレースホルダに、レジストリに登録される
    assert _img_count(agent.messages[1]) == 0
    text = " ".join(p.get("text", "") for p in agent.messages[1]["content"] if isinstance(p, dict))
    assert "img_1" in text and "コンター図Aを確認" in text
    assert agent._image_registry.get("img_1") == "data:imgA"
    # 最新は保持
    assert _img_count(agent.messages[3]) == 1
    # view_image(id) で呼び戻し（次ターンの user 発話として待避）
    tool = build_view_image_tool(ws, agent)
    out = tool.func(id="img_1")
    assert "img_1" in out
    assert agent._pending_user_parts[-1]["image_url"]["url"] == "data:imgA"
    assert "見つかりません" in tool.func(id="img_404")


def test_image_recent_mode_no_registry(tmp_path):
    reg = build_registry([], str(tmp_path / "ws"))
    agent = Agent(
        FakeLLM([]), reg,
        Config(max_context_images=1, image_context_mode="recent"),
        system_prompt="S",
    )
    agent.messages = [{"role": "system", "content": "S"}, _umsg("古い", "data:x"), _umsg("新", "data:y")]
    agent._prune_history()
    assert _img_count(agent.messages[1]) == 0
    assert agent._image_registry == {}  # recent では id を振らない

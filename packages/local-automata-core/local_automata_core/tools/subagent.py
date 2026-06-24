"""サブエージェント探索ツール（dispatch_agent）。

大規模コードベースの調査を、読み取り専用の子エージェントに委譲する。子は多数の
ファイルを grep / glob / read_file で調べ、**結論だけ**を返す。これにより本体
（親エージェント）の会話履歴を膨らませずに、巨大なコードから必要な情報を引ける。

Claude Code の Explore / Task サブエージェントに相当する。
"""
from __future__ import annotations

from dataclasses import replace

from .base import Tool
from .workspace import Workspace

DISPATCH_AGENT_TOOL = "dispatch_agent"

# 子エージェント（探索専用）が読めるツール。書き込み・実行は与えない。
_READONLY_TOOLS = ["read_file", "list_dir", "grep", "glob"]

_EXPLORE_SYSTEM = """\
あなたは大規模コードベースを調査する「探索エージェント」です。
与えられた調査タスクに対し、read_file / list_dir / grep / glob を使って必要な
情報を集め、**結論を簡潔にまとめて**返してください。

方針:
- まず grep / glob で当たりをつけ、関係するファイルだけ read_file で確認する。
- ファイルは変更しない（読み取り専用。書き込み・コマンド実行はできない）。
- 回答には根拠として該当箇所を `path:行番号` で示す。
- 冗長な全文引用は避け、「どこに何があるか・どう繋がるか」を箇条書きで簡潔に。
- 調べ終えたら、依頼内容に直接答える短いまとめを書いて終了する。"""


def build_dispatch_agent_tool(llm, config, ws: Workspace, host_agent) -> Tool:
    """dispatch_agent ツールを作る。

    - llm/config: 親と同じ LLM・設定を子でも使う（max_steps だけ抑える）。
    - ws: 親と同じ作業ディレクトリ（読み取り専用ツールで共有）。
    - host_agent: 親エージェント。子の状況（探索中…）を親のインジケータへ流す。
    """

    def dispatch_agent(task: str, max_steps: int = 20) -> str:
        # 循環 import を避けるため遅延 import する。
        from ..agent import Agent
        from . import build_registry

        steps = max(1, min(int(max_steps or 20), config.max_steps))
        child_config = replace(config, max_steps=steps)
        readonly = build_registry(_READONLY_TOOLS, ws, config.allow_install)
        # 子の本文は GUI に流さない（探索の生ログで画面を埋めない）。状況通知だけ
        # 親のインジケータへ転送し、最終まとめを戻り値（ツール結果）として返す。
        child = Agent(
            llm,
            readonly,
            child_config,
            system_prompt=_EXPLORE_SYSTEM,
            on_status=getattr(host_agent, "on_status", None) or (lambda _s: None),
        )
        try:
            return child.run(task) or "(調査結果が空でした)"
        except Exception as exc:  # noqa: BLE001 - 失敗は親へ文字列で返す
            return f"Error: サブエージェント探索に失敗しました: {exc}"

    return Tool(
        name=DISPATCH_AGENT_TOOL,
        description=(
            "読み取り専用の子エージェントを起動し、大規模コードベースの調査・検索を委譲する。"
            "子が多数のファイルを調べた上で結論だけを返すため、本体の文脈を膨らませずに済む。"
            "例: 『認証処理がどこで実装されているか調べて』『この関数の呼び出し元を全部探して』。"
            "ファイルは変更しない（調査専用）。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "調査してほしい内容（自然文。具体的なほど良い）",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "子エージェントの最大ステップ数（既定20）",
                },
            },
            "required": ["task"],
        },
        func=dispatch_agent,
    )

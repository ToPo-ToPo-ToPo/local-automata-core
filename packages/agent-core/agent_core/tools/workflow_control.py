from __future__ import annotations

from .base import Tool

# オーケストレーション型ワークフローでステップの完了を報告する制御ツール。
COMPLETE_STEP_TOOL = "complete_step"
WORKFLOW_CONTROL_TOOL_NAMES = (COMPLETE_STEP_TOOL,)


def _complete_step(result: str = "pass", note: str = "") -> str:
    if result not in ("pass", "fail"):
        return "Error: result は 'pass' か 'fail' を指定してください"
    label = "成功" if result == "pass" else "やり直し要求"
    msg = f"ステップ完了を記録しました（result={result}: {label}）。"
    return msg + (f" 備考: {note}" if note else "")


def build_workflow_control_tools() -> list[Tool]:
    """ステップの完了を報告する complete_step ツールを返す。

    オーケストレータはこのツールが呼ばれるまで次のステップへ進めない。
    result='pass' で次へ、result='fail' で（定義があれば）指定ステップへ戻る。
    """
    return [
        Tool(
            name=COMPLETE_STEP_TOOL,
            description=(
                "現在のワークフローステップが完了したことを報告する。"
                "ステップの作業を終えたら必ず呼ぶこと。"
                "result='pass' で次のステップへ進む。"
                "やり直しや前段の修正が必要なときは result='fail' で呼ぶ。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "enum": ["pass", "fail"],
                        "description": "ステップの判定結果（成功=pass / やり直し=fail）",
                    },
                    "note": {
                        "type": "string",
                        "description": "判定の根拠や次への申し送り（任意）",
                    },
                },
                "required": ["result"],
            },
            func=_complete_step,
        ),
    ]

from __future__ import annotations

import json
import sys
from typing import Any, Callable

import httpx
from openai import APIError, APITimeoutError

from .config import Config
from .colors import BOLD_CYAN, DIM, colorize
from .images import build_user_content, to_image_url
from .mcp_client import split_image_markers
from local_llm_client import LLMClient, TextSink


def _noop_text(_t: str) -> None:  # pragma: no cover
    pass
from .spinner import Spinner
from .tokenizer import TokenCounter
from .tools.base import ToolRegistry, truncate_middle

# 画像1枚のおおよそのコンテキスト寄与（トークン換算の概算。base64 長は使わない）
_IMAGE_TOKEN_COST = 300

# 履歴の刈り取り（コンテキスト編集）で使うプレースホルダ。
# ディスク上のファイルが真実なので、履歴には各ファイルの最新の全文だけ残せば足りる。
_ELIDED_DELETED = "(deleted content omitted)"
_ELIDED_STALE = "(omitted as stale; this file was re-read/changed later)"
# 古い画像を文脈から外すときのプレースホルダ（情報は残し、生データだけ落とす）。
_ELIDED_IMAGE = "(image omitted to save context; re-read it with view_image if needed)"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _content_images(content: Any) -> int:
    if isinstance(content, list):
        return sum(
            1
            for p in content
            if isinstance(p, dict) and p.get("type") == "image_url"
        )
    return 0

SYSTEM_PROMPT = """\
あなたは local-automata、ターミナル上で動く汎用コーディングエージェントです。
カレントディレクトリ上のプロジェクトに対してユーザーのタスクを達成します。

方針:
- 与えられたツールを使って、自分で調べ・編集し・コマンドで検証してください。
- 推測で答えず、まずファイルを読んでから判断してください。
- 既存ファイルを修正するときは edit_file を使い、変更する箇所だけを最小の
  old/new で指定してください。ファイル全体を write_file で再生成しないこと。
  old は変更箇所を一意に特定できる最小限の文脈だけを含め、変更しない部分まで
  old/new に入れないでください。write_file は新規作成または全面刷新のときだけ使います。
- 変更後はテストやビルドなどのコマンドで検証してください。
- タスクが完了したら、何をしたかを簡潔に日本語で説明して終了してください。
"""

PLANNING_GUIDANCE = """\
## 作業の進め方（計画してから実行する）
複雑なタスクでは、着手する前に手順を計画してください。
1. まず update_plan ツールで、タスクを小さなステップに分解した計画を作る。
2. 各ステップに着手する前に、そのステップを in_progress に更新する。
3. 完了したら done に更新し、必要なら計画を見直す。
4. ごく簡単なタスクでは計画を省略してよい。"""

# (event, detail) を受け取る進捗ログのコールバック
Logger = Callable[[str, str], None]

# 現在の処理状況（「実行中: run_command」「コンテキストを圧縮中…」など）を通知する
# コールバック。空文字でクリア。本文ストリーム(on_text)とは別チャネル。
StatusSink = Callable[[str], None]


def _noop(event: str, detail: str) -> None:  # pragma: no cover
    pass


def _noop_status(status: str) -> None:  # pragma: no cover
    pass


class Agent:
    """ツール呼び出しを伴う最小のエージェントループ。"""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        config: Config,
        log: Logger = _noop,
        on_text: TextSink = _noop_text,
        on_status: StatusSink = _noop_status,
        system_prompt: str = SYSTEM_PROMPT,
        planning: bool = False,
        workflow: str | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.config = config
        self.log = log
        self.on_text = on_text
        self.on_status = on_status
        self._tokens = TokenCounter(config.model)
        self._base_system = system_prompt
        if planning:
            self._base_system += "\n\n" + PLANNING_GUIDANCE
        if workflow:
            self._base_system += (
                "\n\n## 従うべきワークフロー\n"
                + workflow
                + "\n\n上記のワークフローの手順に従って進めてください。"
            )
        self._summary: str | None = None
        # view_image ツールが読み込んだ画像パーツを、次ターンのユーザー発話として
        # 流し込むための待避場所（ツールの戻り値は文字列なので画像を直接返せない）。
        self._pending_user_parts: list[dict[str, Any]] = []
        # on_demand モードで古い画像を id 付きで退避するレジストリ（id -> data URI）。
        # view_image("img_N") で必要な画像だけ呼び戻せる。順番は組み替えない。
        self._image_registry: dict[str, str] = {}
        self._image_seq = 0
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_content()}
        ]

    def reset(self) -> None:
        """会話履歴と要約をクリアする（システムプロンプトは保持）。"""
        self._summary = None
        self._pending_user_parts = []
        self._image_registry = {}
        self._image_seq = 0
        self.messages = [{"role": "system", "content": self._system_content()}]

    def context_usage(self) -> tuple[int, int]:
        """現在の会話履歴の概算トークン数と上限を返す（GUI 表示用）。

        used は会話履歴（system＋要約＋各メッセージ）の概算トークン数、
        max は max_context_tokens（これを超えると古いやり取りの自動要約が走る）。
        """
        return self._total_size(), self.config.max_context_tokens

    def run(
        self,
        user_input: str,
        images: list[str] | None = None,
        files: list[str] | None = None,
        videos: list[str] | None = None,
    ) -> str:
        content = build_user_content(user_input, images, files, videos)
        self.messages.append({"role": "user", "content": content})
        # コンテキスト管理（刈り取り＋必要なら圧縮）は各 LLM 呼び出し直前（_chat）で
        # 毎回行う。タスク途中のステップでも圧縮が走り、状況が GUI に通知される。
        schemas = self.tools.schemas()
        for step in range(self.config.max_steps):
            if self.config.debug:
                print(f"[debug] step {step + 1}/{self.config.max_steps}", file=sys.stderr)
            message, note = self._chat(schemas)
            if note is not None:
                return note

            self.messages.append(self._assistant_dict(message))

            # 本文・ツール呼び出しの表示と末尾改行は llm.chat 側で済んでいる。
            if not message.tool_calls:
                # プロンプトモードでツール呼び出しの解析に失敗した場合は、
                # 訂正を促してループを継続する（ユーザーが促す必要がない）。
                parse_error = getattr(message, "parse_error", None)
                if parse_error:
                    self.on_text(f"  → {parse_error}\n")
                    self.messages.append(
                        {"role": "user", "content": parse_error}
                    )
                    continue
                if not (message.content or "").strip():
                    self.on_text("(empty response)\n")
                return message.content or ""

            for call in message.tool_calls:
                self._dispatch_call(call)
            # view_image や MCP ツールが返した画像があれば、次ターンのユーザー発話として流し込む。
            self._flush_pending()
        stop = "(stopped: reached the maximum number of steps)"
        self.on_text(stop + "\n")
        return stop

    def _chat(self, schemas: list[dict[str, Any]]) -> tuple[Any | None, str | None]:
        """LLM を1回呼ぶ。成功なら (message, None)、失敗なら (None, エラー文) を返す。

        生成の直前に空行を1行入れて直前のツール結果と分離し、接続/タイムアウト
        系の例外を握りつぶしてユーザーに分かる説明文へ変換する（ループ側で return）。
        """
        # モデルへ送る前に、毎回コンテキスト管理を行う（刈り取り＋必要なら要約圧縮）。
        # タスク途中（ツール往復の各ステップ）でも上限超過なら圧縮し、その間は
        # on_status("コンテキストを圧縮中…") で GUI/CLI に必ず状況を通知する。
        self._maybe_compact()
        # 生成（「生成中」表示・本文）を直前のツール結果から1行空けて分離する。
        self.on_text("\n")
        # 最初のトークンが出るまで（prefill＝大きな文脈の処理中。最も待ち時間が長い）も
        # 動作中だと分かるよう状況を通知する。フロントは "prefill" を含む状況を専用表示にする。
        # 最初の本文トークンが来るとフロント側で自動的に解除される。
        self.on_status("Generating response (prefill)")
        try:
            message = self.llm.chat(self.messages, schemas, self.on_text)
            return message, None
        except (APITimeoutError, httpx.TimeoutException) as exc:
            # ストリーミング中は openai が包まない生の httpx 例外も飛んでくる。
            note = (
                f"[Error] The model response timed out ({self.config.base_url}). "
                "Large local models or long prompts can take a while for the first response. "
                "Increase request_timeout in agent.toml (0 for unlimited).\n"
                f"Details: {exc}"
            )
            self.on_text(note + "\n")
            return None, note
        except (APIError, httpx.HTTPError) as exc:
            note = (
                f"[Error] Could not connect to the LLM ({self.config.base_url}). "
                "Check that the server is running and verify base_url / model in agent.toml.\n"
                f"Details: {exc}"
            )
            self.on_text(note + "\n")
            return None, note

    def _dispatch_call(self, call: Any, allowed: set[str] | None = None) -> str:
        """1つのツール呼び出しを表示・実行し、結果を messages に追記して返す。

        allowed が指定された場合、その集合に無いツールは実行せずエラー文を返す
        （ワークフローのステップごとにツールを制限するために使う）。
        """
        verbose = self.config.verbose
        self.on_text(colorize(self._format_call(call, verbose=verbose), BOLD_CYAN) + "\n")
        name = call.function.name
        if allowed is not None and name not in allowed:
            result = (
                f"Error: tool '{name}' cannot be used in this step. "
                f"Available tools: {sorted(allowed)}"
            )
        else:
            # ツール実行中は本文が流れないので、状況を別チャネルで通知する
            self.on_status(f"Running: {name}")
            try:
                with Spinner(f"Running: {name}"):
                    result = self._execute(call)
            finally:
                self.on_status("")
        # MCP ツールが返した画像をテキストから取り出す（IMAGE_MARKER）。画像は
        # role:tool では多くの VLM が受け取れないため、_flush_pending() で次ターンの
        # ユーザー発話として image_url で渡す（vision 時 forward_tool_images=true のみ）。
        result, image_paths = split_image_markers(result)
        # 作業の切れ目として結果を表示する（Claude Code 風）。
        # 既定は1行に要約。verbose のときは全文を字下げして出す（過程を全部見せる）。
        for line in self._result_lines(result, verbose=verbose):
            self.on_text(colorize(line, DIM) + "\n")
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            }
        )
        if image_paths and self.config.forward_tool_images:
            added = 0
            for p in image_paths:
                try:
                    self._pending_user_parts.append(
                        {"type": "image_url", "image_url": {"url": to_image_url(p)}}
                    )
                    added += 1
                except Exception:
                    pass
            if added:
                self.on_text(colorize(
                    f"  ⎿ [passed {added} image(s) to the model]", DIM) + "\n")
        return result

    def _flush_pending(self) -> None:
        """view_image が待避した画像パーツを、ユーザー発話として会話に流し込む。

        ツールの戻り値は文字列なので画像を直接返せない。view_image は画像を
        _pending_user_parts に積み、ここで「ユーザーが画像を提示した」形にして
        次の LLM 呼び出しでモデルが視覚的に確認できるようにする。
        """
        if not self._pending_user_parts:
            return
        # 能動的なディレクティブにする。受動的な「確認して続けてください」だと、画像
        # 転送ターンの直前 assistant 相槌（「確認します」等）を temp=0 でそのまま反復し、
        # 評価・読解タスクを実行しないことがある（ローカル小型 VLM で再現）。直前のツール
        # 結果（＝指示）を明示参照する能動文にすると、指示と画像が結びつき確実に実行される。
        parts: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "次の画像を読み取り、直前のツール結果に示された指示・目的に従って回答してください。",
            },
            *self._pending_user_parts,
        ]
        self.messages.append({"role": "user", "content": parts})
        self._pending_user_parts = []

    def set_system(self, system_prompt: str) -> None:
        """システムプロンプトを差し替え、会話履歴の先頭(system)を更新する。

        会話履歴(messages[1:])と要約は保持する。ワークフローのステップごとに
        異なるシステムプロンプトへ切り替えるために使う。
        """
        self._base_system = system_prompt
        self.messages[0] = {"role": "system", "content": self._system_content()}

    def run_phase(
        self,
        user_content: Any | None,
        *,
        system: str,
        allowed_tools: set[str] | None,
        max_steps: int,
        stop_tools: set[str],
    ) -> tuple[str, dict[str, Any] | None]:
        """1ステップ分だけエージェントを駆動する（オーケストレーション用）。

        - system でこのステップ専用のシステムプロンプトに切り替える。
        - allowed_tools（None なら全ツール）に絞ってツールを公開・実行する。
        - stop_tools のいずれかが呼ばれたらそのステップを終了し、(本文, その引数)
          を返す。max_steps に達しても呼ばれなければ (本文, None) を返す。
        会話履歴はステップ間で保持され、次ステップが前ステップの成果を参照できる。
        """
        self.set_system(system)
        if user_content is not None:
            self.messages.append({"role": "user", "content": user_content})
        # コンテキスト管理（刈り取り＋必要なら圧縮）は _chat が各呼び出し前に毎回行う。
        schemas = self.tools.schemas()
        if allowed_tools is not None:
            schemas = [
                s for s in schemas if s["function"]["name"] in allowed_tools
            ]
        final_text = ""
        for _ in range(max_steps):
            message, note = self._chat(schemas)
            if note is not None:
                return note, None
            self.messages.append(self._assistant_dict(message))
            if message.content:
                final_text = message.content
            if not message.tool_calls:
                parse_error = getattr(message, "parse_error", None)
                if parse_error:
                    self.on_text(f"  → {parse_error}\n")
                    self.messages.append({"role": "user", "content": parse_error})
                    continue
                # ツールを呼ばずに本文だけ返した場合は、完了報告ツールの使用を促す。
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "このステップが完了したら "
                            + " または ".join(sorted(stop_tools))
                            + " を呼んでください。作業が残っているなら続けてください。"
                        ),
                    }
                )
                continue
            for call in message.tool_calls:
                self._dispatch_call(call, allowed=allowed_tools)
                if call.function.name in stop_tools:
                    try:
                        stop_args = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        stop_args = {}
                    return final_text, stop_args if isinstance(stop_args, dict) else {}
            # view_image で読み込んだ画像があれば次ターンへ流し込む。
            self._flush_pending()
        return final_text, None

    @staticmethod
    def _assistant_dict(message: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            out["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        return out

    @staticmethod
    def _format_call(call: Any, limit: int = 80, verbose: bool = False) -> str:
        """ツール呼び出しを「● ツール(引数)」の形に整形する。

        verbose のときは全引数を JSON で省略せず表示する。
        """
        name = call.function.name
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        if verbose:
            try:
                rendered = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                rendered = str(args)
            return f"● {name}({rendered})"
        key = ""
        if isinstance(args, dict):
            key = str(
                args.get("path")
                or args.get("command")
                or args.get("query")
                or (", ".join(f"{k}={v!r}" for k, v in args.items()) if args else "")
            )
        key = key.replace("\n", " ")
        if len(key) > limit:
            key = key[: limit - 1] + "…"
        return f"● {name}({key})"

    @staticmethod
    def _brief_result(result: str, limit: int = 100) -> str:
        text = result.strip()
        if not text:
            return "(no output)"
        first = text.splitlines()[0]
        return first if len(first) <= limit else first[: limit - 1] + "…"

    @classmethod
    def _result_lines(cls, result: str, verbose: bool = False) -> list[str]:
        """ツール結果の表示行を返す。既定は1行要約、verbose は全文（字下げ）。"""
        if not verbose:
            return [f"  ⎿ {cls._brief_result(result)}"]
        text = result.rstrip()
        if not text.strip():
            return ["  ⎿ (no output)"]
        lines = text.splitlines()
        return [f"  ⎿ {lines[0]}"] + [f"     {ln}" for ln in lines[1:]]

    def _execute(self, call: Any) -> str:
        name = call.function.name
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"Error: invalid JSON arguments: {exc}"
        try:
            result = tool.func(**args)
        except Exception as exc:  # noqa: BLE001 - ツール失敗はループに戻して継続
            return f"Error executing {name}: {exc}"
        text = result if isinstance(result, str) else str(result)
        # 全ツール結果が通る唯一の関門で出力長を一律に抑える（中央集約の切り詰め）。
        # run_command/read_file/grep だけでなく MCP・subagent など実装漏れのあるツールも
        # 含め、巨大な出力で履歴が膨張するのを構造的に防ぐ。先頭と末尾の両方を残すため、
        # 末尾に出やすいエラー・トレースバックの結論も失わない。
        return truncate_middle(text, self.config.tool_output_max_chars)

    # --- コンテキスト長の管理（要約による圧縮） -----------------------------

    def _system_content(self) -> str:
        if self._summary:
            return f"{self._base_system}\n\n## これまでの会話の要約\n{self._summary}"
        return self._base_system

    def _msg_size(self, message: dict[str, Any]) -> int:
        """メッセージのおおよそのトークン数（本文＋画像＋ツール呼び出し）。"""
        content = message.get("content")
        size = self._tokens.count(_content_text(content))
        size += _content_images(content) * _IMAGE_TOKEN_COST
        for call in message.get("tool_calls") or []:
            fn = call["function"]
            size += self._tokens.count(fn["name"]) + self._tokens.count(fn["arguments"])
        return size

    def _total_size(self) -> int:
        return sum(self._msg_size(m) for m in self.messages)

    def _tail_split(self) -> int:
        """要約する範囲と残す範囲の境界（残す先頭の index）を返す。

        末尾から積み上げて keep 予算を超えた位置で区切り、ユーザー発話の
        境界（=ターンの先頭）まで前方にずらす。これによりツール呼び出しと
        その結果のペアが分断されない。
        """
        budget = max(self.config.max_context_tokens // 2, 1)
        acc = 0
        split = 1
        for i in range(len(self.messages) - 1, 0, -1):
            acc += self._msg_size(self.messages[i])
            if acc > budget:
                split = i
                break
        # ターンの先頭（user 発話、または assistant 発話＝ツール呼び出し群の先頭）まで
        # 前方へずらし、assistant とその tool 結果のペアを分断しない。user だけに揃えると、
        # 単一の依頼に対してツール往復が続く対話（user 発話が先頭の1件しかない）で境界が
        # 見つからず split が末尾まで進み、tail が空＝圧縮後 [system] だけになってしまう
        # （Qwen 等のテンプレートが「user クエリ無し」で 500 を出す）。assistant 境界も
        # 許せば tail は必ず非空になり、ツールペアも保たれる。
        while split < len(self.messages) and self.messages[split]["role"] not in (
            "user",
            "assistant",
        ):
            split += 1
        return split

    def _prune_history(self) -> None:
        """古くなったファイル全文や削除済みコードを短いプレースホルダに置き換える。

        方針（ディスク上のファイルが真実なので、現在の内容は最新の1つだけ残せば足りる）:
        - edit_file の old（削除したコード）は常に省略する（将来参照しないため）。
        - あるファイルを後で全文読み直し / 書き込みした場合、それより前にある
          そのファイルの read_file 全文・write_file 内容・edit_file の new を省略する。
        最新の全文スナップショットとそれ以降の編集内容は保持するので、モデルは
        現在のファイル状態を読み直しなしで把握できる。べき等（再実行しても安全）。
        """
        msgs = self.messages
        # tool_call_id -> (ツール名, 引数dict)
        id_info: dict[str, tuple[str, dict[str, Any]]] = {}
        for m in msgs:
            for call in m.get("tool_calls") or []:
                fn = call.get("function", {})
                cid = call.get("id")
                if not cid:
                    continue
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                id_info[cid] = (fn.get("name", ""), args if isinstance(args, dict) else {})

        def is_full_read(args: dict[str, Any]) -> bool:
            # offset/limit 付きの部分読みは全文スナップショットとして扱わない。
            return not args.get("offset") and args.get("limit") is None

        # 各パスについて「最後の全文スナップショット」のある位置を求める。
        last_full: dict[str, int] = {}
        for i, m in enumerate(msgs):
            role = m.get("role")
            if role == "assistant":
                for call in m.get("tool_calls") or []:
                    cid = call.get("id")
                    name, args = id_info.get(cid, ("", {}))
                    path = args.get("path")
                    if name == "write_file" and path:
                        last_full[path] = i
            elif role == "tool":
                name, args = id_info.get(m.get("tool_call_id"), ("", {}))
                path = args.get("path")
                if name == "read_file" and path and is_full_read(args):
                    last_full[path] = i

        def superseded(path: object, i: int) -> bool:
            return isinstance(path, str) and path in last_full and i < last_full[path]

        # 刈り取り本体。
        for i, m in enumerate(msgs):
            role = m.get("role")
            if role == "assistant":
                for call in m.get("tool_calls") or []:
                    fn = call.get("function", {})
                    cid = call.get("id")
                    name, args = id_info.get(cid, ("", {}))
                    path = args.get("path")
                    if name == "edit_file":
                        new_args = dict(args)
                        changed = False
                        if new_args.get("old") not in (None, _ELIDED_DELETED):
                            new_args["old"] = _ELIDED_DELETED
                            changed = True
                        if superseded(path, i) and new_args.get("new") not in (
                            None,
                            _ELIDED_STALE,
                        ):
                            new_args["new"] = _ELIDED_STALE
                            changed = True
                        if changed:
                            fn["arguments"] = json.dumps(new_args, ensure_ascii=False)
                    elif name == "write_file" and superseded(path, i):
                        if args.get("content") not in (None, _ELIDED_STALE):
                            new_args = dict(args)
                            new_args["content"] = _ELIDED_STALE
                            fn["arguments"] = json.dumps(new_args, ensure_ascii=False)
            elif role == "tool":
                name, args = id_info.get(m.get("tool_call_id"), ("", {}))
                if name == "read_file" and superseded(args.get("path"), i):
                    if m.get("content") != _ELIDED_STALE:
                        m["content"] = _ELIDED_STALE
        # 画像も古いものを文脈から外す（最近の画像だけ残す）。
        self._prune_old_images()

    def _prune_old_images(self) -> None:
        """直近の画像だけを残し、古い画像の生データをプレースホルダに置き換える。

        画像はユーザー発話として履歴に残り毎ターン再送されるため、放置すると文脈を
        圧迫する。最近 `max_context_images` 個の「画像を含むメッセージ」だけを丸ごと
        残し、それより前の画像は外す（動画フレームのまとまりを壊さないよう、メッセージ
        単位で残す）。モデルが最初に書いた画像の説明テキストは残るので情報は失われない。
        view_image で読み込んだ workspace 画像は、必要なら同じツールで取り直せる。
        べき等（再実行しても変化しない）。
        """
        keep = getattr(self.config, "max_context_images", 2)
        if keep < 0:  # 負で無効（全画像を保持）
            return

        def has_image(m: dict[str, Any]) -> bool:
            content = m.get("content")
            return isinstance(content, list) and any(
                isinstance(p, dict) and p.get("type") == "image_url" for p in content
            )

        img_indices = [i for i, m in enumerate(self.messages) if has_image(self.messages[i])]
        if len(img_indices) <= keep:
            return
        to_elide = set(img_indices) if keep == 0 else set(img_indices[:-keep])
        on_demand = getattr(self.config, "image_context_mode", "recent") == "on_demand"
        for i in to_elide:
            m = self.messages[i]
            # 同じメッセージのテキストから、選択用の短いラベルを作る。
            label = next(
                (
                    p["text"].strip().replace("\n", " ")[:40]
                    for p in m["content"]
                    if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
                ),
                "",
            )
            kept_parts: list[Any] = []
            ids: list[str] = []
            for part in m["content"]:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    if on_demand:
                        self._image_seq += 1
                        iid = f"img_{self._image_seq}"
                        self._image_registry[iid] = part.get("image_url", {}).get("url", "")
                        ids.append(iid)
                    else:
                        ids.append("")  # 枚数を数えるだけ
                else:
                    kept_parts.append(part)
            if not ids:
                continue
            if on_demand:
                if len(ids) == 1:
                    note = (
                        f'(画像 {ids[0]}「{label}」は省略しました。'
                        f'view_image("{ids[0]}") で再表示できます)'
                    )
                else:
                    note = (
                        f'(画像 {ids[0]}〜{ids[-1]}「{label}」計{len(ids)}枚は省略しました。'
                        f'view_image("{ids[0]}") 等で個別に再表示できます)'
                    )
            else:
                note = f"{_ELIDED_IMAGE} ({len(ids)} images)"
            kept_parts.append({"type": "text", "text": note})
            m["content"] = kept_parts

    def _maybe_compact(self) -> None:
        # まず刈り取り（古い全文・削除済みコードの省略）でサイズを減らしてから判定する。
        self._prune_history()
        if self._total_size() <= self.config.max_context_tokens:
            return
        split = self._tail_split()
        old = self.messages[1:split]
        if not old:
            return
        # 要約は本文を流さない無音の LLM 呼び出しなので、状況を通知する
        self.on_status("Compressing context…")
        try:
            summary = self._summarize(old)
        except Exception as exc:  # noqa: BLE001 - 要約失敗時は圧縮せず継続
            self.log("compact", f"Skipped compaction because summarization failed: {exc}")
            return
        finally:
            self.on_status("")
        if not summary:
            return
        self._summary = summary
        self.messages = [
            {"role": "system", "content": self._system_content()},
            *self.messages[split:],
        ]
        # テンプレートによっては user 発話が最低1件必要（Qwen 系は無いと例外）。圧縮で
        # tail から user 発話が消えた場合に備え、最初の依頼（old 内の最初の user 発話）を
        # system 直後に1件だけ復元する。エージェントの目的を保つ意味でも有効。
        if not any(m.get("role") == "user" for m in self.messages):
            first_user = next((m for m in old if m.get("role") == "user"), None)
            if first_user is not None:
                self.messages.insert(1, first_user)
        self.log("compact", f"Compacted the conversation history by summarizing ({len(old)} messages summarized)")

    def _summarize(self, old: list[dict[str, Any]]) -> str:
        instruction = (
            "あなたは会話履歴を要約するアシスタントです。既存の要約と追加のやり取りを"
            "統合し、後続のタスク継続に必要な事実・決定・現在の状態・未完了事項を、"
            "簡潔な箇条書きで日本語にまとめてください。要約のみを出力すること。"
        )
        body = (
            "## 既存の要約\n"
            + (self._summary or "(なし)")
            + "\n\n## 追加のやり取り\n"
            + self._render(old)
        )
        result = self.llm.chat(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": body},
            ],
            [],
        )
        return (result.content or "").strip()

    @staticmethod
    def _render(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for message in messages:
            role = message["role"]
            if role == "assistant":
                if message.get("content"):
                    lines.append(f"[assistant] {message['content']}")
                for call in message.get("tool_calls") or []:
                    fn = call["function"]
                    lines.append(f"[tool_call] {fn['name']}({fn['arguments']})")
            elif role == "tool":
                content = message.get("content") or ""
                if len(content) > 1000:
                    content = content[:1000] + " …(truncated)"
                lines.append(f"[tool_result] {content}")
            else:
                text = _content_text(message.get("content"))
                n_images = _content_images(message.get("content"))
                suffix = f" [{n_images} images]" if n_images else ""
                lines.append(f"[{role}] {text}{suffix}")
        return "\n".join(lines)

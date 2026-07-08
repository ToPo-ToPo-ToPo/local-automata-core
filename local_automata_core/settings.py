from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .workflow import WorkflowSpec

# 構造的な設定ファイル（TOML）と、指示（システムプロンプト）の Markdown ファイル。
# Codex の config.toml / AGENTS.md、Claude Code の settings.json / CLAUDE.md と
# 同じく「構造的設定は TOML、指示は Markdown」に役割を分ける。
DEFAULT_CONFIG_FILENAME = "agent.toml"
DEFAULT_INSTRUCTIONS_FILENAME = "AGENTS.md"
DEFAULT_MEMORY_PATH = ".coder/memory.json"
# エージェントのツールが読み書き・実行を行う既定の作業ディレクトリ
DEFAULT_WORKDIR = "workspace"


@dataclass
class MemoryConfig:
    """長期メモリの設定。enabled のときだけメモリツールが有効になる。"""

    enabled: bool = False
    path: str = DEFAULT_MEMORY_PATH


@dataclass
class VoiceConfig:
    """音声入力（STT）の設定。enabled のときだけ Web GUI にマイク入力が出る。

    音声→テキストはローカルの mlx-whisper で行う。変換後は通常のテキスト入力として
    エージェントへ渡るため、エージェントのコアは音声を意識しない。
    """

    enabled: bool = False
    model: str = "mlx-community/whisper-large-v3-mlx"  # 最高精度（非量子化 full large-v3）
    language: str | None = None  # None で自動判定
    # 文字起こし結果を LLM で校正してから入力欄へ入れる（日本語の同音異義語・漢字の
    # 誤変換を文脈から直す）。1回 LLM 呼び出しが増えるぶん少し遅くなる。既定は無効。
    correct: bool = False


@dataclass
class AgentConfig:
    """解決済みのエージェント定義（システムプロンプト＋ツールのセット）。

      - system_prompt: 指示 Markdown の内容（None なら組み込み既定）
      - tools:         有効ツール名のリスト（None なら全ツール、[] なら無し）
      - memory:        長期メモリ設定（None なら未設定）
      - workdir:       ツールの作業ディレクトリ（None なら呼び出し側の既定）
      - planning:      着手前に計画を立てさせるか
      - workflow:      助言型ワークフロー。外部 Markdown の内容を system に注入し
                       LLM の自律判断に委ねる（None なら未指定）
      - workflow_spec: オーケストレーション型ワークフロー（外部 YAML, mode:
                       orchestrated）。ステップを順番に確実に実行する（None なら未指定）
      - runtime:       モデル・パラメータ・運用設定（agent.toml 由来。model / base_url /
                       tool_mode / temperature など）
      - mcp_servers:   MCP サーバー定義（name -> 設定）。ツールが自動で取り込まれる
    """

    system_prompt: str | None = None
    tools: list[str] | None = None
    memory: MemoryConfig | None = None
    voice: VoiceConfig | None = None
    workdir: str | None = None
    planning: bool = False
    workflow: str | None = None
    workflow_spec: "WorkflowSpec | None" = None
    runtime: dict = field(default_factory=dict)
    mcp_servers: dict = field(default_factory=dict)


def find_config_path(explicit: str | None = None) -> Path | None:
    """使用する TOML 設定ファイルのパスを解決する。

    優先順位: 明示指定 (--config) > ./agent.toml。
    既定パスは存在する場合のみ返す（無ければ None = 既定値で動作）。
    """
    if explicit:
        return Path(explicit)
    default = Path(DEFAULT_CONFIG_FILENAME)
    return default if default.exists() else None


def _read_instructions(path: Path, *, required: bool, label: str = "instructions file") -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8").strip() or None
    if required:
        raise FileNotFoundError(f"{label} not found: {path}")
    return None


def _validate_tools(tools: object) -> list[str] | None:
    if tools is not None and not (
        isinstance(tools, list) and all(isinstance(t, str) for t in tools)
    ):
        raise ValueError("tools must be a list of strings")
    return tools  # type: ignore[return-value]


# agent.toml で指定できる実行時設定（型: bool は int と区別して検証する）
_RUNTIME_KEYS: dict[str, type | tuple[type, ...]] = {
    "base_url": str,
    "model": str,
    "api_key": str,
    "temperature": (int, float),
    "max_steps": int,
    "max_context_tokens": int,
    "max_context_images": int,
    "image_context_mode": str,
    "tool_output_max_chars": int,  # 1ツール結果の最大文字数（超過分は中央省略カット）
    "max_context_chars": int,  # 廃止（後方互換のため受理のみ。圧縮の基準には使わない）
    "max_tokens": int,
    "enable_thinking": bool,
    "stream": bool,
    "request_timeout": (int, float),
    "allow_install": bool,
    "debug": bool,
    "verbose": bool,
    "tool_mode": str,
    "forward_tool_images": bool,  # MCPツールが返す画像をVLMへ転送するか（vision時のみ有効化）
    "web_port": int,      # Web GUI の待受ポート。--port 未指定時の既定
    "mcp_call_timeout": (int, float),  # MCP ツール呼び出しの既定タイムアウト秒数（0/負で無制限）
    "mcp_mode": str,  # MCP の扱い: auto(既定)/on_demand/eager。詳細は _parse_runtime の検証を参照
    "mcp_dir": str,  # 走査して MCP サーバーを自動発見するディレクトリ（AIOS フォルダ方式）
}


def _parse_runtime(section: dict) -> dict:
    """section から実行時・運用設定（model / base_url / tool_mode など）を検証して取り出す。"""
    out: dict = {}
    for key, typ in _RUNTIME_KEYS.items():
        if key not in section:
            continue
        value = section[key]
        if typ is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
        elif isinstance(value, bool) or not isinstance(value, typ):
            name = "a number" if typ != str else "a string"
            raise ValueError(f"{key} must be {name}")
        out[key] = value
    if "tool_mode" in out and out["tool_mode"] not in ("native", "prompt"):
        raise ValueError('tool_mode must be "native" or "prompt"')
    # mcp_mode: auto=全ツールを最初から提示しプロセスは初回呼び出し時に起動（既定）/
    # on_demand=エージェントが list・activate で明示起動 / eager=起動時に全接続。
    if "mcp_mode" in out and out["mcp_mode"] not in ("auto", "on_demand", "eager"):
        raise ValueError('mcp_mode must be "auto", "on_demand", or "eager"')
    if "image_context_mode" in out and out["image_context_mode"] not in (
        "recent", "on_demand",
    ):
        raise ValueError('image_context_mode must be "recent" or "on_demand"')
    if "web_port" in out and not (1 <= out["web_port"] <= 65535):
        raise ValueError("web_port must be between 1 and 65535")
    # モデルを既定から変えるなら、ポート(base_url)も必ず明示する。
    # こうしておくと「モデル→ポート」の対応が一意になり、別モデルが既定ポート
    # （:8799 の既定モデル）と取り違えられる事故を防げる（同一PCで複数モデルを並走させる前提）。
    if "model" in out and "base_url" not in out:
        raise ValueError(
            "When specifying model, you must also specify base_url (port). "
            "Each model is assigned to a separate port (when using the default model you may omit both model and base_url, which uses :8799)."
        )
    return out


def _parse_mcp(section: dict) -> dict:
    """[mcp] セクションから MCP サーバー定義を検証して取り出す。

    形式: [mcp.servers.<name>] に command(+args,env,cwd) か url を書く。
    任意で timeout（このサーバーのツール呼び出しタイムアウト秒。0/負で無制限）も指定できる。
    """
    mcp = section.get("mcp")
    if mcp is None:
        return {}
    if not isinstance(mcp, dict):
        raise ValueError("mcp must be a table")
    servers = mcp.get("servers", {})
    if not isinstance(servers, dict):
        raise ValueError("mcp.servers must be a table")
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"mcp.servers.{name} must be a table")
        if not cfg.get("command") and not cfg.get("url"):
            raise ValueError(
                f"mcp.servers.{name} requires either command or url"
            )
        cwd = cfg.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise ValueError(f"mcp.servers.{name}.cwd must be a string")
        description = cfg.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError(f"mcp.servers.{name}.description must be a string")
        timeout = cfg.get("timeout")
        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float))
        ):
            raise ValueError(
                f"mcp.servers.{name}.timeout must be a number (seconds; 0/negative means unlimited)"
            )
    return servers


def _parse_memory(section: dict) -> MemoryConfig | None:
    mem = section.get("memory")
    if mem is None:
        return None
    if not isinstance(mem, dict):
        raise ValueError("memory must be a table")
    enabled = mem.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("memory.enabled must be a boolean")
    path = mem.get("path", DEFAULT_MEMORY_PATH)
    if not isinstance(path, str):
        raise ValueError("memory.path must be a string")
    return MemoryConfig(enabled=enabled, path=path)


def _parse_voice(section: dict) -> VoiceConfig | None:
    voice = section.get("voice")
    if voice is None:
        return None
    if not isinstance(voice, dict):
        raise ValueError("voice must be a table")
    enabled = voice.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("voice.enabled must be a boolean")
    default = VoiceConfig()
    model = voice.get("model", default.model)
    if not isinstance(model, str):
        raise ValueError("voice.model must be a string")
    language = voice.get("language")
    if language is not None and not isinstance(language, str):
        raise ValueError("voice.language must be a string")
    correct = voice.get("correct", default.correct)
    if not isinstance(correct, bool):
        raise ValueError("voice.correct must be a boolean")
    return VoiceConfig(enabled=enabled, model=model, language=language, correct=correct)


def _select_section(
    data: dict, profile: str | None
) -> tuple[dict, str | None, bool]:
    """使用するセクション（profiles.<name> か トップレベル）を決める。

    返り値: (section, instructions_file, instructions_explicit)
    """
    profiles = data.get("profiles", {})
    if profiles and not isinstance(profiles, dict):
        raise ValueError("profiles must be a table")

    selected = profile or data.get("default_profile")

    if profiles:
        if selected is None:
            raise ValueError(
                "Profiles are defined. Select one with --profile or "
                f"set default_profile. Available: {list(profiles)}"
            )
        if selected not in profiles:
            raise ValueError(
                f"Profile '{selected}' not found. Available: {list(profiles)}"
            )
        section = profiles[selected]
        if not isinstance(section, dict):
            raise ValueError(f"profiles.{selected} must be a table")
        return section, section.get("instructions_file"), "instructions_file" in section

    # プロファイル未定義: トップレベルをそのまま使う（単一エージェント）
    if profile is not None:
        raise ValueError(
            f"Profile '{profile}' not found (no profiles are defined)."
        )
    return data, data.get("instructions_file"), "instructions_file" in data


def load_agent_config(
    config_path: Path | None,
    instructions_override: str | None = None,
    profile: str | None = None,
    workflow_override: str | None = None,
) -> AgentConfig:
    """TOML 設定と指示 Markdown を読み込み、AgentConfig（プロンプト＋ツール）を返す。

    - 単一エージェント: トップレベルの `tools` / `instructions_file` を使う。
    - 複数プロファイル: `[profiles.<name>]` を定義し、profile 引数 /
      CODER_PROFILE / TOML の `default_profile` で1つ選ぶ。各プロファイルが
      システムプロンプト（instructions_file）とツールのセットを持つ。
    - システムプロンプトは Markdown（既定 AGENTS.md）から読む。既定ファイルが
      無ければ組み込み既定を使い、明示指定したファイルが無ければエラー。
    """
    data: dict = {}
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("rb") as fp:
            data = tomllib.load(fp)
        if "system_prompt" in data:
            raise ValueError(
                "system_prompt is no longer supported. Write the system prompt "
                f"in a Markdown file such as {DEFAULT_INSTRUCTIONS_FILENAME}."
            )

    section, instructions_file, instructions_explicit = _select_section(data, profile)
    tools = _validate_tools(section.get("tools"))

    if instructions_file is not None and not isinstance(instructions_file, str):
        raise ValueError("instructions_file must be a string")

    workdir = section.get("workdir")
    if workdir is not None and not isinstance(workdir, str):
        raise ValueError("workdir must be a string")

    planning = section.get("planning", False)
    if not isinstance(planning, bool):
        raise ValueError("planning must be a boolean")

    workflow_file = section.get("workflow_file")
    if workflow_file is not None and not isinstance(workflow_file, str):
        raise ValueError("workflow_file must be a string")

    # Markdown ファイルは agent.toml のあるディレクトリ基準で解決する
    # （--config で別ディレクトリの設定を使っても正しく読めるように）。
    base = config_path.parent if config_path is not None else Path(".")

    def _resolve(name: str) -> Path:
        p = Path(name)
        return p if p.is_absolute() else base / p

    if instructions_override is not None:
        instructions_file = instructions_override
        instructions_explicit = True

    if instructions_file is None:
        instructions_file = DEFAULT_INSTRUCTIONS_FILENAME

    system_prompt = _read_instructions(
        _resolve(instructions_file), required=instructions_explicit
    )

    # ワークフロー（外部ファイルで渡す固定手順）。指定時は必須として読む。
    # 拡張子で方式を切り替える: .yaml/.yml は orchestrated（確実に順番実行）、
    # それ以外（.md など）は advisory（system へ注入し LLM の自律判断に委ねる）。
    if workflow_override is not None:
        workflow_file = workflow_override
    workflow: str | None = None
    workflow_spec: "WorkflowSpec | None" = None
    if workflow_file:
        wf_path = _resolve(workflow_file)
        if not wf_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {wf_path}")
        text = wf_path.read_text(encoding="utf-8")
        if wf_path.suffix.lower() in (".yaml", ".yml"):
            from .workflow import parse_workflow

            workflow_spec = parse_workflow(text, label=f"workflow file ({wf_path.name})")
        else:
            workflow = text.strip() or None

    memory = _parse_memory(section)
    voice = _parse_voice(section)
    runtime = _parse_runtime(section)
    mcp_servers = _parse_mcp(section)
    return AgentConfig(
        system_prompt=system_prompt,
        tools=tools,
        memory=memory,
        voice=voice,
        workdir=workdir,
        planning=planning,
        workflow=workflow,
        workflow_spec=workflow_spec,
        runtime=runtime,
        mcp_servers=mcp_servers,
    )

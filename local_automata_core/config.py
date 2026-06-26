from __future__ import annotations

from dataclasses import dataclass

# 共有定数は中立な constants モジュールに集約（server/ と共有）。
# 互換のため config 名前空間からも参照できるよう再エクスポートする。
from local_llm_client import DEFAULT_MODEL
DEFAULT_VISION_MODEL = DEFAULT_MODEL  # 既定モデルはマルチモーダル（テキストと共通）


@dataclass
class Config:
    """エージェントの実行設定。値は agent.toml と既定から決まる（環境変数は使わない）。

    ローカルLLMは OpenAI 互換サーバー経由で叩く想定:
      - mlx_lm.server   (例: http://localhost:8799/v1)
      - llama.cpp server (例: http://localhost:8799/v1)
    """

    base_url: str = "http://localhost:8799/v1"
    model: str = DEFAULT_MODEL
    api_key: str = "not-needed"
    # 既定 0.0（貪欲・決定論的）。小型ローカルモデルが正確な JSON ツール呼び出しや
    # 画像評価を安定して出すための既定。揺らぎが欲しい用途は agent.toml で上書きする。
    temperature: float = 0.0
    max_steps: int = 50
    # 会話履歴がこのトークン数を超えたら古いやり取りを要約して圧縮する。
    # モデルの文脈窓（トークン単位）に対応。実トークナイザがあれば正確に、無ければ概算で数える。
    # 既定は Qwen3.6 の文脈窓（max_position_embeddings=262144=256K）に合わせ、
    # モデルのコンテキストを使い切るまで要約圧縮を遅らせる。長文脈ほどメモリ・速度負荷が
    # 上がるため、ローカル環境に合わせて agent.toml で小さめに上書きしてよい。
    max_context_tokens: int = 262144
    # 文脈に残す「画像を含むメッセージ」の最大数。これより古い画像は生データを外し、
    # プレースホルダに置き換える（毎ターンの画像再送による文脈圧迫を防ぐ）。
    # 0 で「常に最新ターンの画像も外す」、負（-1 等）で無効化（全画像を保持）。
    max_context_images: int = 2
    # 古い画像の扱い方:
    #   "recent"    … 直近 max_context_images 枚だけ残し、古い画像は単純に省略（既定）。
    #   "on_demand" … 古い画像に id を振り、view_image("img_N") で必要なものだけ呼び戻せる
    #                 ようにする（モデルがどの画像を使うか選べる）。順番は組み替えない。
    image_context_mode: str = "recent"
    # 1ツール結果の最大文字数。これを超える出力は中央（_execute）で切り詰める。先頭だけ
    # 残す素朴な方式と違い、先頭と末尾の両方を残して中央を省略する（エラーは末尾に出やすい）。
    # run_command・read_file・grep だけでなく MCP/subagent など全ツールに一律で効く安全網。
    # 既定は大きめにして read_file 等の通常利用を妨げず、暴走出力だけを抑える。0 以下で無効。
    tool_output_max_chars: int = 100_000
    # Qwen3 系などの思考(thinking)モード。既定で無効。True で有効化する
    enable_thinking: bool = False
    # 応答のストリーミング表示。false で非ストリーミング（ツールが不安定な場合の保険）
    stream: bool = True
    # 1応答あたりの最大生成トークン数。小さいとツール呼び出し（コード生成）が途中で切れる。
    # 既定は Qwen3.6 のコンテキスト長（max_position_embeddings=262144=256K）に合わせ、
    # 出力を人為的に切り詰めない。実際の上限はコンテキスト窓（入力＋出力）側で律速される。
    max_tokens: int = 262144
    # ツール呼び出し方式: "native"（API の function calling）/ "prompt"（仕様を注入し応答を解析）。
    # 既定は "prompt"（自作方式）。モデルの生出力をそのまま見られるため過程を確認しやすい。
    tool_mode: str = "prompt"
    # 詳細表示。True で「過程を全部出す」: prompt モードの生出力（ツール呼び出しの JSON も
    # 隠さない）、ツール呼び出しの全引数、ツール結果の全文を表示する。CLI・Web GUI とも既定で
    # 有効。簡潔な1行要約に戻したいときは agent.toml で verbose = false にする。
    verbose: bool = True
    # LLM 応答の読み取りタイムアウト秒数。None または 0 以下で無制限（ローカルの巨大
    # モデルは初回応答に時間がかかるため既定は無制限）。接続自体のタイムアウトは別途短い。
    request_timeout: float | None = None
    # run_command でのパッケージ導入を許可するか（既定は禁止＝勝手なインストールを防ぐ）
    allow_install: bool = False
    # MCP ツールが返した画像をモデルへ転送するか。vision 対応バックエンド
    # （mlx-vlm / router）のときだけ true にする。テキスト専用サーバーに画像を
    # 送るとエラーになるため既定は false。
    forward_tool_images: bool = False
    # デバッグ出力（LLM 呼び出しの所要時間・サイズ、各ステップ）を stderr に出す
    debug: bool = False

    @classmethod
    def load(cls, file: dict | None = None) -> "Config":
        """既定値に agent.toml 由来の設定（file）を重ねて Config を作る。

        file には model / temperature などを入れる（型は呼び出し側で検証済み想定）。
        環境変数は参照しない。CLI も参照しないため、実行時設定は agent.toml と既定のみ。
        """
        file = file or {}
        d = cls()

        def g(key: str) -> object:
            return file[key] if key in file else getattr(d, key)

        timeout = g("request_timeout")
        return cls(
            base_url=str(g("base_url")),
            model=str(g("model")),
            api_key=str(g("api_key")),
            temperature=float(g("temperature")),
            max_steps=int(g("max_steps")),
            max_context_tokens=int(g("max_context_tokens")),
            max_context_images=int(g("max_context_images")),
            image_context_mode=str(g("image_context_mode")),
            tool_output_max_chars=int(g("tool_output_max_chars")),
            enable_thinking=bool(g("enable_thinking")),
            stream=bool(g("stream")),
            max_tokens=int(g("max_tokens")),
            tool_mode=str(g("tool_mode")),
            verbose=bool(g("verbose")),
            request_timeout=None if timeout is None else float(timeout),
            allow_install=bool(g("allow_install")),
            forward_tool_images=bool(g("forward_tool_images")),
            debug=bool(g("debug")),
        )


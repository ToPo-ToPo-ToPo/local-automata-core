# local-automata-core

エージェント構築用の共有ツールキット（import 名 `local_automata_core`）。汎用ツール（filesystem / shell / memory / plan）・MCP 連携・設定ローダ・画像/PDF 変換・ワークフロー仕様パーサを提供する。**Agent ループ本体は含まない** — ループや合成は利用側が組む。

LLM 接続は [local-llm-client](https://github.com/ToPo-ToPo-ToPo/local-llm-client)、推論サーバーは [local-llm-server](https://github.com/ToPo-ToPo-ToPo/local-llm-server) に委ねる。

## インストール

```bash
uv add local-automata-core
```

## 使い方

```python
from local_automata_core import find_config_path, load_agent_config, default_registry

config = load_agent_config(find_config_path())  # agent.toml + AGENTS.md を読む
registry = default_registry()                   # 汎用ツールを workspace/ に構築
```

## 開発

```bash
uv sync --locked
uv run --locked pytest
```

## セキュリティ更新（0.13.3 / 2026-10-28）

0.13.2 は開発用ロックでは新しい依存を使っていたが、公開要件に古いクライアント・Python・
間接依存を許す範囲が残っていた。0.13.3 は PyPI の wheel / sdist にも修正済みの下限を設定する。

Python の最小版は各系列 **3.11.18 / 3.12.16 / 3.13.18 / 3.14.10 / 3.15.2**。
`Requires-Python` で未修正パッチを除外し、依存先クライアントの読み込み時にも検査する。
uv 0.13.6 は依存の Python 除外区間を無視する場合があるため、メタデータだけには頼らない。
開発環境は従来どおり `.venv` と通常の uv（0.13.6 以降）を使用する。
Python を替える場合は旧 `.venv` を退避してから `uv venv --python 3.13.18` で作成し、
`uv sync --locked` を実行する。PC 全体の既定設定・起動スクリプトは変更しない。

| パッケージ | 公開要件の下限 | ロック・実行版 | 主な理由 |
| --- | --- | --- | --- |
| local-llm-client | 0.10.3 | 0.10.3 | Python・CA・SDK の間接依存にも安全な下限を設定 |
| MCP | 2.2.0 | 2.3.0 | HTTP ヘッダーなどのセキュリティ修正 |
| certifi / httpx2 | 2026.10.19 / 2.12.0 | 2026.10.23 / 2.16.0 | CA 除外・HTTP framing・解凍処理の修正 |
| pypdf | 6.21.3 | 6.21.3 | 2026-10-27 公開の4件の修正を含む |
| PyMuPDF | 2.0 | 2.0 | CVE-2026-82035（旧 1 系では未修正） |
| PyYAML | 6.0 | 6.0.3 | 従来の下限を維持 |
| cryptography | 50.0.0 | 50.0.2 | 証明書制約・PKCS7 などの修正を含む |
| opentelemetry-api | 1.45.1 | 1.45.1 | CVE-2026-114603 の修正 |
| PyJWT | 2.15.0 | 2.15.1 | JWK / JWT 検証などの修正を含む |
| python-multipart | 0.0.31 | 0.0.32 | multipart 処理の DoS 修正 |
| Starlette | 1.3.1 | 1.10.0 | HTTP/2 Cookie ヘッダー処理の修正 |
| ONNX Runtime | 1.18.0 | 1.31.1 | PyMuPDF 2 同梱モデルの IR 10 をサポート |
| mpmath | 1.3.0 | 1.4.1 | 旧 ONNX Runtime → SymPy 経由の ReDoS を除外 |
| Protobuf | 6.33.5 | 7.36.2 | 途中の 6 系にも再帰処理の脆弱版があるため、この版以降に限定 |
| setuptools（実行・ビルド） | 84.1.0 | 84.2.1 | 旧 ONNX Runtime の間接依存と配布ビルドに安全な下限を設定 |

クライアント 0.10.3 は AnyIO>=4.14.2、httpx>=0.28.1、httpcore>=1.0.9、
httpx2>=2.12.0、tqdm>=4.66.3、typing-extensions>=4.12.2、修正済み Pydantic も要求する。
間接依存を経由して古い版が残らないよう、直接利用しないパッケージにも下限を置いている。
依存を旧版で固定している利用側は再解決が必要。PyMuPDF の公開下限は 1 系から 2 系へ変わるが、
0.13.2 の開発ロックは既に 2.0 であり、PDF のテキスト抽出・画像化 API を含む既存テストは維持した。

### 検証結果と範囲

macOS / Apple Silicon で、通常環境は Python 3.13.18（pyenv）/ OpenSSL 3.6.2 を使用。
`uv run --locked` から更新後の `.venv` の依存構成が使われることを確認した。
既存 105 テスト（実 MCP の stdio・Streamable HTTP、PDF の画像化など）は成功。
取り込んだ Windows 対応の専用 2 テストは macOS ではスキップした。
配布 wheel を空の環境へ入れ、Python 3.11.18 / 3.13.18 の最小依存構成でも各 105 テストが成功した。
core の LLMClient（OpenAI 3.39.0）から既存のローカル LLM に接続し、
通常応答「2」とストリーミング応答「東京」も取得した。

`scripts/check-tls.py` は urllib・LLMClient の SDK・MCP の既定非同期 HTTP クライアントを検査する。
各実装で正常対照 `https://julialang.org/` を先に確認し、問題の 12 チェーンと No CT 2 件を検査する。
認証情報を送らず、リダイレクトを追わず、TLS 検証は有効にする。取得資料のスクリプトは実行しない。
受け入れた場合は公表されたリーフ・中間証明書の SHA-256 指紋を照合する。
通常環境と最小構成（Python 3.11.18 / 3.13.18）で、各実装の正常対照成功と 14/14 件の
証明書拒否を確認した。拒否理由は証明書チェーン不信頼・侵害鍵拒否・macOS の失効拒否で、
期限切れや CT 要件だけの拒否ではない。
期限切れ・ホスト名不一致・CT 要件だけの拒否、DNS・タイムアウトは侵害 CA 拒否の確認に含めない。

```bash
uv run --locked python scripts/check-tls.py --output .tools/validation/tls.json
```

ロック環境と最小構成の既知の依存を PyPI の脆弱性情報へ照合し、該当・取得失敗とも 0 件を確認。
pypdf / PyMuPDF は情報反映の遅れがあったため、公式のリリース・脆弱性告知でも補完した。
配布物による依存解決で修正前の 15 構成を拒否し、未修正 Python の 5 系列では pip がインストールを
拒否、実際の旧インタープリターからの import も通信開始前に停止することを確認した。
検査は実施時点の公表情報の範囲。CI・Docker 設定はなく未実行、Windows/Linux、
Python 3.12/3.14/3.15 の機能動作、独自 CA・独自 TLS・PyMuPDF 内部の外部通信は未確認。

ログは `.tools/validation/dependency-security-20261028/` に保存（Git・配布対象外）。
`core-current-versions.json`、`core-current-tests.log`、`core-min-py311-tests.log`、
`core-current-tls.json`、`core-min-tls.json`、`public-resolver-results.json` に実体・理由を記録する。

根拠:

- [Python 各系列の修正版](https://blog.python.org/2026/10/python-3152-31410-31318-31216-31118/)
- [MCP の修正](https://github.com/modelcontextprotocol/python-sdk/security/advisories/GHSA-84m7-p3x7-pcfv)
- [pypdf の修正一覧](https://github.com/py-pdf/pypdf/security/advisories)
- [PyMuPDF 2.0 の変更](https://pymupdf.readthedocs.io/en/latest/changes.html)
- [cryptography の修正](https://github.com/pyca/cryptography/security/advisories/GHSA-g6cj-pr64-35w5)
- [ONNX Runtime と IR の互換性](https://onnxruntime.ai/docs/reference/compatibility.html)
- [setuptools の修正](https://github.com/pypa/setuptools/security/advisories/GHSA-xmj3-9gf7-h52w)
- [uv の Python 要件の扱い](https://docs.astral.sh/uv/reference/resolver-internals/#requires-python)

## ライセンス

Apache-2.0

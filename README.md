# local-automata-core (workspace)

エージェント共有コア（L3）の uv workspace。

| パッケージ | 層 | 役割 |
|---|---|---|
| [`local-automata-core`](packages/local-automata-core) | L3 | ツールフレームワーク・共通ツール・MCP・エージェントループ・文脈管理（import 名 `local_automata_core`） |

L2 のゲートウェイ接続クライアントは独立リポジトリ [local-llm-client](https://github.com/ToPo-ToPo-ToPo/local-llm-client)
へ分離し、PyPI から依存する（`local-llm-client>=0.3.0`）。推論サーバー（ゲートウェイ）も別リポジトリ
[local-llm-server](https://github.com/ToPo-ToPo-ToPo/local-llm-server)。

## 開発

```bash
uv sync            # 全メンバーを editable で同期（クロス依存も自動）
uv run pytest      # 全パッケージのテスト
```

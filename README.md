# agent-core (workspace)

エージェント共有コアの uv workspace（モノレポ）。2 パッケージ:

| パッケージ | 層 | 役割 |
|---|---|---|
| [`local-llm-client`](packages/local-llm-client) | L2 | ゲートウェイ接続クライアント（respond/stream/画像/tool-calling 整形・解析） |
| [`agent-core`](packages/agent-core) | L3 | ツールフレームワーク・共通ツール・MCP・エージェントループ・文脈管理 |

推論サーバー（ゲートウェイ）は別リポジトリ [local-llm-server](https://github.com/ToPo-ToPo-ToPo/local-llm-server)。

## 開発

```bash
uv sync            # 全メンバーを editable で同期（クロス依存も自動）
uv run pytest      # 全パッケージのテスト
```

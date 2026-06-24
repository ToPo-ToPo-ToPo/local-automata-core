"""検証用の実 MCP サーバー（Streamable HTTP）。

scripts/verify_streamable.py からサブプロセスとして起動し、url（リモート）接続の
Streamable HTTP トランスポートを本物の MCP スタック越しに確認するための最小サーバー。
host / port は引数（または環境変数）で受け取り、エンドポイントは既定の /mcp。

    python scripts/_verify_streamable_server.py <host> <port>
"""
import os
import sys

from mcp.server.fastmcp import FastMCP


def main() -> None:
    host = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MCP_HOST", "127.0.0.1")
    port = int(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MCP_PORT", "8765"))

    mcp = FastMCP("verify-streamable", host=host, port=port)

    @mcp.tool()
    def add(a: int, b: int) -> int:
        """2数を足す。"""
        return a + b

    @mcp.tool()
    def greet(name: str) -> str:
        """挨拶する。"""
        return f"Hello, {name}!"

    # Streamable HTTP トランスポートで起動（エンドポイントは既定の /mcp）。
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

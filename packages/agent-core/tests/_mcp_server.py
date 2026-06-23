"""テスト用の最小 MCP サーバー（stdio）。pytest からサブプロセスとして起動する。

先頭が _ なので pytest には収集されない。
"""
import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test-server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """2数を足す"""
    return a + b


@mcp.tool()
def greet(name: str) -> str:
    """挨拶する"""
    return f"Hello, {name}!"


@mcp.tool()
def whereami() -> str:
    """サーバープロセスの現在の作業ディレクトリを返す（cwd 検証用）。"""
    return os.getcwd()


if __name__ == "__main__":
    mcp.run()

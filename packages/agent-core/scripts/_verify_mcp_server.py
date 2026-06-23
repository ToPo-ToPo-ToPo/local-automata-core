"""検証用の実 MCP サーバー（stdio）。scripts/verify_mcp.py から起動する。

実サーバーの挙動（cwd・進捗通知・キャンセル）を本物の MCP スタック越しに
確認するための最小サーバー。テストの疑似サーバーとは別物。
"""
import asyncio
import os

from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("verify-server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """2数を足す。"""
    return a + b


@mcp.tool()
def whereami() -> str:
    """サーバープロセスの作業ディレクトリ（cwd 検証用）。"""
    return os.getcwd()


@mcp.tool()
async def slow(seconds: float, steps: int = 3, marker: str = "", ctx: Context = None) -> str:
    """seconds 秒かけて進捗を報告しながら待つ。

    キャンセルされたら marker パスへ目印を書く（実キャンセル到達の確認用）。
    """
    try:
        for i in range(steps):
            if ctx is not None:
                await ctx.report_progress(
                    progress=float(i), total=float(steps), message=f"step {i + 1}/{steps}"
                )
            await asyncio.sleep(seconds / steps)
        return "done"
    except asyncio.CancelledError:
        if marker:
            try:
                with open(marker, "w", encoding="utf-8") as fp:
                    fp.write("cancelled")
            except OSError:
                pass
        raise


if __name__ == "__main__":
    mcp.run()

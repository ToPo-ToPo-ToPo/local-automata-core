"""view_image ツール: 生成した画像をエージェント自身が“見る”ための入力。

run_command で作った画像（matplotlib のコンター図など）はディスクに保存されるだけで、
モデルには見えていない。view_image はそのファイルを vision 入力としてモデルへ渡し、
「図の見た目を確認してから次の処理に進む」ことを可能にする。vision 対応モデルが前提。

戻り値は文字列なので画像を直接返せない。そこで読み込んだ画像を host_agent の待避
キューに積み、次ターンのユーザー発話として流し込む（Agent._flush_pending）。
"""
from __future__ import annotations

from pathlib import Path

from ..images import pdf_to_image_urls, to_image_url
from .base import Tool
from .workspace import Workspace

VIEW_IMAGE_TOOL = "view_image"
READ_PDF_PAGES_TOOL = "read_pdf_pages"


def build_view_image_tool(ws: Workspace, host_agent) -> Tool:
    """view_image ツールを作る（ws の中だけを読み、host_agent に画像を積む）。"""

    def view_image(path: str | None = None, id: str | None = None) -> str:
        # id 指定: 文脈から外された過去の画像をレジストリから呼び戻す（on_demand モード）。
        if id:
            url = host_agent._image_registry.get(id)
            if not url:
                return (
                    f"Error: 画像 id '{id}' が見つかりません"
                    "（未登録、または既に文脈から外れて再取得できません）。"
                )
            host_agent._pending_user_parts.append(
                {"type": "image_url", "image_url": {"url": url}}
            )
            return f"画像 {id} を読み込みました。次の応答で内容を視覚的に確認できます。"
        if not path:
            return "Error: path（workspace 内の画像）か id（過去画像の参照）を指定してください。"
        root = ws.root
        if Path(path).is_absolute():
            return "Error: 絶対パスは使えません。workspace 相対で指定してください。"
        target = (root / path).resolve()
        if target != root and root not in target.parents:
            return f"Error: workspace の外は読めません: {path}"
        if not target.is_file():
            return f"Error: 画像が見つかりません: {path}"
        try:
            url = to_image_url(str(target))
        except Exception as exc:  # noqa: BLE001 - 失敗は文字列で返す
            return f"Error: 画像を読み込めませんでした: {exc}"
        host_agent._pending_user_parts.append(
            {"type": "image_url", "image_url": {"url": url}}
        )
        return (
            f"画像 {path} を読み込みました。次の応答で内容を視覚的に確認できます。"
            "図の見た目を確認したうえで次の処理を進めてください。"
        )

    return Tool(
        name=VIEW_IMAGE_TOOL,
        description=(
            "画像を読み込んで次の応答で視覚的に確認できるようにする。図が意図通りか確かめて"
            "から処理を続けたいときに使う。vision 対応モデルが必要。"
            "path に workspace 内の画像ファイル（自分で生成したグラフ・コンター図など）を指定するか、"
            "id に過去の画像参照（履歴で『画像 img_N は省略』と示されたもの）を指定して呼び戻す。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "確認したい画像のパス（workspace 相対）"},
                "id": {
                    "type": "string",
                    "description": "文脈から外された過去画像の参照（例: 'img_3'）。path の代わりに指定",
                },
            },
            "required": [],
        },
        func=view_image,
    )


def build_read_pdf_pages_tool(ws: Workspace, host_agent) -> Tool:
    """read_pdf_pages ツールを作る（ws 内の PDF をページ画像として読み込む）。

    PDF の添付はテキスト抽出されるが、図・表・数式・スキャン画像などはテキスト化で
    失われる。このツールは指定ページを画像にレンダリングして vision 入力として渡し、
    レイアウトや図を視覚的に確認できるようにする。view_image と同じく host_agent の
    待避キューに積み、次ターンのユーザー発話として流し込む。vision 対応モデルが前提。
    """

    def read_pdf_pages(path: str | None = None, pages: str | None = None) -> str:
        if not path:
            return "Error: path（workspace 内の PDF ファイル）を指定してください。"
        root = ws.root
        if Path(path).is_absolute():
            return "Error: 絶対パスは使えません。workspace 相対で指定してください。"
        target = (root / path).resolve()
        if target != root and root not in target.parents:
            return f"Error: workspace の外は読めません: {path}"
        if not target.is_file():
            return f"Error: PDF が見つかりません: {path}"
        if target.suffix.lower() != ".pdf":
            return f"Error: PDF ファイルではありません: {path}"
        try:
            urls, page_numbers = pdf_to_image_urls(str(target), pages)
        except Exception as exc:  # noqa: BLE001 - 失敗は文字列で返す
            return f"Error: PDF をレンダリングできませんでした: {exc}"
        for url in urls:
            host_agent._pending_user_parts.append(
                {"type": "image_url", "image_url": {"url": url}}
            )
        shown = ", ".join(str(p) for p in page_numbers)
        return (
            f"{path} のページ {shown}（計 {len(page_numbers)} ページ）を画像として読み込みました。"
            "次の応答で内容を視覚的に確認できます。"
        )

    return Tool(
        name=READ_PDF_PAGES_TOOL,
        description=(
            "PDF のページを画像としてレンダリングし、次の応答で視覚的に確認できるようにする。"
            "添付PDFのテキスト抽出では失われる図・表・数式・レイアウト・スキャン画像を見たいときに使う。"
            "vision 対応モデルが必要。path に workspace 内の PDF を指定し、pages で読むページを"
            "1 始まりで指定する（例: '1,3,5-8'。省略時は先頭から数ページ）。一度に読めるのは数ページまで。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "読みたい PDF のパス（workspace 相対）"},
                "pages": {
                    "type": "string",
                    "description": "読むページ（1 始まり、例: '1,3,5-8'）。省略時は先頭から数ページ",
                },
            },
            "required": ["path"],
        },
        func=read_pdf_pages,
    )

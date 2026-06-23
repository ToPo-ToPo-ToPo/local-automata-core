from __future__ import annotations

import base64
import mimetypes
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# 動画1本から抜き出すフレーム数の既定と上限（文脈長・コスト保護）。
DEFAULT_VIDEO_FRAMES = 8
MAX_VIDEO_FRAMES = 32


def _is_url(ref: str) -> bool:
    return ref.startswith(("http://", "https://", "data:"))


def to_image_url(ref: str) -> str:
    """画像参照（ローカルパス or URL）を OpenAI 互換の image_url 文字列に変換する。

    URL/データURI はそのまま、ローカルファイルは base64 のデータURIに変換する。
    """
    if _is_url(ref):
        return ref
    path = Path(ref)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {ref}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


# 添付テキストの最大文字数（超えたら末尾を省略。文脈長の保護）
MAX_ATTACHMENT_CHARS = 20000


def _extract_pdf_text(path: Path) -> str:
    try:
        import pypdf
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required to read PDF files (core dependency). Run `uv sync` to install dependencies."
        ) from exc
    reader = pypdf.PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


# PDF1ファイルから一度に画像化するページ数の既定と上限（文脈長・コスト保護）。
DEFAULT_PDF_PAGES = 4
MAX_PDF_PAGES = 16
# レンダリング解像度（dpi）。可読性とペイロードサイズの折衷。
PDF_RENDER_DPI = 150


def parse_page_spec(spec: str | None, page_count: int) -> list[int]:
    """ページ指定文字列を 0 始まりのページ番号リストへ変換する。

    "1,3,5" や "2-4"、"1,4-6" のような 1 始まりの指定を受け付ける。spec が空なら
    先頭から DEFAULT_PDF_PAGES ページ。範囲外は無視し、重複は除いて昇順で返す。
    """
    if not spec or not spec.strip():
        n = min(DEFAULT_PDF_PAGES, page_count)
        return list(range(n))
    pages: set[int] = set()
    for token in spec.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            lo_s, _, hi_s = token.partition("-")
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError as exc:
                raise ValueError(f"Invalid page range: '{token}'") from exc
            for p in range(lo, hi + 1):
                if 1 <= p <= page_count:
                    pages.add(p - 1)
        else:
            try:
                p = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid page number: '{token}'") from exc
            if 1 <= p <= page_count:
                pages.add(p - 1)
    return sorted(pages)


def pdf_to_image_urls(
    ref: str, pages: str | None = None, dpi: int = PDF_RENDER_DPI
) -> tuple[list[str], list[int]]:
    """PDF のページを画像にレンダリングして (image_url データURIのリスト, ページ番号) を返す。

    pages（"1,3,5-8" 形式の 1 始まり指定）があればそのページ、無ければ先頭から
    DEFAULT_PDF_PAGES ページ。一度に MAX_PDF_PAGES ページまで。pypdf はテキスト抽出
    専用でレンダリングできないため pymupdf を使う。図・表・スキャン等、テキスト抽出
    では失われる視覚情報を vision 対応モデルへ「画像」として渡すための関数。
    """
    path = Path(ref)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {ref}")
    try:
        import pymupdf  # type: ignore
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore  # 旧バージョンの import 名
        except ImportError as exc:
            raise RuntimeError(
                "pymupdf is required to render PDF pages as images. Run `uv sync` to install dependencies."
            ) from exc
    doc = pymupdf.open(str(path))
    try:
        page_count = doc.page_count
        indices = parse_page_spec(pages, page_count)
        if not indices:
            raise ValueError(
                f"No valid pages selected (PDF has {page_count} page(s)); check the 'pages' spec."
            )
        if len(indices) > MAX_PDF_PAGES:
            indices = indices[:MAX_PDF_PAGES]
        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        urls: list[str] = []
        for i in indices:
            pix = doc.load_page(i).get_pixmap(matrix=matrix)
            data = base64.b64encode(pix.tobytes("png")).decode("ascii")
            urls.append(f"data:image/png;base64,{data}")
        return urls, [i + 1 for i in indices]
    finally:
        doc.close()


def read_attachment_text(ref: str) -> tuple[str, str]:
    """添付ファイルを (ファイル名, テキスト) として読む。

    .pdf は pypdf でテキスト抽出、それ以外は UTF-8 テキストとして読む。
    大きすぎる場合は末尾を省略する。
    """
    path = Path(ref)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {ref}")
    if path.suffix.lower() == ".pdf":
        text = _extract_pdf_text(path)
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Cannot read as text (binary?): {ref}. "
                "Pass PDF files with the .pdf extension."
            ) from exc
    if len(text) > MAX_ATTACHMENT_CHARS:
        omitted = len(text) - MAX_ATTACHMENT_CHARS
        text = text[:MAX_ATTACHMENT_CHARS] + f"\n…({omitted} characters omitted)"
    return path.name, text


def _video_duration_seconds(path: Path) -> float | None:
    """ffprobe で動画の長さ（秒）を取得する。取れなければ None。"""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(proc.stdout.strip())
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def video_to_image_urls(
    ref: str, max_frames: int = DEFAULT_VIDEO_FRAMES, fps: float | None = None
) -> list[str]:
    """動画から代表フレームを抜き出し、image_url 用のデータURIのリストにして返す。

    ffmpeg でフレームを抽出する（要 ffmpeg）。fps 指定があればその間隔で、無ければ
    動画全体から max_frames 枚を均等サンプリングする（ffprobe で長さを取得。取れない
    場合は 1fps）。vision 対応モデルへ「複数画像」として渡して動画入力を代替する。
    """
    path = Path(ref)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {ref}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is required to extract video frames (install with `brew install ffmpeg` or similar)."
        )
    n = max(1, min(int(max_frames or DEFAULT_VIDEO_FRAMES), MAX_VIDEO_FRAMES))
    if fps and fps > 0:
        vf = f"fps={fps}"
    else:
        dur = _video_duration_seconds(path)
        vf = f"fps={n / dur:.6f}" if dur and dur > 0 else "fps=1"
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "f_%04d.jpg")
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-vf", vf, "-frames:v", str(n), "-y", out,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Video frame extraction timed out") from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to extract frames: {proc.stderr.strip()[:300]}"
            )
        frames = sorted(Path(tmp).glob("f_*.jpg"))
        if not frames:
            raise RuntimeError("Could not extract frames (the file may be empty or an unsupported format)")
        urls = []
        for f in frames[:n]:
            data = base64.b64encode(f.read_bytes()).decode("ascii")
            urls.append(f"data:image/jpeg;base64,{data}")
        return urls


def build_user_content(
    text: str,
    images: list[str] | None = None,
    files: list[str] | None = None,
    videos: list[str] | None = None,
) -> Any:
    """ユーザー発話の content を組み立てる。

    files（PDF/テキスト）は抽出してテキスト本文に連結する。videos は ffmpeg で
    代表フレームを抜き出して画像として扱う。images/動画フレームがあれば OpenAI 互換
    のパーツ配列（text ＋ image_url）を返し、無ければ文字列を返す。
    """
    images = list(images or [])
    files = files or []
    videos = videos or []
    body = text
    for ref in files:
        name, content = read_attachment_text(ref)
        body += f"\n\n[Attachment: {name}]\n{content}"
    for vref in videos:
        frames = video_to_image_urls(vref)
        body += f"\n\n[{len(frames)} frames from video {Path(vref).name} (in chronological order)]"
        images.extend(frames)
    if not images:
        return body
    parts: list[dict[str, Any]] = [{"type": "text", "text": body}]
    for ref in images:
        parts.append({"type": "image_url", "image_url": {"url": to_image_url(ref)}})
    return parts

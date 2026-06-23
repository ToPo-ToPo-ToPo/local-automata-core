import io

import pytest

from agent_core.images import (
    build_user_content,
    parse_page_spec,
    pdf_to_image_urls,
    read_attachment_text,
    to_image_url,
)
from agent_core.spinner import Spinner


def _make_pdf(path, pages):
    """テスト用の PDF を pages ページ分つくる。"""
    import pymupdf

    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"Page {i + 1}")
    doc.save(str(path))
    doc.close()


def test_to_image_url_data_uri(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 10)
    uri = to_image_url(str(p))
    assert uri.startswith("data:image/png;base64,")


def test_to_image_url_passthrough():
    assert to_image_url("https://example.com/x.png") == "https://example.com/x.png"


def test_read_attachment_text(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    name, text = read_attachment_text(str(p))
    assert name == "data.csv" and "a,b" in text


def test_read_attachment_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_attachment_text(str(tmp_path / "no.txt"))


def test_read_attachment_binary(tmp_path):
    p = tmp_path / "b.dat"
    p.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(ValueError):
        read_attachment_text(str(p))


def test_build_user_content_text_and_files(tmp_path):
    p = tmp_path / "n.md"
    p.write_text("メモ本文", encoding="utf-8")
    content = build_user_content("質問", files=[str(p)])
    assert isinstance(content, str)
    assert "質問" in content and "[Attachment: n.md]" in content and "メモ本文" in content


def test_build_user_content_with_image(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n0")
    content = build_user_content("見て", images=[str(img)])
    assert isinstance(content, list)
    assert content[0]["type"] == "text" and content[1]["type"] == "image_url"


def test_build_user_content_with_video(tmp_path, monkeypatch):
    # 動画は代表フレームに展開して画像として扱う（ffmpeg はモックする）。
    monkeypatch.setattr(
        "agent_core.images.video_to_image_urls",
        lambda ref, **k: ["data:image/jpeg;base64,AAAA", "data:image/jpeg;base64,BBBB"],
    )
    content = build_user_content("動画を見て", videos=["clip.mp4"])
    assert isinstance(content, list)
    assert content[0]["type"] == "text" and "2 frames" in content[0]["text"]
    imgs = [p for p in content if p["type"] == "image_url"]
    assert len(imgs) == 2


def test_video_to_image_urls_requires_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_core.images.shutil.which", lambda _name: None)
    v = tmp_path / "v.mp4"
    v.write_bytes(b"\x00\x00")
    with pytest.raises(RuntimeError, match="ffmpeg"):
        from agent_core.images import video_to_image_urls

        video_to_image_urls(str(v))


def test_parse_page_spec():
    assert parse_page_spec(None, 10) == [0, 1, 2, 3]  # 既定は先頭から4ページ
    assert parse_page_spec("", 10) == [0, 1, 2, 3]
    assert parse_page_spec("1,3,5", 10) == [0, 2, 4]
    assert parse_page_spec("2-4", 10) == [1, 2, 3]
    assert parse_page_spec("1,4-6", 10) == [0, 3, 4, 5]
    assert parse_page_spec("3,3,3", 10) == [2]  # 重複は除く
    assert parse_page_spec("5-100", 3) == []  # 範囲外は無視
    assert parse_page_spec("1,9", 2) == [0]  # ページ数を超える指定は落とす


def test_parse_page_spec_invalid():
    with pytest.raises(ValueError):
        parse_page_spec("abc", 5)
    with pytest.raises(ValueError):
        parse_page_spec("1-x", 5)


def test_pdf_to_image_urls_default(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, 3)
    urls, nums = pdf_to_image_urls(str(pdf))
    assert nums == [1, 2, 3]  # 全3ページ（既定上限4以内）
    assert all(u.startswith("data:image/png;base64,") for u in urls)
    import base64

    assert base64.b64decode(urls[0].split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"


def test_pdf_to_image_urls_page_spec(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, 5)
    urls, nums = pdf_to_image_urls(str(pdf), "1,3")
    assert nums == [1, 3] and len(urls) == 2


def test_pdf_to_image_urls_empty_selection(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, 3)
    with pytest.raises(ValueError, match="No valid pages"):
        pdf_to_image_urls(str(pdf), "9-20")


def test_pdf_to_image_urls_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        pdf_to_image_urls(str(tmp_path / "nope.pdf"))


def test_spinner_inert_on_non_tty():
    s = Spinner("x", stream=io.StringIO())
    s.start()
    s.stop()  # 例外が出ないこと（非TTYでは何もしない）


def test_spinner_global_disable():
    # GUI 用に set_enabled(False) で全スピナーを無効化できる（TTY でも描かない）。
    from agent_core.spinner import set_enabled

    class _TTY(io.StringIO):
        def isatty(self):
            return True

    try:
        set_enabled(True)
        assert Spinner("x", stream=_TTY())._enabled is True
        set_enabled(False)
        assert Spinner("x", stream=_TTY())._enabled is False  # TTY でも無効
    finally:
        set_enabled(True)  # 後続テストに影響しないよう戻す

import io

import pytest
from ebooklib import epub

from podmemory.services.text_extract import extract_from_epub


def make_book(spine=None, content=None):
    book = epub.EpubBook()
    book.set_identifier("synthetic-order")
    book.set_title("Synthetic order")
    book.set_language("en")
    book.add_author("Synthetic author")
    chapters = {}
    for key in ["third", "first", "second", "unused", "auxiliary"]:
        chapter = epub.EpubHtml(title=key, file_name=key + ".xhtml", uid=key)
        chapter.content = f"<h1>{key}</h1><p>{key} learning content repeated for a meaningful synthetic chapter.</p>"
        if content and key in content:
            chapter.content = content[key]
        book.add_item(chapter)
        chapters[key] = chapter
    book.add_item(epub.EpubItem(uid="image", file_name="image.jpg", media_type="image/jpeg", content=b"synthetic"))
    book.spine = spine if spine is not None else [
        ("first", "yes"), ("auxiliary", "no"), ("second", "yes"), ("third", "yes")
    ]
    book.toc = [chapters[key] for key in ["first", "second", "third"]]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    output = io.BytesIO()
    epub.write_epub(output, book)
    return output.getvalue()


@pytest.mark.asyncio
async def test_primary_chapters_follow_spine_not_manifest():
    result = await extract_from_epub(make_book())
    assert [chapter["title"] for chapter in result.chapters] == ["first", "second", "third"]
    assert result.text.index("first") < result.text.index("second") < result.text.index("third")
    assert "unused" not in result.text
    assert "auxiliary" not in result.text
    assert result.title == "Synthetic order"
    assert result.author == "Synthetic author"
    assert result.word_count == len(result.text.split())


@pytest.mark.asyncio
@pytest.mark.parametrize("spine, expected", [
    ([("second", "yes"), ("first", "yes")], ["second", "first"]),
    ([("first", "yes"), ("missing", "yes"), ("image", "yes"), ("third", "yes")], ["first", "third"]),
    ([("first", "yes"), ("auxiliary", "yes"), ("second", "yes")], ["first", "auxiliary", "second"]),
])
async def test_selected_spine_order_and_document_filter(spine, expected):
    result = await extract_from_epub(make_book(spine))
    assert [chapter["title"] for chapter in result.chapters] == expected
    assert result.text == "\n\n".join(chapter["text"] for chapter in result.chapters)


@pytest.mark.asyncio
@pytest.mark.parametrize("spine", [[], [("missing", "yes")], [("first", "no")]])
async def test_no_primary_text_is_not_replaced_with_manifest_content(spine):
    with pytest.raises(ValueError, match="EPUB contains no meaningful text"):
        await extract_from_epub(make_book(spine))


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["<p></p>", "<p>short</p>"])
async def test_empty_or_short_primary_chapters_remain_rejected(body):
    content = {key: body for key in ["first", "second", "third"]}
    with pytest.raises(ValueError, match="EPUB contains no meaningful text"):
        await extract_from_epub(make_book(content=content))


@pytest.mark.asyncio
async def test_short_chapter_is_filtered_without_reordering_others():
    result = await extract_from_epub(make_book(content={"second": "<p>short</p>"}))
    assert [chapter["title"] for chapter in result.chapters] == ["first", "third"]

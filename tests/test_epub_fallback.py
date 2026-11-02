import io
from xml.sax.saxutils import quoteattr
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import pytest

from podmemory.services.text_extract import extract_from_epub


CASES = [
    ({"foreign": "a"}, "yes", ["A", "B"]),
    ({"foreign": "middle", "middle": "a"}, "yes", ["A", "B"]),
    ({"foreign": "foreign"}, "yes", ["B"]),
    ({"foreign": "middle", "middle": "foreign"}, "yes", ["B"]),
    ({"foreign": "absent"}, "yes", ["B"]),
    ({}, "yes", ["B"]),
    ({"foreign": "a b"}, "yes", ["B"]),
    ({"foreign": "a"}, "no", ["B"]),
]

CASES.append(({
    "foreign": "hop0",
    **{f"hop{i}": f"hop{i+1}" for i in range(49)},
    "hop49": "a",
}, "yes", ["A", "B"]))


def fallback_book(fallbacks, linear="yes", variant="default"):
    output = io.BytesIO()
    root = "OPS/sub/book.opf"
    items = []
    uids = ["b", "a", "foreign", "middle"]
    uids += [key for key in fallbacks if key not in uids]
    foreign_extension = ".xhtml" if variant == "misleading_extension" else ".pdf"
    for uid in uids:
        html = uid in ["a", "b"]
        href = uid + (".xhtml" if html else foreign_extension if uid == "foreign" else ".pdf")
        media = "application/xhtml+xml" if html else "application/pdf"
        fallback = " fallback=" + quoteattr(fallbacks[uid]) if uid in fallbacks else ""
        items.append(f'<item id="{uid}" href="{href}" media-type="{media}"{fallback}/>')
    opf = f'''<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="id">synthetic</dc:identifier>
    <dc:title>Fallback book</dc:title><dc:language>en</dc:language></metadata>
    <manifest>{"".join(items)}<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/></manifest>
    <spine><itemref idref="foreign" linear="{linear}"/><itemref idref="b"/></spine></package>'''
    if variant == "direct":
        opf = opf.replace('idref="foreign"', 'idref="a"')
    if variant == "shared":
        opf = opf.replace('<itemref idref="b"/>', '<itemref idref="middle"/><itemref idref="b"/>')
    container = f'''<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0"><rootfiles>
    <rootfile full-path="{root}" media-type="application/oebps-package+xml"/></rootfiles></container>'''
    if variant == "rootfiles":
        container = container.replace('<rootfile full-path=', '<rootfile full-path="OPS/sub/other.opf" media-type="application/oebps-package+xml"/><rootfile full-path=', 1)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        archive.writestr(root, opf)
        if variant == "rootfiles":
            archive.writestr("OPS/sub/other.opf", opf.replace('fallback="a"', 'fallback="b"'))
        for key in ["a", "b"]:
            if key == "a" and variant in ["empty", "short"]:
                text = "tiny" if variant == "short" else ""
                body = f"<p>{text}</p>"
            else:
                body = f"<h1>{key.upper()}</h1><p>Meaningful synthetic chapter {key} text for extraction and service testing.</p>"
            archive.writestr(f"OPS/sub/{key}.xhtml", f'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>{key}</title></head><body>{body}</body></html>')
        for key in [uid for uid in uids if uid not in ["a", "b"]]:
            extension = foreign_extension if key == "foreign" else ".pdf"
            archive.writestr(f"OPS/sub/{key}{extension}", b"synthetic foreign resource")
        archive.writestr("OPS/sub/nav.xhtml", '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body><nav epub:type="toc"><ol><li><a href="foreign.pdf">A</a></li><li><a href="b.xhtml">B</a></li></ol></nav></body></html>')
    return output.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize("fallbacks, linear, titles", CASES)
async def test_spine_manifest_fallback_extraction(fallbacks, linear, titles):
    result = await extract_from_epub(fallback_book(fallbacks, linear))
    assert [chapter["title"] for chapter in result.chapters] == titles
    assert result.text == "\n\n".join(chapter["text"] for chapter in result.chapters)


@pytest.mark.parametrize("fallbacks, linear, titles", CASES)
def test_spine_manifest_fallback_upload(client, fallbacks, linear, titles):
    response = client.post("/api/extract-epub", files={"file": ("fallback.epub", fallback_book(fallbacks, linear), "application/epub+zip")})
    assert response.status_code == 200
    data = response.json()
    assert [chapter["title"] for chapter in data["chapters"]] == titles
    assert data["text"] == "\n\n".join(chapter["text"] for chapter in data["chapters"])


VARIANTS = [
    ({"foreign": "a", "a": "foreign"}, "direct", ["A", "B"]),
    ({"foreign": "a", "middle": "a"}, "shared", ["A", "A", "B"]),
    ({"foreign": "a"}, "rootfiles", ["A", "B"]),
    ({"foreign": "a"}, "misleading_extension", ["A", "B"]),
    ({"foreign": "a"}, "empty", ["B"]),
    ({"foreign": "a"}, "short", ["B"]),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("fallbacks, variant, titles", VARIANTS)
async def test_fallback_resolution_preserves_package_and_document_semantics(fallbacks, variant, titles):
    result = await extract_from_epub(fallback_book(fallbacks, variant=variant))
    assert [chapter["title"] for chapter in result.chapters] == titles


@pytest.mark.parametrize("fallbacks, variant, titles", VARIANTS)
def test_fallback_variant_upload(client, fallbacks, variant, titles):
    response = client.post("/api/extract-epub", files={"file": ("variant.epub", fallback_book(fallbacks, variant=variant), "application/epub+zip")})
    assert response.status_code == 200
    assert [chapter["title"] for chapter in response.json()["chapters"]] == titles

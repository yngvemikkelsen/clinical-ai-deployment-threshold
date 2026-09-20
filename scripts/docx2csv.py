"""Extract a DOCX table to CSV. Strips cell properties before reading text."""
import re, csv, sys, zipfile

def tables(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    out = []
    for tbl in re.findall(r"<w:tbl>.*?</w:tbl>", xml, re.S):
        rows = []
        for tr in re.findall(r"<w:tr[ >].*?</w:tr>", tbl, re.S):
            cells = []
            for tc in re.findall(r"<w:tc>.*?</w:tc>", tr, re.S):
                body = re.sub(r"<w:tcPr>.*?</w:tcPr>", "", tc, flags=re.S)
                txt = "".join(re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", body, re.S))
                txt = (txt.replace("&amp;", "&").replace("&lt;", "<")
                          .replace("&gt;", ">").replace("\u2212", "-"))
                cells.append(re.sub(r"\s+", " ", txt).strip())
            if cells:
                rows.append(cells)
        if rows:
            out.append(rows)
    return out

if __name__ == "__main__":
    src, idx = sys.argv[1], int(sys.argv[2])
    dest = sys.argv[3] if len(sys.argv) > 3 else None
    t = tables(src)[idx]
    if dest:
        with open(dest, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(t)
        print(f"wrote {dest}: {len(t)} rows x {len(t[0])} cols")
    else:
        for r in t[:3]:
            print(" | ".join(r)[:170])

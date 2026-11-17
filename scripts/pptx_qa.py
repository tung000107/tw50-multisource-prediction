"""Heuristic layout QA for the generated deck (no LibreOffice available):
- shapes outside the slide, text boxes whose estimated wrapped height exceeds the box, overlapping content shapes.
Char-width model: CJK ~1.0em, Latin ~0.52em, digits/space ~0.5em; line height 1.2em; 1pt = 1/72 in."""
import sys, re
from pptx import Presentation
from pptx.util import Emu
EMU = 914400
prs = Presentation(sys.argv[1]); SW, SH = prs.slide_width / EMU, prs.slide_height / EMU
def width_in(text, pt):
    w = 0
    for ch in text:
        w += (1.0 if ord(ch) > 0x2E80 else 0.55 if ch.isalpha() and ch.isupper() else 0.5) * pt / 72
    return w
issues = 0
for si, slide in enumerate(prs.slides, 1):
    boxes = []
    for sh in slide.shapes:
        if sh.left is None: continue
        x, y, w, h = sh.left / EMU, sh.top / EMU, sh.width / EMU, sh.height / EMU
        if x < -0.01 or y < -0.01 or x + w > SW + 0.01 or y + h > SH + 0.01:
            print(f"slide {si}: OUT OF SLIDE {sh.shape_type} '{(sh.text_frame.text[:30] if sh.has_text_frame else '')}' box=({x:.2f},{y:.2f},{w:.2f},{h:.2f})"); issues += 1
        if sh.has_text_frame and sh.text_frame.text.strip():
            lines = 0
            for p in sh.text_frame.paragraphs:
                t = "".join(r.text for r in p.runs); pt = max((r.font.size.pt for r in p.runs if r.font.size), default=12)
                avail = max(w - 0.1, 0.3); lines += max(1, -(-width_in(t, pt) // avail)) * pt * 1.25 / 72 + 0.02
            if lines > h + 0.08:
                print(f"slide {si}: TEXT OVERFLOW est {lines:.2f}in > box {h:.2f}in  '{sh.text_frame.text[:40]}'"); issues += 1
        kind = "pic" if sh.shape_type == 13 else "table" if sh.has_table else "text" if (sh.has_text_frame and sh.text_frame.text.strip()) else "shape"
        if kind != "shape": boxes.append((kind, x, y, w, h, (sh.text_frame.text[:25] if sh.has_text_frame else kind)))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            ox = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]); oy = min(a[2] + a[4], b[2] + b[4]) - max(a[2], b[2])
            if ox > 0.05 and oy > 0.05:
                print(f"slide {si}: OVERLAP {a[0]} '{a[5]}' x {b[0]} '{b[5]}' ({ox:.2f}x{oy:.2f}in)"); issues += 1
print("slides", len(prs.slides), "issues", issues)

# ---- table growth check: PowerPoint grows rows to fit text; estimate final table bottom ----
print("--- table growth check ---")
for si, slide in enumerate(prs.slides, 1):
    for sh in slide.shapes:
        if not sh.has_table: continue
        tbl = sh.table; colw = [c.width / EMU for c in tbl.columns]; total = 0
        for r in tbl.rows:
            need = r.height / EMU
            for ci, c in enumerate(r.cells):
                lines_h = 0.1
                for p in c.text_frame.paragraphs:
                    t = "".join(rr.text for rr in p.runs); pt = max((rr.font.size.pt for rr in p.runs if rr.font.size), default=10)
                    avail = max(colw[ci] - 0.15, 0.2); lines_h += max(1, -(-width_in(t, pt) // avail)) * pt * 1.22 / 72
                need = max(need, lines_h)
            total += need
        bottom = sh.top / EMU + total
        flag = "  <-- EXCEEDS 7.05in" if bottom > 7.05 else ""
        print(f"slide {si}: table est. bottom {bottom:.2f}in (declared {sh.top / EMU + sh.height / EMU:.2f}){flag}")

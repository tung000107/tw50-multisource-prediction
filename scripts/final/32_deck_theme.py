"""Theme assets for the final deck, in the style of the Progress Update (Canva): off-white background with soft lavender / yellow / pink
blobs, a cover and an end slide with a candlestick illustration. Output: figures/final/theme/{bg_main,bg_cover,bg_end,bg_section}.png"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
OUT = Path(__file__).resolve().parents[2] / "figures" / "final" / "theme"; OUT.mkdir(parents=True, exist_ok=True)
W, H = 2667, 1500  # 13.33 x 7.5 in at 200 dpi
BG = (246, 245, 249); LAV = (201, 184, 245); YEL = (246, 231, 170); PINK = (243, 201, 221); LAV2 = (215, 205, 248)

def blobs(spots, blur=170):
    base = Image.new("RGB", (W, H), BG)
    for (cx, cy, rx, ry, col, a) in spots:
        mask = Image.new("L", (W, H), 0); ImageDraw.Draw(mask).ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=a)
        mask = mask.filter(ImageFilter.GaussianBlur(blur)); base.paste(Image.new("RGB", (W, H), col), (0, 0), mask)
    return base

MAIN = [(2350, 150, 700, 420, LAV, 190), (2650, 600, 420, 520, LAV2, 150), (200, 1450, 620, 330, PINK, 150), (900, 1500, 500, 230, YEL, 110), (1250, 80, 380, 160, YEL, 70)]
blobs(MAIN).save(OUT / "bg_main.png")
blobs([(2300, 200, 900, 520, LAV, 200), (2650, 700, 480, 600, LAV2, 170), (150, 1400, 700, 380, PINK, 160), (950, 1500, 600, 260, YEL, 130)]).save(OUT / "bg_section.png")

def candles(img, x0, y0, n=22, step=52, seed=4, scale=1.0):
    d = ImageDraw.Draw(img, "RGBA"); rng = np.random.default_rng(seed); p = 0.0; path = []
    for i in range(n):
        o = p; c = p + rng.normal(0.35, 1.0); hi = max(o, c) + abs(rng.normal(0, 0.5)); lo = min(o, c) - abs(rng.normal(0, 0.5)); path.append((o, c, hi, lo)); p = c
    for i, (o, c, hi, lo) in enumerate(path):
        x = x0 + i * step * scale; up = c >= o; col = (168, 148, 240, 255) if up else (180, 174, 196, 255)
        y = lambda v: y0 - v * 38 * scale
        d.line((x, y(hi), x, y(lo)), fill=col, width=max(2, int(3 * scale)))
        top, bot = y(max(o, c)), y(min(o, c)); bot = max(bot, top + 6 * scale)
        d.rectangle((x - 12 * scale, top, x + 12 * scale, bot), fill=col)

def framed(name, spots, with_candles=True):
    img = blobs(spots); d = ImageDraw.Draw(img, "RGBA")
    if with_candles: candles(img, 1250, 1300, n=20, step=68, seed=7, scale=1.7)
    # frame: lavender -> orange, approximated by two rounded rectangles
    d.rounded_rectangle((70, 70, W - 70, H - 70), radius=40, outline=(170, 150, 235, 255), width=4)
    d.rounded_rectangle((70, 70, W - 70, H - 70), radius=40, outline=(232, 140, 90, 110), width=2)
    img.save(OUT / name)
framed("bg_cover.png", [(2250, 150, 800, 450, LAV, 190), (2600, 650, 450, 560, LAV2, 150), (100, 1300, 700, 400, PINK, 150), (300, 450, 500, 420, YEL, 110)])
framed("bg_end.png", [(2350, 250, 750, 450, LAV, 170), (150, 1200, 700, 420, PINK, 140), (350, 400, 520, 420, YEL, 120)])
print("theme ok", sorted(p.name for p in OUT.glob("*.png")))

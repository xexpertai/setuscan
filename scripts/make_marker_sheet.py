"""Build docs/marker-sheet.pdf: printable SetuScan scale markers + instructions.

Markers are drawn as vector squares (cell by cell from OpenCV's ArUco dictionary),
so the printed size is exact when printed at 100 %.

Usage: uv run python scripts/make_marker_sheet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from setuscan.marker import CARD_MARGIN, MARKER_SIZES_MM, render_marker  # noqa: E402

CELLS = 6  # 4x4 data bits + 1-cell black border on each side


def marker_bits(marker_id: int) -> np.ndarray:
    img = render_marker(marker_id, CELLS * 20)
    return img[10::20, 10::20] < 128  # True = black cell


def draw_card(c: canvas.Canvas, x: float, y: float, marker_id: int) -> float:
    side = MARKER_SIZES_MM[marker_id]
    m = CARD_MARGIN * side
    card = side + 2 * m
    c.setDash(3, 3)
    c.setLineWidth(0.4)
    c.setStrokeGray(0.55)
    c.rect(x * mm, y * mm, card * mm, card * mm)  # cut line
    c.setDash()
    bits = marker_bits(marker_id)
    cell = side / CELLS
    c.setFillGray(0)
    for r in range(CELLS):
        for k in range(CELLS):
            if bits[r, k]:
                c.rect((x + m + k * cell) * mm, (y + m + (CELLS - 1 - r) * cell) * mm, cell * mm, cell * mm,
                       stroke=0, fill=1)
    c.setFont("Helvetica", 6.5)
    c.setFillGray(0.3)
    c.drawString((x + 1.5) * mm, (y + 1.5) * mm, f"SetuScan id {marker_id} . {side:.0f} mm")
    c.setFillGray(0)
    return card


def ruler(c: canvas.Canvas, x: float, y: float, length: float = 100.0) -> None:
    c.setLineWidth(0.6)
    c.line(x * mm, y * mm, (x + length) * mm, y * mm)
    for i in range(int(length) + 1):
        h = 4 if i % 10 == 0 else (2.5 if i % 5 == 0 else 1.5)
        c.line((x + i) * mm, y * mm, (x + i) * mm, (y + h) * mm)
        if i % 10 == 0:
            c.setFont("Helvetica", 6)
            c.drawCentredString((x + i) * mm, (y + 5) * mm, str(i))
    c.setFont("Helvetica-Bold", 8)
    c.drawString((x + length + 3) * mm, y * mm, "must measure exactly 100 mm")


def main() -> None:
    out = ROOT / "docs" / "marker-sheet.pdf"
    out.parent.mkdir(exist_ok=True)
    c = canvas.Canvas(str(out), pagesize=A4)
    c.setTitle("SetuScan scale marker sheet")
    c.setAuthor("Harsha Shinde")
    W, H = A4[0] / mm, A4[1] / mm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(15 * mm, (H - 18) * mm, "SetuScan scale marker sheet")
    c.setFont("Helvetica", 9)
    c.drawString(15 * mm, (H - 24) * mm,
                 "Print on A4 at 100 % / \"Actual size\". Do NOT use \"Fit to page\". Then check the ruler with a real ruler.")
    ruler(c, 15, H - 36)

    big = 40 * (1 + 2 * CARD_MARGIN)
    small = 20 * (1 + 2 * CARD_MARGIN)
    y = H - 46 - big
    x = 15.0
    for mid in (0, 1):  # row 1: two 40 mm cards
        x += draw_card(c, x, y, mid) + 10
    y -= small + 8
    x = 15.0
    for mid in (2, 3):  # row 2: two 20 mm cards for close-ups
        x += draw_card(c, x, y, mid) + 10
    assert 2 * big + 10 + 15 < W - 10, "cards must fit the page width"

    lines = [
        ("How to photograph a crack", True),
        ("1. Cut out one card along the dashed line. Use a big card (40 mm) normally, a small one (20 mm) for close-ups.", False),
        ("2. Stick it FLAT on the same surface as the crack, 2-10 cm from it (masking tape on the corners). Do not cover the crack.", False),
        ("3. Hold the phone roughly parallel to the surface, about 20-40 cm away, so the card and the crack fill most of the frame.", False),
        ("4. Use even light. Avoid flash glare and harsh sun. Tap the screen to focus on the crack, hold steady.", False),
        ("5. For a later visit, put the card back near the same spot and shoot from about the same position.", False),
        ("6. Check the ruler above: if it is not 100 mm, the print was scaled and every measurement will be wrong.", False),
        ("Safety and privacy", True),
        ("Only photograph surfaces you can reach safely from the ground. Never climb or lean over edges for a photo.", False),
        ("Do not photograph people, vehicles' number plates, or private interiors without permission.", False),
        ("SetuScan is decision support. It does not certify that a structure is safe. A qualified engineer decides.", False),
        ("Markers: OpenCV ArUco dictionary DICT_4X4_50, ids 0-3. Sizes are the black square's outer edge.", False),
    ]
    ty = y - 14
    for text, bold in lines:
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 10.5 if bold else 9)
        if bold:
            ty -= 3
        c.drawString(15 * mm, ty * mm, text)
        ty -= 6.2
    c.setFont("Helvetica", 7)
    c.setFillGray(0.4)
    c.drawString(15 * mm, 10 * mm, "SetuScan . MIT licence . github.com/xexpertai (repository link added at submission)")
    c.showPage()
    c.save()
    print("wrote", out)


if __name__ == "__main__":
    main()

"""
Draw the digital-root transition table as a colour heatmap.

Run:  python heatmap.py        (writes transition_heatmap.svg; no extra packages needed)

Each cell counts how often a prime with one digital root is followed by a prime
with another, for the first 1,000 primes. Darker means more often. The counts
come straight from modular_model_step9.py.
"""
from pathlib import Path

from modular_model_step9 import (
    build_mod9_transition_counts,
    build_prime_dataframe,
    clean_dataframe_for_learning,
)

ROOTS = [1, 2, 4, 5, 7, 8]
CELL_W, CELL_H, GAP = 76, 52, 4
LEFT, TOP = 150, 96
WHITE, TEAL, INK, MUTED = (255, 255, 255), (29, 158, 117), "#1f2328", "#59636e"


def shade(fraction: float) -> str:
    """Blend white towards teal; capped so dark text stays readable on every cell."""
    t = 0.06 + 0.64 * fraction
    return "#" + "".join(f"{round(w + (c - w) * t):02x}" for w, c in zip(WHITE, TEAL))


def build_svg(counts: dict[tuple[int, int], int]) -> str:
    grid = [[counts.get((src, dst), 0) for dst in ROOTS] for src in ROOTS]
    peak = max(max(row) for row in grid)
    width = LEFT + 6 * (CELL_W + GAP) + 30
    legend_y = TOP + 6 * (CELL_H + GAP) + 28
    height = legend_y + 40
    font = 'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif"'
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">',
        "<title>Digital-root transitions of consecutive primes</title>",
        "<desc>Heatmap for the first 1,000 primes. A root is rarely followed by the same root; "
        "the most common transitions correspond to prime gaps of 2, 4 and 6.</desc>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{LEFT}" y="34" {font} font-size="17" font-weight="600" fill="{INK}">'
        "Which digital root follows which (first 1,000 primes)</text>",
        f'<text x="{LEFT}" y="62" {font} font-size="13" fill="{MUTED}">Next prime\'s digital root</text>',
        f'<text x="{LEFT - 16}" y="{TOP - 12}" {font} font-size="13" fill="{MUTED}" text-anchor="end">This prime\'s root</text>',
    ]
    for j, root in enumerate(ROOTS):
        x = LEFT + j * (CELL_W + GAP) + CELL_W / 2
        out.append(f'<text x="{x}" y="{TOP - 12}" {font} font-size="14" fill="{INK}" text-anchor="middle">{root}</text>')
    for i, src in enumerate(ROOTS):
        y = TOP + i * (CELL_H + GAP)
        out.append(f'<text x="{LEFT - 16}" y="{y + CELL_H / 2 + 5}" {font} font-size="14" fill="{INK}" text-anchor="end">{src}</text>')
        for j, value in enumerate(grid[i]):
            x = LEFT + j * (CELL_W + GAP)
            outline = f' stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="5 4"' if i == j else ""
            out.append(f'<rect x="{x}" y="{y}" width="{CELL_W}" height="{CELL_H}" rx="6" fill="{shade(value / peak)}"{outline}/>')
            out.append(f'<text x="{x + CELL_W / 2}" y="{y + CELL_H / 2 + 5}" {font} font-size="15" fill="{INK}" text-anchor="middle">{value}</text>')
    out.append(f'<text x="{LEFT}" y="{legend_y + 12}" {font} font-size="13" fill="{MUTED}">Fewer</text>')
    for k, fraction in enumerate((0.05, 0.35, 0.65, 1.0)):
        out.append(f'<rect x="{LEFT + 48 + k * 24}" y="{legend_y}" width="20" height="16" rx="3" fill="{shade(fraction)}"/>')
    out.append(f'<text x="{LEFT + 150}" y="{legend_y + 12}" {font} font-size="13" fill="{MUTED}">More</text>')
    out.append(f'<rect x="{LEFT + 220}" y="{legend_y}" width="20" height="16" rx="3" fill="#ffffff" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    out.append(f'<text x="{LEFT + 248}" y="{legend_y + 12}" {font} font-size="13" fill="{MUTED}">Same root repeats</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    rows = clean_dataframe_for_learning(build_prime_dataframe(1000))
    counts = build_mod9_transition_counts(rows)
    path = Path(__file__).parent / "transition_heatmap.svg"
    path.write_text(build_svg(counts), encoding="utf-8")
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()

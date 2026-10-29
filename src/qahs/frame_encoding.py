"""Text encodings of ARC-AGI-3 frames for language-model prompts.

ARC-AGI-3 already returns frames as integer grids (at most 64x64, values 0-15), so no
image decoding is needed. These functions only choose how compactly that grid is written.
"""

from __future__ import annotations

import json
from collections import Counter, deque

Frame = tuple[tuple[int, ...], ...]

FRAME_FORMATS = ("runs", "objects", "hex")
HEX_DIGITS = "0123456789abcdef"
MAX_OBJECTS = 120
MAX_DIFF_CELLS = 160


def _background(frame: Frame) -> int:
    return Counter(value for row in frame for value in row).most_common(1)[0][0]


def _header(frame: Frame, background: int) -> str:
    return f"shape={len(frame)}x{len(frame[0])}; coordinates use x=column,y=row; inferred background={background}"


def runs_summary(frame: Frame) -> str:
    """Color statistics plus every non-background horizontal run with exact coordinates."""
    if not frame:
        return "empty frame"
    width = len(frame[0])
    counts = Counter(value for row in frame for value in row)
    background = counts.most_common(1)[0][0]
    color_summary = []
    for value, count in sorted(counts.items()):
        locations = [(x, y) for y, row in enumerate(frame) for x, cell in enumerate(row) if cell == value]
        xs = [point[0] for point in locations]
        ys = [point[1] for point in locations]
        color_summary.append(
            f"color {value}: count={count}, bounds=x[{min(xs)}..{max(xs)}],y[{min(ys)}..{max(ys)}]"
        )

    rows = []
    for y, row in enumerate(frame):
        runs = []
        start = 0
        while start < width:
            value = row[start]
            end = start
            while end + 1 < width and row[end + 1] == value:
                end += 1
            if value != background:
                span = f"x{start}" if start == end else f"x{start}-{end}"
                runs.append(f"{span}={value}")
            start = end + 1
        if runs:
            rows.append(f"y{y}: " + ", ".join(runs))
    return (
        _header(frame, background)
        + "\nColor statistics:\n" + "\n".join(color_summary)
        + "\nNon-background horizontal runs (exact coordinates):\n"
        + ("\n".join(rows) if rows else "none")
    )


def find_objects(frame: Frame, background: int | None = None) -> list[dict]:
    """Return 4-connected same-color components, excluding the background."""
    if not frame:
        return []
    background = _background(frame) if background is None else background
    height, width = len(frame), len(frame[0])
    seen = [[False] * width for _ in range(height)]
    objects = []
    for y0 in range(height):
        for x0 in range(width):
            color = frame[y0][x0]
            if seen[y0][x0] or color == background:
                continue
            cells = []
            queue = deque([(x0, y0)])
            seen[y0][x0] = True
            while queue:
                x, y = queue.popleft()
                cells.append((x, y))
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < width and 0 <= ny < height and not seen[ny][nx] and frame[ny][nx] == color:
                        seen[ny][nx] = True
                        queue.append((nx, ny))
            xs = [x for x, _ in cells]
            ys = [y for _, y in cells]
            left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
            box_width, box_height = right - left + 1, bottom - top + 1
            if len(cells) == box_width * box_height:
                shape = "rect"
            elif box_width * box_height <= 36:
                filled = set(cells)
                shape = "/".join(
                    "".join("#" if (x, y) in filled else "." for x in range(left, right + 1))
                    for y in range(top, bottom + 1)
                )
            else:
                shape = f"fill={len(cells) / (box_width * box_height):.2f}"
            objects.append({"c": color, "bbox": [left, top, right, bottom], "n": len(cells), "shape": shape})
    return objects


def objects_summary(frame: Frame) -> str:
    """One compact JSON line per connected object; usually far shorter than run lists."""
    if not frame:
        return "empty frame"
    background = _background(frame)
    objects = find_objects(frame, background)
    lines = [json.dumps(item, separators=(",", ":")) for item in objects[:MAX_OBJECTS]]
    if len(objects) > MAX_OBJECTS:
        lines.append(f"... {len(objects) - MAX_OBJECTS} more objects omitted")
    return (
        _header(frame, background)
        + f"\nObjects ({len(objects)}; c=color, bbox=[x0,y0,x1,y1], n=cells, shape=rect|bitmap rows|fill ratio):\n"
        + ("\n".join(lines) if lines else "none")
    )


def hex_grid(frame: Frame) -> str:
    """Lossless grid: one hex digit per cell, one line per row."""
    if not frame:
        return "empty frame"
    return (
        _header(frame, _background(frame))
        + "\nRows y0..yN, one hex digit per cell from x0:\n"
        + "\n".join("".join(HEX_DIGITS[value] for value in row) for row in frame)
    )


def encode_frame(frame: Frame, fmt: str = "runs") -> str:
    if fmt == "runs":
        return runs_summary(frame)
    if fmt == "objects":
        return objects_summary(frame)
    if fmt == "hex":
        return hex_grid(frame)
    raise ValueError(f"Unknown frame format: {fmt}. Choose one of {FRAME_FORMATS}.")


def frame_diff(before: Frame, after: Frame) -> str:
    """Describe only what changed between two frames."""
    if not before or not after or len(before) != len(after) or len(before[0]) != len(after[0]):
        return "frame shape changed; full new frame:\n" + runs_summary(after)
    changes = [
        (x, y, old, new)
        for y, (old_row, new_row) in enumerate(zip(before, after))
        for x, (old, new) in enumerate(zip(old_row, new_row))
        if old != new
    ]
    if not changes:
        return "no cells changed"
    xs = [item[0] for item in changes]
    ys = [item[1] for item in changes]
    transitions = Counter((old, new) for _, _, old, new in changes)
    lines = [
        f"{len(changes)} cells changed within x[{min(xs)}..{max(xs)}],y[{min(ys)}..{max(ys)}]",
        "color transitions: " + ", ".join(f"{old}->{new}x{count}" for (old, new), count in sorted(transitions.items())),
    ]
    if len(changes) <= MAX_DIFF_CELLS:
        lines.append("cells (x,y:old->new): " + " ".join(f"{x},{y}:{old}->{new}" for x, y, old, new in changes))
    else:
        lines.append("too many cells to list individually; objects after the change:")
        lines.append(objects_summary(after))
    return "\n".join(lines)

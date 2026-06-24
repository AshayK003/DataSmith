"""Shared Lucide SVG icons for DataSmith UI.

All icons are Lucide (MIT-licensed), stroke-based, use currentColor.
Import from here instead of duplicating SVGs across pages.
"""


def _svg(path_d: str, size: int = 18) -> str:
    """Build a Lucide SVG icon string."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f'{path_d}</svg>'
    )


BRAND = _svg(
    '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 '
    '6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 '
    '7.94-7.94l-3.76 3.76z"/>',
    size=22,
)

HOME = _svg(
    '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
    '<polyline points="9 22 9 12 15 12 15 22"/>'
)

INFO = _svg(
    '<circle cx="12" cy="12" r="10"/>'
    '<path d="M12 16v-4"/>'
    '<path d="M12 8h.01"/>'
)

DATABASE = _svg(
    '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
    '<path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>'
    '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'
)

WAVES = _svg('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>')

DOWNLOAD = _svg(
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
    '<polyline points="7 10 12 15 17 10"/>'
    '<line x1="12" y1="15" x2="12" y2="3"/>'
)

SEARCH = _svg(
    '<circle cx="11" cy="11" r="8"/>'
    '<path d="m21 21-4.3-4.3"/>'
)

SPARKLES = _svg(
    '<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 '
    '1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 '
    '1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>'
    '<path d="M5 3v4"/>'
    '<path d="M19 17v4"/>'
    '<path d="M3 5h4"/>'
    '<path d="M17 19h4"/>',
)

FOLDER = _svg(
    '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 '
    '2-2h5l2 3h9a2 2 0 0 1 2 2z"/>'
)

REFRESH = _svg(
    '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>'
    '<path d="M21 3v5h-5"/>'
    '<path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>'
    '<path d="M8 16H3v5"/>',
)

SLIDERS = _svg(
    '<line x1="4" y1="21" x2="4" y2="14"/>'
    '<line x1="4" y1="10" x2="4" y2="3"/>'
    '<line x1="12" y1="21" x2="12" y2="12"/>'
    '<line x1="12" y1="8" x2="12" y2="3"/>'
    '<line x1="20" y1="21" x2="20" y2="16"/>'
    '<line x1="20" y1="12" x2="20" y2="3"/>'
    '<line x1="2" y1="14" x2="6" y2="14"/>'
    '<line x1="10" y1="8" x2="14" y2="8"/>'
    '<line x1="18" y1="16" x2="22" y2="16"/>',
)

CHECK = _svg('<path d="M20 6 9 17l-5-5"/>')

BAR_CHART = _svg(
    '<line x1="12" y1="20" x2="12" y2="10"/>'
    '<line x1="18" y1="20" x2="18" y2="4"/>'
    '<line x1="6" y1="20" x2="6" y2="16"/>',
)

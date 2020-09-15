from . import register_svg_render_function
from ...objects import Line
from ...materials import LineMaterial


@register_svg_render_function(Line, LineMaterial)
def line_renderer(wobject):
    """Render function capable of rendering lines."""

    geometry = wobject.geometry
    positions = geometry.positions.data

    style = "stroke:rgb(255,0,0);stroke-width:2"

    lines = []
    for i in range(len(positions) - 1):
        x1 = positions[i, 0]
        y1 = positions[i, 1]
        x2 = positions[i + 1, 0]
        y2 = positions[i + 1, 1]
        lines.append(
            f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' style='{style}' />"
        )

    return lines

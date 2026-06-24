from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


MATERIALS = {
    1: {"name": "SOI silicon", "key": "soi_si", "color": "#0f6686"},
    2: {"name": "BOX SiO2", "key": "box_sio2", "color": "#8dc7ff"},
    3: {"name": "Si handle", "key": "si_handle", "color": "#0f6686"},
    4: {"name": "Si prism", "key": "si_prism", "color": "#d7f0d3"},
    5: {"name": "Underfill", "key": "underfill", "color": "#eaa1dc"},
    6: {"name": "Handle SiO2", "key": "handle_sio2", "color": "#b7ddff"},
}


DEFAULT_PARAMETERS = {
    "units": "um",
    "mesh_size_um": 1.0,
    "soi_thickness_um": 3.0,
    "box_thickness_um": 1.0,
    "si_handle_thickness_below_second_etch_um": 50.0,
    "handle_oxide_thickness_um": 1.0,
    "first_facet_etch_depth_um": 10.0,
    "first_facet_si_ledge_width_um": 4.0,
    "second_facet_etch_depth_um": 33.5,
    "prism_bottom_clearance_over_sio2_cladding_um": 10.0,
    "prism_bottom_clearance_over_second_facet_trench_um": 21.0,
    "prism_pic_edge_clearance_um": 1.0,
    "left_platform_width_um": 80.0,
    "left_platform_right_x_um": 0.0,
    "center_platform_left_x_um": 41.7,
    "center_platform_width_um": 55.09,
    "right_platform_left_x_um": 230.0,
    "right_platform_width_um": 80.0,
    "prism_left_low_flat_width_um": 5.2,
    "prism_bottom_apex_height_um": 37.3,
    "prism_bottom_left_angle_deg": 45.0,
    "prism_bottom_right_angle_deg": 64.5,
    "right_drop_width_um": 10.83,
    "right_low_flat_width_um": 46.38,
    "right_ramp_angle_deg": 45.0,
    "prism_thickness_um": 100.0,
    "epoxy_cure_temperature_C": 200.0,
    "final_temperature_C": 25.0,
    "temperature_drop_C": -175.0,
    "underfill_young_mpa": 2500.0,
    "underfill_poisson": 0.35,
    "underfill_cte_ppm_per_C": 40.0,
}


def as_float(params: dict, key: str) -> float:
    return float(params[key])


def load_parameters(path: Path) -> dict:
    params = DEFAULT_PARAMETERS.copy()
    if path.exists():
        params.update(json.loads(path.read_text(encoding="utf-8")))
    return params


def platform_regions(params: dict) -> list[tuple[float, float]]:
    left_right = as_float(params, "left_platform_right_x_um")
    left_width = as_float(params, "left_platform_width_um")
    center_left = as_float(params, "center_platform_left_x_um")
    center_width = as_float(params, "center_platform_width_um")
    right_left = as_float(params, "right_platform_left_x_um")
    right_width = as_float(params, "right_platform_width_um")

    return [
        (left_right - left_width, left_right),
        (center_left, center_left + center_width),
        (right_left, right_left + right_width),
    ]


def stack_depths(params: dict) -> dict[str, float]:
    soi = as_float(params, "soi_thickness_um")
    box = as_float(params, "box_thickness_um")
    first = as_float(params, "first_facet_etch_depth_um")
    second = as_float(params, "second_facet_etch_depth_um")
    handle = as_float(params, "si_handle_thickness_below_second_etch_um")
    oxide = as_float(params, "handle_oxide_thickness_um")
    second_stop = -(first + second)
    handle_bottom = second_stop - handle

    return {
        "top": 0.0,
        "soi_bottom": -soi,
        "box_bottom": -(soi + box),
        "first_stop": -first,
        "second_stop": second_stop,
        "handle_bottom": handle_bottom,
        "stack_bottom": handle_bottom - oxide,
    }


def append_point(points: list[tuple[float, float]], point: tuple[float, float]) -> None:
    if points and abs(points[-1][0] - point[0]) < 1.0e-10 and abs(
        points[-1][1] - point[1]
    ) < 1.0e-10:
        return
    points.append(point)


def substrate_top_profile(params: dict) -> list[tuple[float, float]]:
    regions = platform_regions(params)
    depths = stack_depths(params)
    ledge = as_float(params, "first_facet_si_ledge_width_um")
    first = depths["first_stop"]
    second = depths["second_stop"]

    points: list[tuple[float, float]] = []
    append_point(points, (regions[0][0], 0.0))
    append_point(points, (regions[0][1], 0.0))

    for current, next_region in zip(regions[:-1], regions[1:]):
        x_right = current[1]
        x_next = next_region[0]
        gap = x_next - x_right
        if gap <= 0.0:
            continue

        left_ledge_end = min(x_right + ledge, x_next)
        right_ledge_start = max(x_next - ledge, left_ledge_end)

        append_point(points, (x_right, first))
        append_point(points, (left_ledge_end, first))

        if right_ledge_start > left_ledge_end:
            append_point(points, (left_ledge_end, second))
            append_point(points, (right_ledge_start, second))
            append_point(points, (right_ledge_start, first))

        append_point(points, (x_next, first))
        append_point(points, (x_next, 0.0))
        append_point(points, (next_region[1], 0.0))

    return points


def y_on_polyline(points: list[tuple[float, float]], x: float) -> float | None:
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        if abs(x1 - x0) < 1.0e-12:
            if abs(x - x0) < 1.0e-12:
                return max(y0, y1)
            continue

        low = min(x0, x1) - 1.0e-12
        high = max(x0, x1) + 1.0e-12
        if low <= x <= high:
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return None


def substrate_top_y(x: float, params: dict) -> float | None:
    return y_on_polyline(substrate_top_profile(params), x)


def prism_bottom_outline(params: dict) -> list[tuple[float, float]]:
    left_region, center_region, right_region = platform_regions(params)
    x_min = left_region[0]
    x_left_right = left_region[1]
    x_center_left = center_region[0]
    x_center_right = center_region[1]
    x_right_left = right_region[0]
    x_max = right_region[1]

    depths = stack_depths(params)
    high_y = as_float(params, "prism_bottom_clearance_over_sio2_cladding_um")
    low_y = depths["second_stop"] + as_float(
        params, "prism_bottom_clearance_over_second_facet_trench_um"
    )
    edge_clearance = as_float(params, "prism_pic_edge_clearance_um")
    ledge = as_float(params, "first_facet_si_ledge_width_um")
    left_flat_width = as_float(params, "prism_left_low_flat_width_um")
    apex_height = as_float(params, "prism_bottom_apex_height_um")
    left_angle = math.radians(as_float(params, "prism_bottom_left_angle_deg"))
    right_angle = math.radians(as_float(params, "prism_bottom_right_angle_deg"))
    right_drop_width = as_float(params, "right_drop_width_um")
    right_flat_width = as_float(params, "right_low_flat_width_um")
    right_ramp_angle = math.radians(as_float(params, "right_ramp_angle_deg"))

    left_high_end = x_left_right - edge_clearance
    ledge_end = x_left_right + ledge
    left_low_start = ledge_end + edge_clearance
    if low_y < depths["first_stop"]:
        threshold_y = depths["first_stop"] + edge_clearance
        t_max = (threshold_y - high_y) / (low_y - high_y)
        required_low_start = left_high_end + (ledge_end - left_high_end) / t_max
        left_low_start = max(left_low_start, required_low_start)
    left_flat_end = left_low_start + left_flat_width
    apex_y = high_y + apex_height
    apex_x = left_flat_end + (apex_y - low_y) / math.tan(left_angle)
    right_low_start = max(
        apex_x + (apex_y - low_y) / math.tan(right_angle),
        x_center_right + ledge + edge_clearance + right_drop_width,
    )
    right_high_start = x_right_left - edge_clearance
    right_ramp_dx = (high_y - low_y) / math.tan(right_ramp_angle)
    right_ramp_start = right_high_start - right_ramp_dx
    right_flat_start = right_ramp_start - right_flat_width

    points = [
        (x_min, high_y),
        (left_high_end, high_y),
        (left_low_start, low_y),
        (left_flat_end, low_y),
        (apex_x, apex_y),
        (right_low_start, low_y),
    ]
    if right_low_start < right_flat_start:
        points.append((right_flat_start, low_y))
    points.extend(
        [
            (right_ramp_start, low_y),
            (right_high_start, high_y),
            (x_max, high_y),
        ]
    )

    compacted: list[tuple[float, float]] = []
    for point in points:
        append_point(compacted, point)
    return compacted


def prism_polygon(params: dict) -> list[tuple[float, float]]:
    bottom = prism_bottom_outline(params)
    top_y = max(y for _x, y in bottom) + as_float(params, "prism_thickness_um")
    return bottom + [(bottom[-1][0], top_y), (bottom[0][0], top_y)]


def point_on_segment(
    x: float,
    y: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    tol: float = 1.0e-10,
) -> bool:
    cross = (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)
    if abs(cross) > tol:
        return False

    dot = (x - x0) * (x1 - x0) + (y - y0) * (y1 - y0)
    if dot < -tol:
        return False

    length_sq = (x1 - x0) ** 2 + (y1 - y0) ** 2
    return dot <= length_sq + tol


def point_in_polygon(x: float, y: float, vertices: list[tuple[float, float]]) -> bool:
    inside = False
    count = len(vertices)

    for ii in range(count):
        x0, y0 = vertices[ii]
        x1, y1 = vertices[(ii + 1) % count]
        if point_on_segment(x, y, x0, y0, x1, y1):
            return True

        intersects = (y0 > y) != (y1 > y)
        if intersects:
            x_cross = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x < x_cross:
                inside = not inside

    return inside


def material_at(x: float, y: float, params: dict) -> int:
    if point_in_polygon(x, y, prism_polygon(params)):
        return 4

    depths = stack_depths(params)
    surface_y = substrate_top_y(x, params)
    if surface_y is None:
        return 0

    if depths["stack_bottom"] <= y <= surface_y:
        if y < depths["handle_bottom"]:
            return 6

        if abs(surface_y) < 1.0e-9:
            if depths["soi_bottom"] <= y <= depths["top"]:
                return 1
            if depths["box_bottom"] <= y < depths["soi_bottom"]:
                return 2

        return 3

    prism_bottom_y = y_on_polyline(prism_bottom_outline(params), x)
    if prism_bottom_y is not None and surface_y <= y <= prism_bottom_y:
        return 5

    return 0


def build_axis(minimum: float, maximum: float, step: float, anchors: list[float]) -> list[float]:
    values = {round(minimum, 10), round(maximum, 10)}
    count = int(math.ceil((maximum - minimum) / step))
    for ii in range(count + 1):
        values.add(round(minimum + ii * step, 10))

    for anchor in anchors:
        if minimum <= anchor <= maximum:
            values.add(round(anchor, 10))

    ordered = sorted(values)
    compacted = [ordered[0]]
    for value in ordered[1:]:
        if abs(value - compacted[-1]) > 1.0e-9:
            compacted.append(value)
    return compacted


def geometry_bounds(params: dict) -> tuple[float, float, float, float]:
    regions = platform_regions(params)
    depths = stack_depths(params)
    bottom = prism_bottom_outline(params)
    thickness = as_float(params, "prism_thickness_um")
    x_min = min(x0 for x0, _x1 in regions)
    x_max = max(x1 for _x0, x1 in regions)
    y_min = depths["stack_bottom"]
    y_max = max(y for _x, y in bottom) + thickness
    return x_min, x_max, y_min, y_max


def prism_clearance_report(params: dict) -> dict[str, float]:
    regions = platform_regions(params)
    depths = stack_depths(params)
    bottom = prism_bottom_outline(params)
    ledge = as_float(params, "first_facet_si_ledge_width_um")
    edge_clearance = as_float(params, "prism_pic_edge_clearance_um")
    low_y = depths["second_stop"] + as_float(
        params, "prism_bottom_clearance_over_second_facet_trench_um"
    )
    high_y = as_float(params, "prism_bottom_clearance_over_sio2_cladding_um")
    top_y = max(y for _x, y in bottom) + as_float(params, "prism_thickness_um")
    center_left_edge_y = y_on_polyline([bottom[3], bottom[4]], regions[1][0])
    if center_left_edge_y is None:
        center_left_edge_y = float("nan")
    ledge_end_y = y_on_polyline([bottom[1], bottom[2]], regions[0][1] + ledge)
    if ledge_end_y is None:
        ledge_end_y = float("nan")
    center_right_edge_y = y_on_polyline([bottom[4], bottom[5]], regions[1][1])
    if center_right_edge_y is None:
        center_right_edge_y = float("nan")

    return {
        "requested_lateral_edge_clearance_um": edge_clearance,
        "low_prism_bottom_y_um": round(low_y, 6),
        "high_prism_bottom_y_um": round(high_y, 6),
        "flat_prism_top_y_um": round(top_y, 6),
        "low_prism_bottom_above_second_facet_trench_um": round(
            low_y - depths["second_stop"], 6
        ),
        "high_prism_bottom_above_cladding_um": round(high_y - depths["top"], 6),
        "left_high_point_from_pic_edge_um": round(regions[0][1] - bottom[1][0], 6),
        "left_low_section_starts_after_ledge_by_um": round(
            bottom[2][0] - (regions[0][1] + ledge), 6
        ),
        "left_incoming_edge_vertical_clearance_at_ledge_end_um": round(
            ledge_end_y - depths["first_stop"], 6
        ),
        "straight_left_edge_vertical_clearance_at_center_pic_edge_um": round(
            center_left_edge_y - depths["top"], 6
        ),
        "straight_right_edge_vertical_clearance_at_center_pic_edge_um": round(
            center_right_edge_y - depths["top"], 6
        ),
        "right_low_section_starts_after_ledge_by_um": round(
            bottom[5][0] - (regions[1][1] + ledge), 6
        ),
        "right_high_point_from_pic_edge_um": round(
            regions[2][0] - bottom[-2][0], 6
        ),
    }


def build_mesh(params: dict) -> tuple[list[tuple[float, float]], list[tuple[int, int, int, int]]]:
    step = as_float(params, "mesh_size_um")
    x_min, x_max, y_min, y_max = geometry_bounds(params)
    depths = stack_depths(params)
    substrate = substrate_top_profile(params)
    prism_bottom = prism_bottom_outline(params)
    thickness = as_float(params, "prism_thickness_um")

    x_anchors = [x_min, x_max]
    y_anchors = [
        y_min,
        y_max,
        depths["handle_bottom"],
        depths["second_stop"],
        depths["first_stop"],
        depths["box_bottom"],
        depths["soi_bottom"],
        depths["top"],
    ]

    for x0, x1 in platform_regions(params):
        x_anchors.extend([x0, x1])

    for x, y in substrate:
        x_anchors.append(x)
        y_anchors.append(y)

    for x, y in prism_bottom:
        x_anchors.append(x)
        y_anchors.extend([y, y + thickness])

    x_axis = build_axis(x_min, x_max, step, x_anchors)
    y_axis = build_axis(y_min, y_max, step, y_anchors)

    nodes: list[tuple[float, float]] = []
    node_index: dict[tuple[int, int], int] = {}
    triangles: list[tuple[int, int, int, int]] = []

    def get_node(ix: int, iy: int) -> int:
        key = (ix, iy)
        if key not in node_index:
            node_index[key] = len(nodes)
            nodes.append((x_axis[ix], y_axis[iy]))
        return node_index[key]

    for ix in range(len(x_axis) - 1):
        x_mid = 0.5 * (x_axis[ix] + x_axis[ix + 1])
        for iy in range(len(y_axis) - 1):
            y_mid = 0.5 * (y_axis[iy] + y_axis[iy + 1])
            material_id = material_at(x_mid, y_mid, params)
            if material_id == 0:
                continue

            n00 = get_node(ix, iy)
            n10 = get_node(ix + 1, iy)
            n11 = get_node(ix + 1, iy + 1)
            n01 = get_node(ix, iy + 1)
            triangles.append((n00, n10, n11, material_id))
            triangles.append((n00, n11, n01, material_id))

    return nodes, triangles


def triangle_area(
    p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]
) -> float:
    return 0.5 * abs(
        (p1[0] - p0[0]) * (p2[1] - p0[1])
        - (p2[0] - p0[0]) * (p1[1] - p0[1])
    )


def collect_boundary_edges(
    nodes: list[tuple[float, float]],
    triangles: list[tuple[int, int, int, int]],
    params: dict,
) -> list[tuple[int, int, int]]:
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)

    for n0, n1, n2, _material_id in triangles:
        for a, b in ((n0, n1), (n1, n2), (n2, n0)):
            edge_counts[tuple(sorted((a, b)))] += 1

    x_min, x_max, y_min, y_max = geometry_bounds(params)
    tol = 1.0e-8
    edges: list[tuple[int, int, int]] = []

    for (a, b), count in edge_counts.items():
        if count != 1:
            continue

        xa, ya = nodes[a]
        xb, yb = nodes[b]
        if abs(ya - y_min) < tol and abs(yb - y_min) < tol:
            ref = 101
        elif abs(xa - x_min) < tol and abs(xb - x_min) < tol:
            ref = 102
        elif abs(xa - x_max) < tol and abs(xb - x_max) < tol:
            ref = 103
        elif abs(ya - y_max) < tol and abs(yb - y_max) < tol:
            ref = 104
        else:
            ref = 100
        edges.append((a, b, ref))

    return edges


def write_medit_mesh(
    path: Path,
    nodes: list[tuple[float, float]],
    triangles: list[tuple[int, int, int, int]],
    edges: list[tuple[int, int, int]],
) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write("MeshVersionFormatted 1\n")
        stream.write("Dimension 2\n\n")
        stream.write("Vertices\n")
        stream.write(f"{len(nodes)}\n")
        for x, y in nodes:
            stream.write(f"{x:.10g} {y:.10g} 0\n")

        stream.write("\nTriangles\n")
        stream.write(f"{len(triangles)}\n")
        for n0, n1, n2, material_id in triangles:
            stream.write(f"{n0 + 1} {n1 + 1} {n2 + 1} {material_id}\n")

        stream.write("\nEdges\n")
        stream.write(f"{len(edges)}\n")
        for n0, n1, ref in edges:
            stream.write(f"{n0 + 1} {n1 + 1} {ref}\n")

        stream.write("\nEnd\n")


def write_vtk(
    path: Path,
    nodes: list[tuple[float, float]],
    triangles: list[tuple[int, int, int, int]],
) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# vtk DataFile Version 2.0\n")
        stream.write("PIC prism 2D geometry\n")
        stream.write("ASCII\n")
        stream.write("DATASET UNSTRUCTURED_GRID\n")
        stream.write(f"POINTS {len(nodes)} float\n")
        for x, y in nodes:
            stream.write(f"{x:.10g} {y:.10g} 0.0\n")

        stream.write(f"CELLS {len(triangles)} {len(triangles) * 4}\n")
        for n0, n1, n2, _material_id in triangles:
            stream.write(f"3 {n0} {n1} {n2}\n")

        stream.write(f"CELL_TYPES {len(triangles)}\n")
        for _triangle in triangles:
            stream.write("5\n")

        stream.write(f"CELL_DATA {len(triangles)}\n")
        stream.write("SCALARS material_id int 1\n")
        stream.write("LOOKUP_TABLE default\n")
        for _n0, _n1, _n2, material_id in triangles:
            stream.write(f"{material_id}\n")


def svg_point(
    x: float,
    y: float,
    x_min: float,
    y_max: float,
    scale: float,
    margin: float,
) -> tuple[float, float]:
    return (margin + (x - x_min) * scale, margin + (y_max - y) * scale)


def svg_points(
    points: list[tuple[float, float]],
    x_min: float,
    y_max: float,
    scale: float,
    margin: float,
) -> str:
    return " ".join(
        f"{sx:.3f},{sy:.3f}"
        for sx, sy in (svg_point(x, y, x_min, y_max, scale, margin) for x, y in points)
    )


def svg_rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    x_min: float,
    y_max: float,
    scale: float,
    margin: float,
    color: str,
    stroke: str = "#111111",
) -> str:
    sx0, sy0 = svg_point(x0, y1, x_min, y_max, scale, margin)
    sx1, sy1 = svg_point(x1, y0, x_min, y_max, scale, margin)
    return (
        f'<rect x="{sx0:.3f}" y="{sy0:.3f}" width="{sx1 - sx0:.3f}" '
        f'height="{sy1 - sy0:.3f}" fill="{color}" stroke="{stroke}" '
        'stroke-width="0.55"/>'
    )


def svg_line(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    x_min: float,
    y_max: float,
    scale: float,
    margin: float,
    color: str = "#ff5c00",
    width: float = 1.3,
    arrows: bool = True,
) -> str:
    sx0, sy0 = svg_point(x0, y0, x_min, y_max, scale, margin)
    sx1, sy1 = svg_point(x1, y1, x_min, y_max, scale, margin)
    marker = ' marker-start="url(#cd_arrow)" marker-end="url(#cd_arrow)"' if arrows else ""
    return (
        f'<line x1="{sx0:.3f}" y1="{sy0:.3f}" x2="{sx1:.3f}" y2="{sy1:.3f}" '
        f'stroke="{color}" stroke-width="{width}"{marker}/>'
    )


def svg_cd_text(
    x: float,
    y: float,
    label: str,
    x_min: float,
    y_max: float,
    scale: float,
    margin: float,
    size: int = 11,
    anchor: str = "middle",
) -> str:
    sx, sy = svg_point(x, y, x_min, y_max, scale, margin)
    return (
        f'<text x="{sx:.3f}" y="{sy:.3f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="700" fill="#b33100" '
        'paint-order="stroke" stroke="#fff8cc" stroke-width="3" '
        f'stroke-linejoin="round">{label}</text>'
    )


def svg_dim_h(
    x0: float,
    x1: float,
    y: float,
    label: str,
    x_min: float,
    y_max: float,
    scale: float,
    margin: float,
) -> list[str]:
    return [
        svg_line(x0, y, x1, y, x_min, y_max, scale, margin),
        svg_cd_text(0.5 * (x0 + x1), y + 2.2, label, x_min, y_max, scale, margin),
    ]


def svg_dim_v(
    x: float,
    y0: float,
    y1: float,
    label: str,
    x_min: float,
    y_max: float,
    scale: float,
    margin: float,
) -> list[str]:
    return [
        svg_line(x, y0, x, y1, x_min, y_max, scale, margin),
        svg_cd_text(x + 2.2, 0.5 * (y0 + y1), label, x_min, y_max, scale, margin, anchor="start"),
    ]


def svg_cd_overlay(
    params: dict,
    x_min: float,
    y_max: float,
    scale: float,
    margin: float,
) -> list[str]:
    depths = stack_depths(params)
    regions = platform_regions(params)
    substrate = substrate_top_profile(params)
    bottom = prism_bottom_outline(params)
    high_y = as_float(params, "prism_bottom_clearance_over_sio2_cladding_um")
    low_y = depths["second_stop"] + as_float(
        params, "prism_bottom_clearance_over_second_facet_trench_um"
    )
    top_y = max(y for _x, y in bottom) + as_float(params, "prism_thickness_um")
    ledge = as_float(params, "first_facet_si_ledge_width_um")
    edge_clearance = as_float(params, "prism_pic_edge_clearance_um")
    right_high_start = regions[2][0] - edge_clearance
    right_ramp_start = right_high_start - (high_y - low_y) / math.tan(
        math.radians(as_float(params, "right_ramp_angle_deg"))
    )
    right_flat_start = right_ramp_start - as_float(params, "right_low_flat_width_um")

    left_flat_end = bottom[3][0]
    apex_x, apex_y = bottom[4]
    right_low_start = bottom[5][0]
    right_drop_reference_x = regions[1][1] + ledge + edge_clearance

    def line_attrs(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        label: str,
        detail: str,
        value: float | None = None,
        css_class: str = "measure-line",
    ) -> str:
        sx0, sy0 = svg_point(x0, y0, x_min, y_max, scale, margin)
        sx1, sy1 = svg_point(x1, y1, x_min, y_max, scale, margin)
        measured = math.hypot(x1 - x0, y1 - y0) if value is None else value
        return (
            f'<line class="{css_class}" x1="{sx0:.3f}" y1="{sy0:.3f}" '
            f'x2="{sx1:.3f}" y2="{sy1:.3f}" data-label="{label}" '
            f'data-value="{measured:.6g} um" data-detail="{detail}"/>'
        )

    overlay: list[str] = [
        "<style>",
        ".measure-line{stroke:#ff5c00;stroke-width:2.2;stroke-opacity:.58;fill:none;cursor:pointer;pointer-events:stroke;}",
        ".measure-line:hover,.measure-line.active{stroke:#d71920;stroke-width:4;stroke-opacity:.95;}",
        ".geometry-edge{stroke-dasharray:5 4;}",
        ".cd-guide{stroke-dasharray:2 3;}",
        "#cd-panel rect{fill:#fff9df;stroke:#b33100;stroke-width:1;}",
        "#cd-title{font-size:13px;font-weight:700;fill:#202124;}",
        "#cd-value{font-size:20px;font-weight:700;fill:#b33100;}",
        "#cd-detail{font-size:11px;fill:#3c4043;}",
        "#cd-help{font-size:10px;fill:#5f6368;}",
        "</style>",
        '<g id="cd-highlights">',
        '<g id="cd-panel">',
        '<rect x="36" y="54" width="300" height="82" rx="4"/>',
        '<text id="cd-title" x="50" y="78">Click an orange edge or guide</text>',
        '<text id="cd-value" x="50" y="105">measurement</text>',
        '<text id="cd-detail" x="50" y="124">Values are in micrometers.</text>',
        '<text id="cd-help" x="50" y="146">Orange solid lines are prism edges; dashed lines are JSON CDs.</text>',
        "</g>",
    ]

    for index, (p0, p1) in enumerate(zip(bottom[:-1], bottom[1:]), start=1):
        overlay.append(
            line_attrs(
                p0[0],
                p0[1],
                p1[0],
                p1[1],
                f"Prism bottom edge {index}",
                f"Endpoint length from ({p0[0]:.3g}, {p0[1]:.3g}) to ({p1[0]:.3g}, {p1[1]:.3g}).",
                css_class="measure-line geometry-edge",
            )
        )

    for index, (p0, p1) in enumerate(zip(substrate[:-1], substrate[1:]), start=1):
        overlay.append(
            line_attrs(
                p0[0],
                p0[1],
                p1[0],
                p1[1],
                f"PIC topography edge {index}",
                f"Endpoint length from ({p0[0]:.3g}, {p0[1]:.3g}) to ({p1[0]:.3g}, {p1[1]:.3g}).",
                css_class="measure-line geometry-edge",
            )
        )

    cd_lines = [
        (regions[0][0], depths["stack_bottom"] + 7.0, regions[0][1], depths["stack_bottom"] + 7.0, "left_platform_width_um", "Left platform width from geometry_parameters.json.", as_float(params, "left_platform_width_um")),
        (regions[1][0], depths["stack_bottom"] + 13.0, regions[1][1], depths["stack_bottom"] + 13.0, "center_platform_width_um", "Center platform width from geometry_parameters.json.", as_float(params, "center_platform_width_um")),
        (regions[2][0], depths["stack_bottom"] + 7.0, regions[2][1], depths["stack_bottom"] + 7.0, "right_platform_width_um", "Right platform width from geometry_parameters.json.", as_float(params, "right_platform_width_um")),
        (regions[0][1], depths["stack_bottom"] + 20.0, regions[2][0], depths["stack_bottom"] + 20.0, "PIC edge span", "Distance from left platform right edge to right platform left edge.", regions[2][0] - regions[0][1]),
        (regions[0][1], depths["first_stop"] + 3.0, regions[0][1] + ledge, depths["first_stop"] + 3.0, "first_facet_si_ledge_width_um", "First facet lateral silicon ledge.", ledge),
        (bottom[2][0], low_y - 4.0, left_flat_end, low_y - 4.0, "prism_left_low_flat_width_um", "Left low prism underside flat.", as_float(params, "prism_left_low_flat_width_um")),
        (right_drop_reference_x, low_y - 8.5, right_drop_reference_x + as_float(params, "right_drop_width_um"), low_y - 8.5, "right_drop_width_um", "Minimum low-edge span after center PIC ledge before the right straight prism edge may land.", as_float(params, "right_drop_width_um")),
        (right_flat_start, low_y - 4.0, right_ramp_start, low_y - 4.0, "right_low_flat_width_um", "Right low prism underside flat.", as_float(params, "right_low_flat_width_um")),
        (regions[0][1] - edge_clearance, high_y + 5.0, regions[0][1], high_y + 5.0, "prism_pic_edge_clearance_um", "Minimum lateral clearance from PIC edge.", edge_clearance),
        (regions[0][0] + 10.0, depths["soi_bottom"], regions[0][0] + 10.0, depths["top"], "soi_thickness_um", "SOI thickness.", as_float(params, "soi_thickness_um")),
        (regions[0][0] + 18.0, depths["box_bottom"], regions[0][0] + 18.0, depths["soi_bottom"], "box_thickness_um", "BOX thickness.", as_float(params, "box_thickness_um")),
        (regions[0][1] + 9.0, depths["first_stop"], regions[0][1] + 9.0, depths["top"], "first_facet_etch_depth_um", "First etch depth from top surface.", as_float(params, "first_facet_etch_depth_um")),
        (regions[0][1] + 18.0, depths["second_stop"], regions[0][1] + 18.0, depths["first_stop"], "second_facet_etch_depth_um", "Second etch depth measured down from the first etch stop.", as_float(params, "second_facet_etch_depth_um")),
        (0.5 * (regions[1][1] + regions[2][0]), depths["handle_bottom"], 0.5 * (regions[1][1] + regions[2][0]), depths["second_stop"], "si_handle_thickness_below_second_etch_um", "Si handle thickness below second facet stop.", as_float(params, "si_handle_thickness_below_second_etch_um")),
        (regions[2][1] - 8.0, depths["stack_bottom"], regions[2][1] - 8.0, depths["handle_bottom"], "handle_oxide_thickness_um", "Bottom handle oxide thickness.", as_float(params, "handle_oxide_thickness_um")),
        (regions[1][0] + 10.0, depths["top"], regions[1][0] + 10.0, high_y, "prism_bottom_clearance_over_sio2_cladding_um", "Prism underside clearance above cladding/platform top.", as_float(params, "prism_bottom_clearance_over_sio2_cladding_um")),
        (0.5 * (right_flat_start + right_ramp_start), depths["second_stop"], 0.5 * (right_flat_start + right_ramp_start), low_y, "prism_bottom_clearance_over_second_facet_trench_um", "Prism underside clearance above the second facet trench stop.", as_float(params, "prism_bottom_clearance_over_second_facet_trench_um")),
        (apex_x + 12.0, apex_y, apex_x + 12.0, top_y, "prism_thickness_um", "Flat top prism thickness above the underside apex.", as_float(params, "prism_thickness_um")),
        (apex_x - 10.0, high_y, apex_x - 10.0, apex_y, "prism_bottom_apex_height_um", "Prism underside apex height above high underside.", as_float(params, "prism_bottom_apex_height_um")),
    ]

    for x0, y0, x1, y1, label, detail, value in cd_lines:
        overlay.append(
            line_attrs(
                x0,
                y0,
                x1,
                y1,
                label,
                detail,
                value=value,
                css_class="measure-line cd-guide",
            )
        )

    overlay.extend(
        [
            "<script><![CDATA[",
            "(function(){",
            "const title=document.getElementById('cd-title');",
            "const value=document.getElementById('cd-value');",
            "const detail=document.getElementById('cd-detail');",
            "const lines=[...document.querySelectorAll('.measure-line')];",
            "function activate(line){lines.forEach(el=>el.classList.remove('active'));line.classList.add('active');title.textContent=line.dataset.label;value.textContent=line.dataset.value;detail.textContent=line.dataset.detail;}",
            "lines.forEach(line=>line.addEventListener('click',()=>activate(line)));",
            "})();",
            "]]></script>",
        ]
    )
    overlay.append("</g>")
    return overlay


def write_svg(path: Path, params: dict, highlight_cds: bool = False) -> None:
    x_min, x_max, y_min, y_max = geometry_bounds(params)
    margin = 36.0
    canvas_width = 1220.0
    canvas_height = 760.0
    scale = min(
        (canvas_width - 2 * margin - 180.0) / (x_max - x_min),
        (canvas_height - 2 * margin) / (y_max - y_min),
    )

    depths = stack_depths(params)
    substrate = substrate_top_profile(params)
    prism_bottom = prism_bottom_outline(params)
    prism = prism_polygon(params)
    handle_poly = substrate + [
        (substrate[-1][0], depths["handle_bottom"]),
        (substrate[0][0], depths["handle_bottom"]),
    ]
    underfill_poly = prism_bottom + list(reversed(substrate))

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{canvas_width:.0f}" height="{canvas_height:.0f}" '
        f'viewBox="0 0 {canvas_width:.0f} {canvas_height:.0f}">',
        '<defs><marker id="cd_arrow" viewBox="0 0 10 10" refX="5" refY="5" '
        'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#ff5c00"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="24" font-size="16" font-weight="600" fill="#202124">'
        "2D PIC prism stress geometry</text>",
        '<text x="36" y="44" font-size="11" fill="#5f6368">'
        "Second etch is measured from the first etch stop; click orange guides in the CD SVG.</text>",
    ]

    elements.append(
        svg_rect(
            x_min,
            depths["stack_bottom"],
            x_max,
            depths["handle_bottom"],
            x_min,
            y_max,
            scale,
            margin,
            MATERIALS[6]["color"],
        )
    )
    elements.append(
        '<polygon points="'
        + svg_points(handle_poly, x_min, y_max, scale, margin)
        + f'" fill="{MATERIALS[3]["color"]}" stroke="#111111" stroke-width="0.55"/>'
    )
    elements.append(
        '<polygon points="'
        + svg_points(underfill_poly, x_min, y_max, scale, margin)
        + f'" fill="{MATERIALS[5]["color"]}" stroke="#111111" stroke-width="0.45"/>'
    )

    for x0, x1 in platform_regions(params):
        elements.append(
            svg_rect(
                x0,
                depths["box_bottom"],
                x1,
                depths["soi_bottom"],
                x_min,
                y_max,
                scale,
                margin,
                MATERIALS[2]["color"],
            )
        )
        elements.append(
            svg_rect(
                x0,
                depths["soi_bottom"],
                x1,
                depths["top"],
                x_min,
                y_max,
                scale,
                margin,
                MATERIALS[1]["color"],
            )
        )

    elements.append(
        '<polyline points="'
        + svg_points(substrate, x_min, y_max, scale, margin)
        + '" fill="none" stroke="#202124" stroke-width="1.1" '
        'stroke-linejoin="round"/>'
    )
    elements.append(
        '<polygon points="'
        + svg_points(prism, x_min, y_max, scale, margin)
        + f'" fill="{MATERIALS[4]["color"]}" fill-opacity="0.72" '
        'stroke="#477446" stroke-width="0.8"/>'
    )
    elements.append(
        '<polyline points="'
        + svg_points(prism_bottom, x_min, y_max, scale, margin)
        + '" fill="none" stroke="#050505" stroke-width="1.15" '
        'stroke-linejoin="round" stroke-linecap="round"/>'
    )

    if highlight_cds:
        elements.extend(svg_cd_overlay(params, x_min, y_max, scale, margin))

    legend_x = canvas_width - 160.0
    legend_y = 58.0
    elements.append(
        f'<text x="{legend_x:.1f}" y="{legend_y - 18:.1f}" font-size="12" '
        'font-weight="600" fill="#202124">Materials</text>'
    )
    for offset, material in enumerate(MATERIALS.values()):
        y = legend_y + 22.0 * offset
        elements.append(
            f'<rect x="{legend_x:.1f}" y="{y:.1f}" width="14" height="14" '
            f'fill="{material["color"]}" stroke="#202124" stroke-width="0.6"/>'
        )
        elements.append(
            f'<text x="{legend_x + 22.0:.1f}" y="{y + 11.5:.1f}" '
            f'font-size="11" fill="#202124">{material["name"]}</text>'
        )

    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_report(
    path: Path,
    params: dict,
    nodes: list[tuple[float, float]],
    triangles: list[tuple[int, int, int, int]],
    edges: list[tuple[int, int, int]],
) -> None:
    areas: dict[int, float] = defaultdict(float)
    for n0, n1, n2, material_id in triangles:
        areas[material_id] += triangle_area(nodes[n0], nodes[n1], nodes[n2])

    report = {
        "parameters": params,
        "stack_depths_um": {key: round(value, 6) for key, value in stack_depths(params).items()},
        "platform_regions_um": [
            {"x0": round(x0, 6), "x1": round(x1, 6)}
            for x0, x1 in platform_regions(params)
        ],
        "substrate_top_profile_um": [
            {"x": round(x, 6), "y": round(y, 6)}
            for x, y in substrate_top_profile(params)
        ],
        "prism_clearance_report_um": prism_clearance_report(params),
        "prism_bottom_outline_um": [
            {"x": round(x, 6), "y": round(y, 6)}
            for x, y in prism_bottom_outline(params)
        ],
        "mesh": {
            "node_count": len(nodes),
            "triangle_count": len(triangles),
            "boundary_edge_count": len(edges),
        },
        "materials": {
            MATERIALS[material_id]["key"]: {
                "name": MATERIALS[material_id]["name"],
                "material_id": material_id,
                "area_um2": round(area, 6),
            }
            for material_id, area in sorted(areas.items())
        },
        "boundary_refs": {
            "100": "free external/material boundary",
            "101": "bottom",
            "102": "left",
            "103": "right",
            "104": "top",
        },
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate the 2D PIC prism SfePy geometry.")
    parser.add_argument(
        "--parameters",
        type=Path,
        default=root / "input" / "geometry_parameters.json",
        help="Path to the geometry JSON parameters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "rslt",
        help="Directory for generated mesh and preview files.",
    )
    parser.add_argument(
        "--name",
        default="pic_prism_2d",
        help="Base name for generated output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_parameters(args.parameters)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nodes, triangles = build_mesh(params)
    edges = collect_boundary_edges(nodes, triangles, params)

    mesh_path = args.output_dir / f"{args.name}.mesh"
    vtk_path = args.output_dir / f"{args.name}.vtk"
    svg_path = args.output_dir / f"{args.name}.svg"
    cd_svg_path = args.output_dir / f"{args.name}_cd.svg"
    report_path = args.output_dir / f"{args.name}_report.json"

    write_medit_mesh(mesh_path, nodes, triangles, edges)
    write_vtk(vtk_path, nodes, triangles)
    write_svg(svg_path, params)
    write_svg(cd_svg_path, params, highlight_cds=True)
    write_report(report_path, params, nodes, triangles, edges)

    print(f"Wrote {mesh_path}")
    print(f"Wrote {vtk_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {cd_svg_path}")
    print(f"Wrote {report_path}")
    print(f"Nodes: {len(nodes)}")
    print(f"Triangles: {len(triangles)}")


if __name__ == "__main__":
    main()

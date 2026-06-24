from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "rslt" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation


POINT_FIELDS = [
    ("u_magnitude", "Displacement magnitude", "um", False),
    ("u_x", "Displacement X", "um", True),
    ("u_y", "Displacement Y", "um", True),
]

CELL_FIELDS = [
    ("strain_xx", "Strain xx", "dimensionless", True),
    ("strain_yy", "Strain yy", "dimensionless", True),
    ("strain_xy", "Strain xy", "dimensionless", True),
    ("stress_xx_mpa", "Stress xx", "MPa", True),
    ("stress_yy_mpa", "Stress yy", "MPa", True),
    ("stress_xy_mpa", "Stress xy", "MPa", True),
    ("si_prism_delta_n_x", "Si prism delta n x-pol", "dimensionless", True),
    ("si_prism_delta_n_y", "Si prism delta n y-pol", "dimensionless", True),
    ("si_prism_delta_n_z", "Si prism delta n z-pol", "dimensionless", True),
    (
        "si_prism_delta_n_x_minus_y",
        "Si prism delta n x - y",
        "dimensionless",
        True,
    ),
    (
        "si_prism_delta_n_principal_max",
        "Si prism principal delta n max",
        "dimensionless",
        True,
    ),
    (
        "si_prism_delta_n_principal_min",
        "Si prism principal delta n min",
        "dimensionless",
        True,
    ),
]

SUMMARY_GROUPS = [
    (
        "strain_components",
        [
            ("strain_xx", "Strain xx", "dimensionless", True),
            ("strain_yy", "Strain yy", "dimensionless", True),
            ("strain_xy", "Strain xy", "dimensionless", True),
        ],
    ),
    (
        "stress_components_mpa",
        [
            ("stress_xx_mpa", "Stress xx", "MPa", True),
            ("stress_yy_mpa", "Stress yy", "MPa", True),
            ("stress_xy_mpa", "Stress xy", "MPa", True),
        ],
    ),
    (
        "displacement_components",
        [
            ("u_magnitude", "Displacement magnitude", "um", False),
            ("u_x", "Displacement X", "um", True),
            ("u_y", "Displacement Y", "um", True),
        ],
    ),
    (
        "si_prism_index_perturbation",
        [
            ("si_prism_delta_n_x", "Si prism delta n x-pol", "dimensionless", True),
            ("si_prism_delta_n_y", "Si prism delta n y-pol", "dimensionless", True),
            (
                "si_prism_delta_n_x_minus_y",
                "Si prism delta n x - y",
                "dimensionless",
                True,
            ),
        ],
    ),
]


def robust_limits(values: np.ndarray, symmetric: bool) -> tuple[float, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        return 0.0, 1.0

    if symmetric:
        limit = float(np.nanpercentile(np.abs(finite), 99.5))
        if limit == 0.0:
            limit = float(np.nanmax(np.abs(finite)))
        if limit == 0.0:
            limit = 1.0
        return -limit, limit

    low, high = np.nanpercentile(finite, [0.5, 99.5])
    if low == high:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def triangle_areas(points: np.ndarray, tri_cells: np.ndarray) -> np.ndarray:
    p0 = points[tri_cells[:, 0]]
    p1 = points[tri_cells[:, 1]]
    p2 = points[tri_cells[:, 2]]
    return 0.5 * np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    )


def collect_material_boundary_segments(
    points: np.ndarray, tri_cells: np.ndarray, mat_ids: np.ndarray | None
) -> list[list[np.ndarray]]:
    if mat_ids is None:
        return []

    edge_cells: dict[tuple[int, int], list[int]] = {}
    for cell_index, (n0, n1, n2) in enumerate(tri_cells):
        for aa, bb in ((n0, n1), (n1, n2), (n2, n0)):
            edge = tuple(sorted((int(aa), int(bb))))
            edge_cells.setdefault(edge, []).append(cell_index)

    segments = []
    for (aa, bb), cells in edge_cells.items():
        is_external = len(cells) == 1
        is_material_interface = (
            len(cells) == 2 and mat_ids[cells[0]] != mat_ids[cells[1]]
        )
        if is_external or is_material_interface:
            segments.append([points[aa], points[bb]])

    return segments


def extract_plot_data(vtk_path: Path):
    mesh = pv.read(vtk_path)
    triangles = mesh.extract_cells(mesh.celltypes == 5)
    boundary = mesh.extract_cells(mesh.celltypes == 3)

    points = np.asarray(triangles.points[:, :2], dtype=float)
    tri_cells = np.asarray(triangles.cells, dtype=np.int64).reshape((-1, 4))[:, 1:]
    triangulation = Triangulation(points[:, 0], points[:, 1], tri_cells)
    mat_ids = None
    if "mat_id" in triangles.cell_data:
        mat_ids = np.asarray(triangles.cell_data["mat_id"], dtype=int).reshape(-1)

    boundary_segments = []
    if boundary.n_cells:
        boundary_points = np.asarray(boundary.points[:, :2], dtype=float)
        line_cells = np.asarray(boundary.cells, dtype=np.int64).reshape((-1, 3))[:, 1:]
        boundary_segments = [
            [boundary_points[ii], boundary_points[jj]] for ii, jj in line_cells
        ]

    material_boundary_segments = collect_material_boundary_segments(
        points, tri_cells, mat_ids
    )
    if material_boundary_segments:
        boundary_segments = material_boundary_segments

    return mesh, triangles, triangulation, boundary_segments, tri_cells, mat_ids


def add_boundary(ax, boundary_segments) -> None:
    if not boundary_segments:
        return
    collection = LineCollection(
        boundary_segments,
        colors="#202124",
        linewidths=0.22,
        alpha=0.72,
    )
    ax.add_collection(collection)


def style_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")


def plot_field(
    ax,
    triangulation: Triangulation,
    boundary_segments,
    values: np.ndarray,
    title: str,
    units: str,
    symmetric: bool,
    cell_values: bool,
):
    vmin, vmax = robust_limits(values, symmetric=symmetric)
    cmap = "coolwarm" if symmetric else "viridis"

    plot_values = np.ma.masked_invalid(np.asarray(values, dtype=float))

    if cell_values:
        artist = ax.tripcolor(
            triangulation,
            facecolors=plot_values,
            shading="flat",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        artist = ax.tripcolor(
            triangulation,
            plot_values,
            shading="gouraud",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    add_boundary(ax, boundary_segments)
    style_axis(ax, title)
    colorbar = plt.colorbar(artist, ax=ax, fraction=0.034, pad=0.02)
    colorbar.set_label(units)
    return artist


def save_single_plot(
    output_path: Path,
    triangulation: Triangulation,
    boundary_segments,
    values: np.ndarray,
    title: str,
    units: str,
    symmetric: bool,
    cell_values: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.8), constrained_layout=True)
    plot_field(
        ax,
        triangulation,
        boundary_segments,
        values,
        title,
        units,
        symmetric,
        cell_values,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_summary_plot(
    output_path: Path,
    triangulation: Triangulation,
    boundary_segments,
    fields: list[tuple[str, str, str, bool]],
    point_data: dict,
    cell_data: dict,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 13.0), constrained_layout=True)
    for ax, (field_name, title, units, symmetric) in zip(axes, fields):
        if field_name in point_data:
            values = np.asarray(point_data[field_name], dtype=float)
            cell_values = False
        else:
            values = np.asarray(cell_data[field_name], dtype=float)
            cell_values = True

        plot_field(
            ax,
            triangulation,
            boundary_segments,
            values,
            title,
            units,
            symmetric,
            cell_values,
        )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def array_stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "p01": float("nan"),
            "p99": float("nan"),
        }
    return {
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "mean": float(np.nanmean(finite)),
        "p01": float(np.nanpercentile(finite, 1.0)),
        "p99": float(np.nanpercentile(finite, 99.0)),
    }


def displacement_array(point_data: dict[str, np.ndarray]) -> np.ndarray:
    if "u" in point_data:
        values = np.asarray(point_data["u"], dtype=float)
        if values.ndim == 1:
            values = values.reshape((-1, 2))
        return values[:, :2]

    return np.column_stack(
        [
            np.asarray(point_data["u_x"], dtype=float).reshape(-1),
            np.asarray(point_data["u_y"], dtype=float).reshape(-1),
        ]
    )


def fit_line_angle(points: np.ndarray) -> float:
    centered_x = points[:, 0] - np.mean(points[:, 0])
    centered_y = points[:, 1] - np.mean(points[:, 1])
    if np.allclose(centered_x, 0.0):
        return np.pi / 2.0
    slope = np.linalg.lstsq(centered_x[:, None], centered_y, rcond=None)[0][0]
    return float(np.arctan(slope))


def prism_rotation_summary(
    triangles,
    tri_cells: np.ndarray,
    mat_ids: np.ndarray | None,
    point_data: dict[str, np.ndarray],
) -> dict[str, object]:
    if mat_ids is None:
        return {"error": "mat_id cell data was not present in the VTK file."}

    prism_cells = mat_ids == 4
    if not np.any(prism_cells):
        return {"error": "No Si prism cells, material id 4, were found."}

    points = np.asarray(triangles.points[:, :2], dtype=float)
    displacement = displacement_array(point_data)
    prism_triangles = tri_cells[prism_cells]
    areas = triangle_areas(points, prism_triangles)
    cell_points = points[prism_triangles]
    cell_displacement = displacement[prism_triangles]
    centroids = np.mean(cell_points, axis=1)
    centroid_displacements = np.mean(cell_displacement, axis=1)

    weights = areas / np.sum(areas)
    x_bar = np.average(centroids, axis=0, weights=weights)
    u_bar = np.average(centroid_displacements, axis=0, weights=weights)
    x_centered = centroids - x_bar
    u_centered = centroid_displacements - u_bar
    sqrt_weights = np.sqrt(weights)[:, None]

    gradient_columns = np.linalg.lstsq(
        x_centered * sqrt_weights,
        u_centered * sqrt_weights,
        rcond=None,
    )[0]
    displacement_gradient = gradient_columns.T
    deformation_gradient = np.eye(2) + displacement_gradient
    left_svd, _singular, right_svd_t = np.linalg.svd(deformation_gradient)
    rotation_matrix = left_svd @ right_svd_t
    if np.linalg.det(rotation_matrix) < 0.0:
        left_svd[:, -1] *= -1.0
        rotation_matrix = left_svd @ right_svd_t

    small_rotation_rad = 0.5 * (
        displacement_gradient[1, 0] - displacement_gradient[0, 1]
    )
    polar_rotation_rad = float(
        np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    )

    prism_point_ids = np.unique(prism_triangles.reshape(-1))
    prism_points = points[prism_point_ids]
    prism_displacement = displacement[prism_point_ids]
    top_y = float(np.max(prism_points[:, 1]))
    top_tolerance = max(1.0e-6, 0.01)
    top_mask = np.abs(prism_points[:, 1] - top_y) <= top_tolerance
    if np.count_nonzero(top_mask) < 3:
        top_tolerance = 1.0
        top_mask = np.abs(prism_points[:, 1] - top_y) <= top_tolerance

    top_points = prism_points[top_mask]
    top_deformed = top_points + prism_displacement[top_mask]
    original_top_angle_rad = fit_line_angle(top_points)
    deformed_top_angle_rad = fit_line_angle(top_deformed)
    top_edge_rotation_rad = deformed_top_angle_rad - original_top_angle_rad

    def angle_block(value_rad: float) -> dict[str, float]:
        return {
            "rad": float(value_rad),
            "mrad": float(1.0e3 * value_rad),
            "deg": float(np.degrees(value_rad)),
        }

    return {
        "convention": "positive rotation is counter-clockwise in the x-y plot",
        "method": (
            "Whole-prism rotation is from an area-weighted affine fit to prism "
            "cell-centroid displacements. Top-edge rotation is a least-squares "
            "tilt change of the flat prism top surface."
        ),
        "whole_prism_small_rotation": angle_block(small_rotation_rad),
        "whole_prism_polar_rotation": angle_block(polar_rotation_rad),
        "flat_top_edge_rotation": angle_block(top_edge_rotation_rad),
        "centroid_um": [float(x_bar[0]), float(x_bar[1])],
        "centroid_displacement_um": [float(u_bar[0]), float(u_bar[1])],
        "displacement_gradient": {
            "du_x_dx": float(displacement_gradient[0, 0]),
            "du_x_dy": float(displacement_gradient[0, 1]),
            "du_y_dx": float(displacement_gradient[1, 0]),
            "du_y_dy": float(displacement_gradient[1, 1]),
        },
        "prism_cell_count": int(np.count_nonzero(prism_cells)),
        "top_edge_point_count": int(np.count_nonzero(top_mask)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SfePy result fields from VTK.")
    parser.add_argument(
        "--vtk",
        type=Path,
        default=ROOT / "rslt" / "sfepy_output" / "pic_prism_2d.vtk",
        help="SfePy result VTK file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "rslt" / "plots",
        help="Directory for generated PNG plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _mesh, triangles, triangulation, boundary_segments, tri_cells, mat_ids = (
        extract_plot_data(args.vtk)
    )
    point_data = {key: np.asarray(value) for key, value in triangles.point_data.items()}
    cell_data = {key: np.asarray(value) for key, value in triangles.cell_data.items()}

    stats = {}
    rotation = prism_rotation_summary(triangles, tri_cells, mat_ids, point_data)

    for field_name, title, units, symmetric in POINT_FIELDS:
        if field_name not in point_data:
            continue
        values = np.asarray(point_data[field_name], dtype=float)
        stats[field_name] = array_stats(values)
        save_single_plot(
            args.output_dir / f"{field_name}.png",
            triangulation,
            boundary_segments,
            values,
            title,
            units,
            symmetric,
            cell_values=False,
        )

    for field_name, title, units, symmetric in CELL_FIELDS:
        if field_name not in cell_data:
            continue
        values = np.asarray(cell_data[field_name], dtype=float)
        stats[field_name] = array_stats(values)
        save_single_plot(
            args.output_dir / f"{field_name}.png",
            triangulation,
            boundary_segments,
            values,
            title,
            units,
            symmetric,
            cell_values=True,
        )

    for summary_name, fields in SUMMARY_GROUPS:
        if any(
            field_name not in point_data and field_name not in cell_data
            for field_name, _title, _units, _symmetric in fields
        ):
            continue
        save_summary_plot(
            args.output_dir / f"{summary_name}.png",
            triangulation,
            boundary_segments,
            fields,
            point_data,
            cell_data,
        )

    summary_path = args.output_dir / "field_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    rotation_path = args.output_dir / "prism_rotation_summary.json"
    rotation_path.write_text(json.dumps(rotation, indent=2), encoding="utf-8")

    print(f"Wrote plots to {args.output_dir}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote prism rotation summary to {rotation_path}")


if __name__ == "__main__":
    main()

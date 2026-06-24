from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "rslt" / "plots"
OUTPUT_HTML = ROOT / "rslt" / "results_report.html"
GEOMETRY_REPORT = ROOT / "rslt" / "pic_prism_2d_report.json"

SILICON_CTE_PER_C = 2.6e-6

PLOT_GROUPS = [
    (
        "Displacement",
        [
            "displacement_components.png",
            "u_magnitude.png",
            "u_x.png",
            "u_y.png",
        ],
    ),
    (
        "Strain",
        [
            "strain_components.png",
            "strain_xx.png",
            "strain_yy.png",
            "strain_xy.png",
        ],
    ),
    (
        "Stress",
        [
            "stress_components_mpa.png",
            "stress_xx_mpa.png",
            "stress_yy_mpa.png",
            "stress_xy_mpa.png",
        ],
    ),
    (
        "Si Prism Index Perturbation",
        [
            "si_prism_index_perturbation.png",
            "si_prism_delta_n_x.png",
            "si_prism_delta_n_y.png",
            "si_prism_delta_n_z.png",
            "si_prism_delta_n_x_minus_y.png",
            "si_prism_delta_n_principal_max.png",
            "si_prism_delta_n_principal_min.png",
        ],
    ),
]

FIELD_LABELS = {
    "u_magnitude": "Displacement magnitude",
    "u_x": "Displacement x",
    "u_y": "Displacement y",
    "strain_xx": "Strain xx",
    "strain_yy": "Strain yy",
    "strain_xy": "Strain xy",
    "stress_xx_mpa": "Stress xx, MPa",
    "stress_yy_mpa": "Stress yy, MPa",
    "stress_xy_mpa": "Stress xy, MPa",
    "si_prism_delta_n_x": "Si prism delta n x-pol",
    "si_prism_delta_n_y": "Si prism delta n y-pol",
    "si_prism_delta_n_z": "Si prism delta n z-pol",
    "si_prism_delta_n_x_minus_y": "Si prism delta n x - y",
    "si_prism_delta_n_principal_max": "Si prism principal delta n max",
    "si_prism_delta_n_principal_min": "Si prism principal delta n min",
}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.{digits}g}"
    return html.escape(str(value))


def get_nested(data: dict, keys: list[str], default: object = None) -> object:
    current: object = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def link(path: Path, text: str) -> str:
    relative = path.relative_to(OUTPUT_HTML.parent).as_posix()
    return f'<a href="{html.escape(relative)}">{html.escape(text)}</a>'


def asset_card(path: Path, title: str) -> str:
    if not path.exists():
        return ""
    relative = path.relative_to(OUTPUT_HTML.parent).as_posix()
    return f"""
      <figure class="plot-card">
        <a href="{html.escape(relative)}">
          <img src="{html.escape(relative)}" alt="{html.escape(title)}" loading="lazy">
        </a>
        <figcaption>{html.escape(title)}</figcaption>
      </figure>
    """


def image_card(filename: str) -> str:
    image_path = PLOTS_DIR / filename
    if not image_path.exists():
        return ""
    title = filename.removesuffix(".png").replace("_", " ")
    return asset_card(image_path, title)


def rotation_table(rotation: dict) -> str:
    rows = []
    for key, label in [
        ("whole_prism_small_rotation", "Whole-prism small rotation"),
        ("whole_prism_polar_rotation", "Whole-prism polar rotation"),
        ("flat_top_edge_rotation", "Flat top-edge rotation"),
    ]:
        value = rotation.get(key, {})
        rows.append(
            "<tr>"
            f"<th>{html.escape(label)}</th>"
            f"<td>{fmt(value.get('rad'), 9)}</td>"
            f"<td>{fmt(value.get('mrad'), 6)}</td>"
            f"<td>{fmt(value.get('deg'), 6)}</td>"
            "</tr>"
        )

    return """
    <table>
      <thead>
        <tr><th>Quantity</th><th>rad</th><th>mrad</th><th>deg</th></tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    """.format(rows="\n".join(rows))


def displacement_gradient_table(rotation: dict) -> str:
    gradient = rotation.get("displacement_gradient", {})
    rows = []
    for key in ["du_x_dx", "du_x_dy", "du_y_dx", "du_y_dy"]:
        rows.append(f"<tr><th>{html.escape(key)}</th><td>{fmt(gradient.get(key), 9)}</td></tr>")
    return "<table><tbody>{}</tbody></table>".format("\n".join(rows))


def rotation_direction(rotation: dict) -> str:
    angle = as_float(get_nested(rotation, ["whole_prism_small_rotation", "rad"], 0.0))
    if angle < 0.0:
        return "clockwise"
    if angle > 0.0:
        return "counter-clockwise"
    return "neutral"


def rotation_interpretation(rotation: dict, geometry: dict) -> str:
    params = geometry.get("parameters", {})
    prism_outline = geometry.get("prism_bottom_outline_um", [])
    clearance = geometry.get("prism_clearance_report_um", {})
    centroid = rotation.get("centroid_um", [None, None])

    cure_temp = as_float(params.get("epoxy_cure_temperature_C"), 200.0)
    final_temp = as_float(params.get("final_temperature_C"), 25.0)
    delta_t = final_temp - cure_temp
    epoxy_cte = as_float(
        params.get(
            "underfill_cte_per_C",
            as_float(params.get("underfill_cte_ppm_per_C"), 40.0) * 1.0e-6,
        )
    )
    mismatch_strain_pct = (epoxy_cte - SILICON_CTE_PER_C) * abs(delta_t) * 100.0

    low_y = as_float(clearance.get("low_prism_bottom_y_um"))
    centroid_y = as_float(centroid[1] if len(centroid) > 1 else None)
    lever_arm = centroid_y - low_y

    right_low_span = 0.0
    left_low_span = as_float(params.get("prism_left_low_flat_width_um"))
    if len(prism_outline) >= 8:
        right_low_start = as_float(prism_outline[5].get("x"))
        right_low_end = as_float(prism_outline[7].get("x"))
        right_low_span = right_low_end - right_low_start

    direction = rotation_direction(rotation)
    sign_note = (
        "The fitted rotation is negative, so it is clockwise in the x-right, "
        "y-up coordinate system used by the plots."
        if direction == "clockwise"
        else "The fitted rotation is positive, so it is counter-clockwise in the x-right, y-up coordinate system."
    )

    return f"""
    <div class="callout">
      <h3>Physical Interpretation</h3>
      <p>
        {html.escape(sign_note)} The sign is not a display artifact: the prism
        top edge tilts right-side down because the bonded underfill develops an
        unbalanced moment during cooldown.
      </p>
      <div class="summary compact">
        <div class="metric"><b>Epoxy-Si shrink mismatch</b><span>{fmt(mismatch_strain_pct, 3)}%</span></div>
        <div class="metric"><b>Low-interface lever arm</b><span>{fmt(lever_arm, 4)} um</span></div>
        <div class="metric"><b>Right low span / left low flat</b><span>{fmt(right_low_span, 4)} / {fmt(left_low_span, 4)} um</span></div>
      </div>
      <p>
        The epoxy wants to shrink much more than silicon, but it is bonded
        between the prism and the substrate. The low underside is roughly
        {fmt(lever_arm, 4)} um below the prism centroid, so horizontal shear and
        vertical pull there have a large rotational lever arm.
      </p>
      <p>
        The right-side low bonded path is much longer than the short left low
        flat. Along that lower-right path and the adjacent sloped faces, epoxy
        contraction pulls the prism inward and downward. A downward force to the
        right of the centroid, and a leftward force below the centroid, both
        produce a clockwise moment. High flat regions push back, but in this
        geometry they do not fully cancel the lower/sloped interface couple.
      </p>
    </div>
    """


def field_summary_table(summary: dict) -> str:
    rows = []
    for key, values in summary.items():
        label = FIELD_LABELS.get(key, key)
        rows.append(
            "<tr>"
            f"<th>{html.escape(label)}</th>"
            f"<td>{fmt(values.get('min'))}</td>"
            f"<td>{fmt(values.get('max'))}</td>"
            f"<td>{fmt(values.get('mean'))}</td>"
            f"<td>{fmt(values.get('p01'))}</td>"
            f"<td>{fmt(values.get('p99'))}</td>"
            "</tr>"
        )

    return """
    <table>
      <thead>
        <tr><th>Field</th><th>Min</th><th>Max</th><th>Mean</th><th>P01</th><th>P99</th></tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    """.format(rows="\n".join(rows))


def geometry_links() -> str:
    files = [
        (ROOT / "rslt" / "pic_prism_2d.svg", "Geometry SVG"),
        (ROOT / "rslt" / "pic_prism_2d_cd.svg", "Interactive CD SVG"),
        (GEOMETRY_REPORT, "Geometry report JSON"),
        (ROOT / "rslt" / "sfepy_output" / "pic_prism_2d.vtk", "SfePy VTK result"),
        (PLOTS_DIR / "field_summary.json", "Field summary JSON"),
        (PLOTS_DIR / "prism_rotation_summary.json", "Prism rotation JSON"),
    ]
    items = []
    for path, text in files:
        if path.exists():
            items.append(f"<li>{link(path, text)}</li>")
    return "<ul class=\"links\">{}</ul>".format("\n".join(items))


def plot_sections() -> str:
    sections = []
    for title, filenames in PLOT_GROUPS:
        cards = "\n".join(image_card(filename) for filename in filenames)
        if not cards.strip():
            continue
        sections.append(
            f"""
            <section>
              <h2>{html.escape(title)}</h2>
              <div class="plot-grid">{cards}</div>
            </section>
            """
        )
    return "\n".join(sections)


def geometry_preview_section() -> str:
    cards = "\n".join(
        [
            asset_card(ROOT / "rslt" / "pic_prism_2d.svg", "Geometry preview SVG"),
            asset_card(
                ROOT / "rslt" / "pic_prism_2d_cd.svg",
                "Interactive critical-dimension SVG",
            ),
        ]
    )
    if not cards.strip():
        return ""
    return f"""
    <section>
      <h2>Geometry</h2>
      <div class="plot-grid">{cards}</div>
    </section>
    """


def build_html() -> str:
    rotation = read_json(PLOTS_DIR / "prism_rotation_summary.json")
    field_summary = read_json(PLOTS_DIR / "field_summary.json")
    geometry = read_json(GEOMETRY_REPORT)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    centroid = rotation.get("centroid_um", ["", ""])
    centroid_u = rotation.get("centroid_displacement_um", ["", ""])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>2D PIC Prism Stress Results</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #202124;
      --muted: #5f6368;
      --line: #d6d9dc;
      --bg: #ffffff;
      --panel: #f7f9fa;
      --accent: #0f6686;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 28px 48px;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      margin-bottom: 24px;
      padding-bottom: 16px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      line-height: 1.15;
    }}
    h2 {{
      margin: 28px 0 12px;
      font-size: 20px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 7px;
    }}
    h3 {{
      margin: 20px 0 8px;
      font-size: 15px;
    }}
    p {{ margin: 7px 0; }}
    .muted {{ color: var(--muted); }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric b {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .metric span {{
      display: block;
      font-size: 19px;
      font-weight: 700;
      margin-top: 4px;
    }}
    .callout {{
      background: #f7fbfc;
      border: 1px solid var(--line);
      border-left: 4px solid var(--accent);
      border-radius: 8px;
      margin: 18px 0;
      padding: 14px 16px;
    }}
    .callout h3 {{ margin-top: 0; }}
    .summary.compact {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin: 12px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 18px;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 7px 8px;
      text-align: right;
      vertical-align: top;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    thead th {{
      background: #eef3f5;
      font-weight: 700;
    }}
    tbody th {{
      font-weight: 600;
      background: #fbfcfd;
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 16px;
    }}
    .plot-card {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }}
    .plot-card img {{
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
    }}
    figcaption {{
      border-top: 1px solid var(--line);
      padding: 8px 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    .links {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 8px 18px;
      padding-left: 18px;
    }}
    a {{ color: var(--accent); }}
    code {{
      background: #eef3f5;
      border-radius: 4px;
      padding: 1px 4px;
    }}
    @media (max-width: 780px) {{
      main {{ padding: 20px 14px 36px; }}
      .summary {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>2D PIC Prism Stress Results</h1>
    <p class="muted">Generated {html.escape(generated)} from the current SfePy VTK output.</p>
  </header>

  <section>
    <h2>Prism Rotation</h2>
    <p>
      This is a 2D x-y plane-strain model, so the reported prism rotation is about
      the out-of-plane <code>z</code> axis. With <code>x</code> to the right and
      <code>y</code> upward, positive rotation is counter-clockwise by the
      right-hand rule. Negative values are clockwise.
    </p>
    <p>
      The small-rotation value uses
      <code>theta_z = 0.5 * (du_y/dx - du_x/dy)</code> from an area-weighted
      affine fit to the Si prism displacement field. The polar rotation is the
      finite-rotation part of the fitted deformation gradient. The top-edge
      value is the slope change of the flat top prism edge.
    </p>
    <div class="summary">
      <div class="metric"><b>Whole prism</b><span>{fmt(rotation.get("whole_prism_small_rotation", {}).get("mrad"), 6)} mrad</span></div>
      <div class="metric"><b>Flat top edge</b><span>{fmt(rotation.get("flat_top_edge_rotation", {}).get("mrad"), 6)} mrad</span></div>
      <div class="metric"><b>Direction</b><span>{rotation_direction(rotation)}</span></div>
    </div>
    {rotation_table(rotation)}
    <h3>Translation and Gradient</h3>
    <p>
      Area-weighted prism centroid: ({fmt(centroid[0])}, {fmt(centroid[1])}) um.
      Average prism displacement: ({fmt(centroid_u[0])}, {fmt(centroid_u[1])}) um.
    </p>
    {displacement_gradient_table(rotation)}
    {rotation_interpretation(rotation, geometry)}
  </section>

  <section>
    <h2>Source Files</h2>
    {geometry_links()}
  </section>

  {geometry_preview_section()}

  {plot_sections()}

  <section>
    <h2>Field Statistics</h2>
    <p class="muted">
      P01 and P99 are percentiles. The plots use robust color limits so local
      corner singularities do not hide the broader field.
    </p>
    {field_summary_table(field_summary)}
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(build_html(), encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import json
from pathlib import Path

import numpy as nm
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson


ROOT = Path(__file__).resolve().parents[1]
filename_mesh = str(ROOT / "rslt" / "pic_prism_2d.mesh")

output_dir = str(ROOT / "rslt" / "sfepy_output")

PARAMETERS = json.loads((ROOT / "input" / "geometry_parameters.json").read_text())
if "epoxy_cure_temperature_C" in PARAMETERS and "final_temperature_C" in PARAMETERS:
    DT = float(PARAMETERS["final_temperature_C"]) - float(
        PARAMETERS["epoxy_cure_temperature_C"]
    )
else:
    DT = float(PARAMETERS.get("temperature_drop_C", -175.0))
UNDERFILL_CTE_PER_C = float(
    PARAMETERS.get(
        "underfill_cte_per_C",
        float(PARAMETERS.get("underfill_cte_ppm_per_C", 40.0)) * 1.0e-6,
    )
)
BOTTOM_Y = -(
    float(PARAMETERS["first_facet_etch_depth_um"])
    + float(PARAMETERS["second_facet_etch_depth_um"])
    + float(PARAMETERS["si_handle_thickness_below_second_etch_um"])
    + float(PARAMETERS.get("handle_oxide_thickness_um", 0.0))
)
PLANE = "strain"


def elastic_thermal_material(young_mpa: float, poisson: float, cte_per_c: float) -> dict:
    stiffness = stiffness_from_youngpoisson(
        2,
        young=young_mpa,
        poisson=poisson,
        plane=PLANE,
    )
    thermal_strain = nm.array([cte_per_c * DT, cte_per_c * DT, 0.0], dtype=nm.float64)
    prestress = nm.dot(stiffness, thermal_strain)[:, None]
    return {"D": stiffness, "prestress": prestress, "thermal_strain": thermal_strain}


materials = {
    "si": (
        elastic_thermal_material(
            young_mpa=130000.0,
            poisson=0.28,
            cte_per_c=2.6e-6,
        ),
    ),
    "sio2": (
        elastic_thermal_material(
            young_mpa=73000.0,
            poisson=0.17,
            cte_per_c=0.55e-6,
        ),
    ),
    "epoxy": (
        elastic_thermal_material(
            young_mpa=float(PARAMETERS.get("underfill_young_mpa", 2500.0)),
            poisson=float(PARAMETERS.get("underfill_poisson", 0.35)),
            cte_per_c=UNDERFILL_CTE_PER_C,
        ),
    ),
}

regions = {
    "Omega": "all",
    "SOI": "cells of group 1",
    "BOX": "cells of group 2",
    "Handle": "cells of group 3",
    "Prism": "cells of group 4",
    "Underfill": "cells of group 5",
    "HandleOxide": "cells of group 6",
    "Bottom": (f"vertices in (y < {BOTTOM_Y + 1.0e-3:.9g})", "facet"),
}

fields = {
    "displacement": ("real", 2, "Omega", 1),
}

variables = {
    "u": ("unknown field", "displacement", 0),
    "v": ("test field", "displacement", "u"),
}

ebcs = {
    "fixed_bottom": ("Bottom", {"u.all": 0.0}),
}

integrals = {
    "i": 2,
}

equations = {
    "balance_of_forces": """
        dw_lin_elastic.i.SOI(si.D, v, u)
      + dw_lin_elastic.i.Handle(si.D, v, u)
      + dw_lin_elastic.i.BOX(sio2.D, v, u)
      + dw_lin_elastic.i.HandleOxide(sio2.D, v, u)
      + dw_lin_elastic.i.Prism(si.D, v, u)
      + dw_lin_elastic.i.Underfill(epoxy.D, v, u)
      =
        dw_lin_prestress.i.SOI(si.prestress, v)
      + dw_lin_prestress.i.Handle(si.prestress, v)
      + dw_lin_prestress.i.BOX(sio2.prestress, v)
      + dw_lin_prestress.i.HandleOxide(sio2.prestress, v)
      + dw_lin_prestress.i.Prism(si.prestress, v)
      + dw_lin_prestress.i.Underfill(epoxy.prestress, v)
    """,
}

solvers = {
    "ls": ("ls.scipy_superlu", {}),
    "newton": (
        "nls.newton",
        {
            "i_max": 1,
            "eps_a": 1.0e-10,
        },
    ),
}

options = {
    "nls": "newton",
    "ls": "ls",
    "output_dir": output_dir,
    "post_process_hook": "post_process",
}


MATERIAL_BY_GROUP = {
    1: "si",
    2: "sio2",
    3: "si",
    4: "si",
    5: "epoxy",
    6: "sio2",
}
PRISM_GROUP_ID = 4

SILICON_REFRACTIVE_INDEX = float(PARAMETERS.get("silicon_refractive_index", 3.476))
SILICON_P11 = float(PARAMETERS.get("silicon_photoelastic_p11", -0.094))
SILICON_P12 = float(PARAMETERS.get("silicon_photoelastic_p12", 0.017))
SILICON_P44 = float(PARAMETERS.get("silicon_photoelastic_p44", -0.051))


def read_triangle_groups(mesh_path: str) -> nm.ndarray:
    lines = Path(mesh_path).read_text(encoding="utf-8").splitlines()
    for ii, line in enumerate(lines):
        if line.strip() == "Triangles":
            count = int(lines[ii + 1].strip())
            groups = []
            for row in lines[ii + 2 : ii + 2 + count]:
                groups.append(int(row.split()[3]))
            return nm.asarray(groups, dtype=nm.int32)
    raise ValueError(f"No Triangles section found in {mesh_path}")


def as_cell_voigt(data) -> nm.ndarray:
    values = nm.asarray(data)
    if values.ndim == 4:
        return values[:, 0, :, 0]
    if values.ndim == 3:
        return values.reshape((values.shape[0], -1))[:, :3]
    if values.ndim == 2:
        return values[:, :3]
    return values.reshape((values.shape[0], -1))[:, :3]


def as_cell_output(data: nm.ndarray) -> nm.ndarray:
    return nm.asarray(data, dtype=nm.float64)[:, None, :, None]


def add_cell_component(out, name: str, values: nm.ndarray, component: int) -> None:
    from sfepy.base.base import Struct

    out[name] = Struct(
        name="output_data",
        mode="cell",
        data=as_cell_output(values[:, component : component + 1]),
        dofs=None,
    )


def add_cell_scalar(out, name: str, values: nm.ndarray) -> None:
    from sfepy.base.base import Struct

    out[name] = Struct(
        name="output_data",
        mode="cell",
        data=as_cell_output(nm.asarray(values, dtype=nm.float64)[:, None]),
        dofs=None,
    )


def masked_prism_values(values: nm.ndarray, cell_groups: nm.ndarray) -> nm.ndarray:
    masked = nm.full(cell_groups.shape[0], nm.nan, dtype=nm.float64)
    prism = cell_groups == PRISM_GROUP_ID
    masked[prism] = values[prism]
    return masked


def silicon_prism_index_perturbation(
    strain: nm.ndarray, cell_groups: nm.ndarray
) -> dict[str, nm.ndarray]:
    """Linear photoelastic perturbation in the Si prism region.

    The SfePy solve is plane strain. For stress-induced index change, use elastic
    strain, i.e. total displacement strain minus the free thermal strain.
    """

    si = materials["si"][0]
    n0 = SILICON_REFRACTIVE_INDEX
    elastic_strain = strain - si["thermal_strain"][None, :]

    exx = elastic_strain[:, 0]
    eyy = elastic_strain[:, 1]
    gamma_xy = elastic_strain[:, 2]
    ezz = -si["thermal_strain"][0] if PLANE == "strain" else 0.0

    delta_b_x = SILICON_P11 * exx + SILICON_P12 * (eyy + ezz)
    delta_b_y = SILICON_P12 * (exx + ezz) + SILICON_P11 * eyy
    delta_b_z = SILICON_P12 * (exx + eyy) + SILICON_P11 * ezz

    # Voigt B6 = 2 Bxy, so the off-diagonal matrix term is half of p44*gamma_xy.
    delta_b_xy = 0.5 * SILICON_P44 * gamma_xy
    trace = 0.5 * (delta_b_x + delta_b_y)
    radius = nm.sqrt((0.5 * (delta_b_x - delta_b_y)) ** 2 + delta_b_xy**2)
    principal_b_1 = trace - radius
    principal_b_2 = trace + radius

    factor = -0.5 * n0**3
    dn_x = factor * delta_b_x
    dn_y = factor * delta_b_y
    dn_z = factor * delta_b_z
    dn_principal_1 = factor * principal_b_1
    dn_principal_2 = factor * principal_b_2

    return {
        "si_prism_delta_n_x": masked_prism_values(dn_x, cell_groups),
        "si_prism_delta_n_y": masked_prism_values(dn_y, cell_groups),
        "si_prism_delta_n_z": masked_prism_values(dn_z, cell_groups),
        "si_prism_delta_n_x_minus_y": masked_prism_values(dn_x - dn_y, cell_groups),
        "si_prism_delta_n_principal_max": masked_prism_values(
            nm.maximum(dn_principal_1, dn_principal_2), cell_groups
        ),
        "si_prism_delta_n_principal_min": masked_prism_values(
            nm.minimum(dn_principal_1, dn_principal_2), cell_groups
        ),
    }


def add_displacement_components(out, state) -> None:
    from sfepy.base.base import Struct

    if not hasattr(state, "get_state_parts"):
        return

    parts = state.get_state_parts()
    if "u" not in parts:
        return

    displacement = nm.asarray(parts["u"], dtype=nm.float64).reshape((-1, 2))
    magnitude = nm.linalg.norm(displacement, axis=1)[:, None]

    out["u_x"] = Struct(
        name="output_data",
        mode="vertex",
        data=displacement[:, 0:1],
        dofs=None,
    )
    out["u_y"] = Struct(
        name="output_data",
        mode="vertex",
        data=displacement[:, 1:2],
        dofs=None,
    )
    out["u_magnitude"] = Struct(
        name="output_data",
        mode="vertex",
        data=magnitude,
        dofs=None,
    )


def post_process(out, pb, state, extend=False):
    from sfepy.base.base import Struct

    strain_raw = pb.evaluate("ev_cauchy_strain.i.Omega(u)", mode="el_avg")
    strain = as_cell_voigt(strain_raw)
    cell_groups = read_triangle_groups(filename_mesh)
    stress = nm.zeros_like(strain)

    for group_id, material_name in MATERIAL_BY_GROUP.items():
        mask = cell_groups == group_id
        if not mask.any():
            continue
        material = materials[material_name][0]
        stress[mask, :] = nm.dot(strain[mask, :], material["D"].T) - material[
            "prestress"
        ][:, 0]

    out["cauchy_strain"] = Struct(
        name="output_data",
        mode="cell",
        data=as_cell_output(strain),
        dofs=None,
    )
    out["cauchy_stress_mpa"] = Struct(
        name="output_data",
        mode="cell",
        data=as_cell_output(stress),
        dofs=None,
    )
    add_cell_component(out, "strain_xx", strain, 0)
    add_cell_component(out, "strain_yy", strain, 1)
    add_cell_component(out, "strain_xy", strain, 2)
    add_cell_component(out, "stress_xx_mpa", stress, 0)
    add_cell_component(out, "stress_yy_mpa", stress, 1)
    add_cell_component(out, "stress_xy_mpa", stress, 2)
    add_cell_scalar(out, "mat_id", cell_groups)
    for name, values in silicon_prism_index_perturbation(strain, cell_groups).items():
        add_cell_scalar(out, name, values)
    add_displacement_components(out, state)
    return out

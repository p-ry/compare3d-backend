import io
import math
import tempfile
from typing import List, Tuple

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import cadquery as cq

# ----------------------------- CORS ---------------------------------
# Set allowed origins via env var CORS_ORIGINS="https://www.pryland.com,https://pryland.com"
import os
origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
app = FastAPI(title="Compare3D Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if origins == ["*"] else origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Geometry helpers ------------------------

def _rotate_solids(solids: List[cq.Solid], rx: float, ry: float, rz: float) -> List[cq.Solid]:
    """Rotate solids about global X, then Y, then Z by degrees."""
    axX = (cq.Vector(0, 0, 0), cq.Vector(1, 0, 0))
    axY = (cq.Vector(0, 0, 0), cq.Vector(0, 1, 0))
    axZ = (cq.Vector(0, 0, 0), cq.Vector(0, 0, 1))
    out = []
    for s in solids:
        out.append(s.rotate(*axX, rx).rotate(*axY, ry).rotate(*axZ, rz))
    return out

def _bbox(solids: List[cq.Solid]) -> Tuple[float, float, float]:
    if not solids:
        return (0.0, 0.0, 0.0)
    xmin=ymin=zmin= float("inf")
    xmax=ymax=zmax= float("-inf")
    for s in solids:
        b = s.BoundingBox()
        xmin = min(xmin, b.xmin); xmax = max(xmax, b.xmax)
        ymin = min(ymin, b.ymin); ymax = max(ymax, b.ymax)
        zmin = min(zmin, b.zmin); zmax = max(zmax, b.zmax)
    return (xmax - xmin, ymax - ymin, zmax - zmin)

def _props_from_model(model: cq.Workplane, density: float) -> dict:
    solids = model.solids().vals()
    if not solids:
        raise ValueError("No solids found in STEP.")
    vol = sum(s.Volume() for s in solids)
    mass = vol * density
    # mass-weighted COG
    cx=cy=cz=0.0
    for s in solids:
        v = s.Volume()
        c = s.Center()
        cx += c.x*v; cy += c.y*v; cz += c.z*v
    cog = (cx/vol, cy/vol, cz/vol)
    bbox = _bbox(solids)
    return {
        "num_solids": len(solids),
        "total_volume_mm3": vol,
        "total_mass_g": mass,
        "cog_mm": cog,
        "bbox_mm": bbox,
    }

# Discrete orientation search: rx,ry,rz ∈ {0, 90, 180, 270}
ANGLES = [0.0, 90.0, 180.0, 270.0]

def _score(
    a_bbox: Tuple[float,float,float],
    a_cog: Tuple[float,float,float],
    b_bbox: Tuple[float,float,float],
    b_cog: Tuple[float,float,float],
    prefer_cog: bool
) -> float:
    # Sum of absolute bbox deltas
    bbox_term = abs(a_bbox[0]-b_bbox[0]) + abs(a_bbox[1]-b_bbox[1]) + abs(a_bbox[2]-b_bbox[2])
    # Euclidean cog distance
    cog_term = math.sqrt((a_cog[0]-b_cog[0])**2 + (a_cog[1]-b_cog[1])**2 + (a_cog[2]-b_cog[2])**2)
    # Weighting: if prefer_cog, increase the influence of COG to favor 180° flips that align centers
    w_cog = 10.0 if prefer_cog else 1.0
    return bbox_term + w_cog * cog_term

def _autorotate_b_to_match(
    a_model: cq.Workplane,
    b_model: cq.Workplane,
    density: float,
    prefer_cog: bool
):
    a_props = _props_from_model(a_model, density)
    a_bbox = a_props["bbox_mm"]
    a_cog  = a_props["cog_mm"]

    best = None
    best_angles = (0.0,0.0,0.0)
    # Try all 64 orientations
    for rx in ANGLES:
        for ry in ANGLES:
            for rz in ANGLES:
                b_solids = b_model.solids().vals()
                rot_b = _rotate_solids(b_solids, rx, ry, rz)
                b_wp = cq.Workplane("XY").newObject(rot_b)
                b_props = _props_from_model(b_wp, density)
                s = _score(a_bbox, a_cog, b_props["bbox_mm"], b_props["cog_mm"], prefer_cog)
                if (best is None) or (s < best):
                    best = s
                    best_angles = (rx, ry, rz)
                    best_props = b_props
                    best_model = b_wp
    return best_model, best_props, best_angles, a_props

def _ident_flags(a: dict, b: dict, t_vol: float, t_mass: float, t_cog: float, t_bbox: float):
    ident_vol_mass = (
        abs(a["total_volume_mm3"]-b["total_volume_mm3"]) <= t_vol and
        abs(a["total_mass_g"]-b["total_mass_g"]) <= t_mass and
        a["num_solids"] == b["num_solids"]
    )
    ident_cog = all(abs(x-y) <= t_cog for x,y in zip(a["cog_mm"], b["cog_mm"]))
    ident_bbox = all(abs(x-y) <= t_bbox for x,y in zip(a["bbox_mm"], b["bbox_mm"]))
    return ident_vol_mass, ident_cog, ident_bbox

# ------------------------------ API ---------------------------------

class CompareResponse(BaseModel):
    a: dict
    b: dict
    deltas: dict
    ident_vol_mass: bool
    ident_cog: bool
    ident_bbox: bool
    autorotate_applied: bool
    chosen_angles_deg: Tuple[float, float, float]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/compare", response_model=CompareResponse)
async def compare(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    density: float = Form(0.0027),        # g/mm^3
    autorotate: bool = Form(True),
    prefer_cog: bool = Form(False),
    t_vol: float = Form(0.001),
    t_mass: float = Form(0.001),
    t_cog: float = Form(0.002),
    t_bbox: float = Form(0.001),
):
    # Load STEP into CadQuery
    with tempfile.TemporaryDirectory() as td:
        pa = os.path.join(td, "a.step")
        pb = os.path.join(td, "b.step")
        pa_bytes = await file_a.read()
        pb_bytes = await file_b.read()
        with open(pa, "wb") as fa: fa.write(pa_bytes)
        with open(pb, "wb") as fb: fb.write(pb_bytes)

        a_model = cq.importers.importStep(pa)
        b_model = cq.importers.importStep(pb)

        if autorotate:
            b_best, b_props, angles, a_props = _autorotate_b_to_match(a_model, b_model, density, prefer_cog)
            autorotate_applied = True
            chosen_angles = angles
            # Use best-rotated b model for final props
            a = a_props
            b = b_props
        else:
            a = _props_from_model(a_model, density)
            b = _props_from_model(b_model, density)
            autorotate_applied = False
            chosen_angles = (0.0, 0.0, 0.0)

        deltas = {
            "volume_mm3": a["total_volume_mm3"] - b["total_volume_mm3"],
            "mass_g": a["total_mass_g"] - b["total_mass_g"],
            "cog_mm": (
                a["cog_mm"][0] - b["cog_mm"][0],
                a["cog_mm"][1] - b["cog_mm"][1],
                a["cog_mm"][2] - b["cog_mm"][2],
            ),
            "bbox_mm": (
                a["bbox_mm"][0] - b["bbox_mm"][0],
                a["bbox_mm"][1] - b["bbox_mm"][1],
                a["bbox_mm"][2] - b["bbox_mm"][2],
            ),
        }

        ident_vol_mass, ident_cog, ident_bbox = _ident_flags(a, b, t_vol, t_mass, t_cog, t_bbox)

        return {
            "a": a,
            "b": b,
            "deltas": deltas,
            "ident_vol_mass": ident_vol_mass,
            "ident_cog": ident_cog,
            "ident_bbox": ident_bbox,
            "autorotate_applied": autorotate_applied,
            "chosen_angles_deg": chosen_angles,
        }

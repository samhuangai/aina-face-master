#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_aina_v10_fixed as fixed

base = fixed.base


def deform_base_hair_fixed(document, binary):
    changed = []
    processed_accessors = set()
    for mesh_index, mesh in enumerate(document.get("meshes", [])):
        if "body" not in mesh.get("name", "").lower():
            continue
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            if "hair" not in base.material_name(document, primitive).lower():
                continue
            accessor_index = primitive["attributes"]["POSITION"]
            if accessor_index in processed_accessors:
                continue
            processed_accessors.add(accessor_index)
            values = base.accessor_array(document, binary, accessor_index).astype(np.float64)
            hair_indices = base.accessor_array(document, binary, primitive["indices"]).reshape(-1).astype(np.int64)
            hair_indices = np.unique(hair_indices)
            subset = values[hair_indices].copy()
            y_min, y_max = float(subset[:, 1].min()), float(subset[:, 1].max())
            anchor = y_min + (y_max - y_min) * 0.52
            upper = subset[:, 1] > anchor
            subset[upper, 1] = anchor + (subset[upper, 1] - anchor) * 0.88
            center_x = float(np.median(subset[:, 0]))
            subset[:, 0] = center_x + (subset[:, 0] - center_x) * 0.94
            values[hair_indices] = subset
            base.write_accessor(document, binary, accessor_index, values.astype(np.float32))
            changed.append({
                "mesh": mesh_index,
                "primitive": primitive_index,
                "hair_vertices": len(hair_indices),
                "shared_accessor_vertices": len(values),
                "skin_vertices_untouched": len(values) - len(hair_indices),
            })
    return {"changed_primitives": changed, "hair_indices_only": True}


base.deform_base_hair = deform_base_hair_fixed


if __name__ == "__main__":
    base.main()

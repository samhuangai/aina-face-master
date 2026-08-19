#!/usr/bin/env python3
"""Final visual convergence of the actual AINA production mesh.

No reference/effect image is generated. This edits the existing locked FaceVerse
geometry used by Blender, keeps topology/order, rebuilds the visible eye system,
lowers the collar, and preserves the existing 52-control generation pipeline.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from aina_vrm_addon_runtime import ensure_vrm_addon


def main() -> None:
    root = Path.cwd()
    ensure_vrm_addon(root)

    import aina_final_vrm_release as release
    import aina_visual_identity_assembly as visual
    import aina_visual_eye_system as eye_system
    import aina_visual_hair_system as hair_system

    base = visual.base

    def already_enabled(_root: Path):
        if "io_scene_vrm" not in bpy.context.preferences.addons:
            raise RuntimeError("VRM Addon preferences disappeared before final visual assembly")
        return None, None, None

    def ellipsoid_weights(coords, center, radii, inner=0.22, outer=1.15):
        center = np.asarray(center, float)
        radii = np.asarray(radii, float)
        q = np.sqrt(np.sum(((coords - center) / radii) ** 2, axis=1))
        w = np.zeros(len(coords), float)
        w[q <= inner] = 1.0
        m = (q > inner) & (q < outer)
        if np.any(m):
            t = (q[m] - inner) / (outer - inner)
            w[m] = 0.5 * (1.0 + np.cos(np.pi * t))
        return w

    def shift(arr, ids, center, radii, delta, inner=0.22, outer=1.15):
        ids = np.asarray(ids, np.int64)
        p = arr[ids].copy()
        w = ellipsoid_weights(p, center, radii, inner, outer)[:, None]
        p += w * np.asarray(delta, float)
        arr[ids] = p

    def scale(arr, ids, center, radii, factors, inner=0.22, outer=1.15):
        ids = np.asarray(ids, np.int64)
        p = arr[ids].copy()
        center = np.asarray(center, float)
        w = ellipsoid_weights(p, center, radii, inner, outer)[:, None]
        target = center + (p - center) * np.asarray(factors, float)
        arr[ids] = p + w * (target - p)

    def final_polish(mapped, head_ids, eye_groups):
        out = mapped.copy()
        h = np.asarray(head_ids, np.int64)
        lm = out[visual.K].copy()

        # AINA target is a compact youthful V lower-third, not the broad adult
        # FaceVerse jaw. Progressively narrow only below the upper cheek plane.
        mouth_z = float(lm[48:60, 2].mean())
        chin_z = float(lm[8, 2])
        upper = mouth_z + 0.030
        denom = max(upper - chin_z, 1e-6)
        p = out[h].copy()
        t = np.clip((upper - p[:, 2]) / denom, 0.0, 1.0)
        jaw_zone = (p[:, 2] < upper) & (p[:, 2] > chin_z - 0.020) & (p[:, 1] < 0.045)
        p[jaw_zone, 0] *= 1.0 - 0.22 * (t[jaw_zone] ** 1.15)
        out[h] = p

        lm = out[visual.K].copy()
        chin = lm[8]
        scale(out, h, chin, (0.050, 0.045, 0.042), (0.78, 0.98, 0.96), 0.05, 1.13)
        shift(out, h, chin, (0.050, 0.045, 0.040), (0.0, 0.0010, 0.0010), 0.0, 1.05)

        # Slight temple/cranium narrowing; the hair provides the outer silhouette.
        lm = out[visual.K].copy()
        eye_z = float(lm[36:48, 2].mean())
        p = out[h].copy()
        wt = np.clip((p[:, 2] - eye_z) / 0.095, 0.0, 1.0)
        temple_zone = (p[:, 2] > eye_z) & (np.abs(p[:, 0]) < 0.085) & (p[:, 1] < 0.055)
        p[temple_zone, 0] *= 1.0 - 0.050 * wt[temple_zone]
        out[h] = p

        # Large almond eyes and shallow youthful sockets. Blender front is -Y.
        lm = out[visual.K].copy()
        for rr in (range(36, 42), range(42, 48)):
            c = lm[list(rr)].mean(0)
            scale(out, h, c, (0.041, 0.030, 0.023), (1.34, 1.00, 1.48), 0.12, 1.13)
            shift(out, h, c, (0.043, 0.032, 0.029), (0.0, -0.0041, 0.0002), 0.08, 1.16)

        # Lift the two true outer eye corners without turning the whole eye upward.
        lm = out[visual.K].copy()
        for idx in (36, 45):
            shift(out, h, lm[idx], (0.019, 0.018, 0.015), (0.0, -0.0004, 0.0023), 0.05, 1.04)

        # Brows closer to the eyes and slightly forward, matching the reference.
        lm = out[visual.K].copy()
        for rr in (range(17, 22), range(22, 27)):
            c = lm[list(rr)].mean(0)
            shift(out, h, c, (0.041, 0.028, 0.022), (0.0, -0.0016, -0.0030), 0.05, 1.12)

        # Delicate nose: significantly narrower and shorter, but with a real
        # profile. Previous passes mistakenly retracted it until it vanished.
        lm = out[visual.K].copy()
        root_n = lm[27]
        scale(out, h, root_n, (0.031, 0.030, 0.049), (0.72, 1.00, 0.82), 0.05, 1.17)
        lm = out[visual.K].copy()
        tip = lm[30]
        base_n = lm[31:36].mean(0)
        shift(out, h, tip, (0.021, 0.023, 0.022), (0.0, -0.0074, 0.0003), 0.05, 1.08)
        shift(out, h, base_n, (0.026, 0.024, 0.020), (0.0, -0.0048, 0.0007), 0.05, 1.08)
        scale(out, h, base_n, (0.027, 0.024, 0.022), (0.78, 1.0, 0.96), 0.05, 1.10)

        # Small closed-looking lips. Width and vertical aperture both reduce,
        # and the entire perioral surface is pulled behind the nose tip.
        lm = out[visual.K].copy()
        mouth = lm[48:60].mean(0)
        scale(out, h, mouth, (0.047, 0.027, 0.022), (0.82, 0.72, 0.58), 0.18, 1.16)
        shift(out, h, mouth, (0.048, 0.030, 0.025), (0.0, 0.0058, 0.0007), 0.12, 1.14)
        lm = out[visual.K].copy()
        for idx in (48, 54):
            shift(out, h, lm[idx], (0.016, 0.016, 0.013), (0.0, 0.0004, 0.0010), 0.05, 1.0)

        # Soft high apple cheeks, while lower jaw remains narrow.
        lm = out[visual.K].copy()
        cheek_r = (lm[40] + lm[31] + lm[48]) / 3.0
        cheek_l = (lm[46] + lm[35] + lm[54]) / 3.0
        for c in (cheek_r, cheek_l):
            shift(out, h, c, (0.043, 0.038, 0.038), (0.0, -0.0021, 0.0008), 0.02, 1.10)

        # Smaller ears tucked into the hair silhouette.
        lm = out[visual.K].copy()
        for ii in (0, 16):
            c = lm[ii]
            scale(out, h, c, (0.035, 0.044, 0.057), (0.76, 0.84, 0.78), 0.0, 1.10)
            shift(out, h, c, (0.035, 0.044, 0.057), ((0.0045 if c[0] < 0 else -0.0045), 0.0020, 0.0), 0.0, 1.05)

        # Coherent source-eye placement (visible eyes are rebuilt by eye_system).
        lm = out[visual.K].copy()
        for ids in eye_groups:
            ids = np.asarray(ids, np.int64)
            c = out[ids].mean(0)
            target = lm[36:42].mean(0) if c[0] < 0 else lm[42:48].mean(0)
            p = out[ids].copy()
            p += np.array([target[0] - c[0], target[1] - c[1] - 0.0015, target[2] - c[2]])
            c2 = p.mean(0)
            p = c2 + (p - c2) * np.array([1.12, 1.00, 1.10])
            out[ids] = p

        return out

    original_make = visual.ORIGINAL_MAKE_MATERIAL

    def final_material(name, color, metallic=0.0, roughness=0.48, emission=None):
        overrides = {
            "AINA_Skin": ((0.84, 0.70, 0.67, 1), 0.0, 0.40, None),
            "AINA_EyeWhite": ((0.96, 0.97, 0.99, 1), 0.0, 0.20, None),
            "AINA_Iris": ((0.25, 0.53, 0.66, 1), 0.01, 0.17, None),
            "AINA_Pupil": ((0.008, 0.012, 0.020, 1), 0.0, 0.18, None),
            "AINA_Hair_Silver": ((0.76, 0.80, 0.87, 1), 0.05, 0.27, None),
            "AINA_Suit_Pearl": ((0.78, 0.83, 0.90, 1), 0.10, 0.29, None),
            # Neutral lips must not expose the old white FaceVerse tooth block.
            "AINA_Teeth": ((0.34, 0.075, 0.085, 1), 0.0, 0.45, None),
            "AINA_MouthInner": ((0.28, 0.045, 0.060, 1), 0.0, 0.48, None),
            "AINA_Lip": ((0.62, 0.22, 0.25, 1), 0.0, 0.35, None),
        }
        if name in overrides:
            color, metallic, roughness, emission = overrides[name]
        return original_make(name, color, metallic, roughness, emission)

    # Enlarge the real geometry eye cards + iris/pupil, preserving their shape keys.
    old_almond = eye_system._almond
    old_disc = eye_system._disc

    def final_almond(name, center, material, side):
        c = np.asarray(center, float)
        ob = old_almond(name, center, material, side)
        if ob.data.shape_keys:
            for kb in ob.data.shape_keys.key_blocks:
                for p in kb.data:
                    p.co.x = c[0] + (p.co.x - c[0]) * 1.26
                    p.co.z = c[2] + (p.co.z - c[2]) * 1.30
        return ob

    def final_disc(name, center, radius, material, side, pupil=False):
        c = np.asarray(center, float)
        ob = old_disc(name, center, radius, material, side, pupil=pupil)
        factor = 1.34
        if ob.data.shape_keys:
            for kb in ob.data.shape_keys.key_blocks:
                for p in kb.data:
                    p.co.x = c[0] + (p.co.x - c[0]) * factor
                    p.co.z = c[2] + (p.co.z - c[2]) * factor
        return ob

    eye_system._almond = final_almond
    eye_system._disc = final_disc

    visual.polish_real_face = final_polish
    visual.base.enable_addons = already_enabled
    visual.base.create_body = release.create_native_body
    visual.base.make_material = final_material

    eye_system.install(visual, release)
    hair_system.install(visual)

    # Add lashes, brows and actual lip material on the deforming production face.
    eye_face_factory = visual.base.create_face_objects

    def create_face_with_details(face_path, height, skin, eye_mat, teeth_mat, mouth_mat):
        result = eye_face_factory(face_path, height, skin, eye_mat, teeth_mat, mouth_mat)
        head, eyes, mapped, groups, head_root, oral_roots, tongue_ids = result
        lm = mapped[visual.K]
        lash_mat = final_material("AINA_Pupil", (0.008, 0.012, 0.020, 1), 0, 0.18)
        brow_mat = final_material("AINA_Brow", (0.23, 0.22, 0.27, 1), 0, 0.34)
        lip_mat = final_material("AINA_Lip", (0.62, 0.22, 0.25, 1), 0, 0.35)
        head.data.materials.append(lip_mat)
        lip_index = len(head.data.materials) - 1

        def curve(name, pts, radius, mat):
            ob = visual.base.create_curve(name, [tuple(p) for p in pts], radius, mat)
            ob.parent = head
            return ob

        for side, rr in (("R", range(36, 42)), ("L", range(42, 48))):
            c = lm[list(rr)].mean(0)
            rx = 0.0215
            pts = [
                (c[0]-rx, c[1]-0.0040, c[2]+0.0004),
                (c[0]-rx*0.52, c[1]-0.0042, c[2]+0.0050),
                (c[0], c[1]-0.0043, c[2]+0.0061),
                (c[0]+rx*0.52, c[1]-0.0042, c[2]+0.0050),
                (c[0]+rx, c[1]-0.0040, c[2]+0.0004),
            ]
            curve(f"AINA_Lash_{side}", pts, 0.00062, lash_mat)
            outer = np.array(pts[0 if side == "R" else -1], float)
            curve(
                f"AINA_LashTail_{side}",
                [outer, outer + np.array([-0.0042 if side == "R" else 0.0042, -0.0002, 0.0025])],
                0.00050,
                lash_mat,
            )

        for side, ids in (("R", list(range(17, 22))), ("L", list(range(22, 27)))):
            pts = [(float(p[0]), float(p[1]-0.0032), float(p[2]+0.0003)) for p in lm[ids]]
            curve(f"AINA_Brow_{side}", pts, 0.00105, brow_mat)

        mc = lm[48:60].mean(0)
        rx_l, rz_l = 0.0205, 0.0065
        for poly in head.data.polygons:
            if poly.material_index != 0:
                continue
            cen = np.mean([np.array(head.data.vertices[i].co) for i in poly.vertices], axis=0)
            q = ((cen[0]-mc[0])/rx_l)**2 + ((cen[2]-mc[2])/rz_l)**2
            if q < 1.0 and cen[1] < mc[1] + 0.010:
                poly.material_index = lip_index
        return result

    visual.base.create_face_objects = create_face_with_details

    # The previous 55 mm-radius collar physically covered the chin and half the
    # lips in portrait renders. Keep the same production object, but below chin.
    def final_collar_and_accent(rig, suit_mat, accent_mat):
        n = 48
        verts, faces = [], []
        z0, z1 = 1.468, 1.505
        for z in (z0, z1):
            for i in range(n):
                a = 2 * math.pi * i / n
                verts.append((0.047 * math.cos(a), 0.043 * math.sin(a), z))
        for i in range(n):
            j = (i + 1) % n
            faces.extend([(i, j, n+j), (i, n+j, n+i)])
        collar = visual.base.mesh_object("AINA_High_Collar", np.asarray(verts,float), np.asarray(faces,np.int32))
        visual.base.assign_single_material(collar, suit_mat)
        visual.base.bone_parent_preserve(collar, rig, "neck_01")
        v = np.array([[0,-.132,1.405],[.024,-.124,1.365],[0,-.142,1.330],[-.024,-.124,1.365],[0,-.112,1.365]],float)
        f = np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4],[0,3,2],[0,2,1]],np.int32)
        crystal = visual.base.mesh_object("AINA_Core_Crystal", v, f)
        visual.base.assign_single_material(crystal, accent_mat)
        visual.base.bone_parent_preserve(crystal, rig, "spine_03")

    visual.base.create_collar_and_accent = final_collar_and_accent

    # Final real-model expression renders. No reference/effect image generated.
    def final_setup_render(out: Path):
        scene = bpy.context.scene
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.render.image_settings.file_format = "PNG"
        scene.render.film_transparent = False
        scene.world.color = (0.12, 0.13, 0.16)
        for o in scene.objects:
            if o.type == "MESH":
                for p in o.data.polygons:
                    p.use_smooth = True
        for o in list(scene.objects):
            if o.type in {"LIGHT", "CAMERA"}:
                bpy.data.objects.remove(o, do_unlink=True)

        def area(name, loc, energy, size):
            d = bpy.data.lights.new(name, "AREA")
            d.energy = energy
            d.shape = "DISK"
            d.size = size
            o = bpy.data.objects.new(name, d)
            bpy.context.collection.objects.link(o)
            o.location = loc
            o.rotation_euler = (Vector((0,0,1.60)) - o.location).to_track_quat("-Z", "Y").to_euler()

        area("AINA_Key", (1.5,-2.0,2.35), 650, 3.0)
        area("AINA_Fill", (-1.6,-1.5,2.0), 390, 2.6)
        area("AINA_Rim", (0,1.7,2.25), 420, 2.5)

        cd = bpy.data.cameras.new("AINA_Camera")
        cam = bpy.data.objects.new("AINA_Camera", cd)
        bpy.context.collection.objects.link(cam)
        scene.camera = cam
        previews = out / "Preview"
        previews.mkdir(parents=True, exist_ok=True)

        def clear_all():
            for o in scene.objects:
                if o.type == "MESH" and o.data.shape_keys:
                    for kb in o.data.shape_keys.key_blocks:
                        kb.value = 0.0

        def apply(vals):
            for o in scene.objects:
                if o.type != "MESH" or not o.data.shape_keys:
                    continue
                for k, v in vals.items():
                    if k in o.data.shape_keys.key_blocks:
                        o.data.shape_keys.key_blocks[k].value = float(v)

        def render(name, loc, target, vals, res=(768,768)):
            clear_all()
            apply(vals)
            cam.location = loc
            cam.data.lens = 82
            cam.rotation_euler = (Vector(target)-cam.location).to_track_quat("-Z","Y").to_euler()
            scene.render.resolution_x = res[0]
            scene.render.resolution_y = res[1]
            scene.render.resolution_percentage = 100
            scene.render.filepath = str(previews/name)
            bpy.ops.render.render(write_still=True)

        cases = {
            "AINA_REAL_NEUTRAL_FRONT.png": {},
            "AINA_REAL_HAPPY_FRONT.png": {"mouthSmileLeft":.82,"mouthSmileRight":.82,"cheekSquintLeft":.30,"cheekSquintRight":.30},
            "AINA_REAL_SURPRISED_FRONT.png": {"browInnerUp":.55,"eyeWideLeft":.86,"eyeWideRight":.86,"jawOpen":.58},
            "AINA_REAL_BLINK_FRONT.png": {"eyeBlinkLeft":1.0,"eyeBlinkRight":1.0},
            "AINA_REAL_AA_FRONT.png": {"jawOpen":.72,"mouthFunnel":.20},
        }
        for name, vals in cases.items():
            render(name, (0,-1.10,1.612), (0,0,1.607), vals)
        render("AINA_REAL_NEUTRAL_3Q.png", (.34,-1.05,1.620), (0,0,1.602), {})
        render("AINA_REAL_FULL_BODY_FRONT.png", (0,-4.7,1.05), (0,0,.98), {}, (1024,1536))
        clear_all()
        return [str(p) for p in sorted(previews.glob("AINA_REAL_*.png"))]

    visual.base.setup_render = final_setup_render
    visual.main()


if __name__ == "__main__":
    main()

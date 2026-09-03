#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    raw = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    return {raw[index].lstrip('-'): raw[index + 1] for index in range(0, len(raw), 2)}


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def mesh_bounds(obj):
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (
        Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points))),
        Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points))),
    )


def material_indices(obj, keyword):
    names = [slot.material.name.lower() if slot.material else '' for slot in obj.material_slots]
    found = set()
    for polygon in obj.data.polygons:
        if polygon.material_index < len(names) and keyword in names[polygon.material_index]:
            found.update(polygon.vertices)
    return sorted(found)


def landmark_center(obj, keyword):
    indices = material_indices(obj, keyword)
    points = [obj.matrix_world @ obj.data.vertices[index].co for index in indices]
    if not points:
        return None
    return Vector((
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
        sum(point.z for point in points) / len(points),
    ))


def render(scene, camera, target, location, path, lens=90):
    camera.data.type = 'PERSP'
    camera.data.lens = lens
    camera.location = Vector(location)
    look_at(camera, target)
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    output = Path(args['out']).resolve()
    preview = output / 'PreviewCloseup'
    qa = output / 'QA'
    preview.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)

    face = next((obj for obj in bpy.data.objects if obj.type == 'MESH' and obj.name.lower().startswith('face')), None)
    if face is None:
        raise RuntimeError('AINA V10 close-up renderer cannot find Face')
    camera = next((obj for obj in bpy.data.objects if obj.type == 'CAMERA'), None)
    if camera is None:
        bpy.ops.object.camera_add()
        camera = bpy.context.object
    bpy.context.scene.camera = camera

    minimum, maximum = mesh_bounds(face)
    eye = landmark_center(face, 'eyewhite')
    target = (minimum + maximum) * 0.5
    if eye is not None:
        target.z = eye.z - 0.015
        target.y = eye.y - 0.010
    target.x = 0.0
    distance = 0.78

    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 640
    scene.render.resolution_y = 760
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.compression = 24
    scene.render.film_transparent = False
    scene.view_settings.exposure = -0.30
    try:
        scene.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        pass

    render(scene, camera, target, (target.x, target.y + distance, target.z), preview / 'AINA_V10_CLOSE_FRONT.png')
    render(scene, camera, target, (target.x + distance * 0.52, target.y + distance * 0.85, target.z), preview / 'AINA_V10_CLOSE_3Q.png')
    render(scene, camera, target, (target.x + distance, target.y, target.z), preview / 'AINA_V10_CLOSE_PROFILE.png')

    report = {
        'target': list(target),
        'distance': distance,
        'face_min': list(minimum),
        'face_max': list(maximum),
        'eye': list(eye) if eye is not None else None,
        'resolution': [640, 760],
    }
    (qa / 'AINA_V10_CLOSEUP_QA.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

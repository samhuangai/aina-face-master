#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

source = Path(__file__).with_name('finalize_aina_v10.py')
text = source.read_text(encoding='utf-8')
old_selection = "        bpy.context.view_layer.objects.active = duplicate\n        duplicate.select_set(True)\n        bpy.ops.object.convert(target='MESH')\n        duplicate.select_set(False)\n"
new_selection = "        bpy.ops.object.select_all(action='DESELECT')\n        bpy.context.view_layer.objects.active = duplicate\n        duplicate.select_set(True)\n        bpy.ops.object.convert(target='MESH')\n        duplicate.select_set(False)\n"
old_frame = "    review_objects = [face] + [obj for obj in meshes if 'body' in obj.name.lower()] + hair_objects\n"
new_frame = "    review_objects = [face] + hair_objects\n"
if old_selection not in text or old_frame not in text:
    raise RuntimeError('AINA V10 hotfix could not locate the expected source blocks')
text = text.replace(old_selection, new_selection).replace(old_frame, new_frame)
compiled = compile(text, str(source), 'exec')
namespace = {'__name__': '__main__', '__file__': str(source), '__package__': None}
exec(compiled, namespace, namespace)

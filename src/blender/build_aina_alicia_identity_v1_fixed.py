#!/usr/bin/env python3
"""Run the compressed AINA Alicia builder with Blender-runtime fixes applied."""
from __future__ import annotations

import ast
import base64
import zlib
from pathlib import Path

BOOTSTRAP = Path(__file__).with_name("build_aina_alicia_identity_v1.py")
module = ast.parse(BOOTSTRAP.read_text(encoding="utf-8"))
payload = None
for node in module.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_PAYLOAD" for t in node.targets):
        payload = ast.literal_eval(node.value)
        break
if payload is None:
    raise RuntimeError("AINA compressed payload was not found")
source = zlib.decompress(base64.b64decode(payload)).decode("utf-8")
source = source.replace(
    "sc.world.use_nodes=True;bg=sc.world.node_tree.nodes.get('Background')",
    "sc.world=sc.world or bpy.data.worlds.new('AINA_Studio_World');sc.world.use_nodes=True;bg=sc.world.node_tree.nodes.get('Background')",
)
source_path = Path("build_aina_alicia_v1_runtime.py").resolve()
source_path.write_text(source, encoding="utf-8")
exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})

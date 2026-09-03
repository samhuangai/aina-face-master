#!/usr/bin/env python3
from __future__ import annotations
import json, struct, sys
from pathlib import Path

VRM_KEYS = {'VRM', 'VRMC_vrm', 'VRMC_springBone', 'VRMC_materials_mtoon', 'VRMC_node_constraint'}


def read_glb(path: Path):
    data = path.read_bytes()
    magic, version, total = struct.unpack_from('<4sII', data, 0)
    if magic != b'glTF' or version != 2 or total != len(data):
        raise ValueError('Input is not a valid GLB 2.0 file')
    offset = 12
    document = None
    binary = b''
    while offset < total:
        length, chunk_type = struct.unpack_from('<II', data, offset)
        offset += 8
        chunk = data[offset:offset + length]
        offset += length
        if chunk_type == 0x4E4F534A:
            document = json.loads(chunk.decode('utf-8').rstrip(' \0'))
        elif chunk_type == 0x004E4942:
            binary = chunk
    if document is None:
        raise ValueError('GLB JSON chunk is missing')
    return document, binary


def strip_extensions(value):
    if isinstance(value, dict):
        extensions = value.get('extensions')
        if isinstance(extensions, dict):
            for key in list(extensions):
                if key in VRM_KEYS or key.startswith('VRMC_'):
                    extensions.pop(key, None)
            if not extensions:
                value.pop('extensions', None)
        for child in list(value.values()):
            strip_extensions(child)
    elif isinstance(value, list):
        for child in value:
            strip_extensions(child)


def write_glb(path: Path, document: dict, binary: bytes):
    document['extensionsUsed'] = [x for x in document.get('extensionsUsed', []) if x not in VRM_KEYS and not x.startswith('VRMC_')]
    document['extensionsRequired'] = [x for x in document.get('extensionsRequired', []) if x not in VRM_KEYS and not x.startswith('VRMC_')]
    if not document['extensionsUsed']:
        document.pop('extensionsUsed', None)
    if not document.get('extensionsRequired'):
        document.pop('extensionsRequired', None)
    encoded = json.dumps(document, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    encoded += b' ' * ((4 - len(encoded) % 4) % 4)
    binary += b'\0' * ((4 - len(binary) % 4) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    path.write_bytes(
        struct.pack('<4sII', b'glTF', 2, total)
        + struct.pack('<II', len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack('<II', len(binary), 0x004E4942)
        + binary
    )


def main():
    if len(sys.argv) != 3:
        raise SystemExit('strip_vrm_extensions.py INPUT.glb OUTPUT.glb')
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    document, binary = read_glb(source)
    strip_extensions(document)
    document.setdefault('asset', {})['generator'] = 'AINA Blender-safe GLB derived from validated VRM payload'
    write_glb(destination, document, binary)
    print({'source_bytes': source.stat().st_size, 'output_bytes': destination.stat().st_size, 'output': str(destination)})


if __name__ == '__main__':
    main()

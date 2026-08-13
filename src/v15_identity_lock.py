#!/usr/bin/env python3
import argparse
from pathlib import Path
import trimesh

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--base-full',type=Path,required=True)
 p.add_argument('--out',type=Path,default=Path('output_v150'))
 a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 mesh=trimesh.load(a.base_full,process=False,maintain_order=True)
 mesh.export(a.out/'AINA_v15_candidate.obj')

if __name__=='__main__': main()

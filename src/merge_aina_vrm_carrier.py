#!/usr/bin/env python3
"""Merge AINA VRM 1.0 semantics onto Blender's morph-preserving GLB.

Why this exists:
- Blender 4.5 native glTF export preserves all 52 AINA shape keys.
- VRM Addon v4.5.0's staging export in this exact scene drops all morph targets.

The native GLB is therefore the geometry/skin/morph carrier. Only VRM semantic
JSON extensions are transplanted from the semantic VRM. Referenced nodes are
remapped by stable name plus structural role (mesh node vs bone/hierarchy node),
which safely disambiguates the intentionally same-named AINA eye mesh/bone pair.
"""
from __future__ import annotations
import argparse, copy, json, struct
from pathlib import Path
from typing import Any

PRESET_BINDS={
    'happy':[('mouthSmileLeft',.82),('mouthSmileRight',.82),('cheekSquintLeft',.30),('cheekSquintRight',.30)],
    'angry':[('browDownLeft',.82),('browDownRight',.82),('mouthFrownLeft',.45),('mouthFrownRight',.45)],
    'sad':[('browInnerUp',.72),('mouthFrownLeft',.72),('mouthFrownRight',.72)],
    'relaxed':[('mouthSmileLeft',.22),('mouthSmileRight',.22),('eyeSquintLeft',.10),('eyeSquintRight',.10)],
    'surprised':[('browInnerUp',.55),('eyeWideLeft',.86),('eyeWideRight',.86),('jawOpen',.58)],
    'neutral':[],
    'aa':[('jawOpen',.72),('mouthFunnel',.20)],
    'ih':[('mouthStretchLeft',.62),('mouthStretchRight',.62)],
    'ou':[('mouthPucker',.78),('mouthFunnel',.55)],
    'ee':[('mouthSmileLeft',.38),('mouthSmileRight',.38),('mouthStretchLeft',.52),('mouthStretchRight',.52)],
    'oh':[('jawOpen',.48),('mouthFunnel',.78)],
    'blink':[('eyeBlinkLeft',1.0),('eyeBlinkRight',1.0)],
    'blinkLeft':[('eyeBlinkLeft',1.0)],
    'blinkRight':[('eyeBlinkRight',1.0)],
    'lookUp':[('eyeLookUpLeft',1.0),('eyeLookUpRight',1.0)],
    'lookDown':[('eyeLookDownLeft',1.0),('eyeLookDownRight',1.0)],
    'lookLeft':[('eyeLookOutLeft',.72),('eyeLookInRight',.72)],
    'lookRight':[('eyeLookInLeft',.72),('eyeLookOutRight',.72)],
}
EXPECTED_TARGETS=[
    'browDownLeft','browDownRight','browInnerUp','browOuterUpLeft','browOuterUpRight',
    'cheekPuff','cheekSquintLeft','cheekSquintRight','eyeBlinkLeft','eyeBlinkRight',
    'eyeLookDownLeft','eyeLookDownRight','eyeLookInLeft','eyeLookInRight','eyeLookOutLeft','eyeLookOutRight','eyeLookUpLeft','eyeLookUpRight','eyeSquintLeft','eyeSquintRight','eyeWideLeft','eyeWideRight',
    'jawForward','jawLeft','jawOpen','jawRight','mouthClose','mouthDimpleLeft','mouthDimpleRight','mouthFrownLeft','mouthFrownRight','mouthFunnel','mouthLeft','mouthLowerDownLeft','mouthLowerDownRight','mouthPressLeft','mouthPressRight','mouthPucker','mouthRight','mouthRollLower','mouthRollUpper','mouthShrugLower','mouthShrugUpper','mouthSmileLeft','mouthSmileRight','mouthStretchLeft','mouthStretchRight','mouthUpperUpLeft','mouthUpperUpRight','noseSneerLeft','noseSneerRight','tongueOut'
]

def read_glb(path:Path):
    b=path.read_bytes();magic,ver,total=struct.unpack_from('<4sII',b,0)
    if magic!=b'glTF' or ver!=2 or total!=len(b):raise RuntimeError(f'Invalid GLB: {path}')
    off=12;j=None;bin_chunks=[]
    while off<total:
        n,t=struct.unpack_from('<II',b,off);off+=8;chunk=b[off:off+n];off+=n
        if t==0x4E4F534A:j=json.loads(chunk.rstrip(b' \t\r\n\x00').decode('utf-8'))
        elif t==0x004E4942:bin_chunks.append(chunk)
    if j is None:raise RuntimeError(f'GLB JSON missing: {path}')
    return j,bin_chunks

def write_glb(path:Path,j:dict,bin_chunks:list[bytes]):
    jb=json.dumps(j,separators=(',',':'),ensure_ascii=False).encode('utf-8');jb+=b' '*((-len(jb))%4)
    chunks=[struct.pack('<II',len(jb),0x4E4F534A)+jb]
    for bb in bin_chunks:
        bb=bb+b'\x00'*((-len(bb))%4);chunks.append(struct.pack('<II',len(bb),0x004E4942)+bb)
    total=12+sum(map(len,chunks));path.write_bytes(struct.pack('<4sII',b'glTF',2,total)+b''.join(chunks))

def resolve_node_index(sem_node:dict,carrier_nodes:list[dict])->int:
    name=sem_node.get('name')
    if not name:raise RuntimeError('Referenced semantic node has no stable name')
    hits=[i for i,n in enumerate(carrier_nodes) if n.get('name')==name]
    if len(hits)==1:return hits[0]
    if not hits:raise RuntimeError(f'Carrier has no node named {name!r}')
    # First distinguish object/mesh nodes from armature-bone/hierarchy nodes.
    want_mesh='mesh' in sem_node;want_skin='skin' in sem_node
    narrowed=[i for i in hits if ('mesh' in carrier_nodes[i])==want_mesh and ('skin' in carrier_nodes[i])==want_skin]
    if len(narrowed)==1:return narrowed[0]
    if narrowed:hits=narrowed
    # Same-name eye object vs eye bone also differs by child hierarchy.
    want_children=bool(sem_node.get('children'))
    narrowed=[i for i in hits if bool(carrier_nodes[i].get('children'))==want_children]
    if len(narrowed)==1:return narrowed[0]
    if narrowed:hits=narrowed
    # Final deterministic structural signature. We intentionally do not choose
    # arbitrarily: a wrong eye node would make the avatar formally invalid.
    sig=lambda n:(('mesh' in n),('skin' in n),bool(n.get('children')),('camera' in n))
    ss=sig(sem_node);narrowed=[i for i in hits if sig(carrier_nodes[i])==ss]
    if len(narrowed)==1:return narrowed[0]
    raise RuntimeError(f'Ambiguous carrier nodes for {name!r}: {hits}; semantic signature={ss}')

def build_node_remap(sem_nodes:list[dict],carrier_nodes:list[dict]):
    cache={}
    def remap(idx:int)->int:
        if idx in cache:return cache[idx]
        if not (0<=idx<len(sem_nodes)):raise RuntimeError(f'Semantic node index out of range: {idx}')
        out=resolve_node_index(sem_nodes[idx],carrier_nodes);cache[idx]=out;return out
    return remap,cache

def remap_node_fields(value:Any,remap):
    if isinstance(value,dict):
        out={}
        for k,v in value.items():
            if k=='node' and isinstance(v,int):out[k]=remap(v)
            else:out[k]=remap_node_fields(v,remap)
        return out
    if isinstance(value,list):return [remap_node_fields(v,remap) for v in value]
    return copy.deepcopy(value)

def carrier_face(carrier:dict):
    nodes=carrier.get('nodes',[]);meshes=carrier.get('meshes',[]);hits=[]
    for ni,n in enumerate(nodes):
        mi=n.get('mesh')
        if n.get('name')=='AINA_Face_v15_5' and isinstance(mi,int) and 0<=mi<len(meshes):hits.append((ni,mi,meshes[mi]))
    if len(hits)!=1:raise RuntimeError(f'Expected one AINA_Face_v15_5 carrier node, got {[(x[0],x[1]) for x in hits]}')
    ni,mi,m=hits[0];names=m.get('extras',{}).get('targetNames',[])
    if names!=EXPECTED_TARGETS:raise RuntimeError(f'Carrier targetNames are not canonical 52 list: {len(names)}')
    counts=[len(p.get('targets',[])) for p in m.get('primitives',[])]
    if not counts or any(c!=52 for c in counts):raise RuntimeError(f'Carrier primitive morph counts invalid: {counts}')
    return ni,mi,names,counts

def validate_node_refs(value:Any,node_count:int,path='extensions'):
    if isinstance(value,dict):
        for k,v in value.items():
            if k=='node' and isinstance(v,int) and not (0<=v<node_count):raise RuntimeError(f'Bad node ref {v} at {path}.{k}')
            validate_node_refs(v,node_count,f'{path}.{k}')
    elif isinstance(value,list):
        for i,v in enumerate(value):validate_node_refs(v,node_count,f'{path}[{i}]')

def merge(semantic_path:Path,carrier_path:Path,out_path:Path,report_path:Path|None=None):
    sem,_=read_glb(semantic_path);carrier,bins=read_glb(carrier_path)
    sem_ext=sem.get('extensions',{})
    if 'VRMC_vrm' not in sem_ext or 'VRMC_springBone' not in sem_ext:raise RuntimeError('Semantic VRM is missing VRMC_vrm or VRMC_springBone')
    sem_nodes=sem.get('nodes',[]);nodes=carrier.get('nodes',[]);remap,cache=build_node_remap(sem_nodes,nodes)
    face_node,face_mesh,target_names,primitive_counts=carrier_face(carrier);target_index={n:i for i,n in enumerate(target_names)}
    vrm=remap_node_fields(sem_ext['VRMC_vrm'],remap)
    preset=vrm.setdefault('expressions',{}).setdefault('preset',{})
    missing_presets=[p for p in PRESET_BINDS if p not in preset]
    if missing_presets:raise RuntimeError(f'Semantic VRM missing presets: {missing_presets}')
    total_binds=0
    for pname,bindings in PRESET_BINDS.items():
        expr=preset[pname];binds=[]
        for target,weight in bindings:
            if target not in target_index:raise RuntimeError(f'Morph target {target} absent from carrier')
            binds.append({'node':face_node,'index':target_index[target],'weight':float(weight)})
        if binds:expr['morphTargetBinds']=binds
        else:expr.pop('morphTargetBinds',None)
        total_binds+=len(binds)
    spring=remap_node_fields(sem_ext['VRMC_springBone'],remap)
    carrier.setdefault('extensions',{})['VRMC_vrm']=vrm;carrier['extensions']['VRMC_springBone']=spring
    used=list(carrier.get('extensionsUsed',[]))
    for e in ('VRMC_vrm','VRMC_springBone'):
        if e not in used:used.append(e)
    carrier['extensionsUsed']=used
    if 'extensionsRequired' in carrier and not carrier['extensionsRequired']:carrier.pop('extensionsRequired',None)
    validate_node_refs(vrm,len(nodes),'VRMC_vrm');validate_node_refs(spring,len(nodes),'VRMC_springBone')
    human=vrm.get('humanoid',{}).get('humanBones',{});required=('hips','spine','chest','neck','head','leftUpperArm','rightUpperArm','leftLowerArm','rightLowerArm','leftHand','rightHand','leftUpperLeg','rightUpperLeg','leftLowerLeg','rightLowerLeg','leftFoot','rightFoot','leftEye','rightEye')
    missing_h=[x for x in required if not isinstance(human.get(x,{}).get('node'),int)]
    if missing_h:raise RuntimeError(f'Merged humanoid mappings missing: {missing_h}')
    springs=spring.get('springs',[]);spring_joints=sum(len(x.get('joints',[])) for x in springs)
    checks={'carrier_52_target_names':len(target_names)==52,'carrier_52_targets_each_primitive':all(x==52 for x in primitive_counts),'expression_binds_ge_20':total_binds>=20,'humanoid_required_mapped':not missing_h,'spring_chains_ge_3':len(springs)>=3,'spring_joints_ge_6':spring_joints>=6,'vrm_spec_1_0':vrm.get('specVersion')=='1.0'}
    if not all(checks.values()):raise RuntimeError(f'Merged VRM pre-pack checks failed: {checks}')
    write_glb(out_path,carrier,bins)
    verify,_=read_glb(out_path);vface=carrier_face(verify);vpreset=verify['extensions']['VRMC_vrm']['expressions']['preset'];verified_binds=sum(len(x.get('morphTargetBinds',[])) for x in vpreset.values())
    report={'product':'AINA VRM Morph Carrier Merge','pass':True,'semantic_source':str(semantic_path),'carrier_source':str(carrier_path),'output':str(out_path),'output_bytes':out_path.stat().st_size,'face_node':face_node,'face_mesh':face_mesh,'target_names_count':len(vface[2]),'primitive_target_counts':vface[3],'preset_morph_bind_total':verified_binds,'spring_count':len(springs),'spring_joint_count':spring_joints,'node_remap_count':len(cache),'checks':checks}
    if report_path:report_path.parent.mkdir(parents=True,exist_ok=True);report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--semantic-vrm',type=Path,required=True);ap.add_argument('--carrier-glb',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--report',type=Path);a=ap.parse_args();merge(a.semantic_vrm,a.carrier_glb,a.out,a.report)
if __name__=='__main__':main()

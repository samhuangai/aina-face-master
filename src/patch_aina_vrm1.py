#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, struct
from pathlib import Path
from typing import Any

JSON_CHUNK = 0x4E4F534A

FALLBACK_TARGET_NAMES = [
'Fcl_ALL_Neutral','Fcl_ALL_Angry','Fcl_ALL_Fun','Fcl_ALL_Joy','Fcl_ALL_Sorrow','Fcl_ALL_Surprised',
'Fcl_BRW_Angry','Fcl_BRW_Fun','Fcl_BRW_Joy','Fcl_BRW_Sorrow','Fcl_BRW_Surprised',
'Fcl_EYE_Natural','Fcl_EYE_Angry','Fcl_EYE_Close','Fcl_EYE_Close_R','Fcl_EYE_Close_L','Fcl_EYE_Fun','Fcl_EYE_Joy','Fcl_EYE_Joy_R','Fcl_EYE_Joy_L','Fcl_EYE_Sorrow','Fcl_EYE_Surprised','Fcl_EYE_Spread','Fcl_EYE_Iris_Hide','Fcl_EYE_Highlight_Hide',
'Fcl_MTH_Close','Fcl_MTH_Up','Fcl_MTH_Down','Fcl_MTH_Angry','Fcl_MTH_Small','Fcl_MTH_Large','Fcl_MTH_Neutral','Fcl_MTH_Fun','Fcl_MTH_Joy','Fcl_MTH_Sorrow','Fcl_MTH_Surprised','Fcl_MTH_SkinFung','Fcl_MTH_SkinFung_R','Fcl_MTH_SkinFung_L','Fcl_MTH_A','Fcl_MTH_I','Fcl_MTH_U','Fcl_MTH_E','Fcl_MTH_O',
'Fcl_HA_Hide','Fcl_HA_Fung1','Fcl_HA_Fung1_Low','Fcl_HA_Fung1_Up','Fcl_HA_Fung2','Fcl_HA_Fung2_Low','Fcl_HA_Fung2_Up','Fcl_HA_Fung3','Fcl_HA_Fung3_Up','Fcl_HA_Fung3_Low','Fcl_HA_Short','Fcl_HA_Short_Up','Fcl_HA_Short_Low']

BONE_BY_VRM1 = {
    'hips':'J_Bip_C_Hips','spine':'J_Bip_C_Spine','chest':'J_Bip_C_Chest','upperChest':'J_Bip_C_UpperChest','neck':'J_Bip_C_Neck','head':'J_Bip_C_Head',
    'leftEye':'J_Adj_L_FaceEye','rightEye':'J_Adj_R_FaceEye',
    'leftShoulder':'J_Bip_L_Shoulder','leftUpperArm':'J_Bip_L_UpperArm','leftLowerArm':'J_Bip_L_LowerArm','leftHand':'J_Bip_L_Hand',
    'rightShoulder':'J_Bip_R_Shoulder','rightUpperArm':'J_Bip_R_UpperArm','rightLowerArm':'J_Bip_R_LowerArm','rightHand':'J_Bip_R_Hand',
    'leftUpperLeg':'J_Bip_L_UpperLeg','leftLowerLeg':'J_Bip_L_LowerLeg','leftFoot':'J_Bip_L_Foot','leftToes':'J_Bip_L_ToeBase',
    'rightUpperLeg':'J_Bip_R_UpperLeg','rightLowerLeg':'J_Bip_R_LowerLeg','rightFoot':'J_Bip_R_Foot','rightToes':'J_Bip_R_ToeBase',
    'leftThumbMetacarpal':'J_Bip_L_Thumb1','leftThumbProximal':'J_Bip_L_Thumb2','leftThumbDistal':'J_Bip_L_Thumb3',
    'rightThumbMetacarpal':'J_Bip_R_Thumb1','rightThumbProximal':'J_Bip_R_Thumb2','rightThumbDistal':'J_Bip_R_Thumb3'}
for side, cap in [('left','L'),('right','R')]:
    for finger in ('Index','Middle','Ring','Little'):
        BONE_BY_VRM1[f'{side}{finger}Proximal']=f'J_Bip_{cap}_{finger}1'
        BONE_BY_VRM1[f'{side}{finger}Intermediate']=f'J_Bip_{cap}_{finger}2'
        BONE_BY_VRM1[f'{side}{finger}Distal']=f'J_Bip_{cap}_{finger}3'

PRESET_BY_TARGET = {
    'neutral':'Fcl_ALL_Neutral','happy':'Fcl_ALL_Joy','angry':'Fcl_ALL_Angry','sad':'Fcl_ALL_Sorrow','relaxed':'Fcl_ALL_Fun','surprised':'Fcl_ALL_Surprised',
    'aa':'Fcl_MTH_A','ih':'Fcl_MTH_I','ou':'Fcl_MTH_U','ee':'Fcl_MTH_E','oh':'Fcl_MTH_O','blink':'Fcl_EYE_Close','blinkLeft':'Fcl_EYE_Close_L','blinkRight':'Fcl_EYE_Close_R'}

def parse_glb(path: Path):
    blob=path.read_bytes(); magic,version,total=struct.unpack_from('<4sII',blob,0)
    if magic!=b'glTF' or version!=2 or total!=len(blob): raise ValueError('Input is not a valid glTF 2.0 GLB')
    chunks=[]; off=12
    while off<total:
        size,kind=struct.unpack_from('<II',blob,off); off+=8; chunks.append((kind,blob[off:off+size])); off+=size
    if not chunks or chunks[0][0]!=JSON_CHUNK: raise ValueError('First GLB chunk must be JSON')
    doc=json.loads(chunks[0][1].decode('utf-8').rstrip(' \t\r\n\0')); return doc,chunks

def write_glb(path: Path, doc: dict[str,Any], chunks):
    raw=json.dumps(doc,ensure_ascii=False,separators=(',',':')).encode('utf-8'); raw+=b' '*((4-len(raw)%4)%4)
    all_chunks=[(JSON_CHUNK,raw)]+list(chunks[1:]); total=12+sum(8+len(x) for _,x in all_chunks)
    out=bytearray(struct.pack('<4sII',b'glTF',2,total))
    for kind,data in all_chunks:
        if len(data)%4: raise ValueError('GLB chunk alignment error')
        out+=struct.pack('<II',len(data),kind)+data
    path.write_bytes(out)

def node_index(doc,name):
    for i,n in enumerate(doc.get('nodes',[])):
        if n.get('name')==name:return i
    for i,n in enumerate(doc.get('nodes',[])):
        if name in n.get('name',''):return i
    return None

def find_face(doc):
    best=(-1,None,[],0)
    for mi,m in enumerate(doc.get('meshes',[])):
        names=(m.get('extras') or {}).get('targetNames') or []; count=max((len(p.get('targets',[])) for p in m.get('primitives',[])),default=0); score=max(len(names),count)
        if score>best[0]:best=(score,mi,list(names),count)
    if best[1] is None or best[0]<=0:raise ValueError('No morph-target face mesh')
    nodes=[i for i,n in enumerate(doc.get('nodes',[])) if n.get('mesh')==best[1]]
    if not nodes:raise ValueError('No node references face mesh')
    names=best[2]
    if (not names or all(str(n).startswith('target_') for n in names)) and best[3]==len(FALLBACK_TARGET_NAMES):
        names=list(FALLBACK_TARGET_NAMES); doc['meshes'][best[1]].setdefault('extras',{})['targetNames']=names
    return best[1],nodes[0],names

def expression(node,index,preset):
    return {'morphTargetBinds':[{'node':node,'index':index,'weight':1.0}],'isBinary':False,'overrideBlink':'block' if preset in {'blink','blinkLeft','blinkRight'} else 'none','overrideLookAt':'none','overrideMouth':'block' if preset in {'aa','ih','ou','ee','oh'} else 'none'}

def patch(doc,author,version):
    mesh,face_node,names=find_face(doc); indices={n:i for i,n in enumerate(names)}; human={}; required={'hips','spine','head','leftUpperLeg','leftLowerLeg','leftFoot','rightUpperLeg','rightLowerLeg','rightFoot','leftUpperArm','leftLowerArm','leftHand','rightUpperArm','rightLowerArm','rightHand'}; missing=[]
    for vrm_name,node_name in BONE_BY_VRM1.items():
        i=node_index(doc,node_name)
        if i is not None:human[vrm_name]={'node':i}
        elif vrm_name in required:missing.append((vrm_name,node_name))
    if missing:raise ValueError(f'Missing required humanoid nodes: {missing}')
    presets={}; missing_targets=[]
    for preset,target in PRESET_BY_TARGET.items():
        if target in indices:presets[preset]=expression(face_node,indices[target],preset)
        else:missing_targets.append(target)
    meta={'name':'AINA','version':version,'authors':[author],'copyrightInformation':'AINA CORE; VRoid sample base attribution retained in package documentation.','contactInformation':'Shenzhen Uoon Technology Co., Ltd.','references':['AINA approved character reference sheet','VRoid female sample base'],'thirdPartyLicenses':'Technical base derived from the VRoid female sample distributed in madjin/vrm-samples. Review included source terms before external redistribution.','licenseUrl':'https://vrm.dev/licenses/1.0/','avatarPermission':'onlyAuthor','allowExcessivelyViolentUsage':False,'allowExcessivelySexualUsage':False,'commercialUsage':'corporation','allowPoliticalOrReligiousUsage':False,'allowAntisocialOrHateUsage':False,'creditNotation':'required','allowRedistribution':False,'modification':'allowModification','otherLicenseUrl':'https://github.com/madjin/vrm-samples/blob/master/README.md'}
    doc.setdefault('extensions',{})['VRMC_vrm']={'specVersion':'1.0','meta':meta,'humanoid':{'humanBones':human},'firstPerson':{},'lookAt':{'offsetFromHeadBone':[0.0,0.06,0.0],'type':'bone','rangeMapHorizontalInner':{'inputMaxValue':90.0,'outputScale':10.0},'rangeMapHorizontalOuter':{'inputMaxValue':90.0,'outputScale':10.0},'rangeMapVerticalDown':{'inputMaxValue':90.0,'outputScale':10.0},'rangeMapVerticalUp':{'inputMaxValue':90.0,'outputScale':10.0}},'expressions':{'preset':presets,'custom':{}}}
    used=doc.setdefault('extensionsUsed',[]); req=doc.setdefault('extensionsRequired',[])
    if 'VRMC_vrm' not in used:used.append('VRMC_vrm')
    if 'VRMC_vrm' not in req:req.append('VRMC_vrm')
    return {'face_mesh':mesh,'face_node':face_node,'target_count':len(names),'presets':sorted(presets),'missing_expression_targets':missing_targets,'humanoid_bones':len(human),'required_humanoid_ok':True,'specVersion':'1.0'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--report',type=Path); ap.add_argument('--author',default='samhuangai / AINA CORE'); ap.add_argument('--version',default='1.0'); a=ap.parse_args()
    doc,chunks=parse_glb(a.input); report=patch(doc,a.author,a.version); a.output.parent.mkdir(parents=True,exist_ok=True); write_glb(a.output,doc,chunks); verified,_=parse_glb(a.output); assert verified['extensions']['VRMC_vrm']['specVersion']=='1.0'
    if a.report:a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

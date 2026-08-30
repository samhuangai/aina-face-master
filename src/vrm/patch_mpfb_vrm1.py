#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,struct
from pathlib import Path
ARKIT52=['browDownLeft','browDownRight','browInnerUp','browOuterUpLeft','browOuterUpRight','cheekPuff','cheekSquintLeft','cheekSquintRight','eyeBlinkLeft','eyeBlinkRight','eyeLookDownLeft','eyeLookDownRight','eyeLookInLeft','eyeLookInRight','eyeLookOutLeft','eyeLookOutRight','eyeLookUpLeft','eyeLookUpRight','eyeSquintLeft','eyeSquintRight','eyeWideLeft','eyeWideRight','jawForward','jawLeft','jawOpen','jawRight','mouthClose','mouthDimpleLeft','mouthDimpleRight','mouthFrownLeft','mouthFrownRight','mouthFunnel','mouthLeft','mouthLowerDownLeft','mouthLowerDownRight','mouthPressLeft','mouthPressRight','mouthPucker','mouthRight','mouthRollLower','mouthRollUpper','mouthShrugLower','mouthShrugUpper','mouthSmileLeft','mouthSmileRight','mouthStretchLeft','mouthStretchRight','mouthUpperUpLeft','mouthUpperUpRight','noseSneerLeft','noseSneerRight','tongueOut']
def args():
 p=argparse.ArgumentParser();p.add_argument('--glb',type=Path,required=True);p.add_argument('--out',type=Path,required=True);return p.parse_args()
def read(path):
 data=path.read_bytes();magic,ver,total=struct.unpack_from('<4sII',data,0)
 if magic!=b'glTF' or ver!=2:raise ValueError('Not GLB 2.0')
 off=12;g=None;blob=b''
 while off<total:
  n,t=struct.unpack_from('<II',data,off);off+=8;c=data[off:off+n];off+=n
  if t==0x4E4F534A:g=json.loads(c.decode('utf-8').rstrip(' \0'))
  elif t==0x004E4942:blob=c
 return g,blob
def write(path,g,blob):
 g['buffers'][0]['byteLength']=len(blob);js=json.dumps(g,separators=(',',':'),ensure_ascii=False).encode();js+=b' '*((-len(js))%4);blob+=b'\0'*((-len(blob))%4);total=12+8+len(js)+8+len(blob)
 path.write_bytes(struct.pack('<4sII',b'glTF',2,total)+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(blob),0x004E4942)+blob)
def main():
 a=args();g,blob=read(a.glb);nodes=g.get('nodes',[]);meshes=g.get('meshes',[]);names={n.get('name','').lower():i for i,n in enumerate(nodes)}
 def pick(*opts):
  for o in opts:
   if o.lower() in names:return names[o.lower()]
  for o in opts:
   for k,v in names.items():
    if k.endswith(o.lower()) or o.lower() in k:return v
  return None
 bone_candidates={'hips':['pelvis'],'spine':['spine_01'],'chest':['spine_02'],'upperChest':['spine_03'],'neck':['neck_01','neck'],'head':['head'],'leftShoulder':['clavicle_l'],'leftUpperArm':['upperarm_l'],'leftLowerArm':['lowerarm_l'],'leftHand':['hand_l'],'rightShoulder':['clavicle_r'],'rightUpperArm':['upperarm_r'],'rightLowerArm':['lowerarm_r'],'rightHand':['hand_r'],'leftUpperLeg':['thigh_l'],'leftLowerLeg':['calf_l'],'leftFoot':['foot_l'],'leftToes':['ball_l'],'rightUpperLeg':['thigh_r'],'rightLowerLeg':['calf_r'],'rightFoot':['foot_r'],'rightToes':['ball_r']}
 fingers={'Thumb':['thumb_01','thumb_02','thumb_03'],'Index':['index_01','index_02','index_03'],'Middle':['middle_01','middle_02','middle_03'],'Ring':['ring_01','ring_02','ring_03'],'Little':['pinky_01','pinky_02','pinky_03']}
 for side,suf in [('left','l'),('right','r')]:
  for digit,parts in fingers.items():
   labels=['Metacarpal','Proximal','Distal'] if digit=='Thumb' else ['Proximal','Intermediate','Distal']
   for label,part in zip(labels,parts):bone_candidates[side+digit+label]=[part+'_'+suf]
 human={};missing=[]
 for key,cands in bone_candidates.items():
  i=pick(*cands)
  if i is None:missing.append(key)
  else:human[key]={'node':i}
 target_locations={}
 for ni,n in enumerate(nodes):
  mi=n.get('mesh')
  if mi is None or mi>=len(meshes):continue
  for idx,name in enumerate(meshes[mi].get('extras',{}).get('targetNames',[])):target_locations.setdefault(name,(ni,idx))
 def mb(name,weight=1.0):
  loc=target_locations.get(name);return [] if loc is None else [{'node':loc[0],'index':loc[1],'weight':float(weight)}]
 def expr(binds,is_binary=False,ob='none',ol='none',om='none'):return {'morphTargetBinds':binds,'isBinary':is_binary,'overrideBlink':ob,'overrideLookAt':ol,'overrideMouth':om}
 def binds(*items):
  out=[]
  for item in items:out+=mb(item[0],item[1]) if isinstance(item,tuple) else mb(item)
  return out
 presets={'neutral':expr([]),'happy':expr(binds(('mouthSmileLeft',.8),('mouthSmileRight',.8),('cheekSquintLeft',.3),('cheekSquintRight',.3))),'angry':expr(binds(('browDownLeft',.8),('browDownRight',.8),('mouthFrownLeft',.45),('mouthFrownRight',.45))),'sad':expr(binds(('browInnerUp',.8),('mouthFrownLeft',.7),('mouthFrownRight',.7))),'relaxed':expr(binds(('mouthSmileLeft',.25),('mouthSmileRight',.25))),'surprised':expr(binds(('browInnerUp',.8),('eyeWideLeft',.7),('eyeWideRight',.7),('jawOpen',.65))),'aa':expr(binds(('jawOpen',.75),('mouthFunnel',.25)),om='block'),'ih':expr(binds(('mouthStretchLeft',.55),('mouthStretchRight',.55),('jawOpen',.18)),om='block'),'ou':expr(binds(('mouthPucker',.8),('mouthFunnel',.5)),om='block'),'ee':expr(binds(('mouthStretchLeft',.7),('mouthStretchRight',.7),('mouthSmileLeft',.3),('mouthSmileRight',.3)),om='block'),'oh':expr(binds(('mouthFunnel',.8),('jawOpen',.55)),om='block'),'blink':expr(binds('eyeBlinkLeft','eyeBlinkRight'),True,ob='block'),'blinkLeft':expr(binds('eyeBlinkLeft'),True,ob='block'),'blinkRight':expr(binds('eyeBlinkRight'),True,ob='block'),'lookUp':expr(binds(('eyeLookUpLeft',.8),('eyeLookUpRight',.8)),ol='block'),'lookDown':expr(binds(('eyeLookDownLeft',.8),('eyeLookDownRight',.8)),ol='block'),'lookLeft':expr(binds(('eyeLookOutLeft',.75),('eyeLookInRight',.75)),ol='block'),'lookRight':expr(binds(('eyeLookInLeft',.75),('eyeLookOutRight',.75)),ol='block')}
 custom={name:expr(mb(name)) for name in ARKIT52 if name in target_locations};head=pick('head');annotations=[{'node':i,'type':'auto'} for i,n in enumerate(nodes) if 'mesh' in n]
 vrm={'specVersion':'1.0','meta':{'name':'AINA','version':'1.0','authors':['Shenzhen Uoon Technology Co., Ltd.'],'copyrightInformation':'AINA Digital Human','contactInformation':'','references':['AINA Identity Master MPFB V3'],'thirdPartyLicenses':'MPFB2/MakeHuman base assets: CC0 1.0. See source package licenses.','avatarPermission':'onlyAuthor','allowExcessivelyViolentUsage':False,'allowExcessivelySexualUsage':False,'commercialUsage':'corporation','creditNotation':'required','allowRedistribution':False,'modification':'allowModification','otherLicenseUrl':'https://vrm.dev/en/vrm1/license/'},'humanoid':{'humanBones':human},'firstPerson':{'meshAnnotations':annotations},'lookAt':{'offsetFromHeadBone':[0,.06,0],'type':'expression','rangeMapHorizontalInner':{'inputMaxValue':45,'outputScale':1},'rangeMapHorizontalOuter':{'inputMaxValue':45,'outputScale':1},'rangeMapVerticalDown':{'inputMaxValue':35,'outputScale':1},'rangeMapVerticalUp':{'inputMaxValue':35,'outputScale':1}},'expressions':{'preset':presets,'custom':custom}}
 g.setdefault('extensions',{})['VRMC_vrm']=vrm
 spring_names=['Hair_L','Hair_R','Hair_Back'];roots=[];tips=[]
 for name in spring_names:
  r=len(nodes);t=r+1;nodes.append({'name':'AINA_'+name+'_Spring','translation':[0,0,0],'children':[t]});nodes.append({'name':'AINA_'+name+'_Tip','translation':[0,-.075,0],'children':[]})
  if head is not None:nodes[head].setdefault('children',[]).append(r)
  roots.append(r);tips.append(t)
 spring={'specVersion':'1.0','colliders':[],'colliderGroups':[],'springs':[]}
 if head is not None:
  spring['colliders']=[{'node':head,'shape':{'sphere':{'offset':[0,.07,0],'radius':.13}}}];spring['colliderGroups']=[{'name':'AINA_Head','colliders':[0]}]
  for name,r,t in zip(spring_names,roots,tips):spring['springs'].append({'name':name,'joints':[{'node':r,'hitRadius':.008,'stiffness':.58,'gravityPower':.12,'gravityDir':[0,-1,0],'dragForce':.32},{'node':t,'hitRadius':.006,'stiffness':.42,'gravityPower':.18,'gravityDir':[0,-1,0],'dragForce':.36}],'colliderGroups':[0],'center':head})
 g['extensions']['VRMC_springBone']=spring
 g['extensionsUsed']=[x for x in g.get('extensionsUsed',[]) if x not in {'VRM','VRMC_vrm','VRMC_springBone'}]+['VRMC_vrm','VRMC_springBone'];g['extensionsRequired']=[x for x in g.get('extensionsRequired',[]) if x not in {'VRM','VRMC_vrm','VRMC_springBone'}]+['VRMC_vrm','VRMC_springBone'];g.get('extensions',{}).pop('VRM',None);g['asset']['generator']='AINA MPFB VRM 1.0 Production Pipeline'
 a.out.parent.mkdir(parents=True,exist_ok=True);write(a.out,g,blob)
 report={'file':a.out.name,'bytes':a.out.stat().st_size,'vrm_spec':'1.0','humanoid_bones':len(human),'missing_humanoid':missing,'morph_targets':len(target_locations),'arkit_custom':len(custom),'preset_expressions':len(presets),'spring_chains':len(spring['springs']),'head_node':head};a.out.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

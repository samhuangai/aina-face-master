#!/usr/bin/env python3
import bpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'vendor'/'mpfb2'/'src'))
import mpfb
mpfb.register()
from mpfb.services import HumanService,TargetService
from mpfb.entities.objectproperties import HumanObjectProperties
body=HumanService.create_human()
body.name='AINA_Body_Base'
for k,v in {'gender':1.0,'age':0.5,'muscle':0.38,'weight':0.42,'height':0.58,'proportions':0.56}.items():
 HumanObjectProperties.set_value(k,v,entity_reference=body)
TargetService.reapply_macro_details(body)
print('AINA_BODY_READY',len(body.data.vertices))

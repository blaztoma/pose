"""Mesh marker calibration and contact baking for Blender; no rig constraints left behind."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpy
import numpy as np
from mathutils import Quaternion,Vector
from hand_contacts import correct_frame,anchor,sub,length,validate_taps


def getter(rig):
    def get(name):
        m=rig.matrix_world @ rig.pose.bones[name].matrix;q=m.to_quaternion()
        return tuple(m.translation*100),(q.x,q.y,q.z,q.w)
    return get


def fit_anchors(rig,meshes):
    rocket='Bip01 R Hand' in rig.data.bones
    names=(dict(upper='Bip01 R UpperArm',fore='Bip01 R Forearm',hand='Bip01 R Hand',finger='Bip01 R Finger2',
                tip='Bip01 R Finger22',previous='Bip01 R Finger21',surface='Bip01 L Hand',
                index='Bip01 L Finger1',middle='Bip01 L Finger2',pinky='Bip01 L Finger4') if rocket else
           dict(upper='upperarm_r',fore='lowerarm_r',hand='hand_r',finger='middle_01_r',tip='middle_03_r',
                previous='middle_02_r',surface='hand_l',index='index_01_l',middle='middle_01_l',pinky='pinky_01_l'))
    matrices={n:rig.matrix_world @ rig.data.bones[n].matrix_local for n in names.values()}
    points={n:np.array(m.translation)*100 for n,m in matrices.items()}
    wrist=points[names['surface']]
    direction=points[names['middle']]-wrist
    normal=np.cross(points[names['index']]-wrist,points[names['pinky']]-wrist)
    normal/=np.linalg.norm(normal)
    desired=wrist+.18*direction
    finger_direction=points[names['tip']]-points[names['previous']]
    finger_direction/=np.linalg.norm(finger_direction)
    markers={}
    for role,bone in [('tip',names['tip']),('surface',names['surface'])]:
        candidates=[]
        for mesh in meshes:
            group=mesh.vertex_groups.get(bone)
            if group is None: continue
            for v in mesh.data.vertices:
                weight=next((g.weight for g in v.groups if g.group==group.index),0)
                # Dorsal wrist vertices share much of their weight with wrist
                # corrective bones. A high hand-only threshold picks the side
                # of the wrist instead of the intended upper surface.
                if weight<(.65 if role=='tip' else .1): continue
                p=np.array(mesh.matrix_world @ v.co)*100
                if role=='tip':
                    score=-float(np.dot(p-points[bone],finger_direction))
                else:
                    offset=p-desired;altitude=float(np.dot(offset,normal))
                    if altitude<=0: continue
                    tangent=offset-altitude*normal
                    score=float(tangent@tangent+.08*altitude*altitude)
                candidates.append((score,mesh.name,v.index,p,weight))
        if not candidates: raise ValueError(f'No skin marker for {bone}')
        _,mesh_name,index,p,weight=min(candidates,key=lambda row:row[0])
        m=matrices[bone];q=m.to_quaternion()
        local=q.inverted() @ Vector(p-points[bone])
        markers[role]=dict(bone=bone,offset_cm=list(local),mesh=mesh_name,vertex=index,skin_weight=weight)
        mesh=next(o for o in meshes if o.name==mesh_name)
        influences=[]
        for group in mesh.data.vertices[index].groups:
            name=mesh.vertex_groups[group.group].name
            if group.weight<=0 or name not in rig.data.bones: continue
            matrix=rig.matrix_world @ rig.data.bones[name].matrix_local
            offset=matrix.to_quaternion().inverted() @ Vector(p-np.array(matrix.translation)*100)
            influences.append(dict(bone=name,weight=group.weight,offset_cm=list(offset)))
        total=sum(item['weight'] for item in influences)
        for item in influences: item['weight']/=total
        markers[role]['skin']=influences
        if role=='surface': markers[role]['normal']=list(q.inverted() @ Vector(normal))
    markers['bones']={k:names[k] for k in ('upper','fore','hand','finger')}
    markers['hand_length_cm']=float(np.linalg.norm(direction))
    markers['calibration']='Rest-mesh skin vertices, right middle tip and left dorsal wrist/hand'
    return markers


def bake_contacts(rig,meshes,profile,fps,frames):
    anchors=fit_anchors(rig,meshes);get=getter(rig);samples=[]
    scene=bpy.context.scene
    def set_rotation(name,q):
        bone=rig.pose.bones[name];world=rig.matrix_world @ bone.matrix
        desired=Quaternion((q[3],q[0],q[1],q[2])).to_matrix().to_4x4()
        desired.translation=world.translation
        # Keep the imported rig's uniform FBX scale.
        from mathutils import Matrix
        desired=desired @ Matrix.Diagonal((*world.to_scale(),1))
        bone.matrix=rig.matrix_world.inverted() @ desired
        bone.rotation_mode='QUATERNION'
        bone.keyframe_insert('rotation_quaternion',frame=scene.frame_current,group=name)
        bpy.context.view_layer.update()
    for f in range(frames):
        scene.frame_set(f+1);bpy.context.view_layer.update()
        sample=correct_frame(get,set_rotation,anchors,profile,f/fps)
        if sample:
            sample['frame']=f+1
            samples.append(sample)
    if not samples: raise ValueError('Contact profile does not overlap the clip')
    return dict(status='corrected',profile=profile,anchors=anchors,samples=samples,
                taps=validate_taps(samples,profile),
                validation='Skin-weighted markers; FBX markers checked after reimport')


def verify_contacts(rig,info):
    get=getter(rig);maximum=0;evaluated=[]
    for sample in info.get('samples',[]):
        bpy.context.scene.frame_set(sample['frame']);bpy.context.view_layer.update()
        error=length(sub(anchor(get,info['anchors']['tip']),sample['tip_cm']))
        maximum=max(maximum,error)
        evaluated.append(dict(sample,tip_cm=list(anchor(get,info['anchors']['tip'])),
                              surface_cm=list(anchor(get,info['anchors']['surface']))))
    if maximum>.1: raise ValueError(f'FBX contact marker changed by {maximum:.3f} cm')
    return dict(samples=len(info.get('samples',[])),max_export_marker_error_cm=maximum,
                taps=validate_taps(evaluated,info['profile']))


if __name__=='__main__':
    import argparse,sys
    p=argparse.ArgumentParser();p.add_argument('--fbx',required=True);p.add_argument('--out',required=True)
    p.add_argument('--reference',required=True)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.fbx(filepath=args.fbx,use_anim=False)
    rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
    anchors=fit_anchors(rig,[o for o in bpy.context.scene.objects if o.type=='MESH'])
    ref=json.loads(Path(args.reference).read_text())
    # Convert Blender global axes into UE's original exported reference axes.
    names=list(dict.fromkeys([*anchors['bones'].values(),anchors['tip']['bone'],anchors['surface']['bone'],
                            'middle_02_r','index_01_l','middle_01_l','pinky_01_l']))
    a=np.array([np.array((rig.matrix_world @ rig.data.bones[n].matrix_local).translation)*100 for n in names])
    b=np.array([ref[n]['p'] for n in names]);ac=a.mean(0);bc=b.mean(0)
    u,s,v=np.linalg.svd((a-ac).T@(b-bc));r=v.T@u.T
    # UE and Blender have opposite handedness: the axis conversion may reflect.
    translation=bc-r@ac
    fit_error=float(np.max(np.linalg.norm((r@a.T).T+translation-b,axis=1)))
    if fit_error>.01: raise ValueError(f'Mesh reference mismatch: {fit_error:.3f} cm')
    from hand_contacts import rotate,qinv
    used=set(names)
    for role in ('tip','surface'):
        item=anchors[role];bone=item['bone'];m=rig.matrix_world @ rig.data.bones[bone].matrix_local
        world=np.array(m.translation)*100+np.array(m.to_quaternion()@Vector(item['offset_cm']))
        item['offset_cm']=list(rotate(qinv(ref[bone]['q']),r@world+translation-np.array(ref[bone]['p'])))
        if role=='surface':
            item['normal']=list(rotate(qinv(ref[bone]['q']),r@np.array(m.to_quaternion()@Vector(item['normal']))))
        for influence in item['skin']:
            bone=influence['bone'];used.add(bone)
            m=rig.matrix_world @ rig.data.bones[bone].matrix_local
            world=np.array(m.translation)*100+np.array(m.to_quaternion()@Vector(influence['offset_cm']))
            influence['offset_cm']=list(rotate(qinv(ref[bone]['q']),r@world+translation-np.array(ref[bone]['p'])))
    anchors['mesh_asset']='/Game/MetaHumans/MH_Signer/Body/SKM_MH_Signer_BodyMesh'
    anchors['reference_bones']={name:ref[name] for name in sorted(used)}
    anchors['reference_fit_error_cm']=fit_error
    Path(args.out).write_text(json.dumps(anchors,indent=2))

"""Opt-in, timed hand contacts; shared by Blender and Unreal (centimetres)."""
import hashlib
import json
import math
from pathlib import Path


def add(a,b): return tuple(x+y for x,y in zip(a,b))
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def mul(a,s): return tuple(x*s for x in a)
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def length(a): return math.sqrt(dot(a,a))
def unit(a):
    n=length(a)
    if n<1e-10: raise ValueError('Degenerate contact geometry')
    return mul(a,1/n)
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def qmul(a,b):
    xyz=add(add(mul(b[:3],a[3]),mul(a[:3],b[3])),cross(a[:3],b[:3]))
    return (*xyz,a[3]*b[3]-dot(a[:3],b[:3]))
def qinv(q): return (-q[0],-q[1],-q[2],q[3])
def rotate(q,v): return qmul(qmul(q,(*v,0)),qinv(q))[:3]
def mixq(a,b,t):
    d=dot(a,b)
    if d<0: b=mul(b,-1);d=-d
    if d>.9995: return unit(add(mul(a,1-t),mul(b,t)))
    angle=math.acos(min(1,d));s=math.sin(angle)
    return add(mul(a,math.sin((1-t)*angle)/s),mul(b,math.sin(t*angle)/s))
def between(a,b,max_angle=math.pi):
    a,b=unit(a),unit(b);d=max(-1,min(1,dot(a,b)))
    if d>1-1e-10: return (0,0,0,1)
    axis=cross(a,b)
    if length(axis)<1e-8: axis=cross(a,min(((1,0,0),(0,1,0),(0,0,1)),key=lambda v:abs(dot(a,v))))
    angle=min(math.acos(d),max_angle)/2
    return (*mul(unit(axis),math.sin(angle)),math.cos(angle))


def interpolate(keys,t):
    if t<=keys[0][0]: return keys[0][1]
    for (a,x),(b,y) in zip(keys,keys[1:]):
        if t<=b:
            f=(t-a)/(b-a);f=f*f*(3-2*f)
            return x+(y-x)*f
    return keys[-1][1]


def validate_profile(profile):
    if profile.get('schema_version') not in (1,2) or profile.get('contact')!='right_middle_to_left_hand_dorsum':
        raise ValueError('Unsupported hand contact profile')
    for key,maximum in [('strength',1),('clearance_cm',10)]:
        values=profile[key]
        if len(values)<2 or any(len(k)!=2 or not all(math.isfinite(v) for v in k)
                or k[0]<0 or not 0<=k[1]<=maximum for k in values):
            raise ValueError(f'Invalid contact {key}')
        if any(b[0]<=a[0] for a,b in zip(values,values[1:])): raise ValueError('Contact times must increase')
    if profile['strength'][0][1]!=0 or profile['strength'][-1][1]!=0:
        raise ValueError('Contact must ease in and out')
    if profile['schema_version']==2:
        contacts=profile['contact_intervals_seconds']
        if not contacts or any(len(k)!=2 or not all(math.isfinite(v) for v in k)
                               or not 0<=k[0]<k[1] for k in contacts):
            raise ValueError('Invalid contact intervals')
        if any(b[0]<=a[1] for a,b in zip(contacts,contacts[1:])):
            raise ValueError('Contact intervals must be separated by a release')
        releases=profile['release_trajectories']
        if len(releases)!=len(contacts)-1:
            raise ValueError('Each pair of contacts needs its own release trajectory')
        for left,right,release in zip(contacts,contacts[1:],releases):
            keys=release['up_hand_lengths']
            if (len(keys)<3 or any(len(k)!=2 or not all(math.isfinite(v) for v in k)
                                  or not 0<=k[1]<=1 for k in keys)
                    or any(b[0]<=a[0] for a,b in zip(keys,keys[1:]))
                    or keys[0]!=[left[1],0] or keys[-1]!=[right[0],0]
                    or max(k[1] for k in keys)<=0):
                raise ValueError('Release must rise between adjacent contacts and return to zero')
        start,end=contacts[0][0],contacts[-1][1]
        if (interpolate(profile['strength'],start)!=1 or interpolate(profile['strength'],end)!=1
                or any(value!=1 for time,value in profile['strength'] if start<=time<=end)):
            raise ValueError('Annotated contact/release sequence requires full trajectory strength')
    return profile


def load_profile(source):
    path=Path(source).parent/'contact_profile.json'
    if not path.is_file(): return None
    profile=validate_profile(json.loads(path.read_text(encoding='utf-8')))
    # Manual timing belongs to one recording, never silently another signer.
    video=path.parent/profile['source_video']
    if video.parent.resolve()!=path.parent.resolve(): raise ValueError('Contact video must be adjacent')
    with video.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=profile['source_video_sha256']: raise ValueError('Contact source changed: review its timings')
    stem=Path(source).stem.removesuffix('_filtered').removesuffix('_30fps')
    if stem!=video.stem: raise ValueError('Contact profile belongs to a different video')
    return profile


def anchor(get, definition):
    if definition.get('skin'):
        # A wrist skin vertex is influenced by several bones, including wrist
        # correctives. Attaching it only to hand_l can put the target inside skin.
        result=(0.,0.,0.)
        for influence in definition['skin']:
            p,q=get(influence['bone'])
            result=add(result,mul(add(p,rotate(q,influence['offset_cm'])),influence['weight']))
        return result
    p,q=get(definition['bone'])
    return add(p,rotate(q,definition['offset_cm']))


def contact_target(get,anchors,profile,t):
    """Lock only the contact intervals; the intervening target follows a lift.

    Release height is a source-video annotation in units of the passive hand's
    wrist-to-middle-MCP length. World Z is up in both adapters. This deliberately
    avoids reusing a tiny surface-normal gap as the entire tapping movement.
    """
    surface=anchor(get,anchors['surface'])
    phase='legacy'
    if profile['schema_version']==2:
        for start,end in profile['contact_intervals_seconds']:
            if start<=t<=end:
                return surface,surface,'contact',0.
        for release in profile['release_trajectories']:
            keys=release['up_hand_lengths']
            if keys[0][0]<t<keys[-1][0]:
                scale=anchors['hand_length_cm']
                if not math.isfinite(scale) or scale<=0:
                    raise ValueError('Invalid avatar hand length')
                lift=interpolate(keys,t)*scale
                return add(surface,(0,0,lift)),surface,'release',lift
        phase='approach' if t<profile['contact_intervals_seconds'][0][0] else 'exit'
    normal=rotate(get(anchors['surface']['bone'])[1],anchors['surface']['normal'])
    clearance=interpolate(profile['clearance_cm'],t)
    return add(surface,mul(normal,clearance)),surface,phase,clearance


def two_bone(s,t,e,a,b):
    v=sub(t,s);d=length(v);direction=unit(v)
    reach=max(abs(a-b)+1e-5,min(a+b-1e-5,d));w=add(s,mul(direction,reach))
    bend=sub(sub(e,s),mul(direction,dot(sub(e,s),direction)))
    if length(bend)<1e-7:
        bend=cross(direction,min(((1,0,0),(0,1,0),(0,0,1)),key=lambda v:abs(dot(v,direction))))
    along=(a*a-b*b+reach*reach)/(2*reach)
    elbow=add(add(s,mul(direction,along)),mul(unit(bend),math.sqrt(max(0,a*a-along*along))))
    return elbow,w,abs(d-reach)


def correct_frame(get,set_rotation,anchors,profile,t):
    """Preserve link lengths, the passive hand and all unaffected local rotations.

    get returns world position/quaternion (xyzw); setter changes world rotation.
    Surface markers were fitted to this avatar's mesh, not the source person's size.
    """
    strength=interpolate(profile['strength'],t)
    if strength<=0: return None
    names=anchors['bones'];upper,fore,hand,finger=(names[k] for k in ('upper','fore','hand','finger'))
    s,uq=get(upper);e,fq=get(fore);w,hq=get(hand);m,mq=get(finger)
    tip=anchor(get,anchors['tip'])
    goal,surface,phase,clearance=contact_target(get,anchors,profile,t)
    delta=between(sub(tip,m),sub(goal,m),math.radians(55))
    aimed=add(m,rotate(delta,sub(tip,m)))
    wrist_target=add(w,sub(goal,aimed))
    shift=length(sub(wrist_target,w))
    if shift*strength>profile.get('max_wrist_shift_cm',12): raise ValueError(f'Contact wrist correction too large at {t:.3f}s: {shift*strength:.2f} cm')
    elbow,wrist,clamp=two_bone(s,wrist_target,e,length(sub(e,s)),length(sub(w,e)))
    if strength>.9999 and clamp>.1: raise ValueError(f'Unreachable contact ({clamp:.2f} cm)')
    goals={upper:qmul(between(sub(e,s),sub(elbow,s)),uq),
           fore:qmul(between(sub(w,e),sub(wrist,elbow)),fq),hand:hq,finger:qmul(delta,mq)}
    for name,original in [(upper,uq),(fore,fq),(hand,hq),(finger,mq)]:
        set_rotation(name,mixq(original,goals[name],strength))
    after=anchor(get,anchors['tip'])
    error=length(sub(after,goal))
    if strength>.9999 and error>.1: raise ValueError(f'Contact solve residual {error:.3f} cm')
    return dict(time_seconds=t,strength=strength,clearance_cm=clearance,phase=phase,
                before_error_cm=length(sub(tip,goal)),after_error_cm=error,wrist_shift_cm=shift,
                tip_cm=list(after),surface_cm=list(surface),target_cm=list(goal))


def validate_taps(samples,profile,minimum_lift_ratio=.9):
    """Check the sampled lift, not just the solver's own point residual.

    samples must use actual evaluated/reimported marker positions. A missing or
    collapsed release is a failure even when both endpoint contacts look good.
    """
    if profile['schema_version']!=2:
        return {'status':'legacy_profile'}
    contacts=[];releases=[]
    for start,end in profile['contact_intervals_seconds']:
        selected=[s for s in samples if start<=s['time_seconds']<=end]
        if not selected: raise ValueError('Contact interval has no sampled frame')
        error=max(length(sub(s['tip_cm'],s['surface_cm'])) for s in selected)
        if error>.1: raise ValueError(f'Tap contact residual {error:.3f} cm')
        contacts.append(dict(start=start,end=end,sampled_frames=len(selected),max_gap_cm=error))
    for release in profile['release_trajectories']:
        keys=release['up_hand_lengths'];start,end=keys[0][0],keys[-1][0]
        selected=[s for s in samples if start<s['time_seconds']<end]
        if not selected: raise ValueError('Release interval has no sampled frame')
        expected=max(s['target_cm'][2]-s['surface_cm'][2] for s in selected)
        actual=max(s['tip_cm'][2]-s['surface_cm'][2] for s in selected)
        if expected<=0 or actual<expected*minimum_lift_ratio:
            raise ValueError(f'Tap release collapsed: {actual:.3f} / {expected:.3f} cm')
        releases.append(dict(start=start,end=end,sampled_frames=len(selected),
                             expected_peak_up_cm=expected,actual_peak_up_cm=actual))
    return dict(status='passed',contacts=contacts,releases=releases)

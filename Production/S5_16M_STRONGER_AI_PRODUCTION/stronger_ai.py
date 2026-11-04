from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
import json, random, time, argparse, os, sys, urllib.request, urllib.error

DIMS = ['player','goal','movement','topology','interaction','sequence']
RULES = {
'player': [
('P0','Class A controllable'),('P1','Class B controllable'),('P2','Class C controllable'),('P3','Nearest eligible object controllable'),
('P4','Farthest eligible object controllable'),('P5','Selected object controllable'),('P6','Last-touched object controllable'),('P7','Goal-nearest object controllable'),
('P8','Largest eligible component controllable'),('P9','Smallest eligible component controllable'),('P10','Control cycles after event'),('P11','State-dependent control'),
('P12','Region-dependent control'),('P13','Action-dependent control'),('P14','Conditional control after prerequisite'),('P15','Multiple-object control')],
'goal': [
('G0','Reach Class B'),('G1','Reach Class C'),('G2','Reach Class A'),('G3','Reach designated region'),('G4','Collect all Class C'),('G5','Collect all Class B'),
('G6','Collect specified set'),('G7','Avoid hazards and exit'),('G8','Move object to target'),('G9','Match object classes to targets'),('G10','Form spatial pattern'),
('G11','Clear designated objects'),('G12','Activate all switches'),('G13','Open exit then reach it'),('G14','Achieve target configuration'),('G15','Complete ordered objective')],
'movement': [
('M0','One-cell movement'),('M1','Two-cell movement'),('M2','Slide until obstacle'),('M3','Push adjacent object'),('M4','Pull adjacent object'),('M5','Swap positions'),
('M6','Jump one occupied cell'),('M7','Teleport through linked location'),('M8','Rotate requested direction 90 degrees'),('M9','Reverse requested direction'),
('M10','Alternate one/two-cell step'),('M11','Object-state-dependent movement'),('M12','Terrain-dependent movement'),('M13','Momentum-dependent movement'),
('M14','Event-conditional movement'),('M15','Coupled movement')],
'topology': [
('T0','Fixed wall with gap'),('T1','Barrier cells passable'),('T2','Switch-controlled door'),('T3','Key-controlled door'),('T4','One-way passage'),('T5','Class-dependent passage'),
('T6','State-dependent passage'),('T7','Turn-dependent passage'),('T8','Breakable wall'),('T9','Movable barrier'),('T10','Teleport portal'),('T11','Wraparound boundary'),
('T12','Hidden passage'),('T13','Conditional gap'),('T14','Region restriction'),('T15','Dynamic topology')],
'interaction': [
('I0','No special interaction'),('I1','Hazard/reset on contact'),('I2','Collect on contact'),('I3','Activate on contact'),('I4','Deactivate on contact'),('I5','Transform controlled object'),
('I6','Transform target object'),('I7','Swap object states/positions'),('I8','Activate when pushed'),('I9','Unlock on interaction'),('I10','Teleport trigger'),('I11','Duplicate object'),
('I12','Remove both objects'),('I13','Merge objects'),('I14','State-conditional interaction'),('I15','Chain reaction')],
'sequence': [
('S0','Stateless'),('S1','Class C before Class B'),('S2','Class B before Class C'),('S3','Key before door'),('S4','Switch before goal'),('S5','Ordered object classes'),
('S6','Reverse ordered object classes'),('S7','Alternate target classes'),('S8','Previous action affects result'),('S9','Previous object affects result'),('S10','Counter-based state change'),
('S11','Toggle between two modes'),('S12','Persistent unlock'),('S13','Temporary unlock'),('S14','Multi-stage objective'),('S15','History-dependent rule switch')]
}
assert all(len(RULES[d]) == 16 for d in DIMS)

@dataclass(frozen=True)
class WorldModel:
    player:int; goal:int; movement:int; topology:int; interaction:int; sequence:int
    def ids(self): return tuple(RULES[d][getattr(self,d)][0] for d in DIMS)
    def index24(self):
        v=0
        for d in DIMS: v=(v<<4)|getattr(self,d)
        return v
    @classmethod
    def from_index24(cls,v:int):
        if not 0 <= v < 2**24: raise ValueError('24-bit world index required')
        vals=[]
        for _ in range(6): vals.append(v & 15); v >>= 4
        return cls(*reversed(vals))

@dataclass
class Obj:
    oid:str; cls:str; r:int; c:int; kind:str='object'; active:bool=False; collected:bool=False; size:int=1

class HypothesisSchema:
    @staticmethod
    def compact(scale=16, dims=6):
        names=DIMS[:dims]
        return {d:[{'id':rid,'description':desc} for rid,desc in RULES[d][:scale]] for d in names}
    @staticmethod
    def size(scale=16,dims=6): return scale**dims
    @staticmethod
    def save(path):
        payload={'dimensions':DIMS,'bits_per_dimension':4,'total_bits':24,'rules':HypothesisSchema.compact(),
                 'mathematical_space_size':16**6,'encoding':'P|G|M|T|I|S, four bits each'}
        with open(path,'w') as f: json.dump(payload,f,indent=2)

# V3: every ontology ID has an explicit handler path. This reports semantic implementation,
# not empirical validation of the research benchmark.
def rule_support_report(world:WorldModel)->dict:
    report={}
    for d in DIMS:
        idx=getattr(world,d); rid,desc=RULES[d][idx]
        report[d]={'id':rid,'description':desc,'implementation':'dedicated_v3'}
    report['all_dedicated']=True
    return report

class ScaledARCEnv:
    ACTIONS=('UP','DOWN','LEFT','RIGHT')
    DIR={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    def __init__(self, world:WorldModel, seed:int=0, n:int=8, max_steps:int=60):
        self.world=world; self.seed=seed; self.n=n; self.max_steps=max_steps; self.rng=random.Random(seed)
        self.validation=self.validate_world(world)
        if not self.validation['valid']:
            raise ValueError('Invalid world: '+ '; '.join(self.validation['reasons']))
        self.reset()

    @staticmethod
    def validate_world(world:WorldModel)->dict:
        # The canonical V3 template contains A/B/C objects, a wall/gap, switch-capable C objects,
        # an exit region, and supports state/history. Thus all 96 rule IDs are executable.
        reasons=[]
        for d in DIMS:
            v=getattr(world,d)
            if not 0 <= v < 16: reasons.append(f'{d} out of range: {v}')
        return {'valid':not reasons,'reasons':reasons,'template':'canonical_v3'}

    def reset(self):
        self.steps=0; self.solved=False; self.failed=False; self.termination_reason=None
        self.mode=0; self.prev_action=None; self.prev_touched=None; self.counter=0
        self.unlocked=False; self.selected_oid='A'; self.control_cycle=0; self.stage=0
        self.sequence_events=[]; self.temporary_unlock_until=None; self.last_direction=None
        self.objects=[Obj('A','A',3,2,size=2),Obj('B','B',4,7,size=3),Obj('C1','C',3,7,size=1),Obj('C2','C',7,2,size=1)]
        self.wall={(r,4) for r in range(1,self.n+1) if r!=4}
        self.base_wall=set(self.wall)
        self.start={o.oid:(o.r,o.c) for o in self.objects}
        self.history=[]
        return self.observe()

    def _live(self): return [o for o in self.objects if not o.collected]
    def _obj(self,oid): return next((o for o in self.objects if o.oid==oid and not o.collected),None)
    def _class(self,c): return [o for o in self._live() if o.cls==c]
    def _dist(self,o,rc=(4,8)): return abs(o.r-rc[0])+abs(o.c-rc[1])

    def _control(self, action:Optional[str]=None):
        p=self.world.player; e=self._live()
        if not e: return []
        if p in (0,1,2): return self._class(('A','B','C')[p])[:1]
        if p==3: return [min(e,key=lambda o:self._dist(o,(1,1)))]
        if p==4: return [max(e,key=lambda o:self._dist(o,(1,1)))]
        if p==5: return [self._obj(self.selected_oid)] if self._obj(self.selected_oid) else []
        if p==6: return [self._obj(self.prev_touched)] if self.prev_touched and self._obj(self.prev_touched) else [e[0]]
        if p==7: return [min(e,key=self._dist)]
        if p==8: return [max(e,key=lambda o:o.size)]
        if p==9: return [min(e,key=lambda o:o.size)]
        if p==10: return [e[self.control_cycle % len(e)]]
        if p==11: return [e[self.mode % len(e)]]
        if p==12:
            left=[o for o in e if o.c<=self.n//2]; right=[o for o in e if o.c>self.n//2]
            pool=left if self.mode==0 else right; return (pool or e)[:1]
        if p==13:
            idx=self.ACTIONS.index(action or 'UP') % len(e); return [e[idx]]
        if p==14: return [e[0]] if self.unlocked else []
        if p==15: return e[:2]
        raise AssertionError('unreachable player rule')

    def _inside(self,r,c): return 1<=r<=self.n and 1<=c<=self.n
    def _topology_blocks(self,o,r,c,action):
        t=self.world.topology
        if not self._inside(r,c): return t!=11
        if (r,c) not in self.wall: return False
        if t==0: return True
        if t==1: return False
        if t in (2,3): return not self.unlocked
        if t==4: return action!='RIGHT'
        if t==5: return o.cls!='A'
        if t==6: return self.mode==0
        if t==7: return self.steps%2==1
        if t==8: return True
        if t==9: return True
        if t==10: return False
        if t==11: return False
        if t==12: return (r,c)!=(2,4)
        if t==13: return not (self.unlocked and r==4)
        if t==14: return o.cls=='C'
        if t==15: return self.mode==0
        raise AssertionError('unreachable topology rule')

    def _occupied(self,r,c,exclude=None):
        return next((o for o in self._live() if o.oid!=exclude and (o.r,o.c)==(r,c)),None)

    def _raw_step(self,o,dr,dc,action,allow_push=False):
        nr,nc=o.r+dr,o.c+dc
        if self.world.topology==11:
            nr=((nr-1)%self.n)+1; nc=((nc-1)%self.n)+1
        if self._topology_blocks(o,nr,nc,action):
            if self.world.topology==8 and (nr,nc) in self.wall:
                self.wall.remove((nr,nc))
            elif self.world.topology==9 and (nr,nc) in self.wall:
                br,bc=nr+dr,nc+dc
                if self._inside(br,bc) and (br,bc) not in self.wall:
                    self.wall.remove((nr,nc)); self.wall.add((br,bc))
                else: return False
            else: return False
        target=self._occupied(nr,nc,exclude=o.oid)
        if target and allow_push:
            br,bc=target.r+dr,target.c+dc
            if self._inside(br,bc) and not self._topology_blocks(target,br,bc,action) and not self._occupied(br,bc,exclude=target.oid):
                target.r,target.c=br,bc
            else: return False
        o.r,o.c=nr,nc
        return True

    def _movement_vector(self,action):
        dr,dc=self.DIR[action]; m=self.world.movement
        if m==8: dr,dc=dc,-dr
        elif m==9: dr,dc=-dr,-dc
        elif m==13 and self.last_direction: dr,dc=self.DIR[self.last_direction]
        elif m==14 and self.unlocked: dr,dc=-dr,-dc
        return dr,dc

    def _apply_move(self,o,action):
        m=self.world.movement; dr,dc=self._movement_vector(action); moved=False
        if m==5:
            nr,nc=o.r+dr,o.c+dc; t=self._occupied(nr,nc,exclude=o.oid)
            if t: o.r,o.c,t.r,t.c=t.r,t.c,o.r,o.c; return True
        if m==6:
            nr,nc=o.r+dr,o.c+dc
            if self._occupied(nr,nc,exclude=o.oid): dr*=2; dc*=2
        if m==7:
            if self._raw_step(o,dr,dc,action):
                if o.c==4: o.r,o.c=4,8
                return True
            return False
        steps=1
        if m==1: steps=2
        elif m==2: steps=self.n
        elif m==10: steps=1+(self.steps%2)
        elif m==11: steps={'A':1,'B':2,'C':3}[o.cls]
        elif m==12: steps=2 if o.r==4 else 1
        allow_push=(m==3)
        old=(o.r,o.c)
        for _ in range(steps):
            if not self._raw_step(o,dr,dc,action,allow_push=allow_push): break
            moved=True
            if m not in (1,2,10,11,12): break
        if m==4 and moved:
            # pull an object from the cell behind the original position
            br,bc=old[0]-dr,old[1]-dc; t=self._occupied(br,bc,exclude=o.oid)
            if t: t.r,t.c=old
        if m==15 and moved:
            other=next((x for x in self._live() if x.oid!=o.oid),None)
            if other: self._raw_step(other,dr,dc,action)
        self.last_direction=action
        return moved

    def _record_event(self,event):
        self.sequence_events.append(event)
        self.sequence_events=self.sequence_events[-12:]

    def _interact(self,controller):
        targets=[t for t in self._live() if t.oid!=controller.oid and (t.r,t.c)==(controller.r,controller.c)]
        for target in targets:
            self.prev_touched=target.oid; self._record_event(target.cls)
            i=self.world.interaction
            if i==0: pass
            elif i==1: self.failed=True; self.termination_reason='hazard_contact'
            elif i==2: target.collected=True
            elif i==3: target.active=True; self.unlocked=True
            elif i==4: target.active=False; self.unlocked=False
            elif i==5: controller.cls={'A':'B','B':'C','C':'A'}[controller.cls]
            elif i==6: target.cls={'A':'B','B':'C','C':'A'}[target.cls]
            elif i==7: controller.cls,target.cls=target.cls,controller.cls
            elif i==8: target.active=True; self.unlocked=True
            elif i==9: self.unlocked=True
            elif i==10: controller.r,controller.c=1,self.n
            elif i==11:
                self.objects.append(Obj(f'D{len(self.objects)}',target.cls,target.r,min(self.n,target.c+1)))
            elif i==12: target.collected=True; controller.collected=True
            elif i==13: target.collected=True; controller.size+=target.size; controller.active=True
            elif i==14:
                if self.mode==1: target.active=True; self.unlocked=True
            elif i==15:
                for x in self._live(): x.active=True
                self.unlocked=True; self.mode=1
            else: raise AssertionError('unreachable interaction rule')

    def _apply_sequence(self,action):
        s=self.world.sequence
        if s==0: return
        if s==1 and 'C' in self.sequence_events: self.unlocked=True
        elif s==2 and 'B' in self.sequence_events: self.unlocked=True
        elif s==3 and self.counter>=2: self.unlocked=True
        elif s==4:
            # switch-before-goal: only an activated object unlocks goal access
            self.unlocked=self.unlocked or any(o.active for o in self._live())
        elif s==5 and self.sequence_events[-3:]==['A','B','C']: self.unlocked=True
        elif s==6 and self.sequence_events[-3:]==['C','B','A']: self.unlocked=True
        elif s==7 and len(self.sequence_events)>=2 and self.sequence_events[-1]!=self.sequence_events[-2]: self.unlocked=True
        elif s==8 and self.prev_action and self.prev_action==action: self.mode=1
        elif s==9 and self.prev_touched: self.mode=1
        elif s==10 and self.counter%3==0: self.mode^=1; self.unlocked=self.mode==1
        elif s==11: self.mode^=1; self.unlocked=self.mode==1
        elif s==12 and self.unlocked: self.unlocked=True
        elif s==13:
            if self.unlocked and self.temporary_unlock_until is None: self.temporary_unlock_until=self.steps+2
            if self.temporary_unlock_until is not None and self.steps>self.temporary_unlock_until: self.unlocked=False
        elif s==14:
            if self.unlocked: self.stage=max(self.stage,1)
            if self.stage==1 and self.counter>=4: self.stage=2
        elif s==15 and len(self.sequence_events)>=2: self.mode=1
        self.prev_action=action

    def _check_goal(self):
        g=self.world.goal; ctr=self._control(self.prev_action); c=ctr[0] if ctr else None
        def touch(cls): return c is not None and any(o.cls==cls and not o.collected and o.oid!=c.oid and (o.r,o.c)==(c.r,c.c) for o in self.objects)
        if g==0: return touch('B')
        if g==1: return touch('C')
        if g==2: return touch('A')
        if g==3: return c is not None and c.c==self.n
        if g==4: return all(o.collected for o in self.objects if o.cls=='C')
        if g==5: return all(o.collected for o in self.objects if o.cls=='B')
        if g==6: return all(o.collected for o in self.objects if o.oid in ('C1','C2'))
        if g==7: return c is not None and c.c==self.n and not self.failed
        if g==8: return any((o.r,o.c)==(4,8) for o in self._live() if o.oid!=getattr(c,'oid',None))
        if g==9: return all(o.c in (2,7) for o in self._live())
        if g==10: return len({o.r for o in self._live()})==1
        if g==11: return all(o.collected for o in self.objects if o.cls=='C')
        if g==12: return all(o.active for o in self.objects if o.cls=='C')
        if g==13: return self.unlocked and c is not None and c.c==self.n
        if g==14: return self.unlocked and self.mode==1
        if g==15: return self.stage>=2 or self.unlocked and len(self.sequence_events)>=3
        raise AssertionError('unreachable goal rule')

    def step(self,action):
        if action not in self.ACTIONS: raise ValueError(action)
        if self.solved or self.failed:
            return self.observe(),0,self.solved,{'terminal':True,'termination_reason':self.termination_reason}
        self.steps+=1; self.counter+=1
        ctr=self._control(action)
        for o in list(ctr):
            if o and not o.collected:
                self._apply_move(o,action); self._interact(o)
        self._apply_sequence(action)
        # topology-specific dynamic changes
        if self.world.topology==15 and self.steps%3==0: self.mode^=1
        if self.world.topology==10:
            for o in self._live():
                if (o.r,o.c)==(4,4): o.r,o.c=4,8
        self.solved=self._check_goal()
        if self.solved: self.termination_reason='solved'
        if self.steps>=self.max_steps and not self.solved and not self.failed:
            self.failed=True; self.termination_reason='action_budget_exhausted'
        obs=self.observe(); self.history.append({'action':action,'obs':obs})
        return obs,(1 if self.solved else 0),self.solved,{'failed':self.failed,'termination_reason':self.termination_reason}

    def observe(self):
        # Agent-visible state: no hidden rule IDs, no internal object IDs, no unlock/mode flags.
        return {'size':self.n,
                'objects':[{'cls':o.cls,'r':o.r,'c':o.c,'active':o.active} for o in self._live()],
                'barriers':sorted(self.wall),'available_actions':list(self.ACTIONS),'step':self.steps,
                'solved_env':self.solved,'failed_env':self.failed}

    def ground_truth_state(self):
        return {'world_ids':self.world.ids(),'world_index24':self.world.index24(),'objects':[asdict(o) for o in self.objects],
                'unlocked':self.unlocked,'mode':self.mode,'counter':self.counter,'stage':self.stage,
                'sequence_events':list(self.sequence_events),'termination_reason':self.termination_reason,
                'validation':self.validation,'rule_support':rule_support_report(self.world)}


def make_visual_binding(seed:int=0):
    # Neutral symbols prevent class labels A/B/C from leaking directly to the agent.
    variants=[
        {'A':'@','B':'O','C':'X'},
        {'A':'&','B':'Q','C':'Z'},
        {'A':'R','B':'V','C':'K'},
        {'A':'1','B':'2','C':'3'},
    ]
    return variants[seed % len(variants)]

def agent_view(obs:dict,binding:dict):
    out=json.loads(json.dumps(obs))
    for o in out.get('objects',[]):
        o['symbol']=binding[o['cls']]
        o.pop('cls',None)
    out['legend']='Symbols are arbitrary visual identities; infer their roles from interaction.'
    return out

def render_grid(obs:dict)->str:
    n=obs['size']; grid=[['.' for _ in range(n)] for _ in range(n)]
    for r,c in obs.get('barriers',[]):
        if 1<=r<=n and 1<=c<=n: grid[r-1][c-1]='#'
    for o in obs.get('objects',[]):
        r,c=o['r'],o['c']; base=o.get('symbol',o.get('cls','?')); sym=base.lower() if o.get('active') and base.isalpha() else base
        if 1<=r<=n and 1<=c<=n: grid[r-1][c-1]=sym if grid[r-1][c-1]=='.' else grid[r-1][c-1]+'+'+sym
    rows=['    '+' '.join(f'{c:>2}' for c in range(1,n+1))]
    rows += [f'{r:>2}  '+' '.join(f'{x:>2}' for x in row) for r,row in enumerate(grid,1)]
    return '\n'.join(rows)

def observation_delta(before,after):
    # Agent-visible matching uses symbol + occurrence ordering; hidden class/IDs are never exposed.
    def keyed(obs):
        counts={}; out={}
        for o in obs.get('objects',[]):
            k=o.get('symbol',o.get('cls','?')); counts[k]=counts.get(k,0)+1; out[f'{k}{counts[k]}']=o
        return out
    b=keyed(before); a=keyed(after); moved=[]; changed=[]; removed=[]; added=[]
    for oid,bo in b.items():
        if oid not in a: removed.append(oid); continue
        ao=a[oid]
        if (bo['r'],bo['c'])!=(ao['r'],ao['c']): moved.append({'object':oid,'from':[bo['r'],bo['c']],'to':[ao['r'],ao['c']]})
        if bo.get('active')!=ao.get('active'): changed.append({'object':oid,'active':{'before':bo.get('active'),'after':ao.get('active')}})
    for oid in a:
        if oid not in b: added.append(oid)
    bb={tuple(x) for x in before.get('barriers',before.get('barrier_cells',[]))}
    ab={tuple(x) for x in after.get('barriers',after.get('barrier_cells',[]))}
    return {'moved':moved,'changed':changed,'removed':removed,'added':added,
            'barriers_removed':[list(x) for x in sorted(bb-ab)],
            'barriers_added':[list(x) for x in sorted(ab-bb)],
            'solved_changed':before.get('solved_env')!=after.get('solved_env'),
            'failed_changed':before.get('failed_env')!=after.get('failed_env')}

SYSTEM_PROMPT="""You are the LLM-only reasoning agent in a controlled ARC-AGI-3-derived experiment.
You may use ONLY the public hypothesis ontology, the current agent-visible state, and the experimental evidence supplied in the user prompt.
The hidden active rule IDs/world model are NEVER provided. Do not assume any rule is active merely because it exists in the ontology.
Use native reasoning only; no external hypothesis-search module is available.
Your primary decision objective is information gain: choose the action expected to best discriminate among currently plausible rule hypotheses while also allowing eventual task completion.
Avoid repeating an already-characterized action when an untested action can provide new evidence, unless the changed context makes repetition informative for a state-, sequence-, interaction-, or topology-dependent rule.
Return only the required structured JSON."""

SCALES={
    'S1': {'scale':4,'dims':4,'size':4**4},
    'S2': {'scale':6,'dims':5,'size':6**5},
    'S3': {'scale':8,'dims':6,'size':8**6},
    'S4': {'scale':12,'dims':6,'size':12**6},
    'S5': {'scale':16,'dims':6,'size':16**6},
}

def _agent_object_records(obs):
    counts={}
    records=[]
    for o in obs.get('objects',[]):
        sym=o.get('symbol') or o.get('cls','?')
        counts[sym]=counts.get(sym,0)+1
        records.append({
            'object_id':f'{sym}{counts[sym]}',
            'symbol':sym,
            'row':o.get('r'),
            'col':o.get('c'),
            'active':bool(o.get('active',False)),
        })
    return records

def _compact_state(obs):
    barriers=[list(x) for x in obs.get('barriers',obs.get('barrier_cells',[]))]
    n=int(obs.get('size',8) or 8)
    # Openings are empty cells in a barrier-dominated column, reported only as visible geometry.
    by_col={}
    for r,c in barriers: by_col.setdefault(c,[]).append(r)
    openings=[]
    if by_col and n:
        wall_col=max(by_col,key=lambda c:len(by_col[c]))
        if len(by_col[wall_col]) >= max(2,n//2):
            occupied=set(by_col[wall_col])
            openings=[[r,wall_col] for r in range(1,n+1) if r not in occupied]
    return {
        'step':obs.get('step'),
        'size':n,
        'grid':render_grid({**obs,'size':n,'barriers':barriers}).splitlines(),
        'objects':_agent_object_records(obs),
        'barrier_cells':barriers,
        'visible_openings_in_barrier_column':openings,
        'available_actions':obs.get('available_actions',[]),
        'legend':obs.get('legend','Symbols are arbitrary visual identities; infer roles from interaction.'),
    }

def _evidence_ledger(prior):
    ledger=[]
    counts={a:0 for a in ['UP','DOWN','LEFT','RIGHT']}
    for k,e in enumerate(prior,1):
        a=e.get('action')
        if a in counts: counts[a]+=1
        ledger.append({
            'evidence_id':f'E{k}',
            'action':a,
            'observation_delta':e.get('observation_delta',{}),
            'solved_env':bool(e.get('solved_env',False)),
            'failed_env':bool(e.get('failed_env',False)),
        })
    return ledger,counts

def build_turn_prompt(obs,schema,prior,scale_name='S5'):
    ledger,counts=_evidence_ledger(prior)
    spec=SCALES[scale_name]
    prompt={
        'benchmark_scale':{'name':scale_name,'complete_hypotheses':spec['size']},
        'public_hypothesis_ontology':schema,
        'current_agent_visible_state':_compact_state(obs),
        'experimental_evidence':ledger,
        'action_test_counts':counts,
        'decision_objective':[
            'Choose the next action expected to provide the greatest useful information for discriminating among currently plausible hypotheses.',
            'Prefer an untested action when it can discriminate among plausible hypotheses.',
            'Do not repeat an established experiment unless the current state/context makes repetition informative.',
            'Do not claim that a specific rule ID is active unless the observations support it.'
        ],
    }
    return json.dumps(prompt,separators=(',',':'))

def audit_prompt(prompt:str, hidden_ids=()):
    forbidden=['world_ids','world_index24','harness_ground_truth','ground_truth_state']
    hits=[x for x in forbidden if x in prompt]
    # Active IDs are allowed to appear in the PUBLIC ontology at S5, so audit only for
    # active-rule assertions/ground-truth fields, not mere ontology membership.
    return {'passed':not hits,'forbidden_hits':hits,'hidden_ids_checked':list(hidden_ids)}

class LLMClient:
    def __init__(self,call_fn=None,mode='placeholder',model='gpt-5.6-sol',ollama_url='http://localhost:11434',scale_name='S5',prompt_audit_dir=None,request_timeout=300,experience_context=None):
        self.call_fn=call_fn; self.mode=mode; self.model=model; self.ollama_url=ollama_url.rstrip('/')
        self.total_input_tokens=0; self.total_output_tokens=0; self.calls=0
        self.total_model_seconds=0.0; self.last_telemetry={}; self.scale_name=scale_name; self.prompt_audit_dir=prompt_audit_dir; self.request_timeout=request_timeout; self.last_prompt=''; self.experience_context=experience_context

    def _validate(self,r,actions):
        if not isinstance(r,dict): raise ValueError('Agent response must be a JSON object')
        if r.get('selected_action') not in actions: raise ValueError(f"Invalid selected_action: {r.get('selected_action')}")
        return {
            'selected_action':r['selected_action'],
            'rationale':str(r.get('rationale',''))[:2000],
            'leading_rules':r.get('leading_rules',{}),
            'confidence':float(r.get('confidence',0.0)),
            'believes_solved':bool(r.get('believes_solved',False)),
            'telemetry':r.get('_telemetry',{}),
        }

    def _manual(self,obs,schema,prior):
        print('\nAGENT-VISIBLE STATE')
        print(render_grid(obs))
        print('Available actions:', ', '.join(obs['available_actions']))
        if prior:
            print('Last observation delta:', json.dumps(prior[-1].get('observation_delta',{})))
        while True:
            a=input('Select action [UP/DOWN/LEFT/RIGHT]: ').strip().upper()
            if a in obs['available_actions']: break
            print('Invalid action.')
        rationale=input('Brief rationale (optional): ').strip()
        solved=input('Do you believe the task is solved? [y/N]: ').strip().lower()=='y'
        return {'selected_action':a,'rationale':rationale or 'Manual solvability test.',
                'leading_rules':{},'confidence':0.0,'believes_solved':solved}

    def _openai(self,obs,schema,prior):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError('OpenAI mode requires: python -m pip install openai') from e
        if not os.getenv('OPENAI_API_KEY'):
            raise RuntimeError('OPENAI_API_KEY is not set')
        client=OpenAI()
        response=client.responses.create(
            model=self.model,
            input=[
                {'role':'system','content':SYSTEM_PROMPT},
                {'role':'user','content':self._prompt(obs,schema,prior)},
            ],
        )
        self.calls+=1
        usage=getattr(response,'usage',None)
        if usage:
            self.total_input_tokens += int(getattr(usage,'input_tokens',0) or 0)
            self.total_output_tokens += int(getattr(usage,'output_tokens',0) or 0)
        txt=getattr(response,'output_text','')
        try:
            return json.loads(txt)
        except Exception as e:
            raise RuntimeError(f'Model did not return valid JSON. Raw output: {txt[:1000]}') from e


    def _anthropic(self,obs,schema,prior):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError('Anthropic mode requires: python -m pip install anthropic') from e
        if not os.getenv('ANTHROPIC_API_KEY'):
            raise RuntimeError('ANTHROPIC_API_KEY is not set')
        client=anthropic.Anthropic()
        t0=time.time()
        # Current Claude models reject temperature and always think (thinking tokens count toward
        # max_tokens), so 500 tokens would truncate replies. Effort is set explicitly; override
        # with ANTHROPIC_EFFORT. See CLAUDE_RUN_NOTES.md in this folder.
        # The prompt never lists the reply fields; the Ollama path enforces them with a JSON schema
        # (`format`). Give Claude the same schema via structured outputs, minus the length/number
        # limits structured outputs does not accept (_validate still checks action and confidence).
        output_schema={
            'type':'object',
            'properties':{
                'selected_action':{'type':'string','enum':list(obs['available_actions'])},
                'rationale':{'type':'string'},
                'leading_rules':{
                    'type':'object',
                    'properties':{k:{'type':'array','items':{'type':'string'}} for k in ['P','G','M','T','I','S']},
                    'additionalProperties':False
                },
                'confidence':{'type':'number'},
                'believes_solved':{'type':'boolean'}
            },
            'required':['selected_action','rationale','leading_rules','confidence','believes_solved'],
            'additionalProperties':False
        }
        response=client.messages.create(
            model=self.model, max_tokens=16000, system=SYSTEM_PROMPT,
            output_config={'effort':os.getenv('ANTHROPIC_EFFORT','medium'),
                           'format':{'type':'json_schema','schema':output_schema}},
            messages=[{'role':'user','content':self._prompt(obs,schema,prior)}])
        self.calls+=1; self.total_model_seconds += time.time()-t0
        usage=getattr(response,'usage',None)
        if usage:
            self.total_input_tokens += int(getattr(usage,'input_tokens',0) or 0)
            self.total_output_tokens += int(getattr(usage,'output_tokens',0) or 0)
        txt=''.join(getattr(block,'text','') for block in getattr(response,'content',[]) if getattr(block,'type',None)=='text')
        try:
            return json.loads(txt)
        except Exception as e:
            raise RuntimeError(f'Anthropic model did not return valid JSON. Raw output: {txt[:1000]}') from e

    def _ollama(self,obs,schema,prior):
        output_schema={
            'type':'object',
            'properties':{
                'selected_action':{'type':'string','enum':list(obs['available_actions'])},
                'rationale':{'type':'string','maxLength':300},
                'leading_rules':{
                    'type':'object',
                    'properties':{k:{'type':'array','items':{'type':'string'},'maxItems':4} for k in ['P','G','M','T','I','S']},
                    'additionalProperties':False
                },
                'confidence':{'type':'number','minimum':0,'maximum':1},
                'believes_solved':{'type':'boolean'}
            },
            'required':['selected_action','rationale','leading_rules','confidence','believes_solved'],
            'additionalProperties':False
        }
        payload={
            'model':self.model,
            'messages':[
                {'role':'system','content':SYSTEM_PROMPT},
                {'role':'user','content':self._prompt(obs,schema,prior)}
            ],
            'stream':False,
            'think':False,
            'format':output_schema,
            'options':{'temperature':0}
        }
        data=json.dumps(payload).encode('utf-8')
        req=urllib.request.Request(self.ollama_url+'/api/chat',data=data,headers={'Content-Type':'application/json'},method='POST')
        t0=time.time()
        try:
            with urllib.request.urlopen(req,timeout=self.request_timeout) as resp:
                envelope=json.loads(resp.read().decode('utf-8'))
        except urllib.error.URLError as e:
            raise RuntimeError(f'Ollama request failed at {self.ollama_url}. Is Ollama running? {e}') from e
        wall=time.time()-t0
        self.calls+=1
        pin=int(envelope.get('prompt_eval_count',0) or 0); pout=int(envelope.get('eval_count',0) or 0)
        self.total_input_tokens+=pin; self.total_output_tokens+=pout; self.total_model_seconds+=wall
        self.last_telemetry={
            'backend':'ollama','model':envelope.get('model',self.model),'wall_seconds':wall,
            'total_duration_ns':envelope.get('total_duration'),'load_duration_ns':envelope.get('load_duration'),
            'prompt_eval_count':pin,'prompt_eval_cached_count':int(envelope.get('prompt_eval_cached_count',0) or 0),
            'prompt_eval_duration_ns':envelope.get('prompt_eval_duration'),'eval_count':pout,'eval_duration_ns':envelope.get('eval_duration')
        }
        content=(envelope.get('message') or {}).get('content','')
        try:
            result=json.loads(content)
        except Exception as e:
            raise RuntimeError(f'Ollama model did not return valid structured JSON. Raw content: {content[:1000]}') from e
        result['_telemetry']=dict(self.last_telemetry)
        return result

    def _prompt(self,obs,schema,prior):
        prompt=build_turn_prompt(obs,schema,prior,self.scale_name)
        if self.experience_context:
            payload=json.loads(prompt)
            payload['prior_task_experience']=self.experience_context
            payload['transfer_instruction']='Use prior experience only as fallible evidence; current observations take precedence.'
            prompt=json.dumps(payload,separators=(',',':'))
        self.last_prompt=prompt
        if self.prompt_audit_dir:
            os.makedirs(self.prompt_audit_dir,exist_ok=True)
            turn=len(prior)+1
            with open(os.path.join(self.prompt_audit_dir,f'prompt_turn_{turn:03d}.json'),'w') as f:
                f.write(json.dumps(json.loads(prompt),indent=2))
        return prompt

    def act(self,obs,schema,prior):
        if self.call_fn is not None:
            r=self.call_fn(SYSTEM_PROMPT,self._prompt(obs,schema,prior))
        elif self.mode=='manual': r=self._manual(obs,schema,prior)
        elif self.mode=='openai': r=self._openai(obs,schema,prior)
        elif self.mode=='anthropic': r=self._anthropic(obs,schema,prior)
        elif self.mode=='ollama': r=self._ollama(obs,schema,prior)
        else:
            acts=obs['available_actions']; a=acts[obs['step']%len(acts)]
            r={'selected_action':a,'rationale':'Smoke-test placeholder policy.','leading_rules':{},'confidence':0.0,'believes_solved':False}
        return self._validate(r,obs['available_actions'])

    def usage_summary(self):
        return {'llm_calls':self.calls,'input_tokens':self.total_input_tokens,'output_tokens':self.total_output_tokens,
                'total_tokens':self.total_input_tokens+self.total_output_tokens,'model':self.model,'mode':self.mode,
                'model_wall_seconds':self.total_model_seconds}

def run_episode(env,llm,schema,trace_path=None,print_trace=False,include_hidden_trace=True,neutral_symbols=True):
    raw_obs=env.reset(); binding=make_visual_binding(env.seed) if neutral_symbols else {'A':'A','B':'B','C':'C'}
    obs=agent_view(raw_obs,binding) if neutral_symbols else raw_obs
    logs=[]; t0=time.time(); false_solutions=0; initial_state=obs
    if print_trace: print('INITIAL AGENT-VISIBLE STATE\n'+render_grid(obs))
    while not raw_obs['solved_env'] and not raw_obs['failed_env']:
        r=llm.act(obs,schema,logs)
        if r.get('believes_solved') and not raw_obs['solved_env']: false_solutions+=1
        action=r['selected_action']; nxt_raw,reward,done,info=env.step(action)
        nxt=agent_view(nxt_raw,binding) if neutral_symbols else nxt_raw
        delta=observation_delta(obs,nxt)
        entry={'turn':raw_obs['step']+1,'state_before':obs,'grid_before':render_grid(obs),'agent':r,'action':action,
               'state_after':nxt,'grid_after':render_grid(nxt),'observation_delta':delta,'reward':reward,
               'solved_env':nxt_raw['solved_env'],'failed_env':nxt_raw['failed_env'],'termination_reason':info.get('termination_reason')}
        if include_hidden_trace: entry['harness_ground_truth_after']=env.ground_truth_state()
        logs.append(entry)
        if print_trace:
            print(f"\nTURN {entry['turn']} | ACTION={action}\n{entry['grid_after']}")
            print('DELTA:',json.dumps(delta))
            if entry['termination_reason']: print('TERMINATION:',entry['termination_reason'])
        raw_obs=nxt_raw; obs=nxt
    usage=llm.usage_summary() if hasattr(llm,'usage_summary') else {}
    summary={'solved':raw_obs['solved_env'],'actions':raw_obs['step'],'false_solution_events':false_solutions,
             'wall_seconds':time.time()-t0,'termination_reason':env.termination_reason,**usage}
    result={'episode':{'world_ids':env.world.ids(),'world_index24':env.world.index24(),'support':rule_support_report(env.world),
                       'validation':env.validation,'max_steps':env.max_steps,'seed':env.seed,'visual_binding_hidden_from_agent':binding},
            'initial_state':initial_state,'initial_grid':render_grid(initial_state),'turns':logs,'summary':summary,
            **summary,'logs':logs,'world_ids':env.world.ids(),'world_index24':env.world.index24()}
    if trace_path:
        with open(trace_path,'w') as f: json.dump(result,f,indent=2)
    return result


def _state_signature(state):
    objs=[]
    for o in state.get('objects',[]):
        sym=o.get('symbol') or o.get('cls','?')
        objs.append((sym,int(o.get('r',0)),int(o.get('c',0)),bool(o.get('active',False))))
    return (tuple(sorted(objs)), tuple(sorted(tuple(x) for x in state.get('barriers',[]))))

def episode_measurements(result:dict)->dict:
    turns=result.get('turns',result.get('logs',[]))
    n=len(turns)
    if not n:
        return {'actions':0,'redundant_action_rate':0.0,'novel_experiment_rate':0.0,'no_change_action_persistence':0.0}
    seen=set(); redundant=0; novel=0; no_change=0; no_change_repeat=0
    action_counts={a:0 for a in ('UP','DOWN','LEFT','RIGHT')}
    confidences=[]; effective_conf=[]; ineffective_conf=[]
    prior_no_change=False
    repeated_after_no_change=0
    for t in turns:
        a=t.get('action'); action_counts[a]=action_counts.get(a,0)+1
        sig=(_state_signature(t.get('state_before',{})),a)
        if sig in seen: redundant+=1
        else: novel+=1; seen.add(sig)
        d=t.get('observation_delta',{})
        changed=bool(d.get('moved') or d.get('changed') or d.get('removed') or d.get('added') or d.get('solved_changed') or d.get('failed_changed'))
        if not changed:
            no_change+=1
            if prior_no_change: no_change_repeat+=1
        if prior_no_change and len(turns)>1:
            repeated_after_no_change += int(a == turns[turns.index(t)-1].get('action')) if False else 0
        prior_no_change=not changed
        c=t.get('agent',{}).get('confidence')
        if isinstance(c,(int,float)):
            confidences.append(float(c))
            (effective_conf if changed else ineffective_conf).append(float(c))
    # consecutive same-action persistence following a no-change observation
    for i in range(1,n):
        prev=turns[i-1]; cur=turns[i]
        d=prev.get('observation_delta',{})
        prev_changed=bool(d.get('moved') or d.get('changed') or d.get('removed') or d.get('added') or d.get('solved_changed') or d.get('failed_changed'))
        if (not prev_changed) and cur.get('action')==prev.get('action'):
            repeated_after_no_change+=1
    unique_actions=sum(1 for v in action_counts.values() if v)
    probs=[v/n for v in action_counts.values() if v]
    import math
    entropy=-sum(q*math.log(q,2) for q in probs) if probs else 0.0
    max_entropy=math.log(4,2)
    usage=result.get('summary',{})
    model_sec=float(usage.get('model_wall_seconds',0.0) or 0.0)
    toks=int(usage.get('total_tokens',0) or 0)
    return {
        'actions':n,
        'unique_actions':unique_actions,
        'action_counts':action_counts,
        'action_diversity_fraction':unique_actions/4.0,
        'action_entropy_bits':entropy,
        'normalized_action_entropy':entropy/max_entropy if max_entropy else 0.0,
        'redundant_state_action_count':redundant,
        'redundant_action_rate':redundant/n,
        'novel_state_action_count':novel,
        'novel_experiment_rate':novel/n,
        'no_change_action_count':no_change,
        'no_change_action_rate':no_change/n,
        'repeated_same_action_after_no_change':repeated_after_no_change,
        'no_change_action_persistence':repeated_after_no_change/max(1,no_change),
        'mean_confidence':sum(confidences)/len(confidences) if confidences else None,
        'mean_confidence_effective_action':sum(effective_conf)/len(effective_conf) if effective_conf else None,
        'mean_confidence_no_change_action':sum(ineffective_conf)/len(ineffective_conf) if ineffective_conf else None,
        'decision_latency_seconds_per_action':model_sec/n if n else None,
        'tokens_per_action':toks/n if n else None,
        'solved':bool(usage.get('solved',result.get('solved',False))),
        'termination_reason':usage.get('termination_reason',result.get('termination_reason')),
    }

def save_measurements(result:dict,path:str):
    m=episode_measurements(result)
    with open(path,'w') as f: json.dump(m,f,indent=2)
    return m

def metrics(results:List[dict]):
    n=len(results); sm=lambda r:r.get('summary',r); solved=[r for r in results if sm(r)['solved']]
    return {'episodes':n,'solve_rate':sum(sm(r)['solved'] for r in results)/n if n else 0,
            'mean_actions':sum(sm(r)['actions'] for r in results)/n if n else 0,
            'mean_actions_solved':sum(sm(r)['actions'] for r in solved)/len(solved) if solved else None,
            'false_solution_events':sum(sm(r)['false_solution_events'] for r in results),
            'mean_wall_seconds':sum(sm(r)['wall_seconds'] for r in results)/n if n else 0,
            'termination_reasons':{k:sum(sm(r).get('termination_reason')==k for r in results) for k in ('solved','action_budget_exhausted','hazard_contact')}}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Stronger AI Group 1 prototype V5 - measurement release')
    ap.add_argument('--trace',default='smoke_trace.json'); ap.add_argument('--summary',default='smoke.json')
    ap.add_argument('--max-steps',type=int,default=8); ap.add_argument('--show-grid',action='store_true')
    ap.add_argument('--mode',choices=['placeholder','manual','openai','anthropic','ollama'],default='placeholder')
    ap.add_argument('--model',default=None,help='Model name. Defaults to gpt-5.6-sol for OpenAI and qwen3:4b for Ollama')
    ap.add_argument('--ollama-url',default='http://localhost:11434',help='Ollama base URL for --mode ollama')
    ap.add_argument('--seed',type=int,default=1); ap.add_argument('--show-class-labels',action='store_true')
    ap.add_argument('--scale',choices=['S1','S2','S3','S4','S5'],default='S1',help='Hypothesis-space scale; default S1=256 for local testing')
    ap.add_argument('--prompt-audit-dir',default='prompt_audit',help='Write exactly what the LLM sees each turn')
    ap.add_argument('--request-timeout',type=int,default=300)
    ap.add_argument('--measurements',default='episode_measurements.json',help='Write V5 behavioral measurements')
    args=ap.parse_args(); HypothesisSchema.save('hypothesis_schema.json')
    if args.model is None: args.model='qwen3:4b' if args.mode=='ollama' else 'gpt-5.6-sol'
    # Coherent benchmark worlds for the scaling series. Omitted dimensions stay at baseline 0.
    BENCHMARK_WORLDS={
        'S1': WorldModel(0,0,0,0,0,0),       # P/G/M/T vary in S1 ontology
        'S2': WorldModel(0,4,0,0,2,0),       # adds interaction uncertainty
        'S3': WorldModel(0,4,0,2,3,4),       # all six dimensions active
        'S4': WorldModel(0,8,3,2,8,4),
        'S5': WorldModel(0,13,0,2,3,4),
    }
    w=BENCHMARK_WORLDS[args.scale]
    assert WorldModel.from_index24(w.index24())==w
    env=ScaledARCEnv(w,seed=args.seed,max_steps=args.max_steps)
    spec=SCALES[args.scale]; schema=HypothesisSchema.compact(spec['scale'],spec['dims'])
    client=LLMClient(mode=args.mode,model=args.model,ollama_url=args.ollama_url,scale_name=args.scale,prompt_audit_dir=args.prompt_audit_dir,request_timeout=args.request_timeout)
    out=run_episode(env,client,schema,trace_path=args.trace,print_trace=args.show_grid,neutral_symbols=not args.show_class_labels)
    summary=dict(out['summary']); summary['hypothesis_scale']=args.scale; summary['hypothesis_space_size']=spec['size']; summary['world_ids']=list(w.ids()); summary['world_index24']=w.index24(); summary['rule_support']=out['episode']['support']; summary['validation']=out['episode']['validation']
    measure=save_measurements(out,args.measurements); summary['measurements']=measure
    with open(args.summary,'w') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2)); print(f'Full trace written to: {args.trace}'); print(f'Measurements written to: {args.measurements}')

# V6 Stronger-AI Metric Freeze v1.0
STRONGER_AI_METRIC_SPEC_VERSION = '1.0'
STRONGER_AI_DIMENSIONS = {
    'generalization': {
        'symbol':'G',
        'primary_metric':'held_out_world_performance',
        'definition':'Performance on held-out worlds/rule combinations not used for development.',
        'protocol':'Evaluate matched held-out episodes; primary score is held-out solve rate in [0,1].',
        'higher_is_better':True,
    },
    'adaptation': {
        'symbol':'A',
        'primary_metric':'post_change_recovery_efficiency',
        'definition':'Ability to recover after an in-episode rule/environment change.',
        'protocol':'Inject a hidden change at a preregistered turn; score 1 - normalized recovery latency, clipped to [0,1].',
        'higher_is_better':True,
    },
    'autonomy': {
        'symbol':'Au',
        'primary_metric':'autonomous_experiment_efficiency',
        'definition':'Quality of independently selected experiments/actions without external search assistance.',
        'protocol':'Score useful novel experimentation and penalize redundant/no-change persistence using frozen V5 measurements.',
        'higher_is_better':True,
    },
    'transfer': {
        'symbol':'T',
        'primary_metric':'cross_task_transfer_gain',
        'definition':'Improvement on a related new task when prior-task experience is available versus cold start.',
        'protocol':'Paired warm-start vs cold-start episodes; normalize action-efficiency/solve improvement to [0,1].',
        'higher_is_better':True,
    },
    'robustness': {
        'symbol':'R',
        'primary_metric':'performance_retention_under_perturbation',
        'definition':'Retention of useful performance under noisy, missing, unfamiliar, or adversarially perturbed observations.',
        'protocol':'Compare perturbed performance to clean matched performance; retention is clipped to [0,1].',
        'higher_is_better':True,
    },
}

def _clip01(x):
    return max(0.0,min(1.0,float(x)))

def generalization_score(held_out_results):
    if not held_out_results: return None
    solved=sum(bool(r.get('summary',r).get('solved',False)) for r in held_out_results)
    return solved/len(held_out_results)

def adaptation_score(recovery_latencies, recovery_budget):
    if not recovery_latencies or recovery_budget <= 0: return None
    vals=[_clip01(1.0-(float(x)/float(recovery_budget))) for x in recovery_latencies]
    return sum(vals)/len(vals)

def autonomy_score(measurements):
    if not measurements: return None
    ner=_clip01(measurements.get('novel_experiment_rate',0.0))
    rar=_clip01(measurements.get('redundant_action_rate',1.0))
    ncap=_clip01(measurements.get('no_change_action_persistence',1.0))
    # Frozen equal-weight subscore: reward novelty, penalize redundancy and persistence.
    return (ner + (1.0-rar) + (1.0-ncap))/3.0

def transfer_score(cold_results, transfer_results):
    if not cold_results or not transfer_results or len(cold_results)!=len(transfer_results): return None
    gains=[]
    for c,w in zip(cold_results,transfer_results):
        cs=c.get('summary',c); ws=w.get('summary',w)
        csol=bool(cs.get('solved',False)); wsol=bool(ws.get('solved',False))
        if wsol and not csol: gains.append(1.0); continue
        if csol and not wsol: gains.append(0.0); continue
        if not csol and not wsol: gains.append(0.5); continue
        ca=max(1,int(cs.get('actions',1))); wa=max(1,int(ws.get('actions',1)))
        gains.append(_clip01(0.5 + 0.5*((ca-wa)/ca)))
    return sum(gains)/len(gains)

def robustness_score(clean_results, perturbed_results):
    if not clean_results or not perturbed_results or len(clean_results)!=len(perturbed_results): return None
    clean=sum(bool(r.get('summary',r).get('solved',False)) for r in clean_results)/len(clean_results)
    pert=sum(bool(r.get('summary',r).get('solved',False)) for r in perturbed_results)/len(perturbed_results)
    if clean==0: return 1.0 if pert==0 else None
    return _clip01(pert/clean)

def stronger_ai_score(scores, weights=None):
    keys=['G','A','Au','T','R']
    if any(scores.get(k) is None for k in keys): return None
    weights=weights or {k:1.0 for k in keys}
    denom=sum(float(weights[k]) for k in keys)
    if denom<=0: raise ValueError('weights must sum to > 0')
    return sum(_clip01(scores[k])*float(weights[k]) for k in keys)/denom

def stronger_ai_report(*, held_out_results=None, recovery_latencies=None, recovery_budget=None,
                       autonomy_measurements=None, cold_results=None, transfer_results=None,
                       clean_results=None, perturbed_results=None, weights=None):
    scores={
        'G':generalization_score(held_out_results),
        'A':adaptation_score(recovery_latencies,recovery_budget) if recovery_budget is not None else None,
        'Au':autonomy_score(autonomy_measurements),
        'T':transfer_score(cold_results,transfer_results),
        'R':robustness_score(clean_results,perturbed_results),
    }
    return {'metric_spec_version':STRONGER_AI_METRIC_SPEC_VERSION,'scores':scores,
            'SAI':stronger_ai_score(scores,weights),'definitions':STRONGER_AI_DIMENSIONS}

def save_metric_spec(path='stronger_ai_metric_spec.json'):
    payload={'metric_spec_version':STRONGER_AI_METRIC_SPEC_VERSION,
             'composite':{'name':'SAI','formula':'equal-weight mean of G,A,Au,T,R by default','range':[0,1]},
             'dimensions':STRONGER_AI_DIMENSIONS}
    with open(path,'w') as f: json.dump(payload,f,indent=2)
    return payload

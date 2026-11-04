#!/usr/bin/env python3
"""Production-scale QAHS for the 16^6 Stronger-AI benchmark.

Lineage
-------
* Preserves QAHS V3.2's architectural idea: explore -> goal discrimination -> exploit,
  posterior-conditioned multi-step planning, and an information firewall.
* Replaces V3.2's exact 256-world S1 posterior with a scalable factorized posterior
  over 16 alternatives in each of P/G/M/T/I/S (16^6 = 16,777,216 worlds).
* Uses a 24-qubit register encoding (4 bits per rule family) for quantum-assisted
  posterior candidate sampling. The full 16.8M basis is implicit; it is never
  materialized as a Python list.

Important scientific boundary
-----------------------------
The quantum component proposes posterior-plausible 24-bit candidate worlds. The
planner then evaluates experiments/actions on a bounded particle approximation.
This package tests whether the QAHS architecture improves reasoning performance;
it does not claim quantum computational advantage over an equivalent classical
search control.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import stronger_ai as sa

FAMILIES=("P","G","M","T","I","S")
FAMILY_ATTR={"P":"player","G":"goal","M":"movement","T":"topology","I":"interaction","S":"sequence"}
ACTIONS=("UP","DOWN","LEFT","RIGHT")
EPS=1e-12


def _norm(d: Mapping[Any,float]) -> Dict[Any,float]:
    z=sum(max(0.0,float(v)) for v in d.values())
    if z<=EPS:
        n=max(1,len(d)); return {k:1.0/n for k in d}
    return {k:max(0.0,float(v))/z for k,v in d.items()}


def entropy(values: Iterable[float]) -> float:
    return -sum(float(x)*math.log2(float(x)) for x in values if float(x)>EPS)


def clip01(x): return max(0.0,min(1.0,float(x)))

def stable_hash(x: Any) -> str:
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()[:20]


def world_tuple(w: sa.WorldModel) -> Tuple[int,int,int,int,int,int]:
    return (w.player,w.goal,w.movement,w.topology,w.interaction,w.sequence)


def world_from_tuple(v: Sequence[int]) -> sa.WorldModel:
    return sa.WorldModel(*[int(x) for x in v])


def posterior_mass(posterior: Mapping[str,Mapping[int,float]], w: Sequence[int]) -> float:
    z=1.0
    for fam,val in zip(FAMILIES,w): z*=float(posterior[fam].get(int(val),0.0))
    return z


@dataclass(frozen=True)
class Particle:
    world: Tuple[int,int,int,int,int,int]
    binding: Tuple[Tuple[str,str], ...]  # visible symbol -> latent class
    weight: float

    def world_model(self) -> sa.WorldModel: return world_from_tuple(self.world)
    def binding_dict(self) -> Dict[str,str]: return dict(self.binding)


class FactorizedPosterior16:
    def __init__(self, seed:int=1, likelihood_floor:float=0.03, damping:float=0.08):
        self.rng=random.Random(seed)
        self.posterior={f:{i:1/16 for i in range(16)} for f in FAMILIES}
        self.binding_posterior: Dict[Tuple[Tuple[str,str],...],float] = {}
        self.evidence_count=0
        self.likelihood_floor=float(likelihood_floor)
        self.damping=float(damping)
        self.history=[]

    @property
    def implicit_space_size(self): return 16**6

    def entropy_bits(self): return sum(entropy(self.posterior[f].values()) for f in FAMILIES)
    def effective_hypotheses(self): return 2.0**self.entropy_bits()
    def family_entropy_bits(self): return {f:entropy(self.posterior[f].values()) for f in FAMILIES}
    def map_world(self): return tuple(max(self.posterior[f],key=self.posterior[f].get) for f in FAMILIES)

    def initialize_bindings(self, obs:Mapping[str,Any]):
        syms=[]
        for o in obs.get('objects',[]):
            s=str(o.get('symbol','?'))
            if s not in syms: syms.append(s)
        if len(syms)!=3:
            # Canonical environment has three visible identities. Keep a single
            # identity mapping only for malformed/smoke observations.
            b=tuple((s,c) for s,c in zip(syms,("A","B","C")))
            self.binding_posterior={b:1.0}; return
        bs=[]
        for perm in itertools.permutations(("A","B","C")):
            bs.append(tuple(sorted(zip(syms,perm))))
        self.binding_posterior={b:1/len(bs) for b in bs}

    def update(self, particles:Sequence[Particle], likelihoods:Sequence[float], evidence:Mapping[str,Any]):
        if not particles or len(particles)!=len(likelihoods): return
        before=self.entropy_bits()
        # Average transition likelihood conditioned on each rule value.
        fam_num={f:{i:0.0 for i in range(16)} for f in FAMILIES}
        fam_den={f:{i:0.0 for i in range(16)} for f in FAMILIES}
        bnum={b:0.0 for b in self.binding_posterior}; bden={b:0.0 for b in self.binding_posterior}
        for p,lk0 in zip(particles,likelihoods):
            lk=max(self.likelihood_floor,min(1.0,float(lk0)))
            for fam,val in zip(FAMILIES,p.world):
                fam_num[fam][val]+=p.weight*lk; fam_den[fam][val]+=p.weight
            if p.binding in bnum:
                bnum[p.binding]+=p.weight*lk; bden[p.binding]+=p.weight
        for f in FAMILIES:
            raw={}
            for i in range(16):
                cond=fam_num[f][i]/fam_den[f][i] if fam_den[f][i]>EPS else 1.0
                raw[i]=self.posterior[f][i]*max(self.likelihood_floor,cond)
            upd=_norm(raw)
            # Small damping prevents a finite particle miss from annihilating a rule.
            self.posterior[f]=_norm({i:(1-self.damping)*upd[i]+self.damping/16 for i in range(16)})
        if self.binding_posterior:
            raw={}
            for b,p0 in self.binding_posterior.items():
                cond=bnum[b]/bden[b] if bden[b]>EPS else 1.0
                raw[b]=p0*max(self.likelihood_floor,cond)
            self.binding_posterior=_norm(raw)
        self.evidence_count+=1
        self.history.append({
            'evidence':dict(evidence),'entropy_before':before,'entropy_after':self.entropy_bits(),
            'family_entropy_bits':self.family_entropy_bits(),
            'map_world_ids':[f'{f}{v}' for f,v in zip(FAMILIES,self.map_world())],
        })


class Register24QAOASampler:
    """24-qubit QAOA posterior sampler over the complete 2^24 hypothesis basis.

    Each 4-bit register exactly encodes one of 16 alternatives. A quadratic
    pseudo-Boolean approximation is fit to -log posterior probability within
    each register, then compiled to a sparse Ising cost Hamiltonian. QAOA samples
    candidate 24-bit worlds. Final candidate ranking uses the exact factorized
    posterior, not the QUBO approximation.
    """
    def __init__(self, enabled=True, require_quantum=False, reps=1, shots=256,
                 seed=1, optimizer_maxiter=12, objective_shots=128):
        self.enabled=bool(enabled); self.require_quantum=bool(require_quantum)
        self.reps=int(reps); self.shots=int(shots); self.seed=int(seed)
        self.optimizer_maxiter=int(optimizer_maxiter); self.objective_shots=int(objective_shots)
        self.last_params=None

    @staticmethod
    def _features(bits4:Sequence[int]):
        z=list(map(float,bits4)); out=[1.0]+z[:]
        out += [z[i]*z[j] for i in range(4) for j in range(i+1,4)]
        return out

    def _fit_register_qubo(self, probs:Mapping[int,float]):
        import numpy as np
        X=[]; y=[]
        for r in range(16):
            bits=[(r>>j)&1 for j in range(4)]
            X.append(self._features(bits))
            y.append(-math.log(max(EPS,float(probs.get(r,EPS)))))
        X=np.asarray(X,float); y=np.asarray(y,float)
        coef,*_=np.linalg.lstsq(X,y,rcond=None)
        pred=X@coef
        rmse=float(np.sqrt(np.mean((pred-y)**2)))
        linear=list(map(float,coef[1:5])); pair=list(map(float,coef[5:]))
        return float(coef[0]),linear,pair,rmse

    @staticmethod
    def _decode_bits(bits:Sequence[int]) -> Tuple[int,int,int,int,int,int]:
        vals=[]
        for base in range(0,24,4):
            vals.append(sum((int(bits[base+j])&1)<<j for j in range(4)))
        return tuple(vals)

    @staticmethod
    def _exact_energy(bits:Sequence[int], posterior):
        vals=Register24QAOASampler._decode_bits(bits)
        return -sum(math.log(max(EPS,float(posterior[f][v]))) for f,v in zip(FAMILIES,vals))

    def sample(self, posterior:Mapping[str,Mapping[int,float]], top_k:int=96):
        if not self.enabled:
            return [],{'backend':'disabled','used_quantum':False,'qubits':24,'implicit_basis_states':2**24}
        try:
            import numpy as np
            from scipy.optimize import minimize
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
        except Exception as e:
            if self.require_quantum:
                raise RuntimeError(f'24-qubit QAOA required but unavailable: {type(e).__name__}: {e}') from e
            return [],{'backend':'unavailable','used_quantum':False,'reason':f'{type(e).__name__}:{e}','qubits':24,'implicit_basis_states':2**24}

        linear=[0.0]*24; pair_terms=[]; rmses={}
        for fi,f in enumerate(FAMILIES):
            _,a,b,rmse=self._fit_register_qubo(posterior[f]); rmses[f]=rmse
            base=4*fi
            for j in range(4): linear[base+j]+=a[j]
            k=0
            for i in range(4):
                for j in range(i+1,4):
                    pair_terms.append((base+i,base+j,b[k])); k+=1
        # QUBO -> Ising E = const + sum h_i Z_i + sum J_ij ZiZj.
        h=[-x/2.0 for x in linear]; J=[]
        for i,j,b in pair_terms:
            h[i]-=b/4.0; h[j]-=b/4.0; J.append((i,j,b/4.0))
        # Scale cost to a numerically stable range.
        mx=max([abs(x) for x in h]+[abs(x[2]) for x in J]+[1.0])
        h=[x/mx for x in h]; J=[(i,j,w/mx) for i,j,w in J]
        sim=AerSimulator(method='matrix_product_state',seed_simulator=self.seed)

        def circuit(params,measure=True):
            qc=QuantumCircuit(24,24 if measure else 0); qc.h(range(24))
            for layer in range(self.reps):
                gamma=float(params[layer]); beta=float(params[self.reps+layer])
                for q,w in enumerate(h):
                    if abs(w)>1e-12: qc.rz(2*gamma*w,q)
                for i,j,w in J:
                    if abs(w)>1e-12: qc.rzz(2*gamma*w,i,j)
                for q in range(24): qc.rx(2*beta,q)
            if measure: qc.measure(range(24),range(24))
            return qc

        def bits_from_str(s): return [int(c) for c in s[::-1]]
        def objective(params):
            counts=sim.run(circuit(params),shots=max(32,self.objective_shots)).result().get_counts()
            total=sum(counts.values()) or 1
            return sum(c*self._exact_energy(bits_from_str(s),posterior) for s,c in counts.items())/total

        x0=self.last_params if self.last_params is not None else [0.65]*self.reps+[0.35]*self.reps
        opt=minimize(objective,x0,method='COBYLA',options={'maxiter':self.optimizer_maxiter,'rhobeg':0.35})
        self.last_params=list(map(float,opt.x))
        counts=sim.run(circuit(opt.x),shots=self.shots).result().get_counts()
        scored=[]
        for s,c in counts.items():
            bits=bits_from_str(s); w=self._decode_bits(bits)
            pm=posterior_mass(posterior,w)
            scored.append((int(c),pm,w))
        scored.sort(key=lambda x:(-x[0],-x[1],x[2]))
        worlds=[]; seen=set()
        for c,pm,w in scored:
            if w in seen: continue
            seen.add(w); worlds.append(w)
            if len(worlds)>=top_k: break
        return worlds,{
            'backend':'qiskit_aer_qaoa_register24','used_quantum':True,'qubits':24,
            'encoding':'six 4-bit registers: P,G,M,T,I,S; register bits are LSB-first',
            'implicit_basis_states':2**24,'reps':self.reps,'shots':self.shots,
            'objective_shots':self.objective_shots,'optimizer':'COBYLA',
            'optimizer_maxiter':self.optimizer_maxiter,'optimizer_success':bool(opt.success),
            'objective':float(opt.fun),'params':list(map(float,opt.x)),
            'register_qubo_rmse':rmses,'unique_quantum_worlds':len(worlds),
        }


class PublicCandidateSimulator:
    """Candidate-world simulator using only public ontology + agent-visible history.

    The active hidden world and true symbol/class binding are never supplied.
    Candidate symbol/class bindings are nuisance hypotheses and are marginalized.
    """
    def __init__(self, initial_obs:Mapping[str,Any], max_steps:int=12, seed:int=1):
        self.initial_obs=copy.deepcopy(initial_obs); self.max_steps=int(max_steps); self.seed=int(seed)

    @staticmethod
    def binding_candidates(obs:Mapping[str,Any]):
        syms=[]
        for o in obs.get('objects',[]):
            s=str(o.get('symbol','?'))
            if s not in syms: syms.append(s)
        if len(syms)!=3:
            return [tuple(sorted(zip(syms,("A","B","C")[:len(syms)])))]
        return [tuple(sorted(zip(syms,p))) for p in itertools.permutations(("A","B","C"))]

    @staticmethod
    def _inverse_binding(binding): return {cls:sym for sym,cls in dict(binding).items()}

    def _build_env(self, world:sa.WorldModel, binding) -> sa.ScaledARCEnv:
        env=sa.ScaledARCEnv(world,seed=self.seed,max_steps=self.max_steps)
        b=dict(binding); counts={'A':0,'B':0,'C':0}; objs=[]
        for vo in self.initial_obs.get('objects',[]):
            sym=str(vo.get('symbol','?')); cls=b.get(sym,'A'); counts[cls]+=1
            oid=cls if counts[cls]==1 and cls in ('A','B') else f'{cls}{counts[cls]}'
            size={'A':2,'B':3,'C':1}[cls]
            objs.append(sa.Obj(oid,cls,int(vo['r']),int(vo['c']),active=bool(vo.get('active',False)),size=size))
        # Ensure C-specific goal semantics can refer to C1/C2 when two C objects exist.
        cnum=0
        for o in objs:
            if o.cls=='C':
                cnum+=1; o.oid=f'C{cnum}'
        env.objects=objs; env.wall={tuple(x) for x in self.initial_obs.get('barriers',[])}; env.base_wall=set(env.wall)
        env.start={o.oid:(o.r,o.c) for o in env.objects}; env.steps=0; env.solved=False; env.failed=False; env.termination_reason=None
        env.mode=0; env.prev_action=None; env.prev_touched=None; env.counter=0; env.unlocked=False; env.control_cycle=0; env.stage=0
        env.sequence_events=[]; env.temporary_unlock_until=None; env.last_direction=None; env.history=[]
        env.selected_oid=env.objects[0].oid if env.objects else 'A'
        return env

    def _agent_obs(self, env:sa.ScaledARCEnv, binding):
        inv=self._inverse_binding(binding)
        raw=env.observe(); out=copy.deepcopy(raw)
        for o in out.get('objects',[]):
            o['symbol']=inv.get(o.pop('cls',None),'?')
        out['legend']='Symbols are arbitrary visual identities; infer their roles from interaction.'
        return out

    def replay(self, particle:Particle, history:Sequence[Mapping[str,Any]]) -> sa.ScaledARCEnv:
        env=self._build_env(particle.world_model(),particle.binding)
        for t in history:
            if env.solved or env.failed: break
            env.step(str(t.get('action')))
        return env

    def predict_delta(self, particle:Particle, history:Sequence[Mapping[str,Any]], action:str):
        env=self.replay(particle,history)
        before=self._agent_obs(env,particle.binding)
        if not env.solved and not env.failed: env.step(action)
        after=self._agent_obs(env,particle.binding)
        return sa.observation_delta(before,after), bool(env.solved), bool(env.failed)

    def simulate_plan(self, particle:Particle, history:Sequence[Mapping[str,Any]], plan:Sequence[str]):
        env=self.replay(particle,history); before_h=self._goal_heuristic(env)
        for a in plan:
            if env.solved or env.failed: break
            env.step(a)
        after_h=self._goal_heuristic(env)
        return env, before_h, after_h

    @staticmethod
    def _goal_heuristic(env:sa.ScaledARCEnv) -> float:
        if env.solved: return 0.0
        g=int(env.world.goal); live=env._live(); ctr=env._control(env.prev_action); c=ctr[0] if ctr else None
        if not live: return 50.0
        def md(a,b): return abs(a.r-b.r)+abs(a.c-b.c)
        if g in (0,1,2):
            tc={0:'B',1:'C',2:'A'}[g]
            ts=[o for o in live if o.cls==tc and (c is None or o.oid!=c.oid)]
            return min([md(c,t) for t in ts] or [40.0]) if c else 40.0
        if g in (3,7,13): return (abs(env.n-c.c) if c else 30.0) + (4.0 if g==13 and not env.unlocked else 0.0)
        if g in (4,11): return 5.0*sum(not o.collected for o in env.objects if o.cls=='C')
        if g==5: return 5.0*sum(not o.collected for o in env.objects if o.cls=='B')
        if g==6: return 5.0*sum(not o.collected for o in env.objects if o.oid in ('C1','C2'))
        if g==8:
            return min(abs(o.r-4)+abs(o.c-8) for o in live)
        if g==9: return float(sum(o.c not in (2,7) for o in live))
        if g==10: return float(max(0,len({o.r for o in live})-1))
        if g==12: return float(sum(not o.active for o in live if o.cls=='C'))
        if g==14: return float((0 if env.unlocked else 2)+(0 if env.mode==1 else 2))
        if g==15: return float(max(0,2-env.stage)+(0 if env.unlocked else 1)+max(0,3-len(env.sequence_events)))
        return 20.0

    def search_plan(self, particle:Particle, history:Sequence[Mapping[str,Any]], remaining:int,
                    horizon:int=8, beam_width:int=48):
        root=self.replay(particle,history)
        if root.solved: return []
        depth=max(1,min(int(remaining),int(horizon)))
        beam=[(self._goal_heuristic(root),[],root)]
        for _ in range(depth):
            nxt=[]
            for _,path,e in beam:
                for a in ACTIONS:
                    x=copy.deepcopy(e)
                    if not x.solved and not x.failed: x.step(a)
                    pp=path+[a]
                    if x.solved: return pp
                    if x.failed: continue
                    h=self._goal_heuristic(x)
                    nxt.append((h+0.02*len(pp),pp,x))
            if not nxt: break
            nxt.sort(key=lambda z:(z[0],z[1]))
            # Deduplicate visible+hidden candidate states cheaply.
            kept=[]; seen=set()
            for rec in nxt:
                e=rec[2]
                sig=(tuple((o.cls,o.r,o.c,o.active,o.collected) for o in e.objects),tuple(sorted(e.wall)),e.unlocked,e.mode,e.prev_action,e.stage,tuple(e.sequence_events))
                if sig in seen: continue
                seen.add(sig); kept.append(rec)
                if len(kept)>=beam_width: break
            beam=kept
        return None


def _delta_features(d:Mapping[str,Any]):
    moved=[]
    for m in d.get('moved',[]) or []:
        fr=m.get('from',[0,0]); to=m.get('to',[0,0])
        moved.append((str(m.get('object','?'))[:1],int(to[0])-int(fr[0]),int(to[1])-int(fr[1])))
    return {
        'moved':tuple(sorted(moved)),
        'changed':len(d.get('changed',[]) or []),'removed':len(d.get('removed',[]) or []),'added':len(d.get('added',[]) or []),
        'barriers_removed':len(d.get('barriers_removed',[]) or []),'barriers_added':len(d.get('barriers_added',[]) or []),
        'solved':bool(d.get('solved_changed')),'failed':bool(d.get('failed_changed')),
    }


def delta_similarity(pred:Mapping[str,Any], actual:Mapping[str,Any]) -> float:
    p=_delta_features(pred); a=_delta_features(actual)
    score=0.0; den=0.0
    for key,w in [('solved',2.0),('failed',2.0),('moved',4.0),('changed',1.0),('removed',1.0),('added',1.0),('barriers_removed',1.0),('barriers_added',1.0)]:
        den+=w
        if p[key]==a[key]: score+=w
        elif key=='moved':
            # Partial credit for matching moved-count even when symbol identity differs.
            if len(p[key])==len(a[key]): score+=0.35*w
        elif isinstance(p[key],int) and isinstance(a[key],int):
            score+=w*max(0.0,1.0-abs(p[key]-a[key])/max(1,p[key],a[key]))
    return max(0.02,min(1.0,score/max(EPS,den)))


class QAHSV32S5:
    VERSION='QAHS-V3.2-S5-production'
    def __init__(self, initial_obs:Mapping[str,Any], seed:int=1, max_steps:int=12,
                 particles:int=384, eval_particles:int=64, quantum=True, require_quantum=False,
                 qaoa_reps:int=1, qaoa_shots:int=256, qaoa_optimizer_maxiter:int=12,
                 plan_horizon:int=8, beam_width:int=48, llm_weight:float=0.20):
        self.seed=int(seed); self.rng=random.Random(seed); self.max_steps=int(max_steps)
        self.posterior=FactorizedPosterior16(seed=seed); self.posterior.initialize_bindings(initial_obs)
        self.sim=PublicCandidateSimulator(initial_obs,max_steps=max_steps,seed=seed)
        self.sampler=Register24QAOASampler(quantum,require_quantum,qaoa_reps,qaoa_shots,seed,qaoa_optimizer_maxiter)
        self.particles_n=max(96,int(particles)); self.eval_particles_n=max(24,int(eval_particles))
        self.plan_horizon=int(plan_horizon); self.beam_width=int(beam_width); self.llm_weight=float(llm_weight)
        self.history=[]; self.last_particles=[]; self.last_quantum={}; self.decisions=[]; self.observations=[]
        self.require_quantum=require_quantum

    def apply_llm_prior(self, leading_rules:Mapping[str,Any], confidence:float=0.0):
        """Fuse the LLM's structured rule shortlist as a deliberately weak prior.

        The LLM never supplies ground truth; this only operationalizes Group 2's
        requirement that the LLM may interpret observations and nominate plausible
        rule vocabulary while QAHS performs the search/action selection.
        """
        if not isinstance(leading_rules,Mapping): return
        alpha=min(0.30,max(0.08,0.10+0.20*clip01(confidence)))
        for f in FAMILIES:
            raw=leading_rules.get(f,[]) or []
            vals=[]
            for x in raw:
                x=str(x).upper().strip()
                if x.startswith(f): x=x[1:]
                try: v=int(x)
                except Exception: continue
                if 0<=v<16 and v not in vals: vals.append(v)
            if not vals: continue
            target={i:0.30/16 for i in range(16)}
            for v in vals: target[v]+=0.70/len(vals)
            self.posterior.posterior[f]=_norm({i:(1-alpha)*self.posterior.posterior[f][i]+alpha*target[i] for i in range(16)})

    def _sample_binding(self):
        bs=list(self.posterior.binding_posterior); ps=[self.posterior.binding_posterior[b] for b in bs]
        return self.rng.choices(bs,weights=ps,k=1)[0]

    def _sample_world_classical(self):
        vals=[]
        for f in FAMILIES:
            ids=list(range(16)); ps=[self.posterior.posterior[f][i] for i in ids]
            vals.append(self.rng.choices(ids,weights=ps,k=1)[0])
        return tuple(vals)

    def _particle_pool(self):
        qworlds,qmeta=self.sampler.sample(self.posterior.posterior,top_k=min(128,self.particles_n//2))
        worlds=[]; seen=set()
        # Quantum proposals first.
        for w in qworlds:
            if w not in seen: seen.add(w); worlds.append(w)
        # MAP and family-axis coverage guarantee every rule value can be updated.
        mw=list(self.posterior.map_world()); wt=tuple(mw)
        if wt not in seen: seen.add(wt); worlds.append(wt)
        for fi,f in enumerate(FAMILIES):
            for v in range(16):
                x=list(mw); x[fi]=v; x=tuple(x)
                if x not in seen: seen.add(x); worlds.append(x)
        while len(worlds)<self.particles_n:
            x=self._sample_world_classical()
            if x not in seen: seen.add(x); worlds.append(x)
        worlds=worlds[:self.particles_n]
        raw=[]
        for w in worlds:
            b=self._sample_binding(); pw=posterior_mass(self.posterior.posterior,w)*self.posterior.binding_posterior.get(b,1.0)
            raw.append((w,b,pw))
        z=sum(x[2] for x in raw) or 1.0
        particles=[Particle(w,b,p/z) for w,b,p in raw]
        particles.sort(key=lambda p:p.weight,reverse=True)
        self.last_quantum=qmeta
        return particles

    @staticmethod
    def _cluster_entropy(records:Sequence[Tuple[float,Any]]) -> float:
        groups={}
        for w,key in records: groups[key]=groups.get(key,0.0)+float(w)
        z=sum(groups.values()) or 1.0
        return entropy(v/z for v in groups.values())

    def _mode(self, remaining:int):
        if self.posterior.evidence_count<2: return 'explore'
        gh=self.posterior.family_entropy_bits()['G']
        if remaining<=4 or gh<2.2: return 'exploit'
        return 'goal_discrimination'

    def _visible_geometry_progress(self, obs:Mapping[str,Any], action:str) -> float:
        objs=list(obs.get('objects',[]))
        if not objs: return 0.0
        # Infer the currently controlled visible identity only from the last observed move.
        ctrl=None
        if self.history:
            moved=(self.history[-1].get('observation_delta',{}).get('moved',[]) or [])
            if moved:
                sym=str(moved[0].get('object','?'))[:1]
                ctrl=next((o for o in objs if str(o.get('symbol','?'))==sym),None)
        ctrl=ctrl or objs[0]
        r,c=int(ctrl.get('r',0)),int(ctrl.get('c',0)); dr,dc=sa.ScaledARCEnv.DIR[action]; nr,nc=r+dr,c+dc
        n=int(obs.get('size',8))
        if not (1<=nr<=n and 1<=nc<=n): return 0.0
        bars={tuple(x) for x in obs.get('barriers',[])}
        # Visible wall opening is a useful, semantics-free navigation landmark.
        bycol={}
        for br,bc in bars: bycol.setdefault(bc,[]).append(br)
        gap=None
        if bycol:
            wc=max(bycol,key=lambda x:len(bycol[x])); occ=set(bycol[wc])
            holes=[(rr,wc) for rr in range(1,n+1) if rr not in occ]
            if holes: gap=min(holes,key=lambda x:abs(r-x[0])+abs(c-x[1]))
        if gap and c<=gap[1]:
            b=abs(r-gap[0])+abs(c-gap[1]); a=abs(nr-gap[0])+abs(nc-gap[1])
            return 0.8 if a<b else (0.45 if a==b else 0.1)
        others=[o for o in objs if o is not ctrl]
        if not others: return 0.4
        b=min(abs(r-int(o['r']))+abs(c-int(o['c'])) for o in others)
        a=min(abs(nr-int(o['r']))+abs(nc-int(o['c'])) for o in others)
        return 0.75 if a<b else (0.4 if a==b else 0.15)

    def _plan_metrics(self, plan:Sequence[str], particles:Sequence[Particle]):
        solved=0.0; progresses=0.0; outcomes=[]; z=sum(p.weight for p in particles) or 1.0
        for p in particles:
            env,bh,ah=self.sim.simulate_plan(p,self.history,plan)
            w=p.weight/z; solved += w*float(env.solved)
            if bh<=0: prog=1.0
            else: prog=max(-1.0,min(1.0,(bh-ah)/max(1.0,bh)))
            progresses += w*(0.5+0.5*prog)
            obs=self.sim._agent_obs(env,p.binding)
            key=stable_hash({'objects':obs.get('objects',[]),'barriers':obs.get('barriers',[]),'solved':env.solved,'failed':env.failed})
            outcomes.append((w,key))
        h=self._cluster_entropy(outcomes)
        info=min(1.0,h/max(1.0,math.log2(max(2,len(particles)))))
        return {'solve_probability':solved,'goal_progress':progresses,'information_gain':info,'predicted_outcomes':len(set(k for _,k in outcomes))}

    def choose_action(self, obs:Mapping[str,Any], llm_action:str):
        t0=time.time(); remaining=max(1,self.max_steps-int(obs.get('step',0)))
        pool=self._particle_pool(); evalp=pool[:min(self.eval_particles_n,len(pool))]
        mode=self._mode(remaining)
        plans={(a,):None for a in ACTIONS}
        # Generate candidate solve plans from posterior-plausible worlds.
        for p in evalp[:min(10,len(evalp))]:
            plan=self.sim.search_plan(p,self.history,remaining,horizon=self.plan_horizon,beam_width=self.beam_width)
            if plan: plans[tuple(plan)]=None
        scored=[]
        state_sig=stable_hash({'objects':obs.get('objects',[]),'barriers':obs.get('barriers',[]),'step':obs.get('step')})
        prior_pairs={(stable_hash({'objects':t.get('state_before',{}).get('objects',[]),'barriers':t.get('state_before',{}).get('barriers',[]),'step':t.get('state_before',{}).get('step')}),t.get('action')) for t in self.history}
        for plan in plans:
            m=self._plan_metrics(plan,evalp); cost=len(plan)/max(1,remaining)
            novelty=1.0 if (state_sig,plan[0]) not in prior_pairs else 0.0
            llm=1.0 if plan[0]==llm_action else 0.0
            geom=self._visible_geometry_progress(obs,plan[0])
            if mode=='explore':
                score=.42*m['information_gain']+.16*m['goal_progress']+.08*m['solve_probability']+.10*novelty+.14*geom+.20*self.llm_weight*llm-.10*cost
            elif mode=='goal_discrimination':
                score=.31*m['information_gain']+.25*m['solve_probability']+.16*m['goal_progress']+.07*novelty+.10*geom+.15*self.llm_weight*llm-.12*cost
            else:
                score=.52*m['solve_probability']+.16*m['goal_progress']+.08*m['information_gain']+.05*novelty+.07*geom+.12*self.llm_weight*llm-.12*cost
            m=dict(m); m['visible_geometry_progress']=geom
            scored.append((score,plan,m,novelty,cost))
        scored.sort(key=lambda x:(-x[0],len(x[1]),x[1]))
        score,plan,m,novelty,cost=scored[0]
        rec={
            'version':self.VERSION,'selected_action':plan[0],'plan':list(plan),'planning_mode':mode,
            'utility':score,'expected_information_gain':m['information_gain'],'solve_probability':m['solve_probability'],
            'goal_progress':m['goal_progress'],'novelty':novelty,'cost_penalty':cost,
            'posterior_entropy_bits':self.posterior.entropy_bits(),'effective_hypotheses':self.posterior.effective_hypotheses(),
            'family_entropy_bits':self.posterior.family_entropy_bits(),
            'map_world_ids':[f'{f}{v}' for f,v in zip(FAMILIES,self.posterior.map_world())],
            'quantum':dict(self.last_quantum),'llm_agreement':plan[0]==llm_action,
            'candidate_plan_count':len(scored),'candidate_particle_count':len(pool),'evaluated_particles':len(evalp),
            'information_firewall':{'active_hidden_world_used':False,'true_symbol_mapping_used':False,'candidate_bindings_marginalized':True},
            'wall_seconds':time.time()-t0,
        }
        self.last_particles=pool; self.decisions.append(rec)
        return plan[0],rec

    def observe_result(self, action:str, actual_delta:Mapping[str,Any], after_obs:Mapping[str,Any], before_obs:Optional[Mapping[str,Any]]=None):
        particles=self.last_particles or self._particle_pool(); likes=[]
        for p in particles:
            pred,_,_=self.sim.predict_delta(p,self.history,action)
            likes.append(delta_similarity(pred,actual_delta))
        self.posterior.update(particles,likes,{'action':action,'observation_delta':dict(actual_delta)})
        # The actual transition is appended only after likelihood evaluation.
        self.history.append({'action':action,'observation_delta':dict(actual_delta),'state_before':copy.deepcopy(before_obs or {}),'state_after':copy.deepcopy(after_obs)})
        rec={'mean_likelihood':sum(likes)/len(likes) if likes else None,'max_likelihood':max(likes) if likes else None,
             'posterior_entropy_bits':self.posterior.entropy_bits(),'family_entropy_bits':self.posterior.family_entropy_bits(),
             'map_world_ids':[f'{f}{v}' for f,v in zip(FAMILIES,self.posterior.map_world())]}
        self.observations.append(rec); return rec

    def export_transfer_state(self):
        return {
            'posterior':{f:{str(k):float(v) for k,v in self.posterior.posterior[f].items()} for f in FAMILIES},
            'evidence_count':self.posterior.evidence_count,
        }

    def import_transfer_state(self, state:Optional[Mapping[str,Any]], strength:float=0.65):
        if not state: return
        strength=max(0.0,min(1.0,float(strength)))
        src=state.get('posterior',{})
        for f in FAMILIES:
            if f not in src: continue
            vals={i:float(src[f].get(str(i),src[f].get(i,0.0))) for i in range(16)}
            vals=_norm(vals)
            self.posterior.posterior[f]=_norm({i:strength*vals[i]+(1-strength)/16 for i in range(16)})
        # Symbol binding is deliberately NOT transferred across tasks.

    def telemetry(self):
        q=[d.get('quantum',{}) for d in self.decisions]; used=sum(bool(x.get('used_quantum')) for x in q)
        return {
            'version':self.VERSION,'implicit_hypothesis_space_size':16**6,'hypothesis_bits':24,'rules_per_family':16,'families':list(FAMILIES),
            'posterior_entropy_bits':self.posterior.entropy_bits(),'effective_hypotheses':self.posterior.effective_hypotheses(),
            'family_entropy_bits':self.posterior.family_entropy_bits(),'map_world_ids':[f'{f}{v}' for f,v in zip(FAMILIES,self.posterior.map_world())],
            'evidence_count':self.posterior.evidence_count,'quantum_calls':len(q),'quantum_used_calls':used,
            'quantum_use_rate':used/len(q) if q else 0.0,'decisions':self.decisions,'observations':self.observations,
            'information_firewall':{'active_hidden_world_used':False,'true_symbol_mapping_used':False,'candidate_bindings_marginalized':True},
            'posterior':{f:{str(k):v for k,v in self.posterior.posterior[f].items()} for f in FAMILIES},
        }

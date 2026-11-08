#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, hashlib, json, math, os, random, time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import stronger_ai as sa
from qahs_s5 import QAHSV32S5
from collaborative_qahs_s5 import CollaborativeQAHSV33S5

ROOT=Path(__file__).resolve().parent
MANIFEST_PATH=ROOT/'benchmark_manifest.json'
METRIC_SPEC_PATH=ROOT/'stronger_ai_metric_spec_v1_1.json'


def load_json(p): return json.loads(Path(p).read_text())

def file_sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def world_from(v): return sa.WorldModel(*map(int,v))

def clip01(x): return max(0.0,min(1.0,float(x)))

def state_sig(state):
    objs=[]
    for o in state.get('objects',[]):
        objs.append((str(o.get('symbol','?')),int(o.get('r',0)),int(o.get('c',0)),bool(o.get('active',False))))
    return (tuple(sorted(objs)),tuple(sorted(tuple(x) for x in state.get('barriers',[]))))


class FixedPerturber:
    def __init__(self, initial_obs:Mapping[str,Any], rate:float, seed:int):
        self.rate=max(0.0,min(1.0,float(rate))); rng=random.Random(seed)
        syms=sorted({str(o.get('symbol','?')) for o in initial_obs.get('objects',[])})
        k=max(1,round(self.rate*len(syms))) if self.rate>0 and syms else 0
        self.mask_symbols=set(rng.sample(syms,min(k,len(syms)))) if k else set()
        bars=[tuple(x) for x in initial_obs.get('barriers',[])]
        kb=round(self.rate*len(bars)) if self.rate>0 else 0
        self.hide_barriers=set(rng.sample(bars,min(kb,len(bars)))) if kb else set()
    def apply(self,obs):
        out=copy.deepcopy(obs)
        for o in out.get('objects',[]):
            if str(o.get('symbol','?')) in self.mask_symbols: o['symbol']='?'
        out['barriers']=[x for x in out.get('barriers',[]) if tuple(x) not in self.hide_barriers]
        out['observation_quality']={'kind':'fixed_missing_symbol_or_barrier','perturbation_rate':self.rate,
                                    'masked_symbol_count':len(self.mask_symbols),'hidden_barrier_count':len(self.hide_barriers)}
        return out


class LLMOnlyAgent:
    def __init__(self,base): self.base=base
    def act(self,obs,schema,prior): return self.base.act(obs,schema,prior)
    def observe_result(self,*args,**kwargs): return None
    def usage_summary(self): return self.base.usage_summary()
    def extra_summary(self): return {'condition':'llm-only'}
    def export_transfer_state(self): return None


class QAHSHybridAgent:
    def __init__(self,base,initial_obs,args,transfer_state=None):
        self.base=base
        self.qahs=QAHSV32S5(
            initial_obs,seed=args.seed_for_agent,max_steps=args.max_steps,
            particles=args.qahs_particles,eval_particles=args.qahs_eval_particles,
            quantum=not args.no_quantum,require_quantum=args.require_quantum,
            qaoa_reps=args.qaoa_reps,qaoa_shots=args.qaoa_shots,
            qaoa_optimizer_maxiter=args.qaoa_optimizer_maxiter,
            plan_horizon=args.plan_horizon,beam_width=args.beam_width,llm_weight=args.llm_weight)
        if transfer_state: self.qahs.import_transfer_state(transfer_state,args.transfer_state_strength)
        self.hybrid=[]
    def act(self,obs,schema,prior):
        llm=self.base.act(obs,schema,prior); la=llm['selected_action']
        self.qahs.apply_llm_prior(llm.get('leading_rules',{}),llm.get('confidence',0.0))
        a,q=self.qahs.choose_action(obs,la)
        out=dict(llm); out['llm_selected_action']=la; out['selected_action']=a; out['qahs']=q
        out['rationale']='QAHS V3.2-S5 arbitration over the same LLM proposal; original rationale: '+str(llm.get('rationale',''))
        self.hybrid.append(q); return out
    def observe_result(self,action,delta,after_obs,before_obs): return self.qahs.observe_result(action,delta,after_obs,before_obs)
    def usage_summary(self): return self.base.usage_summary()
    def extra_summary(self): return {'condition':'qahs','qahs':self.qahs.telemetry(),'transfer_state':self.qahs.export_transfer_state()}
    def export_transfer_state(self): return self.qahs.export_transfer_state()


class CollaborativeQAHSS5Agent:
    def __init__(self,base,initial_obs,args,transfer_state=None):
        self.base=base
        self.collab=CollaborativeQAHSV33S5(
            initial_obs,seed=args.seed_for_agent,max_steps=args.max_steps,
            particles=args.qahs_particles,eval_particles=args.qahs_eval_particles,
            quantum=not args.no_quantum,require_quantum=args.require_quantum,
            qaoa_reps=args.qaoa_reps,qaoa_shots=args.qaoa_shots,
            qaoa_optimizer_maxiter=args.qaoa_optimizer_maxiter,
            plan_horizon=args.plan_horizon,beam_width=args.beam_width,llm_weight=args.llm_weight,
            arbitration_margin=args.arbitration_margin,diagnostic_threshold=args.diagnostic_threshold,
            llm_confidence_weight=args.llm_confidence_weight)
        if transfer_state:
            self.collab.import_transfer_state(transfer_state,args.transfer_state_strength)
    def act(self,obs,schema,prior):
        llm=self.base.act(obs,schema,prior)
        a,rec=self.collab.act(obs,llm)
        out=dict(llm)
        out['llm_selected_action']=llm.get('selected_action')
        out['selected_action']=a
        out['qahs']=rec.get('qahs',{})
        out['collaboration']=rec
        out['rationale']='Collaborative QAHS V3.3 arbitration; LLM and QAHS are treated as peer hypotheses.'
        return out
    def observe_result(self,action,delta,after_obs,before_obs):
        return self.collab.observe_result(action,delta,after_obs,before_obs)
    def usage_summary(self): return self.base.usage_summary()
    def extra_summary(self): return {'condition':'collaborative-qahs','qahs':self.collab.telemetry(),'transfer_state':self.collab.export_transfer_state()}
    def export_transfer_state(self): return self.collab.export_transfer_state()


def build_agent(condition,args,initial_obs,audit_dir=None,experience_context=None,transfer_state=None):
    base=sa.LLMClient(mode=args.provider,model=args.model,ollama_url=args.ollama_url,scale_name='S5',
                      prompt_audit_dir=str(audit_dir) if audit_dir else None,request_timeout=args.request_timeout,
                      experience_context=experience_context)
    if condition=='llm-only': return LLMOnlyAgent(base)
    if condition=='qahs': return QAHSHybridAgent(base,initial_obs,args,transfer_state=transfer_state)
    return CollaborativeQAHSS5Agent(base,initial_obs,args,transfer_state=transfer_state)


def episode_measurements(result): return sa.episode_measurements(result)

def autonomy_episode_score(result):
    m=result.get('measurements') or episode_measurements(result)
    ner=clip01(m.get('novel_experiment_rate',0)); rar=clip01(m.get('redundant_action_rate',1)); ncap=clip01(m.get('no_change_action_persistence',1))
    return (ner+(1-rar)+(1-ncap))/3


def run_episode(condition,args,world,seed,trace_path,audit_dir,change=None,perturb=0.0,experience_context=None,transfer_state=None):
    env=sa.ScaledARCEnv(world,seed=seed,max_steps=args.max_steps); raw=env.reset(); binding=sa.make_visual_binding(seed)
    clean_obs=sa.agent_view(raw,binding); perturber=FixedPerturber(clean_obs,perturb,seed+9091) if perturb else None
    obs=perturber.apply(clean_obs) if perturber else clean_obs
    local=copy.copy(args); local.seed_for_agent=seed
    agent=build_agent(condition,local,obs,audit_dir,experience_context,transfer_state)
    schema=sa.HypothesisSchema.compact(16,6); logs=[]; false_solutions=0; t0=time.time()
    recovery_latency=None; change_turn=None; change_applied=False
    if args.show_grid:
        print(f'\nINITIAL {condition} | {world.ids()} | seed={seed}',flush=True); print(sa.render_grid(obs),flush=True)
    while not raw['solved_env'] and not raw['failed_env']:
        turn=raw['step']+1
        if change and not change_applied and turn==int(change['turn']):
            env.world=world_from(change['changed']); change_turn=turn; change_applied=True
            if args.show_grid: print(f'*** hidden rule change injected at turn {turn} ***',flush=True)
        before=obs
        r=agent.act(obs,schema,logs)
        if r.get('believes_solved') and not raw['solved_env']: false_solutions+=1
        a=r['selected_action']; nxt_raw,reward,done,info=env.step(a)
        clean_nxt=sa.agent_view(nxt_raw,binding); nxt=perturber.apply(clean_nxt) if perturber else clean_nxt
        delta=sa.observation_delta(obs,nxt)
        qobs=agent.observe_result(a,delta,nxt,before)
        e={'turn':turn,'state_before':before,'agent':r,'action':a,'state_after':nxt,'observation_delta':delta,
           'reward':reward,'solved_env':nxt_raw['solved_env'],'failed_env':nxt_raw['failed_env'],'termination_reason':info.get('termination_reason')}
        if qobs is not None: e['qahs_observation']=qobs
        logs.append(e)
        if change_applied and recovery_latency is None and turn>=change_turn:
            changed=bool(delta.get('moved') or delta.get('changed') or delta.get('removed') or delta.get('added') or delta.get('barriers_removed') or delta.get('barriers_added') or delta.get('solved_changed'))
            sig=(state_sig(before),a); earlier={(state_sig(x['state_before']),x['action']) for x in logs[:-1]}
            if changed and sig not in earlier: recovery_latency=turn-change_turn
        if args.show_grid:
            label=f"LLM={r.get('llm_selected_action',r.get('selected_action'))} EXEC={a}" if condition=='qahs' else f"ACTION={a}"
            print(f'\nTURN {turn} | {label}',flush=True); print(sa.render_grid(nxt),flush=True); print('DELTA:',json.dumps(delta),flush=True)
            if condition=='qahs': print('QAHS:',json.dumps(r.get('qahs',{})),flush=True)
        raw=nxt_raw; obs=nxt
    summary={'solved':bool(raw['solved_env']),'actions':int(raw['step']),'false_solution_events':false_solutions,
             'wall_seconds':time.time()-t0,'termination_reason':env.termination_reason,**agent.usage_summary()}
    result={'episode':{'seed':seed,'initial_world_ids':list(world.ids()),'final_world_ids':list(env.world.ids()),'scale':'S5','max_steps':args.max_steps,
                       'hypothesis_space_size':16**6,'hypothesis_bits':24,'perturbation_rate':perturb},
            'turns':logs,'summary':summary,'change_turn':change_turn,'recovery_latency':recovery_latency,'agent_summary':agent.extra_summary()}
    result['measurements']=episode_measurements(result); result['autonomy_score']=autonomy_episode_score(result)
    trace_path.parent.mkdir(parents=True,exist_ok=True); trace_path.write_text(json.dumps(result,indent=2))
    return result,agent.export_transfer_state()


def maybe_run(path,resume,fn):
    if resume and path.exists(): return load_json(path),None,True
    r,state=fn(); return r,state,False


def solve_rate(xs): return sum(bool(x['summary']['solved']) for x in xs)/len(xs) if xs else None

def adaptation_score(xs,budget):
    vals=[]
    for r in xs:
        L=r.get('recovery_latency'); L=budget if L is None else int(L); vals.append(clip01(1-L/budget))
    return sum(vals)/len(vals) if vals else None

def transfer_pair_score(cold,warm):
    c=cold['summary']; w=warm['summary']; cs=bool(c['solved']); ws=bool(w['solved'])
    if ws and not cs:return 1.0
    if cs and not ws:return 0.0
    if not cs and not ws:return 0.5
    ca=max(1,int(c['actions'])); wa=max(1,int(w['actions']))
    return clip01(0.5+0.5*((ca-wa)/ca))

def robustness_score(clean,pert):
    c=solve_rate(clean); p=solve_rate(pert)
    if c is None or p is None:return None,False
    if c<=0:return 0.0,True
    return clip01(p/c),False


def experience_from(r):
    return [{'action':t['action'],'observation_delta':t['observation_delta']} for t in r.get('turns',[])][-12:]


def run_suite(args):
    man=load_json(MANIFEST_PATH); profile=man['profiles'][args.profile]; outroot=Path(args.output_dir)/args.condition
    outroot.mkdir(parents=True,exist_ok=True)
    report={'benchmark_id':man['benchmark_id'],'manifest_sha256':file_sha256(MANIFEST_PATH),'metric_spec_sha256':file_sha256(METRIC_SPEC_PATH),
            'metric_spec_version':'1.1-S5-production','condition':args.condition,'provider':args.provider,'model':args.model,'profile':args.profile,
            'hypothesis_space_size':16**6,'hypothesis_bits':24,'max_steps':args.max_steps,'scores':{},'details':{}}

    if args.experiment=='smoke' or args.profile=='smoke':
        rec=man['smoke_world']; path=outroot/'smoke.json'
        r,_,_=maybe_run(path,args.resume,lambda:run_episode(args.condition,args,world_from(rec['world']),args.seed,path,outroot/'smoke_audit'))
        report['smoke']=r; Path(args.report).write_text(json.dumps(report,indent=2)); return report

    wanted=['generalization','adaptation','autonomy','transfer','robustness'] if args.experiment=='all' else [args.experiment]
    generalization=[]
    if any(x in wanted for x in ('generalization','autonomy','robustness')):
        n=int(profile['generalization']); worlds=man['generalization_worlds'][:n]
        for i,rec in enumerate(worlds,1):
            path=outroot/f'generalization/{rec["id"]}.json'; seed=args.seed+100+i
            r,_,reused=maybe_run(path,args.resume,lambda rec=rec,path=path,seed=seed:run_episode(args.condition,args,world_from(rec['world']),seed,path,outroot/f'generalization/audit_{rec["id"]}'))
            generalization.append(r); print(f"{args.condition} G {i}/{n} {rec['id']} solved={r['summary']['solved']} actions={r['summary']['actions']} reused={reused}",flush=True)
        if 'generalization' in wanted: report['scores']['G']=solve_rate(generalization); report['details']['generalization']={'episodes':generalization}
        if 'autonomy' in wanted:
            vals=[float(r['autonomy_score']) for r in generalization]; report['scores']['Au']=sum(vals)/len(vals) if vals else None; report['details']['autonomy']={'per_episode':vals}
        Path(args.report+'.partial').write_text(json.dumps(report,indent=2))

    if 'adaptation' in wanted:
        n=int(profile['adaptation']); xs=[]
        for i,rec in enumerate(man['adaptation_scenarios'][:n],1):
            path=outroot/f'adaptation/{rec["id"]}.json'; seed=args.seed+200+i
            change={'turn':man['adaptation_change_turn'],'changed':rec['changed']}
            r,_,reused=maybe_run(path,args.resume,lambda rec=rec,path=path,seed=seed,change=change:run_episode(args.condition,args,world_from(rec['base']),seed,path,outroot/f'adaptation/audit_{rec["id"]}',change=change))
            xs.append(r); print(f"{args.condition} A {i}/{n} {rec['id']} latency={r.get('recovery_latency')} solved={r['summary']['solved']} reused={reused}",flush=True)
        report['scores']['A']=adaptation_score(xs,man['recovery_budget']); report['details']['adaptation']={'episodes':xs,'recovery_budget':man['recovery_budget']}
        Path(args.report+'.partial').write_text(json.dumps(report,indent=2))

    if 'transfer' in wanted:
        n=int(profile['transfer']); source_rec=man['transfer']['source']; spath=outroot/'transfer/source.json'; sseed=args.seed+300
        source,source_state,reused=maybe_run(spath,args.resume,lambda:run_episode(args.condition,args,world_from(source_rec['world']),sseed,spath,outroot/'transfer/audit_source'))
        if reused and args.condition=='qahs':
            source_state=(source.get('agent_summary') or {}).get('transfer_state')
        exp=experience_from(source); pairs=[]
        # If a resumed source lacks explicit export state, warm QAHS still receives the same textual experience; this is logged.
        for i,rec in enumerate(man['transfer']['targets'][:n],1):
            seed=args.seed+400+i; cp=outroot/f'transfer/{rec["id"]}_cold.json'; wp=outroot/f'transfer/{rec["id"]}_warm.json'
            cold,_,rc=maybe_run(cp,args.resume,lambda rec=rec,cp=cp,seed=seed:run_episode(args.condition,args,world_from(rec['world']),seed,cp,outroot/f'transfer/audit_{rec["id"]}_cold'))
            warm,_,rw=maybe_run(wp,args.resume,lambda rec=rec,wp=wp,seed=seed:run_episode(args.condition,args,world_from(rec['world']),seed,wp,outroot/f'transfer/audit_{rec["id"]}_warm',experience_context=exp,transfer_state=source_state))
            ps=transfer_pair_score(cold,warm); pairs.append({'id':rec['id'],'score':ps,'cold':cold,'warm':warm})
            print(f"{args.condition} T {i}/{n} {rec['id']} pair_score={ps:.3f} cold={cold['summary']['solved']} warm={warm['summary']['solved']} reused={rc and rw}",flush=True)
        report['scores']['T']=sum(x['score'] for x in pairs)/len(pairs) if pairs else None; report['details']['transfer']={'source':source,'pairs':pairs}
        Path(args.report+'.partial').write_text(json.dumps(report,indent=2))

    if 'robustness' in wanted:
        n=int(profile['robustness']); clean=[]; pert=[]
        # Reuse matching generalization clean episodes when available.
        by_ids={tuple(r['episode']['initial_world_ids']):r for r in generalization}
        for i,rec in enumerate(man['generalization_worlds'][:n],1):
            w=world_from(rec['world']); key=tuple(w.ids()); seed=args.seed+100+i
            c=by_ids.get(key)
            if c is None:
                cp=outroot/f'robustness/{rec["id"]}_clean.json'; c,_,_=maybe_run(cp,args.resume,lambda w=w,cp=cp,seed=seed:run_episode(args.condition,args,w,seed,cp,outroot/f'robustness/audit_{rec["id"]}_clean'))
            pp=outroot/f'robustness/{rec["id"]}_perturbed.json'
            p,_,rp=maybe_run(pp,args.resume,lambda w=w,pp=pp,seed=seed:run_episode(args.condition,args,w,seed,pp,outroot/f'robustness/audit_{rec["id"]}_perturbed',perturb=man['perturbation_rate']))
            clean.append(c); pert.append(p); print(f"{args.condition} R {i}/{n} {rec['id']} clean={c['summary']['solved']} pert={p['summary']['solved']} reused_pert={rp}",flush=True)
        R,absent=robustness_score(clean,pert); report['scores']['R']=R; report['details']['robustness']={'clean':clean,'perturbed':pert,'clean_capability_absent':absent,'perturbation_rate':man['perturbation_rate']}
        Path(args.report+'.partial').write_text(json.dumps(report,indent=2))

    keys=['G','A','Au','T','R']
    report['SAI']=sum(report['scores'][k] for k in keys)/5 if all(report['scores'].get(k) is not None for k in keys) else None
    Path(args.report).write_text(json.dumps(report,indent=2))
    print('\nFINAL',json.dumps({'scores':report['scores'],'SAI':report['SAI']},indent=2),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser(description='S5 16.8M Stronger-AI five-domain benchmark')
    ap.add_argument('--condition',choices=['llm-only','qahs','collaborative'],required=True)
    ap.add_argument('--profile',choices=['smoke','paper','extended'],default='paper')
    ap.add_argument('--experiment',choices=['all','smoke','generalization','adaptation','autonomy','transfer','robustness'],default='all')
    ap.add_argument('--provider',choices=['placeholder','ollama','openai','anthropic'],default='ollama')
    ap.add_argument('--model',default='qwen3:8b'); ap.add_argument('--ollama-url',default='http://localhost:11434')
    ap.add_argument('--max-steps',type=int,default=12); ap.add_argument('--seed',type=int,default=16062026); ap.add_argument('--request-timeout',type=int,default=900)
    ap.add_argument('--show-grid',action='store_true'); ap.add_argument('--resume',action='store_true')
    ap.add_argument('--output-dir',default='s5_16m_results'); ap.add_argument('--report',default='s5_16m_report.json')
    # QAHS production parameters.
    ap.add_argument('--qahs-particles',type=int,default=384); ap.add_argument('--qahs-eval-particles',type=int,default=64)
    ap.add_argument('--qaoa-reps',type=int,default=1); ap.add_argument('--qaoa-shots',type=int,default=256); ap.add_argument('--qaoa-optimizer-maxiter',type=int,default=12)
    ap.add_argument('--plan-horizon',type=int,default=8); ap.add_argument('--beam-width',type=int,default=48); ap.add_argument('--llm-weight',type=float,default=0.20)
    ap.add_argument('--no-quantum',action='store_true'); ap.add_argument('--require-quantum',action='store_true'); ap.add_argument('--transfer-state-strength',type=float,default=0.65)
    ap.add_argument('--arbitration-margin',type=float,default=0.15,help='Minimum evidence margin required before preferring one hypothesis.')
    ap.add_argument('--diagnostic-threshold',type=float,default=0.12,help='If hypotheses are this close, test them with a diagnostic action.')
    ap.add_argument('--llm-confidence-weight',type=float,default=0.20,help='Bounded weight for LLM self-reported confidence in arbitration.')
    args=ap.parse_args()
    if args.require_quantum and args.no_quantum: ap.error('--require-quantum and --no-quantum are mutually exclusive')
    if args.condition=='llm-only': args.no_quantum=True; args.require_quantum=False
    Path(args.output_dir).mkdir(parents=True,exist_ok=True); run_suite(args)

if __name__=='__main__': main()

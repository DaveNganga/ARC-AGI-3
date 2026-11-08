#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import stronger_ai as sa

ROOT=Path(__file__).resolve().parent

def run_witness(world_vals,actions,max_steps=12):
    w=sa.WorldModel(*map(int,world_vals)); env=sa.ScaledARCEnv(w,seed=0,max_steps=max_steps); env.reset()
    for a in actions:
        env.step(a)
        if env.solved or env.failed: break
    return {'world_ids':list(w.ids()),'world_index24':w.index24(),'solved':bool(env.solved),'failed':bool(env.failed),
            'actions_executed':env.steps,'termination_reason':env.termination_reason,'support':sa.rule_support_report(w)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--report',default='s5_manifest_audit.json'); args=ap.parse_args()
    mp=ROOT/'benchmark_manifest.json'; m=json.loads(mp.read_text()); checks=[]
    # Basic hypothesis-space invariants.
    invariants={'space_size_ok':16**6==m['hypothesis_space']['complete_worlds'],
                'bits_ok':m['hypothesis_space']['hypothesis_bits']==24,
                'all_rule_families_16':all(len(sa.RULES[d])==16 for d in sa.DIMS)}
    # Smoke + generalization.
    rec=m['smoke_world']; checks.append({'kind':'smoke','id':rec['id'],**run_witness(rec['world'],rec['witness'],m['action_budget'])})
    for rec in m['generalization_worlds']:
        checks.append({'kind':'generalization','id':rec['id'],**run_witness(rec['world'],rec['witness'],m['action_budget'])})
    for rec in m['adaptation_scenarios']:
        checks.append({'kind':'adaptation_base','id':rec['id'],**run_witness(rec['base'],rec['base_witness'],m['action_budget'])})
        checks.append({'kind':'adaptation_changed','id':rec['id'],**run_witness(rec['changed'],rec['changed_witness'],m['action_budget'])})
    tr=m['transfer']; checks.append({'kind':'transfer_source','id':tr['source']['id'],**run_witness(tr['source']['world'],tr['source']['witness'],m['action_budget'])})
    for rec in tr['targets']:
        checks.append({'kind':'transfer_target','id':rec['id'],**run_witness(rec['world'],rec['witness'],m['action_budget'])})
    failures=[x for x in checks if not x['solved']]
    out={'benchmark_id':m['benchmark_id'],'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'invariants':invariants,
         'checks':checks,'witness_checks':len(checks),'witness_failures':len(failures),'passed':all(invariants.values()) and not failures}
    Path(args.report).write_text(json.dumps(out,indent=2)); print(json.dumps({'passed':out['passed'],'witness_checks':len(checks),'witness_failures':len(failures),'invariants':invariants},indent=2))
    if not out['passed']: raise SystemExit(2)
if __name__=='__main__': main()

#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path

def load(p): return json.loads(Path(p).read_text())
def exact_sign_p(pos,neg):
    n=pos+neg
    if n==0:return 1.0
    k=min(pos,neg)
    tail=sum(math.comb(n,i) for i in range(k+1))/(2**n)
    return min(1.0,2*tail)
def cmp_values(a,b):
    pos=sum(y>x for x,y in zip(a,b)); neg=sum(y<x for x,y in zip(a,b)); ties=len(a)-pos-neg
    return {'n':len(a),'qahs_better':pos,'llm_better':neg,'ties':ties,'two_sided_exact_sign_p':exact_sign_p(pos,neg),
            'llm_mean':sum(a)/len(a) if a else None,'qahs_mean':sum(b)/len(b) if b else None,
            'mean_delta':(sum(b)/len(b)-sum(a)/len(a)) if a else None}
def ad_vals(r):
    b=float(r['details']['adaptation']['recovery_budget']); out=[]
    for e in r['details']['adaptation']['episodes']:
        L=e.get('recovery_latency'); L=b if L is None else float(L); out.append(max(0,min(1,1-L/b)))
    return out
def au_vals(r): return list(map(float,r['details']['autonomy']['per_episode']))
def t_vals(r): return [float(x['score']) for x in r['details']['transfer']['pairs']]
def r_vals(r):
    out=[]
    for c,p in zip(r['details']['robustness']['clean'],r['details']['robustness']['perturbed']):
        # Capability-aware pair score: no clean capability cannot earn robustness credit.
        out.append(1.0 if c['summary']['solved'] and p['summary']['solved'] else 0.0)
    return out
def g_vals(r): return [1.0 if x['summary']['solved'] else 0.0 for x in r['details']['generalization']['episodes']]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--llm',default='s5_16m_llm_report.json'); ap.add_argument('--qahs',default='s5_16m_qahs_report.json'); ap.add_argument('--out',default='s5_16m_comparison.json'); args=ap.parse_args()
    L=load(args.llm); Q=load(args.qahs)
    if L['benchmark_id']!=Q['benchmark_id'] or L['manifest_sha256']!=Q['manifest_sha256']:
        raise SystemExit('Reports do not use the same frozen benchmark manifest')
    if L.get('profile')!=Q.get('profile'): raise SystemExit('Reports use different profiles')
    extract={'G':g_vals,'A':ad_vals,'Au':au_vals,'T':t_vals,'R':r_vals}
    paired={}
    for k,fn in extract.items():
        if k in L.get('scores',{}) and k in Q.get('scores',{}):
            a,b=fn(L),fn(Q)
            if len(a)==len(b): paired[k]=cmp_values(a,b)
    scores={k:{'llm':L.get('scores',{}).get(k),'qahs':Q.get('scores',{}).get(k),
               'delta':(Q.get('scores',{}).get(k)-L.get('scores',{}).get(k)) if L.get('scores',{}).get(k) is not None and Q.get('scores',{}).get(k) is not None else None}
            for k in ['G','A','Au','T','R']}
    out={'benchmark_id':L['benchmark_id'],'profile':L.get('profile'),'manifest_sha256':L['manifest_sha256'],'scores':scores,
         'SAI':{'llm':L.get('SAI'),'qahs':Q.get('SAI'),'delta':(Q.get('SAI')-L.get('SAI')) if L.get('SAI') is not None and Q.get('SAI') is not None else None},
         'paired_domain_tests':paired,
         'claim_discipline':'Improvement supports stronger reasoning characteristics under the predefined five-domain operational definition; it is not evidence of AGI/Strong AI or of quantum computational advantage.'}
    Path(args.out).write_text(json.dumps(out,indent=2))
    print('=== S5 16M FIVE-DOMAIN COMPARISON ===')
    for k in ['G','A','Au','T','R']:
        x=scores[k]; print(f"{k:>2}: LLM={x['llm']} QAHS={x['qahs']} delta={x['delta']}")
    print('SAI:',out['SAI'])
    for k,v in paired.items(): print(k,'paired',json.dumps(v))
    print('Wrote',args.out)
if __name__=='__main__': main()

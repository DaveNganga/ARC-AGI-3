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
def vals(r,k):
    if k=='G': return [1.0 if x['summary']['solved'] else 0.0 for x in r['details']['generalization']['episodes']]
    if k=='A':
        b=float(r['details']['adaptation']['recovery_budget']); out=[]
        for e in r['details']['adaptation']['episodes']:
            L=e.get('recovery_latency'); L=b if L is None else float(L); out.append(max(0,min(1,1-L/b)))
        return out
    if k=='Au': return list(map(float,r['details']['autonomy']['per_episode']))
    if k=='T': return [float(x['score']) for x in r['details']['transfer']['pairs']]
    if k=='R':
        return [1.0 if c['summary']['solved'] and p['summary']['solved'] else 0.0 for c,p in zip(r['details']['robustness']['clean'],r['details']['robustness']['perturbed'])]
    raise KeyError(k)
def cmp(a,b):
    pos=sum(y>x for x,y in zip(a,b)); neg=sum(y<x for x,y in zip(a,b)); ties=len(a)-pos-neg
    return {'n':len(a),'collab_better':pos,'baseline_better':neg,'ties':ties,'two_sided_exact_sign_p':exact_sign_p(pos,neg),'baseline_mean':sum(a)/len(a) if a else None,'collaborative_mean':sum(b)/len(b) if b else None,'mean_delta':(sum(b)/len(b)-sum(a)/len(a)) if a else None}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--baseline',default='s5_16m_qahs_report.json'); ap.add_argument('--collaborative',default='s5_16m_collaborative_report.json'); ap.add_argument('--llm',default='s5_16m_llm_report.json'); ap.add_argument('--out',default='s5_16m_collaborative_comparison.json'); a=ap.parse_args()
    B=load(a.baseline); C=load(a.collaborative); L=load(a.llm)
    for r in (B,C,L):
        if r['benchmark_id']!=B['benchmark_id'] or r['manifest_sha256']!=B['manifest_sha256'] or r.get('profile')!=B.get('profile'):
            raise SystemExit('Reports are not paired on the same frozen benchmark/profile')
    out={'benchmark_id':B['benchmark_id'],'profile':B.get('profile'),'manifest_sha256':B['manifest_sha256'],'scores':{},'paired_domain_tests':{},'SAI':{}}
    for k in ['G','A','Au','T','R']:
        out['scores'][k]={'llm':L['scores'].get(k),'qahs':B['scores'].get(k),'collaborative':C['scores'].get(k),
                          'qahs_delta_vs_llm':B['scores'].get(k)-L['scores'].get(k) if L['scores'].get(k) is not None and B['scores'].get(k) is not None else None,
                          'collab_delta_vs_llm':C['scores'].get(k)-L['scores'].get(k) if L['scores'].get(k) is not None and C['scores'].get(k) is not None else None,
                          'collab_delta_vs_qahs':C['scores'].get(k)-B['scores'].get(k) if C['scores'].get(k) is not None and B['scores'].get(k) is not None else None}
        for base,name in ((B,'qahs'),(L,'llm')):
            x,y=vals(base,k),vals(C,k)
            if len(x)==len(y): out['paired_domain_tests'][f'collaborative_vs_{name}_{k}']=cmp(x,y)
    out['SAI']={'llm':L.get('SAI'),'qahs':B.get('SAI'),'collaborative':C.get('SAI'),
                'collab_delta_vs_llm':C.get('SAI')-L.get('SAI') if C.get('SAI') is not None and L.get('SAI') is not None else None,
                'collab_delta_vs_qahs':C.get('SAI')-B.get('SAI') if C.get('SAI') is not None and B.get('SAI') is not None else None}
    out['claim_discipline']='V3.3 tests collaborative arbitration; it does not establish AGI/Strong AI or quantum advantage.'
    Path(a.out).write_text(json.dumps(out,indent=2))
    print('=== S5 COLLABORATIVE COMPARISON ===')
    for k,v in out['scores'].items(): print(k,v)
    print('SAI',out['SAI']); print('Wrote',a.out)
if __name__=='__main__': main()

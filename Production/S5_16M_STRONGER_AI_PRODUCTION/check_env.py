#!/usr/bin/env python3
import argparse, importlib, json, os, sys, urllib.request

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--provider',choices=['ollama','openai','anthropic','placeholder'],default='ollama'); ap.add_argument('--ollama-url',default='http://localhost:11434'); ap.add_argument('--model',default='qwen3:8b'); ap.add_argument('--require-quantum',action='store_true'); args=ap.parse_args()
    print('Python',sys.version.split()[0])
    ok=True
    for m in ['numpy','scipy']:
        try: mod=importlib.import_module(m); print(m,'OK',getattr(mod,'__version__',''))
        except Exception as e: print(m,'MISSING',e); ok=False
    for m in ['qiskit','qiskit_aer']:
        try: mod=importlib.import_module(m); print(m,'OK',getattr(mod,'__version__',''))
        except Exception as e:
            print(m,'MISSING',e)
            if args.require_quantum: ok=False
    if args.provider=='ollama':
        try:
            with urllib.request.urlopen(args.ollama_url.rstrip('/')+'/api/tags',timeout=5) as r: data=json.loads(r.read().decode())
            names=[x.get('name') or x.get('model') for x in data.get('models',[])]
            print('Ollama OK; models:',', '.join(names[:20]))
            if not any(str(x).split(':')[0]==args.model.split(':')[0] for x in names): print('WARNING requested model not found:',args.model)
        except Exception as e: print('Ollama CHECK FAILED',e); ok=False
    elif args.provider=='openai':
        if not os.getenv('OPENAI_API_KEY'): print('OPENAI_API_KEY missing'); ok=False
        try: importlib.import_module('openai'); print('openai SDK OK')
        except Exception as e: print('openai SDK missing',e); ok=False
    elif args.provider=='anthropic':
        if not os.getenv('ANTHROPIC_API_KEY'): print('ANTHROPIC_API_KEY missing'); ok=False
        try: importlib.import_module('anthropic'); print('anthropic SDK OK')
        except Exception as e: print('anthropic SDK missing',e); ok=False
    if args.require_quantum:
        try:
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
            qc=QuantumCircuit(24,24); qc.h(range(24)); qc.measure(range(24),range(24))
            counts=AerSimulator(method='matrix_product_state',seed_simulator=1).run(qc,shots=8).result().get_counts()
            print('24-qubit Aer MPS smoke OK; unique samples=',len(counts))
        except Exception as e: print('24-qubit quantum smoke FAILED',e); ok=False
    raise SystemExit(0 if ok else 2)
if __name__=='__main__': main()

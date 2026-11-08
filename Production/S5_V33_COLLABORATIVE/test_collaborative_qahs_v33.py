import json
from pathlib import Path

from collaborative_qahs_s5 import CollaborativeQAHSV33S5
import stronger_ai as sa


def make_obs(seed=777):
    w=sa.WorldModel(0,0,0,0,0,0)
    env=sa.ScaledARCEnv(w,seed=seed,max_steps=12); raw=env.reset(); b=sa.make_visual_binding(seed)
    return sa.agent_view(raw,b)


def test_llm_candidate_parser():
    llm={"P":["P2"],"G":["G0"],"M":["M1"],"T":["T0"],"I":["I0"],"S":["S0"]}
    w,m=CollaborativeQAHSV33S5._parse_rule_candidate(llm)
    assert w==(2,0,1,0,0,0) and m['complete']


def test_no_hidden_world_in_arbitration(tmp_path):
    obs=make_obs()
    c=CollaborativeQAHSV33S5(obs,seed=1,max_steps=12,quantum=False,require_quantum=False)
    llm={"selected_action":"RIGHT","rationale":"x","leading_rules":{},"confidence":0.5,"believes_solved":False}
    action,rec=c.act(obs,llm)
    assert action in ("UP","DOWN","LEFT","RIGHT")
    txt=json.dumps(rec)
    assert 'world_index24' not in txt
    assert 'true_world' not in txt
    assert rec['collaboration']['active_hidden_world_used'] is False


def test_arbitration_paths():
    obs=make_obs()
    c=CollaborativeQAHSV33S5(obs,seed=2,max_steps=12,quantum=False)
    c.qahs.posterior.posterior={f:{i:1/16 for i in range(16)} for f in c.qahs.posterior.posterior}
    llm={"selected_action":"RIGHT","leading_rules":{"P":["P0"],"G":["G0"],"M":["M0"],"T":["T0"],"I":["I0"],"S":["S0"]},"confidence":0.9}
    a,rec=c.act(obs,llm)
    assert rec['decision'] in {'llm','qa_hs','llm_action','agreement','diagnostic'}


if __name__=='__main__':
    test_llm_candidate_parser(); test_no_hidden_world_in_arbitration(Path('.')); test_arbitration_paths(); print('collaborative tests PASS')

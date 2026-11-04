import json, random, unittest
from pathlib import Path
import stronger_ai as sa
from qahs_s5 import QAHSV32S5, delta_similarity

ROOT=Path(__file__).resolve().parent

class S5ProductionTests(unittest.TestCase):
    def test_space_and_encoding(self):
        self.assertEqual(16**6,16777216)
        rng=random.Random(9)
        for _ in range(100):
            w=sa.WorldModel(*[rng.randrange(16) for _ in range(6)])
            self.assertEqual(sa.WorldModel.from_index24(w.index24()),w)
    def test_manifest_witnesses(self):
        m=json.loads((ROOT/'benchmark_manifest.json').read_text())
        recs=[(m['smoke_world']['world'],m['smoke_world']['witness'])]
        recs += [(x['world'],x['witness']) for x in m['generalization_worlds']]
        recs += [(x['base'],x['base_witness']) for x in m['adaptation_scenarios']]
        recs += [(x['changed'],x['changed_witness']) for x in m['adaptation_scenarios']]
        recs += [(m['transfer']['source']['world'],m['transfer']['source']['witness'])]
        recs += [(x['world'],x['witness']) for x in m['transfer']['targets']]
        for vals,acts in recs:
            e=sa.ScaledARCEnv(sa.WorldModel(*vals),seed=0,max_steps=12); e.reset()
            for a in acts:
                e.step(a)
                if e.solved or e.failed: break
            self.assertTrue(e.solved,(vals,acts,e.termination_reason))
    def test_delta_similarity_exact(self):
        d={'moved':[{'object':'X1','from':[1,1],'to':[1,2]}],'changed':[],'removed':[],'added':[],
           'barriers_removed':[],'barriers_added':[],'solved_changed':False,'failed_changed':False}
        self.assertAlmostEqual(delta_similarity(d,d),1.0)
    def test_qahs_firewall_and_size(self):
        w=sa.WorldModel(0,0,0,1,0,0); e=sa.ScaledARCEnv(w,seed=1,max_steps=12); raw=e.reset(); obs=sa.agent_view(raw,sa.make_visual_binding(1))
        q=QAHSV32S5(obs,seed=1,max_steps=12,particles=96,eval_particles=24,quantum=False,plan_horizon=3,beam_width=16)
        tel=q.telemetry(); self.assertEqual(tel['implicit_hypothesis_space_size'],16777216); self.assertEqual(tel['hypothesis_bits'],24)
        self.assertFalse(tel['information_firewall']['active_hidden_world_used']); self.assertFalse(tel['information_firewall']['true_symbol_mapping_used'])
    def test_prompt_no_ground_truth(self):
        w=sa.WorldModel(15,15,15,15,15,15); e=sa.ScaledARCEnv(w,seed=2,max_steps=12); obs=sa.agent_view(e.reset(),sa.make_visual_binding(2))
        prompt=sa.build_turn_prompt(obs,sa.HypothesisSchema.compact(16,6),[],'S5')
        a=sa.audit_prompt(prompt,w.ids()); self.assertTrue(a['passed'])

if __name__=='__main__': unittest.main()

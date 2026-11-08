#!/usr/bin/env python3
"""Collaborative QAHS V3.3 for the S5 16^6 hypothesis space.

V3.3 treats the LLM and QAHS as peer reasoning sources. The LLM may nominate a
complete or partial hidden-world hypothesis plus an action. QAHS maintains its
own posterior and generates an evidence-supported candidate. An arbitrator
selects LLM, QAHS, or a diagnostic action based only on agent-visible evidence.
The hidden true world and true symbol mapping are never supplied.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from qahs_s5 import ACTIONS, FAMILIES, QAHSV32S5, clip01, posterior_mass, Particle, _norm


class CollaborativeQAHSV33S5:
    VERSION = "QAHS-V3.3-COLLABORATIVE-S5"

    def __init__(self, initial_obs: Mapping[str, Any], seed: int = 1, max_steps: int = 12,
                 particles: int = 384, eval_particles: int = 64, quantum: bool = True,
                 require_quantum: bool = False, qaoa_reps: int = 1, qaoa_shots: int = 256,
                 qaoa_optimizer_maxiter: int = 12, plan_horizon: int = 8,
                 beam_width: int = 48, llm_weight: float = 0.20,
                 arbitration_margin: float = 0.15, diagnostic_threshold: float = 0.12,
                 llm_confidence_weight: float = 0.20):
        self.qahs = QAHSV32S5(
            initial_obs, seed=seed, max_steps=max_steps, particles=particles,
            eval_particles=eval_particles, quantum=quantum, require_quantum=require_quantum,
            qaoa_reps=qaoa_reps, qaoa_shots=qaoa_shots,
            qaoa_optimizer_maxiter=qaoa_optimizer_maxiter,
            plan_horizon=plan_horizon, beam_width=beam_width, llm_weight=llm_weight)
        self.arbitration_margin = float(arbitration_margin)
        self.diagnostic_threshold = float(diagnostic_threshold)
        self.llm_confidence_weight = float(llm_confidence_weight)
        self.arbitrations = []

    @staticmethod
    def _parse_rule_candidate(leading_rules: Mapping[str, Any]) -> Tuple[Optional[Tuple[int, ...]], Dict[str, Any]]:
        vals: Dict[str, int] = {}
        if not isinstance(leading_rules, Mapping):
            return None, {'complete': False, 'families_present': []}
        for f in FAMILIES:
            raw = leading_rules.get(f, []) or []
            candidates = []
            if isinstance(raw, (str, int)):
                raw = [raw]
            for x in raw:
                s = str(x).upper().strip()
                if s.startswith(f):
                    s = s[1:]
                try:
                    v = int(s)
                except Exception:
                    continue
                if 0 <= v < 16 and v not in candidates:
                    candidates.append(v)
            if candidates:
                vals[f] = candidates[0]
        complete = all(f in vals for f in FAMILIES)
        tup = tuple(vals[f] for f in FAMILIES) if complete else None
        return tup, {'complete': complete, 'families_present': sorted(vals), 'candidate_values': vals}

    def _llm_candidate(self, llm: Mapping[str, Any]) -> Tuple[Optional[Tuple[int, ...]], Dict[str, Any]]:
        return self._parse_rule_candidate(llm.get('leading_rules', {}))

    def _binding_for_world(self, world: Sequence[int]):
        bs = list(self.qahs.posterior.binding_posterior)
        if not bs:
            return tuple()
        # Binding is nuisance information. Use the current MAP binding only for
        # predictive comparison; it is not hidden ground truth.
        return max(bs, key=self.qahs.posterior.binding_posterior.get)

    def _candidate_log_posterior(self, world: Optional[Sequence[int]]) -> float:
        if world is None:
            return float('-inf')
        return math.log(max(1e-300, posterior_mass(self.qahs.posterior.posterior, world)))

    def _candidate_predictive_fit(self, world: Optional[Sequence[int]]) -> float:
        if world is None:
            return float('-inf')
        binding = self._binding_for_world(world)
        p = Particle(tuple(world), binding, 1.0)
        history = self.qahs.history
        # Use observed transition likelihoods under the candidate world. This is
        # deliberately computed from public history only.
        vals = []
        for h in history:
            action = str(h.get('action'))
            actual = h.get('observation_delta', {})
            pred, _, _ = self.qahs.sim.predict_delta(p, history[:history.index(h)], action)
            vals.append(max(1e-6, self.qahs._delta_likelihood(pred, actual) if hasattr(self.qahs, '_delta_likelihood') else 0.5))
        if not vals:
            return 0.0
        return sum(math.log(v) for v in vals) / len(vals)

    def _candidate_score(self, world: Optional[Sequence[int]], llm_conf: float = 0.0) -> Dict[str, float]:
        lp = self._candidate_log_posterior(world)
        pred = self._candidate_predictive_fit(world)
        conf = clip01(llm_conf) if world is not None else 0.0
        # Posterior support is the primary evidence. Predictive fit is secondary
        # and bounded so it cannot dominate posterior evidence.
        score = 0.78 * lp + 0.22 * pred
        if world is not None and llm_conf > 0:
            score += self.llm_confidence_weight * (conf - 0.5)
        return {'log_posterior': lp, 'predictive_fit': pred, 'composite': score}

    def _action_utility(self, action: str, obs: Mapping[str, Any]) -> float:
        pool = self.qahs.last_particles or self.qahs._particle_pool()
        evalp = pool[:min(self.qahs.eval_particles_n, len(pool))]
        m = self.qahs._plan_metrics((action,), evalp)
        mode = self.qahs._mode(max(1, self.qahs.max_steps - int(obs.get('step', 0))))
        geom = self.qahs._visible_geometry_progress(obs, action)
        remaining = max(1, self.qahs.max_steps - int(obs.get('step', 0)))
        cost = 1.0 / remaining
        if mode == 'explore':
            return .46*m['information_gain'] + .18*m['goal_progress'] + .10*m['solve_probability'] + .16*geom - .10*cost
        if mode == 'goal_discrimination':
            return .34*m['information_gain'] + .28*m['solve_probability'] + .18*m['goal_progress'] + .12*geom - .12*cost
        return .56*m['solve_probability'] + .18*m['goal_progress'] + .10*m['information_gain'] + .08*geom - .12*cost

    def _diagnostic_action(self, obs: Mapping[str, Any], llm_world: Optional[Sequence[int]], qahs_world: Sequence[int]):
        if llm_world is None:
            return None, 0.0
        llm_binding = self._binding_for_world(llm_world)
        q_binding = self._binding_for_world(qahs_world)
        lp = Particle(tuple(llm_world), llm_binding, 1.0)
        qp = Particle(tuple(qahs_world), q_binding, 1.0)
        scores = []
        for a in ACTIONS:
            ld, _, _ = self.qahs.sim.predict_delta(lp, self.qahs.history, a)
            qd, _, _ = self.qahs.sim.predict_delta(qp, self.qahs.history, a)
            lk = self.qahs._delta_likelihood(ld, qd) if hasattr(self.qahs, '_delta_likelihood') else 0.5
            separation = 1.0 - float(lk)
            utility = self._action_utility(a, obs)
            scores.append((0.70*separation + 0.30*utility, a, separation, utility))
        scores.sort(reverse=True)
        best = scores[0]
        return best[1], best[0]

    def act(self, obs: Mapping[str, Any], llm: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
        llm_action = str(llm.get('selected_action', 'UP')).upper()
        if llm_action not in ACTIONS:
            llm_action = 'UP'
        llm_world, llm_meta = self._llm_candidate(llm)
        llm_conf = clip01(float(llm.get('confidence', 0.0) or 0.0))

        # Do not inject the LLM prior before measuring agreement. This prevents
        # the arbiter from grading the LLM hypothesis using its own intervention.
        qahs_action, qahs_rec = self.qahs.choose_action(obs, llm_action)
        qahs_world = tuple(self.qahs.posterior.map_world())
        lscore = self._candidate_score(llm_world, llm_conf)
        qscore = self._candidate_score(qahs_world, 0.0)

        llm_action_utility = self._action_utility(llm_action, obs)
        qahs_action_utility = self._action_utility(qahs_action, obs)
        world_gap = abs(lscore['composite'] - qscore['composite'])

        decision = 'qa_hs'
        selected = qahs_action
        diagnostic_action = None
        diagnostic_score = 0.0

        if qahs_action == llm_action and world_gap <= self.arbitration_margin:
            decision = 'agreement'
            selected = llm_action
        elif llm_world is not None and lscore['composite'] > qscore['composite'] + self.arbitration_margin:
            decision = 'llm'
            selected = llm_action
        elif llm_world is not None and world_gap <= self.diagnostic_threshold:
            diagnostic_action, diagnostic_score = self._diagnostic_action(obs, llm_world, qahs_world)
            if diagnostic_action is not None:
                decision = 'diagnostic'
                selected = diagnostic_action
        else:
            # If the LLM hypothesis is competitive but incomplete, compare its
            # proposed action directly against the QAHS action utility.
            if llm_world is not None and lscore['composite'] >= qscore['composite'] - self.arbitration_margin and llm_action_utility > qahs_action_utility:
                decision = 'llm_action'
                selected = llm_action

        if selected not in ACTIONS:
            selected = qahs_action

        rec = {
            'version': self.VERSION,
            'selected_action': selected,
            'llm_action': llm_action,
            'qahs_action': qahs_action,
            'decision': decision,
            'llm_hypothesis': [f'{f}{v}' for f, v in zip(FAMILIES, llm_world)] if llm_world else None,
            'qahs_hypothesis': [f'{f}{v}' for f, v in zip(FAMILIES, qahs_world)],
            'llm_hypothesis_meta': llm_meta,
            'llm_confidence': llm_conf,
            'llm_hypothesis_score': lscore,
            'qahs_hypothesis_score': qscore,
            'hypothesis_score_margin': lscore['composite'] - qscore['composite'],
            'world_gap': world_gap,
            'llm_action_utility': llm_action_utility,
            'qahs_action_utility': qahs_action_utility,
            'diagnostic_action': diagnostic_action,
            'diagnostic_score': diagnostic_score,
            'qahs': qahs_rec,
            'collaboration': {
                'arbitration_margin': self.arbitration_margin,
                'diagnostic_threshold': self.diagnostic_threshold,
                'active_hidden_world_used': False,
                'true_symbol_mapping_used': False,
            }
        }
        self.arbitrations.append(rec)
        return selected, rec

    def observe_result(self, action: str, delta: Mapping[str, Any], after_obs: Mapping[str, Any], before_obs: Mapping[str, Any]):
        return self.qahs.observe_result(action, delta, after_obs, before_obs)

    def export_transfer_state(self):
        return self.qahs.export_transfer_state()

    def import_transfer_state(self, state, strength=0.65):
        return self.qahs.import_transfer_state(state, strength)

    def telemetry(self):
        t = self.qahs.telemetry()
        t['version'] = self.VERSION
        t['collaboration'] = {
            'arbitrations': self.arbitrations,
            'counts': {k: sum(1 for r in self.arbitrations if r.get('decision') == k) for k in ('agreement','llm','llm_action','qa_hs','diagnostic')},
        }
        return t

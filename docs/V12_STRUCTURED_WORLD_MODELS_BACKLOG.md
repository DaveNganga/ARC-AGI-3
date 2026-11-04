# V12 — Structured latent world models

Status: **vertical slice implemented; smoke evaluation pending**. V11 remains frozen and selectable for matched comparisons.

## Implemented in the first V12 slice

- a separate `v12-structured-world-models` harness and dashboard selector;
- six factorized marginal posteriors with an explicit open-world `unknown` component;
- uncertain controller, target, focus, and affected-entity role bindings;
- an empirical local-effect memory learned only from visible action outcomes;
- Explore -> Goal-Discrimination -> Exploit mode selection based on family entropy;
- mode-conditioned additions to the shared ordered-plan objective;
- identical candidate costs and evidence for classical and quantum search;
- trace diagnostics for family posteriors, entropy, bindings, mode, and learned effects;
- prompt guidance that asks Luna for compositional complete hypotheses without supplying a fixed ontology.
- Luna-declared stable goal IDs with explicit `reuse`, `revise`, and `new` lineage operations;
- goal evidence carried across genuine catalog refreshes without Python comparing goal prose;
- goal-discrimination value based on observable prediction differences and suppressed when all branches predict inactivity.
- independent evidence channels: physical effects update controller/interaction/dynamics, boundary evidence updates topology,
  procedure evidence updates sequence, and only measured goal relations or explicit game outcomes update goals;
- complete-hypothesis probabilities reconstructed from the product of the six component marginals so evidence cannot
  leak from a confirmed mechanism into an unconfirmed objective.

The first slice deliberately reuses V11's measured grid, stable entity tracking, action catalog, outcome classifier,
and evidence-bound procedure machinery. This makes the V11/V12 difference attributable to the structured latent
representation and mode-conditioned search rather than a simultaneous perception rewrite.

## Purpose

Introduce a shared, extensible compositional language between visual perception and hypothesis search without
pre-encoding the solution to a particular ARC game. The intended claim is that QAHS may improve search over
compositional hypotheses in unfamiliar interactive environments—not that the system discovers every abstraction
from an unconstrained blank slate.

## Fair comparison boundary

All three conditions receive the same measured state, entity catalog, learned action effects, and hypothesis
components. They differ only in how hypotheses or procedures are selected:

- LLM-only uses the model's own selection.
- Classical QAHS searches the shared structured space exactly or approximately.
- Quantum QAHS searches the same objective and representation with QAOA.

Python may extract facts, preserve identity, measure transitions, and replay learned local effects. Python must not
declare the intended game objective or silently supply a solution plan. The environment remains the source of truth.

## Proposed latent dimensions

- controller or acted-upon entity;
- goal or desired relationship;
- action effect / movement dynamics;
- topology and boundaries;
- interaction mechanism;
- temporal or sequence condition.

Each dimension must support `unknown` and model-proposed extensions. A fixed exhaustive game ontology is explicitly
out of scope.

## Candidate architecture

1. Luna extracts candidate structured components and may propose a new relation when the vocabulary is insufficient.
2. Python maintains separate posteriors over dimensions and uncertain entity-role bindings.
3. Python learns small local transition models from observed action effects rather than emulating the full game.
4. Candidate components are composed into falsifiable latent world models.
5. Short procedures are replayed against those learned local models.
6. LLM, classical QAHS, and quantum QAHS select from the same candidates and evidence.
7. Abstract rules may transfer between levels or related tasks; entity identities and bindings do not.

## Ideas borrowed from the teacher-provided implementation

- compositional rule families;
- uncertain symbol/entity-role bindings;
- family-level posterior entropy;
- Explore -> Goal-Discrimination -> Exploit modes;
- deliberately weak LLM priors that evidence can overturn;
- posterior-conditioned receding-horizon planning;
- information-firewall and prompt-audit checks;
- frozen matched comparisons, resumable traces, and paired metrics.

## Explicit non-goals

- Do not copy the fixed 96-rule ontology as an exhaustive description of ARC-AGI-3.
- Do not use the benchmark's hidden generator or an equivalent perfect simulator.
- Do not treat a larger encoded space or more qubits as evidence of quantum advantage.
- Do not remove the matched classical QAHS control.
- Do not transfer entity identities between games or levels.

## Questions for smoke evaluation and later slices

- Does canonicalizing Luna's free-text components merge paraphrases aggressively enough without merging distinct rules?
- How and when should a Luna-proposed primitive persist beyond the current catalog?
- What minimum evidence permits a local transition model to predict counterfactual actions?
- How should dependencies between latent dimensions be represented in the quantum objective?
- What ablation proves that Python's representation is not doing the substantive game solving?
- Which V11 traces provide appropriate frozen before/after comparisons for V12?

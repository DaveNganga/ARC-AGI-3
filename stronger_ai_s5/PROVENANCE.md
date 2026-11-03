# QAHS production lineage

This release is **QAHS V3.2-S5 production scale-out**.

It preserves the best-performing V3.2 architectural ideas demonstrated in the frozen S1 ablation: posterior-conditioned decision making, explicit Explore -> Goal-Discrimination -> Exploit modes, multi-step planning, a fallible LLM semantic/action prior, and an information firewall.

It is not byte-for-byte V3.2. Exact V3.2 maintained all 256 S1 worlds explicitly and could evaluate plans across the complete S1 posterior. That is not computationally appropriate for 16^6 worlds. S5 therefore replaces the exact S1 posterior with:

- a factorized Bayesian posterior over six 16-valued rule families;
- bounded joint particles for interaction reasoning;
- a 24-qubit, six-register QAOA posterior sampler;
- receding-horizon posterior-conditioned plan evaluation over the sampled candidates;
- explicit marginalization of symbol/class bindings rather than using the hidden true binding.

The active hidden world is never passed to QAHS. Harness logs retain ground truth only for post-hoc evaluation.

This experiment can test whether the **QAHS architecture** improves reasoning relative to the same LLM alone. It does not isolate a quantum computational advantage; a matched CAHS control remains the appropriate later ablation for that claim.

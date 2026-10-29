# Group 2 — Quantum-Assisted Hypothesis Search (QAHS)

**Bottom-up research track: formalize, implement, and evaluate classical and quantum-assisted hypothesis search.**

References:
1. ARC-AGI-3 Technical Report — <https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf>
2. ARC-AGI-3 task `ar25` — <https://arcprize.org/tasks/ar25>

> This file records the assignment as given. The project's current state and the plan to meet this assignment are in [`CORE.md`](CORE.md) and [`IMPROVEMENTS.md`](IMPROVEMENTS.md).

## Mission

Group 2 owns the quantum algorithm and search-engine side of the project. Its fundamental question is:

> **Can quantum-assisted hypothesis search improve the adaptive reasoning performance of a frontier LLM?**

This replaces the earlier emphasis on finding a difficult cybersecurity classification dataset and trying to beat a classical classifier with quantum feature mappings. A better classifier alone would not be strong evidence for a fundamentally "stronger" form of AI. QAHS instead targets novel-environment reasoning and adaptive hypothesis discrimination directly.

---

## 1. Formalize the hypothesis representation

The LLM may first generate natural-language hypotheses such as "Blue is the player and red objects are hazards." Group 2 must determine how to convert each hypothesis into a finite, computable representation suitable for search and optimization.

### Rule dimensions (Figure 1)

| Symbol | Rule dimension | Question it answers | Example | Alternatives |
|---|---|---|---|---:|
| P | Player / Control | What object(s) does the agent control? | P0: Blue is controllable | 16 |
| G | Goal | What must happen to solve the environment? | G0: Reach green | 16 |
| M | Movement / Dynamics | How does the controlled object move? | M0: Move one cell | 16 |
| T | Topology / Barrier | How do walls, gaps, doors, regions, etc. behave? | T0: Gray wall blocks movement except at gap | 16 |
| I | Interaction | What happens when objects interact? | I0: Touch red = hazard / game over | 16 |
| S | State / Sequence | Does history or event order affect the rules? | S0: Stateless | 16 |

### Example candidate world models

- **H1** = (P0, G0, M0, T0, I0, S0), confidence 0.42: Blue is controllable; the goal is to reach green; movement is one cell at a time; gray is an impassable wall except for its gap; red is hazardous; the environment is stateless.
- **H2** = (P0, G0, M0, T0, I0, S0)*, confidence 0.28: Blue is controllable; the goal is to reach red; movement slides until an obstacle; gray is an impassable wall except for its gap; red is collectible; touching red collects it; red must be reached before green.
- **H3** = (P0, G0, M0, T0, I0, S0)*, confidence 0.18: The nearest object is controllable; the goal is to reach green; movement can push objects; gray is a door controlled by a switch; pushing red activates the switch; the environment has a sequence-dependent rule.
- …plus additional candidate world models drawn from the full hypothesis space.

\* Tuples as written in the source; see [Open questions](#open-questions-about-the-source-text).

### Full hypothesis space

$$
\mathcal{H} = H_P \times H_G \times H_M \times H_T \times H_I \times H_S, \qquad |\mathcal{H}| = 16^6 = 16{,}777{,}216 \text{ complete candidate world models}
$$

Available actions: UP, DOWN, LEFT, RIGHT. Selected action: RIGHT.

The reasoning in Figure 1 can be summarized as:

> H1 (0.42) → Blue is controllable + Green is goal → RIGHT tests movement/topology and moves toward Green → **RIGHT**

The LLM defines the six-dimensional hypothesis space, which produces about 16M possible complete world models. It then reasons over the factorized space and may identify a few leading candidates such as H1, H2, and H3. The LLM does not need to print 16,777,216 English paragraphs. Give it the complete factorized specification that defines the full space, and tell it explicitly that its action decisions must consider that complete space. How well the LLM reasons over this compact representation is part of the experiment.

### Role of the LLM in the QAHS condition

The LLM's role is to perceive and interpret the environment and to instantiate or confirm the factorized rule vocabulary. **It "must not perform the final hypothesis-space search or select the intermediate action; those functions belong to QAHS."**

### Binary / qubit encoding

Each dimension has 16 alternatives, so each can be indexed by 4 bits/qubits. A complete candidate world model needs a **24-bit/qubit hypothesis index**, before any ancillary or work qubits the chosen algorithm requires:

$$
x = |p\rangle\,|g\rangle\,|m\rangle\,|t\rangle\,|i\rangle\,|s\rangle, \qquad 4 \text{ qubits/register} \times 6 \text{ registers} = 24 \text{ hypothesis qubits}
$$

For example, one complete world model can be written as the six indices (P2, G9, M1, T6, I3, S5).

Group 2 must document:
- the exact reversible/binary encoding;
- invalid-state handling, for when a dimension has fewer than 16 alternatives at smaller scales;
- all ancillary qubits.

---

## 2. Define the search objective

The search module should **not** simply choose the action favoured by the highest-confidence hypothesis. It should find the action or experiment that best discriminates among competing hypotheses and so improves the agent's world model. In other words, it uses the large hypothesis space to choose the next experiment that is expected to reduce uncertainty and reveal the hidden world model efficiently.

$$
a^* = \arg\max_{a \in A_t} \; \mathrm{IG}(a, \mathcal{H} \mid O_t) \qquad \text{(Equation 2)}
$$

Equation 2 asks: *"Of all available actions, which one is expected to give the most information for distinguishing among the current competing hypotheses?"*

- $A_t$: the available actions
- $\mathcal{H}$: the complete factorized hypothesis space
- $O_t$: all evidence observed through turn $t$
- $\mathrm{IG}(a, \mathcal{H} \mid O_t)$: information gain
- $\arg\max$: the action that produces the maximum value

IG can be expected information gain or another carefully justified utility function. Objective terms to investigate:
- consistency with prior observations
- expected uncertainty reduction
- prediction disagreement among hypotheses
- expected progress toward inferred goals
- complexity penalties
- exploration cost

Central question: **"Given everything the agent might currently believe, what action should it perform next to learn which belief is correct?"**

---

## 3. Develop Quantum-Assisted Hypothesis Search

Pipeline:

```
ARC → LLM → H → QAHS → Action → ARC → repeat
```

Everything outside the action-selection/search mechanism must stay identical to the matching classical condition.

The group should investigate which quantum formulation is **mathematically appropriate**, rather than choosing an algorithm because it sounds promising. Candidates:
- QUBO / Ising optimization
- QAOA
- Quantum annealing
- Amplitude amplification / Grover-style search
- Other quantum optimization or search formulations that can be justified mathematically

Group 2 must show how an LLM-generated natural-language hypothesis becomes a numerical/binary representation a quantum algorithm can process. They must also define mathematically how the algorithm scores candidate actions to find the action most useful for distinguishing among competing hypotheses.

### Suggested approach

Take hypothesis $h_i$, convert it into a machine/quantum-computable representation $x_i$, and compute how useful action $a_j$ would be given that hypothesis and everything observed so far, $O_t$:

$$
h_i \;\rightarrow\; x_i \;\rightarrow\; f(x_i, a_j, O_t)
$$

**Worked example.** $h_i$ = (P0, G0, M0, T0, I1, S0): "Blue is the controllable player; the goal is to reach the green object; movement occurs one cell at a time; gray cells form an impassable wall with a traversable gap; contact with a red object is hazardous; and the environment is stateless."

| Dimension | Selected rule | Meaning |
|---|---|---|
| P | P0 | Blue is controllable |
| G | G0 | Reach green |
| M | M0 | Move one cell |
| T | T0 | Gray is a wall with a passable gap |
| I | I1 | Red contact is hazardous |
| S | S0 | Stateless |

Each dimension has 16 alternatives (0…15), so each needs $\log_2 16 = 4$ bits/qubits:

```
P0 = 0000   G0 = 0000   M0 = 0000   T0 = 0000   I1 = 0001   S0 = 0000
x_i = 0000 0000 0000 0000 0001 0000      (24-bit representation of the complete hypothesis)
```

Actions: $a_j \in \{a_1 = \text{UP},\ a_2 = \text{DOWN},\ a_3 = \text{LEFT},\ a_4 = \text{RIGHT}\}$, e.g. $a_4 = \text{RIGHT}$.

Observations: $O_t = \{o_1, o_2, \dots, o_t\}$, for example:
- $o_1$ = (RIGHT, blue moved right)
- $o_2$ = (UP, blue moved up)
- $o_3$ = moving into gray produced no movement

---

## 4. Build the Python/Jupyter prototype

Implement the search engine as an **externally callable module**. A simulator is enough for the first prototype. Real quantum hardware should only be considered once the formulation and experimental loop work.

**Primary QAHS loop:** environment $S_t$ → the same frontier LLM perceives the state and supplies the factorized $\mathcal{H}$ / $O_t$ representation → QAHS searches $\mathcal{H}$ and selects $a_t$ → the environment executes $a_t$ → new state $S_{t+1}$ → the LLM interprets the observation and updates the representation → QAHS searches again → repeat.

- Input: current environment state / metadata $S_t$
- Input: accumulated observations $O$
- Input: current state metadata
- Input: fixed hypothesis set $\mathcal{H}_t$
- Input: available finite action set $A_t$
- Input: predicted outcomes under each hypothesis
- QAHS output: selected action + objective/search score
- Return the selected action
- Receive the resulting observation and repeat
- **Termination:** environment-verified success, or the action/step budget is used up. Also log the LLM's own solved / not-solved declaration.

---

## 5. Quantum-specific evaluation

Measure whether QAHS gives a genuine search/optimization advantage and what resources it needs:
- search / objective quality
- number of hypotheses or candidate experiments evaluated
- actions to solution after integration
- wall-clock time
- classical compute / resources
- quantum circuit depth, shots, qubits, iterations, or annealing resources, as applicable
- sensitivity to hypothesis-space size $N$
- sensitivity to action-sequence / search depth $d$
- performance on noisy simulators versus real hardware, if hardware testing is attempted

A useful later scaling experiment: vary $N$ = 8, 16, 32, 64, 128, 256, … and see whether relative performance changes as the hypothesis space grows. If action sequences are searched, complexity can also be varied through look-ahead depth $d$.

A simulator is enough for the first prototype. The target $16^6$ space corresponds to 24 hypothesis qubits, but practical simulation cost also depends on ancillas, circuit depth, entanglement, shots, and oracle complexity. Start with smaller spaces and scale to the target only after correctness is established.

| Scale | Hypothesis space | Index qubits | Purpose |
|---|---:|---:|---|
| S1 | 256 | 8 | Debug / validate |
| S2 | 7,776 | ≈13 | Early scaling |
| S3 | 262,144 | 18 | Intermediate |
| S4 | 2,985,984 | ≈22 | Large |
| S5 (target) | 16,777,216 | 24 | Primary Part 1 target |

### Claim boundary

If LLM + QAHS significantly outperforms LLM Only under **Group 1's pre-registered definition**, the result may support the claim that the combined architecture shows "Stronger AI" characteristics. It does **not** by itself show that the improvement comes from a quantum computational advantage.

---

## Group 2 deliverables

- [ ] Formal finite hypothesis representation compatible with LLM output
- [ ] Mathematically justified search / utility objective
- [ ] Working QAHS implementation in Python/Jupyter, initially on a quantum simulator
- [ ] Justification for the selected quantum algorithm / formulation
- [ ] Resource and scaling analysis
- [ ] Correctness tests at small scales, then scaling results up to the largest feasible space (target ≈16.8M hypotheses)
- [ ] Experimental comparison of LLM Only and QAHS using identical hypothesis/action inputs
- [ ] Experimental results comparing LLM Only with LLM + QAHS

---

## Open questions about the source text

*These are implementation notes, not part of the assignment. Confirm them with the instructor or Group 1.*

1. **H2 and H3 tuples.** Both are written as (P0, G0, M0, T0, I0, S0), the same as H1, but their descriptions differ (goal = red, sliding movement, nearest object controllable, …). This is probably a copy error; they should use different option indices.
2. **Uneven scale sizes.** With six dimensions, S3 = 8^6, S4 = 12^6, and S5 = 16^6 work. But S1 = 256 is not k^6 for any integer k, and S2 = 7,776 is 6^5 (6^6 would be 46,656). So S1 and S2 imply dimensions of different sizes, or fewer than six dimensions. The per-dimension option counts for each scale need to be fixed. This is the case where the invalid-state handling in §1 applies.
3. **Figure 1 and Equation 1** are referenced but not included in the text received.
4. **Group 1 dependencies.** The LLM-only baseline protocol and the pre-registered success definition belong to Group 1 and must be obtained before the final experiments.

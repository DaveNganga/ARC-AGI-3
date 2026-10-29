from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf" / "QAHS_Prototype_Plain_Language_Guide_and_Test_Plan.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

PAGE_W, PAGE_H = letter
MARGIN = 0.66 * inch
NAVY = colors.HexColor("#18324A")
TEAL = colors.HexColor("#287F78")
INK = colors.HexColor("#202A33")
MUTED = colors.HexColor("#5A6874")
LINE = colors.HexColor("#CBD3D9")
LIGHT = colors.HexColor("#F4F6F7")
AMBER = colors.HexColor("#9B6824")
RED = colors.HexColor("#9A3E38")


class Rule(Flowable):
    def __init__(self, width: float, color=LINE, thickness=0.7, space_before=5, space_after=9):
        super().__init__()
        self.width = width
        self.color = color
        self.thickness = thickness
        self.spaceBefore = space_before
        self.spaceAfter = space_after
        self.height = thickness

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="TitleQAHS", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=24, leading=28, textColor=NAVY, alignment=TA_LEFT, spaceAfter=10,
))
styles.add(ParagraphStyle(
    name="SubtitleQAHS", parent=styles["Normal"], fontName="Helvetica",
    fontSize=11, leading=16, textColor=MUTED, spaceAfter=14,
))
styles.add(ParagraphStyle(
    name="H1QAHS", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=17, leading=21, textColor=NAVY, spaceBefore=2, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="H2QAHS", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=12.5, leading=15, textColor=TEAL, spaceBefore=8, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="BodyQAHS", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.4, leading=13.3, textColor=INK, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="SmallQAHS", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=8.2, leading=11.2, textColor=MUTED, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="CalloutQAHS", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=10, leading=14, textColor=NAVY, leftIndent=10, rightIndent=10,
    borderColor=TEAL, borderWidth=0, borderPadding=0, spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="StepQAHS", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=8.8, leading=12.2, textColor=INK, leftIndent=16, firstLineIndent=-16, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="CheckQAHS", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=8.6, leading=11.8, textColor=INK, leftIndent=13, firstLineIndent=-13, spaceAfter=3,
))
styles.add(ParagraphStyle(
    name="TableHeadQAHS", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=8.2, leading=10.2, textColor=NAVY,
))
styles.add(ParagraphStyle(
    name="TableBodyQAHS", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=7.8, leading=10, textColor=INK,
))


def P(text: str, style="BodyQAHS") -> Paragraph:
    return Paragraph(text, styles[style])


def step(number: int, title: str, text: str) -> Paragraph:
    return P(f"<b>{number}. {title}:</b> {text}", "StepQAHS")


def check(text: str) -> Paragraph:
    return P(f"[ ] {text}", "CheckQAHS")


def simple_table(rows, widths, header=True):
    prepared = []
    for r, row in enumerate(rows):
        prepared.append([P(str(cell), "TableHeadQAHS" if header and r == 0 else "TableBodyQAHS") for cell in row])
    t = Table(prepared, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.45, LINE),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
            ("LINEBELOW", (0, 0), (-1, 0), 0.9, NAVY),
        ])
    t.setStyle(TableStyle(commands))
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_H - 0.42 * inch, PAGE_W - MARGIN, PAGE_H - 0.42 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 0.35 * inch, "QAHS Prototype - plain-language code guide and test plan")
    canvas.drawRightString(PAGE_W - MARGIN, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


doc = BaseDocTemplate(
    str(OUT), pagesize=letter,
    leftMargin=MARGIN, rightMargin=MARGIN, topMargin=0.62 * inch, bottomMargin=0.58 * inch,
    title="QAHS Prototype: Plain-Language Code Guide and Test Plan",
    author="Dave Ng'ang'a and Abylay Bissary",
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=header_footer)])

story = []
story += [
    Spacer(1, 0.18 * inch),
    P("Quantum-Assisted Hypothesis Search", "TitleQAHS"),
    P("A plain-language guide to what the current code does, where QAOA fits, what has already been verified, and how to test the research claim.", "SubtitleQAHS"),
    Rule(doc.width, TEAL, 1.3, 2, 14),
    P("The one-sentence version", "H1QAHS"),
    P("The program faces a hidden-answer problem. It lists possible explanations and possible actions, scores which action could teach it the most, then lets either a classical search method or QAOA choose a hypothesis-action candidate. After the environment responds, the program removes explanations that no longer fit and repeats."),
    P("The researchers know the hidden answer so that the run can be graded. The agent does not receive that answer. QAOA is useful only if it can guide the agent through the search more effectively than a matched classical method.", "CalloutQAHS"),
    P("The full loop", "H1QAHS"),
    step(1, "Observe", "The harness receives the visible state and the history of earlier actions."),
    step(2, "Propose", "An interpreter supplies candidate explanations. The current default is rule-based; an OpenAI adapter exists but is not wired into the runners."),
    step(3, "Predict", "For every explanation, the code predicts what each available action should cause."),
    step(4, "Score", "Actions receive higher scores when they could reduce uncertainty or make progress toward the goal."),
    step(5, "Encode", "The candidate explanation and action choices are translated into bits. These bits are the decision variables used by classical search or QAOA."),
    step(6, "Search", "The selected backend searches for a low-cost bitstring. Lower cost means better according to the scoring rules."),
    step(7, "Act and learn", "The environment executes the selected action, returns an observation, and the program updates its belief in each explanation."),
    step(8, "Repeat", "The new evidence changes the scores, so the next search solves a slightly different problem."),
    Spacer(1, 7),
    P("What QAOA contributes", "H2QAHS"),
    P("QAOA does not invent the possible explanations, understand the game, or decide what counts as useful evidence. Those parts are supplied classically. QAOA receives a cost function and tries to put more probability on low-cost bitstrings. The research question is whether this search step leads to the hidden answer with fewer actions or better reliability under the same budget."),
    PageBreak(),
]

story += [
    P("1. The small red-tile teaching example", "H1QAHS"),
    P("This is the clearest place to understand the program. A blue player sees a red tile and a green door. The red tile has one hidden role."),
    simple_table([
        ["Possible explanation", "What touching red would do"],
        ["Red is a hazard", "Destroy the player"],
        ["Red is a switch", "Open the green door"],
        ["Red is the goal", "Finish the game"],
        ["Red is irrelevant", "Cause no visible change"],
    ], [2.15 * inch, 4.0 * inch]),
    Spacer(1, 8),
    P("There are four hypotheses and two actions. Eight hypothesis-action combinations fit into three decision bits because 2^3 = 8.", "CalloutQAHS"),
    P("What the main files do", "H2QAHS"),
    simple_table([
        ["File", "Plain-language job"],
        ["environment.py", "Hides the true red-tile rule and returns the result of an action."],
        ["interpreter.py", "Creates the four candidate explanations. It also defines the API-ready language-model interface."],
        ["predictor.py", "Says what each hypothesis predicts for each action."],
        ["objective.py", "Calculates information gain, progress, costs, and updated beliefs."],
        ["compiler.py", "Maps a hypothesis and action to a three-bit index and decodes it later."],
        ["agent.py", "Runs the observe -> score -> search -> act -> update loop."],
    ], [1.55 * inch, 4.6 * inch]),
    P("Example: the hidden rule is 'red is a switch'", "H2QAHS"),
    step(1, "Start uncertain", "All four explanations begin at probability 0.25."),
    step(2, "Score actions", "Approaching red is valuable because its four possible outcomes separate all four explanations. Approaching green distinguishes much less."),
    step(3, "Select and act", "The search backend chooses approach_red."),
    step(4, "Receive evidence", "The environment reports that the green door opened."),
    step(5, "Update beliefs", "The switch explanation rises above 99.9 percent; contradictory explanations receive almost no belief."),
    step(6, "Finish", "The next scores favor approach_green, which reaches the goal."),
    P("This example validates the loop. It is too small and too obvious to demonstrate a quantum advantage."),
    PageBreak(),
]

story += [
    P("2. What the three search backends actually do", "H1QAHS"),
    simple_table([
        ["Backend", "What it does", "What it proves"],
        ["Greedy baseline", "Picks the action with the largest immediate utility, then pairs it with the best-believed hypothesis.", "A simple policy can use the same scores."],
        ["Exact classical", "Calculates every valid candidate cost and takes the minimum.", "The scoring and encoding produce the expected optimum on small problems."],
        ["QAOA statevector", "Tunes a simulated quantum circuit so low-cost bitstrings become more likely, samples it, then keeps the best valid sample.", "The QAOA plumbing can recover useful candidates on small instances."],
    ], [1.25 * inch, 3.05 * inch, 1.85 * inch]),
    P("Inside the current QAOA backend", "H2QAHS"),
    step(1, "Uniform start", "Every bitstring starts with equal amplitude. No answer is preferred yet."),
    step(2, "Cost phase", "The known scoring function changes the phase of each bitstring according to its cost."),
    step(3, "Mixer", "Rotation gates move amplitude among bitstrings."),
    step(4, "Classical tuning", "COBYLA changes the beta and gamma angles to reduce the circuit's expected cost. The current depth is normally p = 2."),
    step(5, "Measurement simulation", "The code samples bitstrings according to the final statevector probabilities."),
    step(6, "Valid result", "It rejects invalid encodings and chooses the lowest-cost sampled candidate."),
    P("Important distinction", "H2QAHS"),
    P("We know how to score a proposed answer; we do not know which proposal will score best until search is performed. QAOA can only optimize relationships represented in the cost. It cannot discover a rule that the hypotheses, predictions, and objective never expressed."),
    P("Current simulation shortcut", "H2QAHS"),
    P("The implementation creates the complete cost list and complete quantum state in ordinary NumPy memory. That is appropriate for debugging, but it grows as 2^qubits. It therefore demonstrates the algorithm's logic, not scalable quantum computing."),
    simple_table([
        ["Current object", "Meaning"],
        ["costs", "One classical number for every possible bitstring"],
        ["statevector", "One complex amplitude for every possible bitstring"],
        ["beta, gamma", "Circuit settings tuned anew for the current cost problem"],
        ["shots", "Repeated samples from the final probability distribution"],
    ], [1.65 * inch, 4.5 * inch]),
    PageBreak(),
]

story += [
    P("3. The larger 64-world experiment", "H1QAHS"),
    P("The second path replaces the four red-tile explanations with six binary rule dimensions. Choosing option 0 or 1 for every dimension creates 2^6 = 64 complete world models."),
    simple_table([
        ["Rule dimension", "Two possible values"],
        ["Player", "Blue is controlled / nearest object is controlled"],
        ["Goal", "Reach green / reach red"],
        ["Movement", "Move one cell / slide until an obstacle"],
        ["Topology", "Gray is a wall / gray contains a passable gap"],
        ["Interaction", "Red is a hazard / red is a switch"],
        ["State", "Stateless / sequence dependent"],
    ], [1.55 * inch, 4.6 * inch]),
    P("Encoding", "H2QAHS"),
    P("Six bits identify one of the 64 hypotheses. Seven additional one-hot bits represent six probe actions and one solution attempt. Exactly one action bit should be 1. The complete QAOA problem therefore uses 13 decision qubits and a simulated statevector with 2^13 = 8,192 amplitudes."),
    P("What QUBO means here", "H2QAHS"),
    P("A Quadratic Unconstrained Binary Optimization model is a score written using binary variables, individual-bit terms, and bit-pair terms. The code adds a large penalty if zero or several action bits are selected. It adds consistency penalties for rule values already observed and rewards informative actions."),
    P("How a run behaves", "H2QAHS"),
    P("Each probe directly reveals one of the six hidden bits. Because all six begin equally uncertain, the agent normally performs six probes and then attempts the complete solution on action seven. This is why the policy proxy and exact classical search both solve all 64 worlds in seven actions."),
    P("The main scientific weakness", "H2QAHS"),
    P("The current QUBO mostly chooses the best hypothesis bits and the best action independently. It contains penalties between action bits, but it does not yet contain rich terms connecting a particular hypothesis pattern to a particular action's outcome. The environment also reveals one rule bit per probe. This makes the experiment excellent for checking software and weak for showing that QAOA found a difficult reasoning path."),
    P("What a stronger next environment needs", "H2QAHS"),
    P("Actions should produce partial or ambiguous observations that support several hypotheses at once. Some actions should be locally attractive but poor over multiple steps. The hidden answer must remain available to the evaluator while being absent from the agent's inputs."),
    PageBreak(),
]

story += [
    P("4. How the codebase fits together", "H1QAHS"),
    simple_table([
        ["Layer", "Files", "Responsibility"],
        ["Teaching environment", "models.py, environment.py", "Defines the small hidden-rule game and shared records."],
        ["Candidate generation", "interpreter.py, providers/openai_provider.py", "Creates hypotheses deterministically or through an API-ready language-model call."],
        ["Prediction and scoring", "predictor.py, objective.py", "Predicts outcomes, calculates information gain and progress, and updates beliefs."],
        ["Small encoding", "compiler.py", "Encodes four hypotheses and two actions into three bits."],
        ["Search", "search/greedy.py, classical.py, qaoa.py", "Provides three interchangeable methods for selecting a candidate."],
        ["Loop and storage", "agent.py, run_store.py, cli.py", "Runs episodes, saves traces, and provides the command-line demo."],
        ["64-world representation", "factorized.py, rule_environment.py", "Defines six binary rule dimensions, seven actions, and the controlled hidden world."],
        ["64-world objective", "factorized_objective.py, qubo.py", "Scores probes, builds the 13-variable QUBO, and converts it to Ising form."],
        ["Experiments", "factorized_agent.py, benchmark.py, benchmark_cli.py", "Runs matched conditions and records aggregate metrics."],
        ["Future integration", "environment_adapter.py", "Defines the minimal reset/step interface for ARC or another environment."],
        ["Interfaces", "dashboard/server.py, dashboard/app.py", "Displays and saves runs from the small teaching example."],
    ], [1.2 * inch, 2.25 * inch, 2.7 * inch]),
    Spacer(1, 8),
    P("What is connected today", "H2QAHS"),
    P("The command line and dashboards run the small red-tile example. The benchmark command runs the 64-world experiment. The OpenAI caller and generic environment adapter are available building blocks, but neither is connected to the main experimental path. The current quantum backend is a custom NumPy statevector simulator; Qiskit is not yet a dependency."),
    P("What was verified during this review", "H2QAHS"),
    P("All nine unit tests passed. The classical and QAOA small demos both solved the switch world in two turns: approach red, observe the door open, then approach green. Across all four hidden rules, all three backends succeeded on the switch and goal worlds but failed on the hazard and does-nothing worlds. The information-seeking action is fatal in the hazard world; after learning that red does nothing, the loop can repeat zero-value actions. These are objective and stopping-rule limitations shared by every backend."),
    P("The preserved 64-world results show both classical conditions at 64/64 with seven actions on average. The preserved eight-world QAOA smoke run solved 8/8 with 6.875 actions on average, but one early correct guess and post-development tuning prevent an advantage claim."),
    PageBreak(),
]

story += [
    P("5. Checklist A - test what already exists", "H1QAHS"),
    P("Run these from the QAHS-Prototype folder."),
    P("Software correctness", "H2QAHS"),
    check("Run: <font name='Courier'>PYTHONPATH=src python3 -m unittest discover -s tests -v</font>"),
    check("Confirm all nine tests pass."),
    check("Run the classical and QAOA small demos with the hidden rule <font name='Courier'>red_is_switch</font>."),
    check("Confirm both choose red first, observe an open door, and then choose green."),
    check("Repeat the small demo for all four hidden rules and record success, turns, and actions."),
    check("Reproduce the current limitation: every backend fails the hazard and does-nothing worlds. This is a shared objective/stopping issue, not a QAOA-specific failure."),
    P("Visual inspection", "H2QAHS"),
    check("Start: <font name='Courier'>PYTHONPATH=src python3 dashboard/server.py</font>"),
    check("Open <font name='Courier'>http://127.0.0.1:8000</font>."),
    check("Run the same hidden rule with Greedy, Classical exact, and QAOA."),
    check("Confirm action scores are identical before search; only the search method should change."),
    check("Confirm the belief update agrees with the returned observation."),
    check("Confirm a JSON trace appears under <font name='Courier'>output/runs/</font>."),
    P("Controlled benchmark", "H2QAHS"),
    check("Run the policy proxy and exact classical search across all 64 worlds."),
    check("Confirm 64/64 success and about seven actions per world."),
    check("Run QAOA on eight worlds first; record the seed, depth, restarts, optimizer limit, shots, time, and actions."),
    check("Treat this as a smoke test. Eight worlds cannot establish an advantage."),
    P("Recommended commands", "H2QAHS"),
    simple_table([
        ["Purpose", "Command"],
        ["64 classical worlds", "PYTHONPATH=src python3 -m qahs.benchmark_cli --conditions policy_proxy classical --worlds 64 --output output/experiments/reproduction_classical_64.json"],
        ["8 QAOA worlds", "PYTHONPATH=src python3 -m qahs.benchmark_cli --conditions qaoa --worlds 8 --output output/experiments/reproduction_qaoa_8.json"],
    ], [1.35 * inch, 4.8 * inch]),
    PageBreak(),
]

story += [
    P("6. Checklist B - turn it into a fair experiment", "H1QAHS"),
    P("Freeze the comparison", "H2QAHS"),
    check("State the primary claim before final testing. Example: QAOA-assisted search reduces actions to solution under an equal search budget."),
    check("Give every condition the same state, history, hypotheses, actions, scoring rule, and stopping rule."),
    check("Compare a real LLM-only action choice, greedy classical, exact classical, and QAOA-assisted search."),
    check("Freeze QAOA depth, restarts, optimizer limit, shots, penalties, and seeds."),
    check("Separate development tasks from held-out test tasks."),
    check("Use solve rate and actions to solution as primary metrics."),
    check("Also report time, model calls, shots, optimizer evaluations, qubits, and circuit depth."),
    check("Repeat stochastic conditions across multiple seeds and report uncertainty."),
    P("Connect the language model", "H2QAHS"),
    check("Install the optional LLM dependency and set the API key only in the environment."),
    check("Wire OpenAIJSONCaller into a runner; it is not currently activated by the command line or dashboard."),
    check("Freeze the prompt and output schema before held-out tests."),
    check("Save every proposed hypothesis, candidate action, model response, and parse failure."),
    check("Add an LLM-only baseline that chooses the next action without QAHS."),
    P("Use a meaningful environment", "H2QAHS"),
    check("Implement an ARC or other task through the reset/step adapter."),
    check("Keep hidden truth in the evaluator and out of the agent prompt and scoring inputs."),
    check("Remove direct probes that reveal exactly one hidden bit."),
    check("Use ambiguous outcomes that support several competing explanations."),
    check("Include cases where the best immediate action is not the best multi-step plan."),
    check("Test enough unseen tasks to distinguish conditions."),
    PageBreak(),
]

story += [
    P("7. Checklist C - validate the quantum part", "H1QAHS"),
    P("Formulation checks", "H2QAHS"),
    check("Confirm every valid bitstring decodes to exactly one hypothesis and one action."),
    check("Confirm invalid encodings cost more than any possible reward."),
    check("Check QUBO and Ising energies on many random bitstrings, not only the four unit-test examples."),
    check("Add meaningful hypothesis-action interaction terms when the environment supports them."),
    check("On small problems, compare the exact minimum with the best candidate QAOA sampled."),
    check("Record the probability placed on optimal and near-optimal candidates."),
    check("Increase qubits gradually and record simulator memory, optimizer work, and failure rate."),
    P("Qiskit checks", "H2QAHS"),
    check("Build the Ising cost operator in Qiskit."),
    check("Run Qiskit QAOA on a noiseless simulator and compare it with the current NumPy result."),
    check("Transpile and report logical qubits, circuit depth, one-qubit gates, and two-qubit gates."),
    check("Run a noise model and report how often the best answer is recovered."),
    check("If hardware is available, use a small frozen instance and record calibration, shots, and errors."),
    P("Interpretation checks", "H2QAHS"),
    check("Describe working demos as software validation."),
    check("Claim better search only when held-out matched comparisons improve the primary metrics."),
    check("Report where QAOA is slower, less stable, or no better."),
    check("Keep 'quantum-assisted reasoning' separate from 'quantum speedup.'"),
    P("Current readiness", "H2QAHS"),
    simple_table([
        ["Question", "Current answer"],
        ["Does the full observe-search-act-update loop run?", "Yes, on synthetic teaching environments."],
        ["Can classical and QAOA use the same encoded objective?", "Yes."],
        ["Does the current code use a real LLM?", "No. An adapter exists but is not connected."],
        ["Does it run genuine ARC tasks?", "No. The adapter is only a contract."],
        ["Does it use Qiskit or quantum hardware?", "No. It uses a NumPy statevector simulator."],
        ["Does it establish stronger AI or quantum advantage?", "No. That requires the held-out matched study above."],
    ], [3.0 * inch, 3.15 * inch]),
    Spacer(1, 8),
    P("The next useful milestone", "H2QAHS"),
    P("Connect one real model, one non-trivial interactive task, and all four matched conditions. Freeze the prompt and QAOA settings, then run a small development study. That result will tell you whether to improve the objective, the environment, or the quantum search before scaling the experiment."),
]

doc.build(story)
print(OUT)


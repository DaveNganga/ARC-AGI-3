import Link from "next/link";

export default function Home() {
  return (
    <main className="about-page">
      <header className="about-hero">
        <p className="eyebrow">Quantum-Assisted Hypothesis Search</p>
        <h1>Can quantum search help an AI reason more effectively?</h1>
        <p className="hero-copy">This prototype tests whether a quantum-assisted decision step can help an AI choose better experiments while learning the rules of an unfamiliar problem.</p>
        <Link className="primary-link" href="/results">Explore the experiments <span aria-hidden="true">→</span></Link>
      </header>

      <section className="about-section intro-grid">
        <div><p className="section-number">01</p><h2>The question</h2></div>
        <div>
          <p className="lead">AI often fails because it follows a weak explanation—not because it lacks possible actions.</p>
          <p>We ask whether representing competing explanations and actions as an optimization problem can improve which test the AI chooses next. ARC-AGI-3 supplies unfamiliar interactive puzzles where the rules must be discovered through observation.</p>
        </div>
      </section>

      <section className="about-section">
        <p className="section-number">02 · The learning loop</p>
        <h2>One shared reasoner. Three ways to choose.</h2>
        <div className="loop-flow" aria-label="Observe, hypothesize, choose, act, reflect, and repeat">
          {[
            ["Observe", "Read the current puzzle state"],
            ["Hypothesize", "Form competing explanations"],
            ["Choose", "Select an informative action"],
            ["Act", "Test it in ARC-AGI-3"],
            ["Reflect", "Update the belief ledger"],
          ].map(([title, copy], index) => <div className="loop-step" key={title}><span>{String(index + 1).padStart(2, "0")}</span><strong>{title}</strong><small>{copy}</small></div>)}
        </div>
        <p className="loop-return">The updated beliefs become evidence for the next turn.</p>
      </section>

      <section className="about-section">
        <p className="section-number">03 · The comparison</p>
        <h2>Only the selection method changes.</h2>
        <div className="mode-grid">
          <article><span className="mode-dot llm-dot" /><h3>LLM only</h3><p>The language model recommends an action from its own hypotheses.</p></article>
          <article><span className="mode-dot classical-dot" /><h3>Classical QAHS</h3><p>An exact classical search scores every valid hypothesis–action pairing.</p></article>
          <article><span className="mode-dot quantum-dot" /><h3>Quantum QAHS</h3><p>A simulated quantum optimization circuit searches the same scored pairings.</p></article>
        </div>
        <p className="fairness-note">All modes use the same model, puzzle, seed, action budget, reflection step, and persistent beliefs.</p>
      </section>

      <section className="about-section architecture-section">
        <p className="section-number">04 · Code flow</p>
        <h2>From browser to evidence</h2>
        <p className="architecture-intro">The public site explains the project and displays sanitized results. Experiments run locally so API keys and the Python research environment remain private.</p>
        <div className="architecture" role="img" aria-label="Architecture showing the Next.js interface, local experiment runner, reasoning loop, three selectors, ARC environment, saved results, and public dashboard">
          <div className="arch-node arch-interface"><small>Next.js</small><strong>Interface</strong><span>About · Results · Local run panel</span></div>
          <div className="arch-arrow">↓</div>
          <div className="arch-node arch-runner"><small>Python</small><strong>Experiment runner</strong><span>Same game, seed and action budget</span></div>
          <div className="arch-arrow">↓</div>
          <div className="arch-node arch-reasoner"><small>OpenAI + belief ledger</small><strong>Reasoning loop</strong><span>Propose → test → reflect → persist</span></div>
          <div className="arch-split">
            <div className="arch-node"><strong>LLM choice</strong><span>Baseline</span></div>
            <div className="arch-node"><strong>Exact search</strong><span>Classical QAHS</span></div>
            <div className="arch-node"><strong>QAOA search</strong><span>Quantum QAHS</span></div>
          </div>
          <div className="arch-merge">↓</div>
          <div className="arch-node arch-environment"><small>ARC-AGI-3 API</small><strong>Interactive environment</strong><span>Action changes the state; evidence returns</span></div>
          <div className="arch-arrow two-way">↕</div>
          <div className="arch-node arch-data"><small>Sanitized JSON</small><strong>Results dashboard</strong><span>Progress · efficiency · exploration · compute</span></div>
        </div>
      </section>

      <section className="about-section closing-callout">
        <p className="section-number">05 · Research hypothesis</p>
        <blockquote>Quantum-assisted hypothesis selection can improve an AI system’s ability to choose useful tests and solve unfamiliar problems under a fixed action budget.</blockquote>
        <Link className="text-link" href="/results">View the comparison dashboard →</Link>
      </section>
    </main>
  );
}

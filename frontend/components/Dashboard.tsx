"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

type Frame = number[][];
type Candidate = { candidate_id: string; action_type: string; x?: number | null; y?: number | null; description?: string };
type Observation = { frame?: Frame; levels_completed?: number; event?: string };
type Reflection = {
  summary?: string;
  next_focus?: string;
  beliefs?: { id: number; statement: string; confidence: number; status: string; evidence: string }[];
};
type Turn = {
  turn: number;
  observation_before: Observation;
  observation_after: Observation;
  selected_candidate: Candidate;
  proposal: { recommended_candidate_id: string; summary?: string };
  search_diagnostics?: { quantum_used?: boolean; shots?: number };
  reflection?: Reflection | null;
  model_calls?: number;
  token_usage?: { input_tokens?: number; output_tokens?: number };
};
type Episode = { condition: string; success?: boolean; turns?: Turn[]; wall_seconds?: number; error?: string | null; stop_reason?: string };
type Run = {
  filename: string;
  source?: string;
  game_id?: string;
  seed?: number;
  action_budget?: number;
  model?: string;
  harness_version?: string;
  without_qahs: Episode;
  with_classical?: Episode;
  with_qahs: Episode;
};
type ManifestItem = { filename: string; label: string };

const ARC_COLORS = [
  "#ffffff", "#f0f0f0", "#c8c8c8", "#969696", "#4b4b4b", "#000000",
  "#e53aa3", "#ff7fcc", "#e53b2c", "#2e5cb8", "#62a5ff", "#f5d442",
  "#f28c28", "#7f1d1d", "#35a853", "#7e57c2",
];
const RUNNER = "http://127.0.0.1:8765";
const MODE_COLORS: Record<string, string> = { LLM: "#55b6d9", Classical: "#efad4a", Quantum: "#36b3a7" };

function episodes(run: Run): [string, Episode][] {
  const values: [string, Episode][] = [["LLM", run.without_qahs]];
  if (run.with_classical) values.push(["Classical", run.with_classical]);
  values.push(["Quantum", run.with_qahs]);
  return values;
}

function turns(episode: Episode) { return episode.turns ?? []; }
function levels(episode: Episode) { const t = turns(episode); return Number(t.at(-1)?.observation_after.levels_completed ?? 0); }
function uniqueActions(episode: Episode) {
  return new Set(turns(episode).map(({ selected_candidate: c }) => `${c.action_type}:${c.x ?? ""}:${c.y ?? ""}`)).size;
}
function changed(turn: Turn) {
  const before = turn.observation_before.frame ?? [];
  const after = turn.observation_after.frame ?? [];
  let count = 0;
  before.forEach((row, y) => row.forEach((cell, x) => { if (after[y]?.[x] !== cell) count += 1; }));
  return count;
}
function usefulActions(episode: Episode) { return turns(episode).filter((turn) => changed(turn) > 0).length; }
function overrides(episode: Episode) {
  return turns(episode).filter((turn) => turn.selected_candidate.candidate_id !== turn.proposal.recommended_candidate_id).length;
}
function tokens(episode: Episode) {
  const total = turns(episode).reduce((sum, turn) => sum + Number(turn.token_usage?.input_tokens ?? 0) + Number(turn.token_usage?.output_tokens ?? 0), 0);
  return total ? total.toLocaleString() : "—";
}
function modelCalls(episode: Episode) { return turns(episode).reduce((sum, turn) => sum + Number(turn.model_calls ?? 1), 0); }

const METRICS = [
  { name: "Progress", about: "Number of game levels completed.", value: levels },
  { name: "Efficiency", about: "Actions used; fewer is better when progress is equal.", value: (episode: Episode) => turns(episode).length },
  { name: "Exploration", about: "Distinct action-and-location combinations attempted.", value: uniqueActions },
  { name: "Useful actions", about: "Actions that visibly changed at least one grid cell.", value: usefulActions },
];

function SetupMap({ run }: { run: Run }) {
  const modes = episodes(run).length;
  const budget = run.action_budget ?? Math.max(...episodes(run).map(([, episode]) => turns(episode).length));
  return (
    <section className="setup-map" aria-label="Run setup">
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <marker id="setup-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" />
          </marker>
        </defs>
        <line x1="46" y1="47" x2="22" y2="29" /><line x1="50" y1="42" x2="50" y2="19" />
        <line x1="54" y1="47" x2="78" y2="29" /><line x1="46" y1="54" x2="31" y2="73" />
        <line x1="54" y1="54" x2="69" y2="73" />
      </svg>
      <div className="setup-node setup-center">SETUP</div>
      <div className="setup-node setup-game"><small>Game</small><strong>{run.game_id ?? "Unknown"}</strong></div>
      <div className="setup-node setup-seed"><small>Seed</small><strong>{run.seed ?? "Unknown"}</strong></div>
      <div className="setup-node setup-actions"><small>Actions</small><strong>{budget}</strong></div>
      <div className="setup-node setup-modes"><small>Modes</small><strong>{modes}</strong></div>
      <div className="setup-node setup-model"><small>Model</small><strong>{run.model || "gpt-6-luna"}</strong></div>
    </section>
  );
}

function ResultsChart({ run }: { run: Run }) {
  const conditions = episodes(run);
  const max = Math.max(1, ...conditions.flatMap(([, episode]) => METRICS.map((metric) => metric.value(episode))));
  return (
    <section className="card results-card">
      <h2>Results</h2>
      <div className="chart" role="img" aria-label="Grouped experiment result bars">
        {METRICS.map((metric) => (
          <div className="metric-group" key={metric.name}>
            <div className="bars">
              {conditions.map(([label, episode]) => {
                const value = metric.value(episode);
                return (
                  <div className="bar-slot" key={label} title={`${label}: ${value}. ${metric.about}`}>
                    <span className="bar-value">{value}</span>
                    <div className="bar" style={{ height: `${Math.max(value ? 8 : 2, (value / max) * 180)}px`, background: MODE_COLORS[label] }} />
                  </div>
                );
              })}
            </div>
            <strong>{metric.name}</strong>
          </div>
        ))}
      </div>
      <div className="legend">{conditions.map(([label]) => <span key={label}><i style={{ background: MODE_COLORS[label] }} />{label}</span>)}</div>
      <div className="summary-table" role="table" aria-label="Computation summary">
        <div className="summary-row summary-head"><span>Mode</span><span>LLM calls</span><span>Tokens</span><span>Searches</span><span>Overrides</span><span>Time</span></div>
        {conditions.map(([label, episode]) => (
          <div className="summary-row" role="row" key={label}>
            <strong>{label}</strong><span>{modelCalls(episode)}</span><span>{tokens(episode)}</span>
            <span>{episode.condition === "llm_only" ? 0 : turns(episode).length}</span>
            <span>{overrides(episode)}</span><span>{Number(episode.wall_seconds ?? 0).toFixed(1)}s</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function Grid({ frame }: { frame?: Frame }) {
  if (!frame?.length) return <div className="empty-grid">No frame</div>;
  const columns = frame[0]?.length || 1;
  return <div className="arc-grid" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>{frame.flat().map((cell, i) => <i key={i} style={{ background: ARC_COLORS[cell] ?? "#000" }} />)}</div>;
}

function EpisodeDetails({ label, episode }: { label: string; episode: Episode }) {
  return (
    <div className="episode">
      <div className="episode-stats"><span>{episode.success ? "Solved" : "Unsolved"}</span><span>{levels(episode)} levels</span><span>{turns(episode).length} actions</span>{episode.stop_reason && <span>Stopped: {episode.stop_reason.replace("_", " ")}</span>}</div>
      {episode.error && <p className="error">Run failed; internal details omitted.</p>}
      {turns(episode).map((turn) => (
        <details key={turn.turn}>
          <summary>#{turn.turn} · {turn.selected_candidate.action_type}</summary>
          <p>{turn.proposal.summary}</p>
          {turn.reflection && <p className="reflection-note"><strong>Reflection:</strong> {turn.reflection.summary} <span>Next: {turn.reflection.next_focus}</span></p>}
          <div className="frames"><div><small>Before</small><Grid frame={turn.observation_before.frame} /></div><div><small>After · {turn.observation_after.event}</small><Grid frame={turn.observation_after.frame} /></div></div>
          <p className="turn-note">Pick: {turn.selected_candidate.description || turn.selected_candidate.candidate_id} · LLM: {turn.proposal.recommended_candidate_id} · Changed cells: {changed(turn)}</p>
        </details>
      ))}
      {!turns(episode).length && <p>No turns recorded for {label}.</p>}
    </div>
  );
}

function Details({ run }: { run: Run }) {
  const conditions = episodes(run);
  const [active, setActive] = useState(conditions[0][0]);
  useEffect(() => setActive(conditions[0][0]), [run.filename]);
  const selected = conditions.find(([label]) => label === active) ?? conditions[0];
  return (
    <section className="card details-card">
      <h2>Details</h2>
      <div className="tabs">{conditions.map(([label]) => <button className={active === label ? "active" : ""} onClick={() => setActive(label)} key={label}>{label}</button>)}</div>
      <EpisodeDetails label={selected[0]} episode={selected[1]} />
    </section>
  );
}

function LocalRunPanel({ onComplete }: { onComplete: (filename: string) => Promise<void> }) {
  const [source, setSource] = useState("demo");
  const [game, setGame] = useState("ls20");
  const [seed, setSeed] = useState(0);
  const [actions, setActions] = useState(8);
  const [frameFormat, setFrameFormat] = useState("runs");
  const [arcMode, setArcMode] = useState("offline");
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState("");
  const [unfinished, setUnfinished] = useState<string[]>([]);

  const loadSessions = useCallback(() => {
    fetch(`${RUNNER}/sessions`).then((response) => response.json())
      .then((result: { unfinished?: string[] }) => setUnfinished(result.unfinished ?? []))
      .catch(() => setUnfinished([]));
  }, []);
  useEffect(loadSessions, [loadSessions]);

  async function call(path: string, body: object, pending: string) {
    setRunning(true);
    setMessage(pending);
    try {
      const response = await fetch(`${RUNNER}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "The experiment failed.");
      await onComplete(result.filename);
      setMessage("Run saved.");
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "The experiment failed.");
    } finally {
      setRunning(false);
      loadSessions();
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void call("/run", { source, game, seed, max_steps: actions, frame_format: frameFormat, arc_mode: arcMode }, "Running all three modes…");
  }

  return (
    <aside className="local-panel">
      <div className="local-panel-inner">
        <h2>Run</h2>
        <form onSubmit={submit}>
          <label>Source<select value={source} onChange={(event) => setSource(event.target.value)} disabled={running}><option value="demo">Demo</option><option value="arc">ARC</option></select></label>
          <label>Game<input value={game} onChange={(event) => setGame(event.target.value)} disabled={running || source === "demo"} maxLength={40} /></label>
          <label>Mode<select value={arcMode} onChange={(event) => setArcMode(event.target.value)} disabled={running || source === "demo"}><option value="offline">Offline</option><option value="online">Online</option></select></label>
          <label>Grid<select value={frameFormat} onChange={(event) => setFrameFormat(event.target.value)} disabled={running || source === "demo"}><option value="runs">Runs</option><option value="objects">Objects</option><option value="hex">Hex</option></select></label>
          <label>Seed<input type="number" min="0" max="1000000" value={seed} onChange={(event) => setSeed(Number(event.target.value))} disabled={running} /></label>
          <label>Actions<input type="range" min="2" max="30" value={actions} onChange={(event) => setActions(Number(event.target.value))} disabled={running} /><output>{actions}</output></label>
          <button type="submit" disabled={running}>{running ? "Running…" : "Start"}</button>
        </form>
        {unfinished.length > 0 && <button type="button" className="resume-button" disabled={running} onClick={() => void call("/resume", { run_id: unfinished[0] }, "Resuming interrupted run…")}>Resume {unfinished[0]}</button>}
        {message && <p className="run-message" aria-live="polite">{message}</p>}
        <small>Local only</small>
      </div>
    </aside>
  );
}

export default function Dashboard() {
  const [manifest, setManifest] = useState<ManifestItem[]>([]);
  const [filename, setFilename] = useState("");
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState("");
  const [isLocal, setIsLocal] = useState(false);

  const loadManifest = useCallback(async (preferred?: string) => {
    setError("");
    return fetch(`/runs/index.json?v=${Date.now()}`, { cache: "no-store" }).then((response) => {
      if (!response.ok) throw new Error("No published runs found.");
      return response.json();
    }).then((items: ManifestItem[]) => {
      setManifest(items);
      setFilename(preferred && items.some((item) => item.filename === preferred) ? preferred : items[0]?.filename ?? "");
    }).catch((reason) => setError(reason.message));
  }, []);
  useEffect(() => {
    const host = window.location.hostname;
    setIsLocal(
      ["localhost", "127.0.0.1"].includes(host) ||
      /^192\.168\.\d{1,3}\.\d{1,3}$/.test(host) ||
      /^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(host),
    );
    void loadManifest();
  }, [loadManifest]);
  useEffect(() => {
    if (!filename) return;
    setRun(null);
    fetch(`/runs/${filename}?v=${Date.now()}`, { cache: "no-store" }).then((response) => response.json()).then((data: Run) => setRun({ ...data, filename })).catch(() => setError("The selected run could not be loaded."));
  }, [filename]);
  const selectedLabel = useMemo(() => manifest.find((item) => item.filename === filename)?.label ?? filename, [manifest, filename]);

  return (
    <div className={isLocal ? "app-shell has-local-panel" : "app-shell"}>
      {isLocal && <LocalRunPanel onComplete={loadManifest} />}
      <main>
        <header className="banner"><h1>ARC-AGI-3: comparing LLM, classical QAHS, and quantum QAHS</h1></header>
        {manifest.length > 0 && <label className="run-select"><span className="sr-only">Saved experiment</span><select value={filename} onChange={(event) => setFilename(event.target.value)} aria-label="Saved experiment">{manifest.map((item) => <option value={item.filename} key={item.filename}>{item.label}</option>)}</select></label>}
        {error && <p className="error">{error}</p>}
        {!run && !error && <p className="loading">Loading experiment…</p>}
        {run && !run.harness_version && !run.with_classical && <p className="legacy-note">Legacy harness run: recorded before the current reasoning loop. Do not pool it with newer runs.</p>}
        {run && <><SetupMap run={run} /><ResultsChart run={run} /><Details run={run} /></>}
        <footer>{selectedLabel}</footer>
      </main>
    </div>
  );
}

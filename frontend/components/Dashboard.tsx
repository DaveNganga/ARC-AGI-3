"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

const LOCAL_RUNNER_URL = process.env.NEXT_PUBLIC_QAHS_RUNNER_URL ?? "http://127.0.0.1:8765";

type Frame = number[][];
type Candidate = { candidate_id: string; action_type: string; x?: number | null; y?: number | null; next_action_type?: string | null; next_x?: number | null; next_y?: number | null; description?: string };
type Atom = { key?: string; kind: string; subject: string; value: string; target?: string | null; source?: string };
type EntityMarker = { id: string; center: [number, number]; bounds?: [number, number, number, number]; color?: number; size?: number };
type Hypothesis = { id?: number; statement?: string; confidence?: number; predictions?: Record<string, string> };
type WorkingHypothesis = { statement?: string; focus_entity_ids?: string[]; target_candidate_id?: string; expected_effect?: string; success_signal?: string; falsification?: string };
type PredictionRow = { candidate_id: string; outcome: string; confidence?: number };
type PredictionMatrixRow = { hypothesis_id: number; statement: string; focus_entity_ids?: string[]; probability: number; predictions: PredictionRow[]; test_sequence?: string[]; sequence_target_state?: string; stop_condition?: string; persistence_candidate_id?: string; tracked_entity_id?: string; target_entity_id?: string; actuator_entity_id?: string; affected_entity_ids?: string[]; expected_relation_change?: string; confirmation_outcomes?: string[]; refutation_outcomes?: string[]; goal_component_id?: string; goal_lineage_action?: string };
type Observation = { frame?: Frame; levels_completed?: number; win_levels?: number; state?: string; event?: string };
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
  proposal: { recommended_candidate_id: string; recommendation_reason?: string; summary?: string; hypotheses?: Hypothesis[]; action_candidates?: Candidate[] };
  search_diagnostics?: { quantum_used?: boolean; shots?: number; regret?: number; selected_rank?: number; exact_optimum_probability?: number; selector_candidate_id?: string; selector_advantage?: number; override_category?: string; override_permitted?: boolean; composed_statement?: string; selected_atoms?: Atom[]; atom_catalog?: Atom[]; entity_catalog?: EntityMarker[]; after_entity_catalog?: EntityMarker[]; luna_working_hypothesis?: WorkingHypothesis | null; trajectory_status?: string; tracked_entity_id?: string | null; target_entity_id?: string | null; distance_delta?: number | null; frontier_position?: boolean; color_changed_cells?: number; prediction_matrix?: PredictionMatrixRow[]; hypothesis_probabilities_before?: Record<string, number>; hypothesis_probabilities_after?: Record<string, number>; selected_plan?: string[]; observed_outcome?: string; physical_outcome?: string; retired_hypotheses?: string[]; active_hypothesis_id?: number | null; forced_reconsideration?: boolean; procedure_target_state?: string; procedure_continued?: boolean; procedure_reframed?: boolean; procedure_evidence_status?: string; validated_for_level?: { hypothesis_id?: number; statement?: string; validated_for_level?: number; candidate_id?: string; predicted_outcome?: string; physical_outcome?: string; game_outcome?: string } | null; success_hypothesis_audit?: { verdict?: string; corrected_hypothesis?: string; transferable_relation?: string; interface_change?: string; next_level_test?: string } };
  reflection?: Reflection | null;
  model_calls?: number;
  timings?: Record<string, number>;
};
type Episode = { condition: string; success?: boolean; turns?: Turn[]; wall_seconds?: number; comparable_seconds?: number; error?: string | null };
type Run = {
  filename: string;
  benchmark?: string;
  source?: string;
  game_id?: string;
  seed?: number;
  action_budget?: number;
  model?: string;
  reasoning_effort?: string;
  harness_version?: string;
  input_representation?: string;
  batch_id?: string | null;
  model_settings?: { comparison_integrity?: { comparison_valid?: boolean; initial_states_match?: boolean } };
  without_qahs: Episode;
  with_classical?: Episode;
  with_qahs: Episode;
};
type ManifestItem = {
  filename: string;
  label: string;
  benchmark?: string;
  game_id?: string;
  seed?: number;
  action_budget?: number;
  model?: string;
  reasoning_effort?: string;
  harness_version?: string;
  input_representation?: string;
  batch_id?: string | null;
};

const ARC_COLORS = [
  "#ffffff", "#f0f0f0", "#c8c8c8", "#969696", "#4b4b4b", "#000000",
  "#e53aa3", "#ff7fcc", "#e53b2c", "#2e5cb8", "#62a5ff", "#f5d442",
  "#f28c28", "#7f1d1d", "#35a853", "#7e57c2",
];
const MODE_COLORS: Record<string, string> = { LLM: "#55b6d9", Classical: "#efad4a", Quantum: "#36b3a7" };
const ARC2_LUNA_MEDIUM = ["136b0064", "1818057f", "291dc1e1", "2ba387bc", "31f7f899", "3dc255db", "45a5af55", "58f5dbd5", "8f3a5a89", "981571dc", "b5ca7ac4", "b99e7126", "bf45cf4b", "c4d067a0", "d59b0160", "db0c5428", "e8686506"];

function episodes(run: Run): [string, Episode][] {
  const values: [string, Episode][] = [["LLM", run.without_qahs]];
  if (run.with_classical) values.push(["Classical", run.with_classical]);
  values.push(["Quantum", run.with_qahs]);
  return values;
}

function turns(episode: Episode) { return episode.turns ?? []; }
function levels(episode: Episode) { const t = turns(episode); return Number(t.at(-1)?.observation_after.levels_completed ?? 0); }
function actionKey(candidate?: Candidate) {
  if (!candidate) return "missing";
  return [candidate.action_type, candidate.x ?? "", candidate.y ?? "", candidate.next_action_type ?? "", candidate.next_x ?? "", candidate.next_y ?? ""].join(":");
}
function uniqueActions(episode: Episode) { return new Set(turns(episode).map((turn) => actionKey(turn.selected_candidate))).size; }
function explorationCoverage(episode: Episode) {
  const attempted = new Set(turns(episode).map((turn) => actionKey(turn.selected_candidate)));
  const offered = new Set(turns(episode).flatMap((turn) => turn.proposal.action_candidates ?? []).map(actionKey));
  attempted.delete("missing");
  offered.delete("missing");
  // Older traces sometimes omit the offered catalog. Their attempted set is
  // still an auditable lower bound on the action space seen in the trace.
  const denominator = Math.max(attempted.size, offered.size, 1);
  return 100 * attempted.size / denominator;
}
function changed(turn: Turn) {
  const before = turn.observation_before.frame ?? [];
  const after = turn.observation_after.frame ?? [];
  let count = 0;
  before.forEach((row, y) => row.forEach((cell, x) => { if (after[y]?.[x] !== cell) count += 1; }));
  return count;
}
function posteriorShift(turn: Turn) {
  const before = turn.search_diagnostics?.hypothesis_probabilities_before;
  const after = turn.search_diagnostics?.hypothesis_probabilities_after;
  if (!before || !after) return 0;
  const ids = new Set([...Object.keys(before), ...Object.keys(after)]);
  let distance = 0;
  ids.forEach((id) => { distance += Math.abs(Number(before[id] ?? 0) - Number(after[id] ?? 0)); });
  return distance / 2;
}
function evidenceYielding(turn: Turn) {
  const diagnostics = turn.search_diagnostics;
  const levelsChanged = Number(turn.observation_after.levels_completed ?? 0) !== Number(turn.observation_before.levels_completed ?? 0);
  const terminalSignal = ["WIN", "GAME_OVER", "FAILED"].includes(String(turn.observation_after.state ?? "").toUpperCase());
  const physical = String(diagnostics?.physical_outcome ?? diagnostics?.observed_outcome ?? "").toUpperCase();
  const informativePhysicalChange = Boolean(physical && !["NO_CHANGE", "UNKNOWN", "NONE"].includes(physical));
  const procedureEvidence = String(diagnostics?.procedure_evidence_status ?? "").toLowerCase();
  const explicitBeliefUpdate = ["refuted", "contradicted", "supported", "confirmed", "validated", "validated_for_level", "weakened"].includes(procedureEvidence)
    || Boolean(diagnostics?.validated_for_level)
    || Boolean(diagnostics?.retired_hypotheses?.length)
    || posteriorShift(turn) >= .05;
  // Frame change is only a fallback for old traces that predate classified
  // physical outcomes; it is not allowed to override an explicit NO_CHANGE.
  const legacyFrameEvidence = !physical && changed(turn) > 0;
  return levelsChanged || terminalSignal || informativePhysicalChange || explicitBeliefUpdate || legacyFrameEvidence;
}
function usefulActions(episode: Episode) { return turns(episode).filter(evidenceYielding).length; }
function usefulActionRate(episode: Episode) { return turns(episode).length ? 100 * usefulActions(episode) / turns(episode).length : 0; }
function actionsPerLevel(episode: Episode) { return levels(episode) > 0 ? turns(episode).length / levels(episode) : null; }
function overrides(episode: Episode) {
  return turns(episode).filter((turn) => {
    // Older V11/V12 traces marked every persisted step as another override.
    // A continued procedure is execution of the original decision, not a new
    // disagreement with Luna, so repair that interpretation at presentation.
    if (turn.search_diagnostics?.procedure_continued) return false;
    return typeof turn.search_diagnostics?.override_permitted === "boolean"
      ? turn.search_diagnostics.override_permitted
      : turn.selected_candidate.candidate_id !== turn.proposal.recommended_candidate_id;
  }).length;
}
function comparableTime(episode: Episode) {
  const normalized = Number(episode.comparable_seconds ?? 0);
  return normalized > 0 ? normalized : Number(episode.wall_seconds ?? 0);
}
function modelCalls(episode: Episode) { return turns(episode).reduce((sum, turn) => sum + Number(turn.model_calls ?? 1), 0); }
function meanRegret(episode: Episode) {
  const values = turns(episode).map((turn) => turn.search_diagnostics?.regret).filter((value): value is number => typeof value === "number");
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}
function supportsCombined(run: Run) { return episodes(run).every(([, episode]) => turns(episode).some((turn) => typeof turn.search_diagnostics?.regret === "number")); }

type ScoreComponents = { progress: number; actions: number; search: number; calls: number; combined: number };
function combinedScores(run: Run): Map<Episode, ScoreComponents> {
  const conditions = episodes(run).map(([, episode]) => episode);
  const maxLevels = Math.max(0, ...conditions.map(levels));
  const progressRates = conditions.map((episode) => levels(episode) / Math.max(1, turns(episode).length));
  const callRates = conditions.map((episode) => levels(episode) / Math.max(1, modelCalls(episode)));
  const regrets = conditions.map(meanRegret);
  const maxProgressRate = Math.max(0, ...progressRates);
  const maxCallRate = Math.max(0, ...callRates);
  const maxRegret = Math.max(0, ...regrets);
  const result = new Map<Episode, ScoreComponents>();
  conditions.forEach((episode, index) => {
    const progress = maxLevels ? levels(episode) / maxLevels : 0;
    const actions = maxProgressRate ? progressRates[index] / maxProgressRate : 0;
    const search = maxRegret ? Math.max(0, 1 - regrets[index] / maxRegret) : 1;
    const calls = maxCallRate ? callRates[index] / maxCallRate : 0;
    result.set(episode, { progress, actions, search, calls, combined: 100 * (.5 * progress + .2 * actions + .2 * search + .1 * calls) });
  });
  return result;
}

type DisplayMetric = { name: string; about: string; value: (episode: Episode) => number | null; suffix?: string; decimals?: number; lowerIsBetter?: boolean };
const METRICS: DisplayMetric[] = [
  { name: "Progress", about: "Game-reported levels completed; visible movement alone does not count.", value: levels, decimals: 0 },
  { name: "Efficiency", about: "Executed actions per completed level; lower is better and an unsolved run is not assigned a score.", value: actionsPerLevel, decimals: 1, lowerIsBetter: true },
  { name: "Exploration", about: "Share of distinct physical actions attempted from the actions offered during this trace; breadth is descriptive, not automatically better.", value: explorationCoverage, suffix: "%", decimals: 0 },
  { name: "Useful actions", about: "Share of actions yielding objective evidence: progress, terminal feedback, a classified state change, or a material belief update.", value: usefulActionRate, suffix: "%", decimals: 0 },
];

const BATCH_METRICS = [
  { name: "Combined", about: "Exploratory v4 score: 50% progress, 20% action efficiency, 20% search quality, 10% call efficiency", value: (episode: Episode, run?: Run) => Math.round(run && supportsCombined(run) ? combinedScores(run).get(episode)?.combined ?? 0 : 0), suffix: "" },
  { name: "Progress", about: "Levels completed", value: levels, suffix: "" },
  { name: "Actions", about: "Actions used; lower is better when progress is equal", value: (episode: Episode) => turns(episode).length, suffix: "" },
  { name: "Exploration %", about: "Share of the offered physical action space tested; breadth is descriptive rather than inherently good", value: (episode: Episode) => Math.round(explorationCoverage(episode)), suffix: "%" },
  { name: "Useful %", about: "Share of actions that yielded objective environmental or belief-update evidence", value: (episode: Episode) => Math.round(usefulActionRate(episode)), suffix: "%" },
  { name: "LLM calls", about: "Logical model requests; a shared cache hit still counts for each condition.", value: modelCalls, suffix: "" },
  { name: "Time", about: "Comparable compute time with shared model-cache latency normalized.", value: (episode: Episode) => Math.round(comparableTime(episode)), suffix: "s" },
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
      <div className="setup-node setup-game"><small>{run.benchmark === "arc_agi_2" ? "Task" : "Game"}</small><strong>{run.game_id ?? "Unknown"}</strong></div>
      <div className="setup-node setup-seed"><small>Seed</small><strong>{run.seed ?? "Unknown"}</strong></div>
      <div className="setup-node setup-actions"><small>Actions</small><strong>{budget}</strong></div>
      <div className="setup-node setup-modes"><small>Modes</small><strong>{modes}</strong></div>
      <div className="setup-node setup-model"><small>Model</small><strong>{run.model || "gpt-6-luna"}{run.reasoning_effort ? ` · ${run.reasoning_effort}` : ""}</strong></div>
    </section>
  );
}

function ResultsChart({ run }: { run: Run }) {
  const conditions = episodes(run);
  const scores = combinedScores(run);
  return (
    <section className="card results-card">
      <h2>Results</h2>
      {supportsCombined(run) && <><div className="score-strip">{conditions.map(([label, episode]) => { const score = scores.get(episode)!; return <div key={label}><small>{label}</small><strong>{score.combined.toFixed(1)}</strong><span>P {(score.progress * 100).toFixed(0)} · A {(score.actions * 100).toFixed(0)} · S {(score.search * 100).toFixed(0)} · C {(score.calls * 100).toFixed(0)}</span></div>; })}</div><p className="metric-about">Exploratory score: 50% progress · 20% action efficiency · 20% search quality · 10% model-call efficiency.</p></>}
      <div className="chart" role="img" aria-label="Grouped experiment result bars">
        {METRICS.map((metric) => (
          <div className="metric-group" key={metric.name}>
            <div className="bars">
              {conditions.map(([label, episode]) => {
                const value = metric.value(episode);
                const available = conditions.map(([, item]) => metric.value(item)).filter((item): item is number => typeof item === "number" && Number.isFinite(item));
                const largest = Math.max(1, ...available);
                const positive = available.filter((item) => item > 0);
                const smallestPositive = positive.length ? Math.min(...positive) : 1;
                const relative = value === null ? 0 : metric.lowerIsBetter
                  ? (value > 0 ? smallestPositive / value : 0)
                  : value / largest;
                const shown = value === null ? "—" : `${value.toFixed(metric.decimals ?? 0)}${metric.suffix ?? ""}`;
                return (
                  <div className="bar-slot" key={label} title={`${label}: ${shown}. ${metric.about}`}>
                    <span className="bar-value">{shown}</span>
                    <div className="bar" style={{ height: `${value === null ? 2 : Math.max(value ? 8 : 2, relative * 180)}px`, background: MODE_COLORS[label] }} />
                  </div>
                );
              })}
            </div>
            <strong>{metric.name}</strong>
            <small className="metric-definition">{metric.about}</small>
          </div>
        ))}
      </div>
      <div className="legend">{conditions.map(([label]) => <span key={label}><i style={{ background: MODE_COLORS[label] }} />{label}</span>)}</div>
      <div className="summary-table" role="table" aria-label="Computation summary">
        <div className="summary-row summary-head"><span>Mode</span><span>LLM calls</span><span>Searches</span><span>Overrides</span><span>Time</span></div>
        {conditions.map(([label, episode]) => (
          <div className="summary-row" role="row" key={label}>
            <strong>{label}</strong><span>{modelCalls(episode)}</span>
            <span>{episode.condition === "llm_only" ? 0 : turns(episode).length}</span>
            <span>{overrides(episode)}</span><span>{comparableTime(episode).toFixed(1)}s</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function BatchResults({ runs, batchId }: { runs: Run[]; batchId: string }) {
  const [metricName, setMetricName] = useState(BATCH_METRICS[0].name);
  const [mode, setMode] = useState("All");
  const metric = BATCH_METRICS.find((item) => item.name === metricName) ?? BATCH_METRICS[0];
  const conditionLabels = ["LLM", "Classical", "Quantum"];
  const shownLabels = mode === "All" ? conditionLabels : [mode];
  const values = runs.flatMap((run) => episodes(run).filter(([label]) => shownLabels.includes(label)).map(([, episode]) => metric.value(episode, run)));
  const max = Math.max(1, ...values);
  const signatures = new Set(runs.map((run) => [run.seed, run.action_budget, run.model, run.reasoning_effort, run.harness_version, run.input_representation].join("|")));
  const solved = runs.reduce((sum, run) => sum + episodes(run).filter(([, episode]) => episode.success).length, 0);
  const invalid = runs.filter((run) => run.model_settings?.comparison_integrity?.comparison_valid === false);

  return (
    <section className="card batch-card">
      <div className="batch-heading">
        <div><p className="section-number">Batch comparison</p><h2>{batchId}</h2></div>
        <div className="batch-controls">
          <label>Metric<select value={metricName} onChange={(event) => setMetricName(event.target.value)}>{BATCH_METRICS.map((item) => <option key={item.name}>{item.name}</option>)}</select></label>
          <label>Mode<select value={mode} onChange={(event) => setMode(event.target.value)}><option>All</option>{conditionLabels.map((label) => <option key={label}>{label}</option>)}</select></label>
        </div>
      </div>
      <div className="batch-facts"><span><strong>{runs.length}</strong> games</span><span><strong>{solved}</strong> solved conditions</span><span><strong>{runs[0]?.seed ?? "—"}</strong> seed</span><span><strong>{runs[0]?.harness_version ?? "—"}</strong> harness</span></div>
      {signatures.size > 1 && <p className="batch-warning">This batch mixes seed, budget, model, harness, or input-representation settings. Interpret it cautiously.</p>}
      {invalid.length > 0 && <p className="batch-warning">{invalid.length} comparison{invalid.length === 1 ? " is" : "s are"} marked invalid and should be excluded from conclusions.</p>}
      <p className="metric-about">{metric.about}</p>
      <div className="batch-chart" role="img" aria-label={`${metric.name} across games in ${batchId}`}>
        {runs.map((run) => (
          <div className="batch-game" key={run.filename}>
            <div className="batch-bars">
              {episodes(run).filter(([label]) => shownLabels.includes(label)).map(([label, episode]) => {
                const value = metric.value(episode, run);
                return <div className="batch-bar-slot" key={label} title={`${run.game_id} · ${label}: ${value}${metric.suffix}`}><span>{value}{metric.suffix}</span><i style={{ height: `${Math.max(value ? 8 : 2, value / max * 205)}px`, background: MODE_COLORS[label] }} /></div>;
              })}
            </div>
            <strong>{run.game_id}</strong>
          </div>
        ))}
      </div>
      <div className="legend">{shownLabels.map((label) => <span key={label}><i style={{ background: MODE_COLORS[label] }} />{label}</span>)}</div>
    </section>
  );
}

function candidateMarkers(candidates?: Candidate[]): EntityMarker[] {
  return (candidates ?? []).flatMap((candidate) => {
    const match = /^click-(e\d+)$/.exec(candidate.candidate_id);
    return match && candidate.x !== undefined && candidate.x !== null && candidate.y !== undefined && candidate.y !== null
      ? [{ id: match[1], center: [candidate.x, candidate.y] as [number, number] }] : [];
  });
}

function Grid({ frame, entities = [] }: { frame?: Frame; entities?: EntityMarker[] }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!frame?.length || !canvas.current) return;
    const height = frame.length;
    const width = frame[0]?.length || 1;
    const scale = Math.max(6, Math.floor(512 / Math.max(width, height)));
    const margin = 28;
    const context = canvas.current.getContext("2d");
    if (!context) return;
    canvas.current.width = width * scale + margin;
    canvas.current.height = height * scale + margin;
    context.fillStyle = "#071017";
    context.fillRect(0, 0, canvas.current.width, canvas.current.height);
    frame.forEach((row, y) => row.forEach((cell, x) => {
      context.fillStyle = ARC_COLORS[cell] ?? "#000000";
      context.fillRect(margin + x * scale, margin + y * scale, scale, scale);
    }));
    context.strokeStyle = "rgba(4, 12, 18, .28)";
    context.lineWidth = Math.max(.5, scale / 14);
    context.beginPath();
    for (let x = 0; x <= width; x += 1) { context.moveTo(margin + x * scale, margin); context.lineTo(margin + x * scale, margin + height * scale); }
    for (let y = 0; y <= height; y += 1) { context.moveTo(margin, margin + y * scale); context.lineTo(margin + width * scale, margin + y * scale); }
    context.stroke();
    const tick = Math.max(1, Math.ceil(Math.max(width, height) / 12));
    context.fillStyle = "#9fb4bf";
    context.font = "11px sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    for (let x = 0; x < width; x += tick) context.fillText(String(x), margin + (x + .5) * scale, margin / 2);
    context.textAlign = "right";
    for (let y = 0; y < height; y += tick) context.fillText(String(y), margin - 7, margin + (y + .5) * scale);
    entities.forEach((entity) => {
      const [x, y] = entity.center;
      if (x < 0 || y < 0 || x >= width || y >= height) return;
      const px = margin + (x + .5) * scale;
      const py = margin + (y + .5) * scale;
      if (entity.bounds) {
        const [x0, y0, x1, y1] = entity.bounds;
        context.strokeStyle = "rgba(255,255,255,.88)";
        context.lineWidth = 1.5;
        context.strokeRect(margin + x0 * scale + 1, margin + y0 * scale + 1,
          (x1 - x0 + 1) * scale - 2, (y1 - y0 + 1) * scale - 2);
      }
      context.font = "bold 12px sans-serif";
      const labelWidth = context.measureText(entity.id).width + 8;
      context.fillStyle = "rgba(3, 12, 18, .88)";
      context.fillRect(px - labelWidth / 2, py - 9, labelWidth, 18);
      context.strokeStyle = "#71e1d7";
      context.strokeRect(px - labelWidth / 2, py - 9, labelWidth, 18);
      context.fillStyle = "#ffffff";
      context.textAlign = "center";
      context.fillText(entity.id, px, py);
    });
  }, [frame, entities]);
  if (!frame?.length) return <div className="empty-grid">No frame</div>;
  const columns = frame[0]?.length || 1;
  return <canvas ref={canvas} className="arc-grid" style={{ aspectRatio: `${columns * Math.max(6, Math.floor(512 / Math.max(columns, frame.length))) + 28} / ${frame.length * Math.max(6, Math.floor(512 / Math.max(columns, frame.length))) + 28}` }} role="img" aria-label={`${columns} by ${frame.length} game state with entity labels`} />;
}

function actionLabel(candidate: Candidate) {
  const coordinates = !candidate.description?.includes("(") && candidate.x !== undefined && candidate.x !== null && candidate.y !== undefined && candidate.y !== null
    ? ` (${candidate.x}, ${candidate.y})` : "";
  return `${candidate.description || candidate.candidate_id}${coordinates}`;
}

function normalizeBatchId(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+/, "").slice(0, 32);
}

function LunaProposal({ turn }: { turn: Turn }) {
  const lunaAtoms = (turn.search_diagnostics?.atom_catalog ?? []).filter((atom) => atom.source === "luna");
  const hypotheses = turn.proposal.hypotheses ?? [];
  const working = turn.search_diagnostics?.luna_working_hypothesis;
  return <section className="turn-block luna-block">
    <h4>Luna</h4>
    {turn.proposal.summary && <p>{turn.proposal.summary}</p>}
    {working && <div className="working-hypothesis">
      <small>Working hypothesis</small><strong>{working.statement}</strong>
      <dl><div><dt>Looking at</dt><dd>{working.focus_entity_ids?.join(", ") || "Unspecified"}</dd></div><div><dt>Targeting</dt><dd>{working.target_candidate_id || turn.proposal.recommended_candidate_id}</dd></div><div><dt>Expects</dt><dd>{working.expected_effect}</dd></div><div><dt>Success</dt><dd>{working.success_signal}</dd></div><div><dt>Falsified by</dt><dd>{working.falsification}</dd></div></dl>
    </div>}
    {!working && lunaAtoms.length > 0 && <p className="turn-note">This older trace recorded Luna’s hypothesis components but not a complete Luna-authored working hypothesis. The selected composition appears under Decision.</p>}
    {lunaAtoms.length > 0 && <div className="concept-groups">{["role", "relation", "effect"].map((kind) => {
      const atoms = lunaAtoms.filter((atom) => atom.kind === kind);
      return atoms.length > 0 && <div key={kind}><small>{kind}</small>{atoms.map((atom, index) => <span className="concept-chip" key={atom.key ?? `${kind}-${index}`}>{atom.subject} · {atom.value.replaceAll("_", " ")}{atom.target ? ` · ${atom.target}` : ""}</span>)}</div>;
    })}</div>}
    {lunaAtoms.length === 0 && hypotheses.length > 0 && <div className="hypothesis-list">{hypotheses.map((hypothesis, index) => <p key={hypothesis.id ?? index}><strong>H{hypothesis.id ?? index + 1}</strong> {hypothesis.statement}</p>)}</div>}
    {turn.proposal.recommendation_reason && <p className="turn-note">Recommendation: {turn.proposal.recommendation_reason}</p>}
  </section>;
}

function ActionSet({ turn }: { turn: Turn }) {
  const actions = turn.proposal.action_candidates ?? [];
  return <section className="turn-block action-set">
    <h4>Available actions</h4>
    <div>{actions.map((candidate) => {
      const selected = candidate.candidate_id === turn.selected_candidate.candidate_id;
      const recommended = candidate.candidate_id === turn.proposal.recommended_candidate_id;
      return <span className={`action-chip${selected ? " selected" : ""}`} key={candidate.candidate_id} title={candidate.action_type}>
        {actionLabel(candidate)}{recommended && <b> Luna</b>}{selected && <b> Pick</b>}
      </span>;
    })}</div>
  </section>;
}

function EvidenceSearch({ turn }: { turn: Turn }) {
  const matrix = turn.search_diagnostics?.prediction_matrix ?? [];
  if (!matrix.length) return null;
  const posterior = turn.search_diagnostics?.hypothesis_probabilities_after ?? {};
  const plan = turn.search_diagnostics?.selected_plan ?? [];
  return <section className="turn-block evidence-search">
    <h4>Hypothesis search</h4>
    <p className="turn-note">Plan: {plan.join(" → ") || "—"} · Observed: {turn.search_diagnostics?.observed_outcome?.replaceAll("_", " ") || "—"}</p>
    <div className="hypothesis-list">{matrix.map((row) => {
      const after = posterior[String(row.hypothesis_id)];
      return <div key={row.hypothesis_id}>
        <p><strong>H{row.hypothesis_id}</strong> {row.statement}</p>
        <small>{(100 * row.probability).toFixed(0)}% → {typeof after === "number" ? `${(100 * after).toFixed(0)}%` : "—"}</small>
        {row.test_sequence && row.test_sequence.length > 0 && <p className="turn-note">Test: {row.test_sequence.join(" → ")} · Target: {row.sequence_target_state || "measurable evidence"} · Stop: {row.stop_condition || "contradiction"}</p>}
        {row.persistence_candidate_id && <p className="turn-note">Persist: {row.persistence_candidate_id}{row.actuator_entity_id ? ` via ${row.actuator_entity_id}` : ""} · Affects: {row.affected_entity_ids?.join(", ") || row.tracked_entity_id || "board"} · Expect: {row.expected_relation_change || "predicted change"}. Support: {row.confirmation_outcomes?.join(", ") || "specified evidence"}. Refute: {row.refutation_outcomes?.join(", ") || "contradiction"}.</p>}
        <p className="turn-note">{row.predictions.map((prediction) => `${prediction.candidate_id}: ${prediction.outcome.replaceAll("_", " ")}`).join(" · ")}</p>
      </div>;
    })}</div>
  </section>;
}

function EpisodeDetails({ label, episode }: { label: string; episode: Episode }) {
  const timing = (key: string) => turns(episode).reduce((total, turn) => total + Number(turn.timings?.[key] ?? 0), 0).toFixed(1);
  return (
    <div className="episode">
      <div className="episode-stats"><span>{episode.success ? "Solved" : "Unsolved"}</span><span>{levels(episode)} levels</span><span>{turns(episode).length} actions</span></div>
      <p className="turn-note">LLM {timing("proposal_seconds")}s · Reflection {timing("reflection_seconds")}s · ARC {timing("environment_seconds")}s · Search {timing("selection_seconds")}s</p>
      {episode.error && <p className="error">Run failed; internal details omitted.</p>}
      {turns(episode).map((turn) => (
        <details key={turn.turn}>
          <summary><span>#{turn.turn}</span><strong>{actionLabel(turn.selected_candidate)}</strong><small>{changed(turn)} cells</small></summary>
          <div className="turn-proposal-grid"><LunaProposal turn={turn} /><ActionSet turn={turn} /></div>
          <EvidenceSearch turn={turn} />
          <section className="turn-block decision-block"><h4>Decision</h4>
            {turn.search_diagnostics?.composed_statement && <p className="composition-note">{turn.search_diagnostics.composed_statement}</p>}
            <p className="decision-action"><small>Chosen action</small><strong>{actionLabel(turn.selected_candidate)}</strong><span>{turn.selected_candidate.action_type}</span></p>
            {turn.search_diagnostics?.procedure_continued && <p className="turn-note">Continuing H{turn.search_diagnostics.active_hypothesis_id} toward {turn.search_diagnostics.procedure_target_state || "its target state"}.</p>}
            {turn.search_diagnostics?.procedure_evidence_status && turn.search_diagnostics.procedure_evidence_status !== "inactive" && <p className="turn-note">Procedure: {turn.search_diagnostics.procedure_evidence_status}{turn.search_diagnostics.procedure_reframed ? " · updated hypothesis" : ""}.</p>}
            {turn.search_diagnostics?.validated_for_level && <p className="turn-note">Validated for level {turn.search_diagnostics.validated_for_level.validated_for_level}: H{turn.search_diagnostics.validated_for_level.hypothesis_id} · {turn.search_diagnostics.validated_for_level.statement}</p>}
            {turn.search_diagnostics?.selected_atoms && <p className="turn-note">{turn.search_diagnostics.selected_atoms.map((atom) => `${atom.kind}: ${atom.subject} ${atom.value.replaceAll("_", " ")}${atom.target ? ` ${atom.target}` : ""}${atom.source === "python_mutation" ? " ★" : ""}`).join(" · ")}</p>}
          </section>
          {turn.search_diagnostics?.trajectory_status && turn.search_diagnostics.trajectory_status !== "not_applicable" && <p className="turn-note">Path: {turn.search_diagnostics.trajectory_status.replaceAll("_", " ")} · Object: {turn.search_diagnostics.tracked_entity_id || "—"}{turn.search_diagnostics.target_entity_id ? ` → ${turn.search_diagnostics.target_entity_id}` : ""}{typeof turn.search_diagnostics.distance_delta === "number" ? ` · Distance: ${turn.search_diagnostics.distance_delta >= 0 ? "−" : "+"}${Math.abs(turn.search_diagnostics.distance_delta).toFixed(1)}` : ""}</p>}
          <section className="turn-block state-block"><h4>Evidence</h4><div className="frames"><figure><figcaption>Before · Level {turn.observation_before.levels_completed ?? 0}</figcaption><Grid frame={turn.observation_before.frame} entities={turn.search_diagnostics?.entity_catalog ?? candidateMarkers(turn.proposal.action_candidates)} /></figure><figure><figcaption>After · Level {turn.observation_after.levels_completed ?? 0}</figcaption><Grid frame={turn.observation_after.frame} entities={turn.search_diagnostics?.after_entity_catalog ?? (turn.observation_before.levels_completed === turn.observation_after.levels_completed ? candidateMarkers(turn.proposal.action_candidates) : [])} /><small>{turn.observation_after.event}</small></figure></div></section>
          {turn.reflection && <p className="reflection-note"><strong>Outcome:</strong> {turn.reflection.summary} <span>Next: {turn.reflection.next_focus}</span></p>}
          {turn.search_diagnostics?.success_hypothesis_audit && <section className="turn-block audit-block"><h4>Success audit · {turn.search_diagnostics.success_hypothesis_audit.verdict?.replaceAll("_", " ")}</h4><p>{turn.search_diagnostics.success_hypothesis_audit.corrected_hypothesis}</p><p className="turn-note">Transfer: {turn.search_diagnostics.success_hypothesis_audit.transferable_relation}</p><p className="turn-note">Interface: {turn.search_diagnostics.success_hypothesis_audit.interface_change}</p></section>}
          <p className="turn-note">Luna pick: {turn.proposal.recommended_candidate_id} · Changed: {changed(turn)}{turn.search_diagnostics?.regret !== undefined ? ` · Regret: ${turn.search_diagnostics.regret.toFixed(3)} · Rank: ${turn.search_diagnostics.selected_rank}` : ""}</p>
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

function winLevels(episode: Episode) {
  return Math.max(0, ...turns(episode).flatMap((turn) => [
    Number(turn.observation_before.win_levels ?? 0),
    Number(turn.observation_after.win_levels ?? 0),
  ]));
}

function generalizationScore(episode: Episode) {
  const completed = levels(episode);
  const total = winLevels(episode);
  if (total > 0) return Math.min(1, completed / total);
  return episode.success ? 1 : 0;
}

function adaptationScore(episode: Episode) {
  const allTurns = turns(episode);
  const budget = Math.max(1, allTurns.length);
  const transitions = allTurns
    .map((turn, index) => ({ turn, index }))
    .filter(({ turn }) => Number(turn.observation_after.levels_completed ?? 0) > Number(turn.observation_before.levels_completed ?? 0));
  if (!transitions.length) return 0;
  const scores = transitions.map(({ index }) => {
    const recoveredAt = allTurns.findIndex((turn, candidateIndex) => candidateIndex > index && objectiveActionEvidence(turn));
    const latency = recoveredAt > index ? recoveredAt - index : budget;
    return Math.max(0, 1 - latency / budget);
  });
  return scores.reduce((sum, value) => sum + value, 0) / scores.length;
}

function noChange(turn: Turn) {
  const physical = String(turn.search_diagnostics?.physical_outcome ?? turn.search_diagnostics?.observed_outcome ?? "").toUpperCase();
  if (physical) return physical === "NO_CHANGE";
  return changed(turn) === 0
    && Number(turn.observation_after.levels_completed ?? 0) === Number(turn.observation_before.levels_completed ?? 0);
}

function objectiveActionEvidence(turn: Turn) {
  const physical = String(turn.search_diagnostics?.physical_outcome ?? turn.search_diagnostics?.observed_outcome ?? "").toUpperCase();
  const levelChanged = Number(turn.observation_after.levels_completed ?? 0) !== Number(turn.observation_before.levels_completed ?? 0);
  const terminal = ["WIN", "GAME_OVER", "FAILED"].includes(String(turn.observation_after.state ?? "").toUpperCase());
  if (levelChanged || terminal) return true;
  if (physical) return !["NO_CHANGE", "UNKNOWN", "NONE"].includes(physical);
  return changed(turn) > 0;
}

function autonomyScore(episode: Episode) {
  const allTurns = turns(episode);
  if (!allTurns.length) return 0;
  let noChangeActions = 0;
  let repeatedAfterNoChange = 0;
  allTurns.forEach((turn) => {
    if (noChange(turn)) noChangeActions += 1;
  });
  for (let index = 1; index < allTurns.length; index += 1) {
    const previous = allTurns[index - 1];
    const current = allTurns[index];
    if (noChange(previous) && actionKey(previous.selected_candidate) === actionKey(current.selected_candidate)) repeatedAfterNoChange += 1;
  }
  const novelty = uniqueActions(episode) / allTurns.length;
  const persistenceAvoidance = 1 - repeatedAfterNoChange / Math.max(1, noChangeActions);
  const evidenceYield = allTurns.filter(objectiveActionEvidence).length / allTurns.length;
  return (novelty + persistenceAvoidance + evidenceYield) / 3;
}

const CAPABILITIES = [
  { name: "Generalization", symbol: "G", score: generalizationScore },
  { name: "Adaptation", symbol: "A", score: adaptationScore },
  { name: "Autonomy", symbol: "Au", score: autonomyScore },
  { name: "Transfer", symbol: "T", score: () => 0 },
  { name: "Robustness", symbol: "R", score: () => 0 },
] as const;

function capabilityScores(episode: Episode) {
  return CAPABILITIES.map((capability) => Math.max(0, Math.min(1, capability.score(episode))));
}

function signedScore(value: number) {
  if (Math.abs(value) < .0005) return "0.000";
  return `${value > 0 ? "+" : ""}${value.toFixed(3)}`;
}

function CapabilityComparison({ run }: { run: Run }) {
  const llm = run.without_qahs;
  const quantum = run.with_qahs;
  const llmScores = capabilityScores(llm);
  const quantumScores = capabilityScores(quantum);
  const llmSai = llmScores.reduce((sum, value) => sum + value, 0) / llmScores.length;
  const quantumSai = quantumScores.reduce((sum, value) => sum + value, 0) / quantumScores.length;
  return <section className="card capability-card">
    <div className="capability-heading"><div><p className="section-number">Run-level capability scores</p><h2>Quantum compared with LLM</h2></div><p>Scores range from 0.000 to 1.000. Classical remains in the result charts as the selector baseline.</p></div>
    <div className="capability-table" role="table" aria-label="Quantum and LLM capability comparison">
      <div className="capability-row capability-head" role="row"><span>Metric</span><span>LLM only</span><span>LLM + Quantum QAHS</span><span>Difference</span></div>
      {CAPABILITIES.map((capability, index) => {
        const llmScore = llmScores[index];
        const quantumScore = quantumScores[index];
        return <div className="capability-row" role="row" key={capability.name}>
          <strong>{capability.name} ({capability.symbol})</strong>
          <span className="capability-number">{llmScore.toFixed(3)}</span>
          <span className="capability-number">{quantumScore.toFixed(3)}</span>
          <span className="capability-number">{signedScore(quantumScore - llmScore)}</span>
        </div>;
      })}
      <div className="capability-row capability-total" role="row">
        <strong>SAI proxy</strong>
        <span className="capability-number">{llmSai.toFixed(3)}</span>
        <span className="capability-number">{quantumSai.toFixed(3)}</span>
        <span className="capability-number">{signedScore(quantumSai - llmSai)}</span>
      </div>
    </div>
    <p className="metric-about capability-note">G is completed-level share; A is normalized recovery after a level transition; Au averages distinct physical-action rate, avoidance of repetition after no change, and objective environmental-evidence yield. T and R are 0.000 because this run does not contain the required paired conditions.</p>
  </section>;
}

function CounterfactualPanel({ run }: { run: Run }) {
  const options = episodes(run).flatMap(([label, episode]) => turns(episode).flatMap((turn) => {
    const alternative = turn.search_diagnostics?.selector_candidate_id;
    if (!alternative || alternative === turn.selected_candidate.candidate_id) return [];
    const candidate = turn.proposal.action_candidates?.find((item) => item.candidate_id === alternative);
    return candidate ? [{ label, condition: episode.condition === "qahs_classical" ? "with_classical" : "with_qahs", turn: turn.turn, candidate, advantage: turn.search_diagnostics?.selector_advantage }] : [];
  }));
  const [choice, setChoice] = useState(0);
  const [horizon, setHorizon] = useState(1);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState("");
  if (run.source !== "arc" || !options.length) return null;
  const selected = options[Math.min(choice, options.length - 1)];
  async function replay() {
    setRunning(true); setResult("Replaying…");
    try {
      const response = await fetch(`${LOCAL_RUNNER_URL}/counterfactual`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ filename: run.filename, condition: selected.condition, turn: selected.turn, horizon }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Replay failed.");
      setResult(`${data.alternative_levels_delta > 0 ? "Progress" : "No level gain"} · ${data.immediate_frame_difference} cells differ from original · saved`);
    } catch (reason) { setResult(reason instanceof Error ? reason.message : "Replay failed."); }
    finally { setRunning(false); }
  }
  return <section className="card counterfactual-card"><h2>Alternative replay</h2><p className="metric-about">Replay the recorded prefix, then test one rejected QAHS action without another model call.</p><div className="counterfactual-controls"><label>Alternative<select value={choice} onChange={(event) => setChoice(Number(event.target.value))}>{options.map((item, index) => <option value={index} key={`${item.condition}-${item.turn}`}>{item.label} · turn {item.turn} · {item.candidate.description || item.candidate.action_type} · +{Number(item.advantage ?? 0).toFixed(2)}</option>)}</select></label><label>Follow-up<select value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}><option value={1}>Immediate</option><option value={3}>3 actions</option><option value={5}>5 actions</option></select></label><button type="button" onClick={replay} disabled={running}>{running ? "Replaying…" : "Test path"}</button></div>{result && <p className="run-message">{result}</p>}</section>;
}

type RunProgress = { phase?: string; condition?: string; turn?: number; max_steps?: number; percent?: number; elapsed_seconds?: number; estimated_remaining_seconds?: number | null; levels?: number };
type BatchJob = { job_id: string; status: string; batch_id: string; total: number; completed: number; current?: string | null; files?: string[]; errors?: { game: string; error: string }[]; message?: string; progress?: RunProgress; run_started_at?: string };

const ACTIVE_BATCH_KEY = "qahs-active-batch";

function LocalRunPanel({ onComplete }: { onComplete: (filename?: string) => Promise<void> }) {
  const [runType, setRunType] = useState<"single" | "batch">("single");
  const [benchmark, setBenchmark] = useState<"arc2" | "arc3">("arc3");
  const [source, setSource] = useState("demo");
  const [game, setGame] = useState("ls20");
  const [games, setGames] = useState("ar25, vc33, lp85, r11l, s5i5");
  const [seed, setSeed] = useState(0);
  const [actions, setActions] = useState(8);
  const [model, setModel] = useState("gpt-6-luna");
  const [reasoningEffort, setReasoningEffort] = useState("medium");
  const [harness, setHarness] = useState("v12");
  const [batch, setBatch] = useState("batch-01");
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState("");
  const [job, setJob] = useState<BatchJob | null>(null);
  const [clock, setClock] = useState(Date.now());

  function chooseBenchmark(value: "arc2" | "arc3") {
    setBenchmark(value);
    if (value === "arc2") {
      setSource("arc2"); setGame(ARC2_LUNA_MEDIUM[0]); setGames(ARC2_LUNA_MEDIUM.slice(0, 8).join(", "));
      setModel("gpt-6-luna"); setReasoningEffort("medium"); setActions(1); setBatch("arc2-luna-medium");
    } else {
      setSource("arc"); setGame("ar25"); setGames("ar25, vc33, lp85, r11l, s5i5"); setActions(8); setBatch("batch-01");
    }
  }

  function toggleArc2Task(task: string) {
    const selected = new Set(games.split(/[\s,]+/).filter(Boolean));
    if (selected.has(task)) selected.delete(task); else selected.add(task);
    setGames(ARC2_LUNA_MEDIUM.filter((item) => selected.has(item)).join(", "));
  }

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => setClock(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [running]);

  const formatDuration = (seconds?: number | null) => {
    if (seconds === undefined || seconds === null) return "—";
    const rounded = Math.max(0, Math.round(seconds));
    return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, "0")}`;
  };
  const phaseLabel: Record<string, string> = { starting: "Starting", condition_started: "Starting mode", reasoning: "Reasoning", objective: "Scoring", selection: "Selecting", environment: "Acting", review: "Reviewing", reviewing: "Reviewing success", turn_completed: "Turn done", condition_completed: "Mode done", done: "Done" };
  const liveElapsed = job?.run_started_at && ["running", "starting"].includes(job.status)
    ? (clock - new Date(job.run_started_at).getTime()) / 1000
    : job?.progress?.elapsed_seconds;

  function createBatchId() {
    const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
    setBatch(`batch-${stamp}`);
  }

  async function waitForBatch(jobId: string): Promise<BatchJob> {
    while (true) {
      await new Promise((resolve) => setTimeout(resolve, 3000));
      const response = await fetch(`${LOCAL_RUNNER_URL}/jobs/${jobId}`, { cache: "no-store" });
      const job: BatchJob = await response.json();
      if (!response.ok) throw new Error("The batch status could not be loaded.");
      setJob(job);
      const current = job.current ? ` · ${job.current}` : "";
      setMessage(`${job.completed}/${job.total}${current}`);
      if (!["running", "starting"].includes(job.status)) return job;
    }
  }

  async function followBatch(jobId: string) {
    setRunning(true);
    try {
      const finalJob = await waitForBatch(jobId);
      if (finalJob.status === "paused") {
        setMessage(`${finalJob.completed}/${finalJob.total} · paused`);
        return;
      }
      localStorage.removeItem(ACTIVE_BATCH_KEY);
      await onComplete(finalJob.files?.[0]);
      setMessage(finalJob.errors?.length ? `${finalJob.completed}/${finalJob.total} · ${finalJob.errors.length} failed` : `${finalJob.completed}/${finalJob.total} · saved`);
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "The batch status could not be loaded.");
    } finally {
      setRunning(false);
    }
  }

  useEffect(() => {
    const savedJob = localStorage.getItem(ACTIVE_BATCH_KEY);
    if (!savedJob) return;
    fetch(`${LOCAL_RUNNER_URL}/jobs/${savedJob}`, { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error();
        const restored: BatchJob = await response.json();
        setJob(restored);
        setMessage(`${restored.completed}/${restored.total} · ${restored.status}`);
        if (["running", "starting"].includes(restored.status)) void followBatch(restored.job_id);
      })
      .catch(() => localStorage.removeItem(ACTIVE_BATCH_KEY));
  // The local job ID is intentionally restored once when the panel mounts.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function resumeBatch() {
    if (!job) return;
    setRunning(true);
    setMessage("Resuming…");
    try {
      const response = await fetch(`${LOCAL_RUNNER_URL}/jobs/${job.job_id}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      const restored: BatchJob & { error?: string } = await response.json();
      if (!response.ok) throw new Error(restored.error || "The batch could not resume.");
      localStorage.setItem(ACTIVE_BATCH_KEY, restored.job_id);
      setJob(restored);
      await followBatch(restored.job_id);
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "The batch could not resume.");
      setRunning(false);
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setRunning(true);
    setMessage(runType === "batch" ? "Starting batch…" : "Running all three modes…");
    try {
      const response = await fetch(`${LOCAL_RUNNER_URL}/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: benchmark === "arc2" ? "arc2" : runType === "batch" ? "arc" : source, game, games: runType === "batch" ? games : [game], seed, max_steps: benchmark === "arc2" ? 1 : actions, batch, model, reasoning_effort: reasoningEffort, harness }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "The experiment failed.");
      localStorage.setItem(ACTIVE_BATCH_KEY, result.job_id);
      setJob(result);
      await followBatch(result.job_id);
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "The experiment failed.");
    } finally { setRunning(false); }
  }

  return (
    <aside className="local-panel">
      <div className="local-panel-inner">
        <h2>Run</h2>
        <form onSubmit={submit}>
          <label>Benchmark<select value={benchmark} onChange={(event) => chooseBenchmark(event.target.value as "arc2" | "arc3")} disabled={running}><option value="arc3">ARC-AGI-3</option><option value="arc2">ARC-AGI-2 · Luna set</option></select></label>
          <div className="run-type"><button type="button" className={runType === "single" ? "active" : ""} onClick={() => setRunType("single")} disabled={running}>Single</button><button type="button" className={runType === "batch" ? "active" : ""} onClick={() => { setRunType("batch"); setSource("arc"); }} disabled={running}>Batch</button></div>
          {benchmark === "arc3" && runType === "single" && <label>Source<select value={source} onChange={(event) => setSource(event.target.value)} disabled={running}><option value="demo">Demo</option><option value="arc">ARC</option></select></label>}
          {runType === "single" && (benchmark === "arc2" ? <label>Task<select value={game} onChange={(event) => setGame(event.target.value)} disabled={running}>{ARC2_LUNA_MEDIUM.map((task) => <option key={task}>{task}</option>)}</select></label> : <label>Game<input value={game} onChange={(event) => setGame(event.target.value)} disabled={running || source === "demo"} maxLength={40} /></label>)}
          {runType === "batch" && benchmark === "arc3" && <label>Games<textarea value={games} onChange={(event) => setGames(event.target.value.toLowerCase())} rows={4} disabled={running} /></label>}
          {runType === "batch" && benchmark === "arc2" && <fieldset className="task-picker"><legend>Luna Medium passes</legend>{ARC2_LUNA_MEDIUM.map((task) => <label key={task}><input type="checkbox" checked={games.split(/[\s,]+/).includes(task)} onChange={() => toggleArc2Task(task)} disabled={running} />{task}</label>)}</fieldset>}
          {(benchmark === "arc2" || runType === "batch" || source === "arc") && <label>Model<select value={model} onChange={(event) => setModel(event.target.value)} disabled={running}><option value="gpt-6-luna">GPT-6 Luna</option><option value="gpt-6-sol">GPT-6 Sol</option>{benchmark === "arc3" && <><option value="claude-haiku-5-5">Claude Haiku 5.5</option><option value="claude-opus-5-5">Claude Opus 5.5</option></>}</select></label>}
          {(benchmark === "arc2" || runType === "batch" || source === "arc") && <label>Effort<select value={reasoningEffort} onChange={(event) => setReasoningEffort(event.target.value)} disabled={running}><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option><option value="xhigh">XHigh</option><option value="max">Max</option></select></label>}
          {benchmark === "arc3" && (runType === "batch" || source === "arc") && <label>Harness<select value={harness} onChange={(event) => setHarness(event.target.value)} disabled={running}><option value="v12">V12 · structured worlds</option><option value="v11">V11 · information gain</option><option value="v10">V10 · frozen</option></select></label>}
          {benchmark === "arc3" && <label>Seed<input type="number" min="0" max="1000000" value={seed} onChange={(event) => setSeed(Number(event.target.value))} disabled={running} /></label>}
          <label>Batch ID<span className="batch-id-field"><input value={batch} onChange={(event) => setBatch(normalizeBatchId(event.target.value))} pattern="[a-z0-9][a-z0-9-]{0,31}" maxLength={32} required disabled={running} /><button type="button" onClick={createBatchId} disabled={running} title="Create batch ID">↻</button></span></label>
          {benchmark === "arc3" && <label>Actions<input type="range" min="2" max="100" value={actions} onChange={(event) => setActions(Number(event.target.value))} disabled={running} /><output>{actions}</output></label>}
          <button type="submit" disabled={running}>{running ? "Running…" : runType === "batch" ? "Start batch" : "Start"}</button>
        </form>
        {job?.status === "paused" && <button type="button" className="resume-button" onClick={resumeBatch} disabled={running}>Resume</button>}
        {(job || running) && <div className="run-progress" aria-live="polite">
          <div className="run-progress-head"><strong>{job?.status.startsWith("completed") ? "✓ Done" : phaseLabel[job?.progress?.phase ?? "starting"] ?? job?.status ?? "Starting"}</strong><span>{Math.round(job?.progress?.percent ?? 0)}%</span></div>
          <div className="run-progress-track"><span style={{ width: `${job?.progress?.percent ?? 0}%` }} /></div>
          <p>{job?.current ?? (running ? "Preparing" : "Batch")}{job?.progress?.condition ? ` · ${job.progress.condition.replaceAll("_", " ")}` : ""}{job?.progress?.turn ? ` · ${job.progress.turn}/${job.progress.max_steps}` : ""}</p>
          <small>{formatDuration(liveElapsed)} elapsed · {formatDuration(job?.progress?.estimated_remaining_seconds)} left</small>
        </div>}
        {message && <p className="run-message" aria-live="polite">{message}</p>}
        <small>Local only</small>
      </div>
    </aside>
  );
}

type PickerOption = { value: string; label: string };

function ResultPicker({ value, options, label, onChange }: { value: string; options: PickerOption[]; label: string; onChange: (value: string) => void }) {
  const [open, setOpen] = useState(false);
  const [opensUp, setOpensUp] = useState(false);
  const [menuHeight, setMenuHeight] = useState(320);
  const root = useRef<HTMLDivElement>(null);
  const selected = options.find((option) => option.value === value);

  useEffect(() => {
    if (!open) return;
    const closeOutside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("keydown", closeOnEscape);
    requestAnimationFrame(() => root.current?.querySelector<HTMLElement>('[aria-selected="true"]')?.scrollIntoView({ block: "nearest" }));
    return () => {
      document.removeEventListener("pointerdown", closeOutside);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  function toggle() {
    if (!open && root.current) {
      const bounds = root.current.getBoundingClientRect();
      const below = window.innerHeight - bounds.bottom - 14;
      const above = bounds.top - 14;
      const useAbove = below < 260 && above > below;
      setOpensUp(useAbove);
      setMenuHeight(Math.max(160, Math.min(420, useAbove ? above : below)));
    }
    setOpen((current) => !current);
  }

  return <div className="run-picker" ref={root}>
    <span className="sr-only">{label}</span>
    <button type="button" className="run-picker-trigger" aria-label={label} aria-haspopup="listbox" aria-expanded={open} onClick={toggle}>
      <span>{selected?.label ?? value}</span><i aria-hidden="true" />
    </button>
    {open && <div className={`run-picker-menu ${opensUp ? "opens-up" : ""}`} style={{ maxHeight: menuHeight }} role="listbox" aria-label={label}>
      {options.map((option) => <button
        type="button"
        role="option"
        aria-selected={option.value === value}
        className={option.value === value ? "selected" : ""}
        key={option.value}
        onClick={() => { onChange(option.value); setOpen(false); }}
      >{option.label}</button>)}
    </div>}
  </div>;
}

export default function Dashboard() {
  const [manifest, setManifest] = useState<ManifestItem[]>([]);
  const [filename, setFilename] = useState("");
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState("");
  const [isLocal, setIsLocal] = useState(false);
  const [view, setView] = useState<"single" | "batch">("single");
  const [batchId, setBatchId] = useState("");
  const [batchRuns, setBatchRuns] = useState<Run[]>([]);
  const [batchLoading, setBatchLoading] = useState(false);

  const loadManifest = useCallback(async (preferred?: string) => {
    setError("");
    return fetch(`/runs/index.json?v=${Date.now()}`, { cache: "no-store" }).then((response) => {
      if (!response.ok) throw new Error("No published runs found.");
      return response.json();
    }).then((items: ManifestItem[]) => {
      if (!items.length) throw new Error("No saved experiments are available yet.");
      setManifest(items);
      setFilename(preferred && items.some((item) => item.filename === preferred) ? preferred : items[0]?.filename ?? "");
      const firstBatch = items.find((item) => item.batch_id)?.batch_id;
      setBatchId((current) => current || firstBatch || "");
    }).catch((reason) => setError(reason.message));
  }, []);
  useEffect(() => {
    const host = window.location.hostname;
    setIsLocal(
      ["localhost", "127.0.0.1"].includes(host) ||
      /^192\.168\.\d{1,3}\.\d{1,3}$/.test(host) ||
      /^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(host),
    );
    const preferredRun = new URLSearchParams(window.location.search).get("run") ?? undefined;
    void loadManifest(preferredRun);
  }, [loadManifest]);
  useEffect(() => {
    if (!filename) return;
    setRun(null);
    fetch(`/runs/${filename}?v=${Date.now()}`, { cache: "no-store" }).then((response) => {
      if (!response.ok) throw new Error(`Run request failed (${response.status}).`);
      return response.json();
    }).then((data: Run) => setRun({ ...data, filename })).catch(() => setError("The selected run could not be loaded."));
  }, [filename]);
  const batchIds = useMemo(() => Array.from(new Set(manifest.map((item) => item.batch_id).filter((item): item is string => Boolean(item)))), [manifest]);
  useEffect(() => {
    if (view !== "batch" || !batchId) return;
    const seen = new Set<string>();
    const items = manifest.filter((item) => {
      if (item.batch_id !== batchId || !item.harness_version) return false;
      const game = item.game_id ?? item.filename;
      if (seen.has(game)) return false;
      seen.add(game);
      return true;
    });
    setBatchLoading(true);
    setError("");
    Promise.all(items.map((item) => fetch(`/runs/${item.filename}?v=${Date.now()}`, { cache: "no-store" }).then((response) => response.json()).then((data: Run) => ({ ...data, filename: item.filename }))))
      .then((runs) => setBatchRuns(runs.sort((left, right) => String(left.game_id).localeCompare(String(right.game_id)))))
      .catch(() => setError("The selected batch could not be loaded."))
      .finally(() => setBatchLoading(false));
  }, [batchId, manifest, view]);
  const selectedLabel = useMemo(() => manifest.find((item) => item.filename === filename)?.label ?? filename, [manifest, filename]);

  return (
    <div className={isLocal ? "app-shell has-local-panel" : "app-shell"}>
      {isLocal && <LocalRunPanel onComplete={loadManifest} />}
      <main>
        <header className="banner"><h1>QAHS: comparing LLM, classical search, and quantum search</h1></header>
        <div className="view-switch" role="group" aria-label="Result view"><button className={view === "single" ? "active" : ""} onClick={() => setView("single")}>Single run</button><button className={view === "batch" ? "active" : ""} onClick={() => setView("batch")} disabled={!batchIds.length}>Batch</button></div>
        {view === "single" && manifest.length > 0 && <ResultPicker value={filename} onChange={setFilename} label="Saved experiment" options={manifest.map((item) => ({ value: item.filename, label: item.label }))} />}
        {view === "batch" && batchIds.length > 0 && <ResultPicker value={batchId} onChange={setBatchId} label="Saved batch" options={batchIds.map((item) => ({ value: item, label: item }))} />}
        {error && <p className="error">{error}</p>}
        {view === "single" && !run && !error && <p className="loading">Loading experiment…</p>}
        {view === "single" && run && <>{run.model_settings?.comparison_integrity?.comparison_valid === false && <p className="batch-warning">This matched comparison failed an integrity check and should not be used for conclusions.</p>}<SetupMap run={run} /><ResultsChart run={run} />{isLocal && <CounterfactualPanel run={run} />}<Details run={run} /><CapabilityComparison run={run} /></>}
        {view === "batch" && batchLoading && <p className="loading">Loading batch…</p>}
        {view === "batch" && !batchLoading && batchRuns.length > 0 && <BatchResults runs={batchRuns} batchId={batchId} />}
        {view === "batch" && !batchLoading && !batchIds.length && <p className="loading">New versioned batches will appear here.</p>}
        <footer>{view === "single" ? selectedLabel : batchId}</footer>
      </main>
    </div>
  );
}

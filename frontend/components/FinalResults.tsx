"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type Candidate = { candidate_id?: string; action_type?: string; x?: number | null; y?: number | null; next_action_type?: string | null; next_x?: number | null; next_y?: number | null };
type Observation = { frame?: number[][]; levels_completed?: number; win_levels?: number; state?: string };
type Turn = {
  turn: number;
  observation_before: Observation;
  observation_after: Observation;
  selected_candidate: Candidate;
  proposal?: { recommended_candidate_id?: string };
  model_calls?: number;
  search_diagnostics?: { quantum_used?: boolean; backend?: string; override_permitted?: boolean; physical_outcome?: string; observed_outcome?: string; procedure_continued?: boolean };
};
type Episode = { success?: boolean; turns?: Turn[]; wall_seconds?: number; error?: string | null };
type FinalRun = {
  filename: string;
  seed: number;
  action_budget?: number;
  condition_order?: string[];
  without_qahs: Episode;
  with_classical: Episode;
  with_qahs: Episode;
};
type ModeName = "LLM" | "Classical" | "Quantum";

const FINAL_FILES = [
  "20261103T195204Z__v12-structured-world-models__v12-final-lp85__lp85__seed-06__actions-40__gpt-6-luna__comparison.json",
  "20261103T205521Z__v12-structured-world-models__v12-final-lp85__lp85__seed-07__actions-40__gpt-6-luna__comparison.json",
  "20261103T231803Z__v12-structured-world-models__v12-final-lp85__lp85__seed-08__actions-40__gpt-6-luna__comparison.json",
  "20261104T022151Z__v12-structured-world-models__v12-final-lp85__lp85__seed-09__actions-40__gpt-6-luna__comparison.json",
  "20261104T035320Z__v12-structured-world-models__v12-final-lp85__lp85__seed-10__actions-40__gpt-6-luna__comparison.json",
] as const;

const MODES: { name: ModeName; key: "without_qahs" | "with_classical" | "with_qahs" }[] = [
  { name: "LLM", key: "without_qahs" },
  { name: "Classical", key: "with_classical" },
  { name: "Quantum", key: "with_qahs" },
];

function turns(episode: Episode) { return episode.turns ?? []; }
function levels(episode: Episode) { return Number(turns(episode).at(-1)?.observation_after.levels_completed ?? 0); }
function actionKey(candidate?: Candidate) { return [candidate?.action_type, candidate?.x ?? "", candidate?.y ?? "", candidate?.next_action_type ?? "", candidate?.next_x ?? "", candidate?.next_y ?? ""].join(":"); }
function changed(turn: Turn) {
  let count = 0;
  const after = turn.observation_after.frame ?? [];
  (turn.observation_before.frame ?? []).forEach((row, y) => row.forEach((cell, x) => { if (after[y]?.[x] !== cell) count += 1; }));
  return count;
}
function physicalOutcome(turn: Turn) { return String(turn.search_diagnostics?.physical_outcome ?? turn.search_diagnostics?.observed_outcome ?? "").toUpperCase(); }
function noChange(turn: Turn) {
  const physical = physicalOutcome(turn);
  if (physical) return physical === "NO_CHANGE";
  return changed(turn) === 0 && Number(turn.observation_after.levels_completed ?? 0) === Number(turn.observation_before.levels_completed ?? 0);
}
function objectiveEvidence(turn: Turn) {
  const physical = physicalOutcome(turn);
  const levelChanged = Number(turn.observation_after.levels_completed ?? 0) !== Number(turn.observation_before.levels_completed ?? 0);
  const terminal = ["WIN", "GAME_OVER", "FAILED"].includes(String(turn.observation_after.state ?? "").toUpperCase());
  if (levelChanged || terminal) return true;
  if (physical) return !["NO_CHANGE", "UNKNOWN", "NONE"].includes(physical);
  return changed(turn) > 0;
}
function firstProgress(episode: Episode) { return turns(episode).find((turn) => Number(turn.observation_after.levels_completed ?? 0) > Number(turn.observation_before.levels_completed ?? 0)); }
function modelCalls(episode: Episode) { return turns(episode).reduce((sum, turn) => sum + Number(turn.model_calls ?? 1), 0); }

function generalizationScore(episode: Episode) {
  const total = Math.max(0, ...turns(episode).flatMap((turn) => [Number(turn.observation_before.win_levels ?? 0), Number(turn.observation_after.win_levels ?? 0)]));
  if (total > 0) return Math.min(1, levels(episode) / total);
  return episode.success ? 1 : 0;
}
function adaptationScore(episode: Episode) {
  const allTurns = turns(episode);
  const budget = Math.max(1, allTurns.length);
  const transitions = allTurns.map((turn, index) => ({ turn, index })).filter(({ turn }) => Number(turn.observation_after.levels_completed ?? 0) > Number(turn.observation_before.levels_completed ?? 0));
  if (!transitions.length) return 0;
  return transitions.map(({ index }) => {
    const recoveredAt = allTurns.findIndex((turn, candidateIndex) => candidateIndex > index && objectiveEvidence(turn));
    return Math.max(0, 1 - (recoveredAt > index ? recoveredAt - index : budget) / budget);
  }).reduce((sum, score) => sum + score, 0) / transitions.length;
}
function autonomyScore(episode: Episode) {
  const allTurns = turns(episode);
  if (!allTurns.length) return 0;
  let noChangeActions = 0;
  let repeatedAfterNoChange = 0;
  allTurns.forEach((turn) => { if (noChange(turn)) noChangeActions += 1; });
  for (let index = 1; index < allTurns.length; index += 1) {
    if (noChange(allTurns[index - 1]) && actionKey(allTurns[index - 1].selected_candidate) === actionKey(allTurns[index].selected_candidate)) repeatedAfterNoChange += 1;
  }
  const novelty = new Set(allTurns.map((turn) => actionKey(turn.selected_candidate))).size / allTurns.length;
  const persistenceAvoidance = 1 - repeatedAfterNoChange / Math.max(1, noChangeActions);
  const evidenceYield = allTurns.filter(objectiveEvidence).length / allTurns.length;
  return (novelty + persistenceAvoidance + evidenceYield) / 3;
}
function capabilityScores(episode: Episode) {
  const values = [generalizationScore(episode), adaptationScore(episode), autonomyScore(episode), 0, 0];
  return [...values, values.reduce((sum, value) => sum + value, 0) / values.length];
}

function episodeSummary(episode: Episode) {
  const allTurns = turns(episode);
  const progress = firstProgress(episode);
  return {
    levels: levels(episode),
    actions: allTurns.length,
    firstProgress: progress?.turn ?? null,
    finalState: allTurns.at(-1)?.observation_after.state ?? "Unknown",
    calls: modelCalls(episode),
    minutes: Number(episode.wall_seconds ?? 0) / 60,
    qaoaTurns: allTurns.filter((turn) => turn.search_diagnostics?.quantum_used).length,
  };
}

export default function FinalResults() {
  const [runs, setRuns] = useState<FinalRun[]>([]);
  const [selectedSeed, setSelectedSeed] = useState(6);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all(FINAL_FILES.map((filename) => fetch(`/runs/${filename}`, { cache: "no-store" })
      .then((response) => { if (!response.ok) throw new Error(`Missing final run: ${filename}`); return response.json(); })
      .then((run: FinalRun) => ({ ...run, filename }))))
      .then((loaded) => setRuns(loaded.sort((left, right) => left.seed - right.seed)))
      .catch((reason: Error) => setError(reason.message));
  }, []);

  const aggregate = useMemo(() => MODES.map(({ name, key }) => {
    const episodes = runs.map((run) => run[key]);
    const actions = episodes.reduce((sum, episode) => sum + turns(episode).length, 0);
    const completed = episodes.reduce((sum, episode) => sum + levels(episode), 0);
    return {
      name,
      progressed: episodes.filter((episode) => levels(episode) > 0).length,
      levels: completed,
      actions,
      efficiency: actions ? 100 * completed / actions : 0,
      calls: episodes.reduce((sum, episode) => sum + modelCalls(episode), 0),
    };
  }), [runs]);

  const capability = useMemo(() => [
    { name: "LLM only", episodes: runs.map((run) => run.without_qahs) },
    { name: "Quantum QAHS", episodes: runs.map((run) => run.with_qahs) },
  ].map(({ name, episodes }) => ({
    name,
    scores: [0, 1, 2, 3, 4, 5].map((index) => episodes.length ? episodes.reduce((sum, episode) => sum + capabilityScores(episode)[index], 0) / episodes.length : 0),
  })), [runs]);

  const selected = runs.find((run) => run.seed === selectedSeed) ?? runs[0];
  const directOverride = useMemo(() => runs.flatMap((run) => turns(run.with_qahs).map((turn) => ({ run, turn })))
    .find(({ turn }) => Number(turn.observation_after.levels_completed ?? 0) > Number(turn.observation_before.levels_completed ?? 0)
      && turn.search_diagnostics?.quantum_used
      && turn.search_diagnostics?.override_permitted
      && Boolean(turn.proposal?.recommended_candidate_id)
      && turn.proposal?.recommended_candidate_id !== turn.selected_candidate.candidate_id), [runs]);

  if (error) return <main className="final-page"><p className="error">{error}</p></main>;
  if (!runs.length) return <main className="final-page"><p className="loading">Loading final evaluation…</p></main>;

  return <main className="final-page">
    <header className="final-hero">
      <p className="eyebrow">Final LP85 evaluation</p>
      <h1>Quantum QAHS produced progress more often across five seeds</h1>
      <blockquote>Across five LP85 seeds, Quantum QAHS produced game progress on more seeds and more progress per executed action than either Luna alone or Classical QAHS. One successful transition directly followed a QAOA override of Luna’s recommended action.</blockquote>
      <div className="final-facts"><span><strong>5</strong> seeds</span><span><strong>40</strong> action maximum</span><span><strong>15</strong> matched episodes</span><span><strong>V12</strong> frozen harness</span></div>
    </header>

    <section className="final-section">
      <p className="section-number">Aggregate outcome</p>
      <h2>Progress favored the quantum condition</h2>
      <div className="final-table" role="table" aria-label="Aggregate final evaluation">
        <div className="final-table-row final-table-head" role="row"><span>Condition</span><span>Progressing seeds</span><span>Levels</span><span>Actions</span><span>Levels / 100 actions</span><span>Model calls</span></div>
        {aggregate.map((row) => <div className={`final-table-row mode-${row.name.toLowerCase()}`} role="row" key={row.name}>
          <strong>{row.name}</strong><span>{row.progressed}/5</span><span>{row.levels}</span><span>{row.actions}</span><span>{row.efficiency.toFixed(3)}</span><span>{row.calls}</span>
        </div>)}
      </div>
      {directOverride && <div className="final-key-event"><small>Direct QAOA result · seed {directOverride.run.seed}, action {directOverride.turn.turn}</small><strong>Luna proposed {directOverride.turn.proposal?.recommended_candidate_id ?? "another action"}; QAOA selected {directOverride.turn.selected_candidate.candidate_id}; the game reported progress.</strong></div>}
    </section>

    <section className="final-section">
      <p className="section-number">Capability proxy averages</p>
      <h2>Run-level scores aggregated across the five seeds</h2>
      <div className="final-table capability-final-table" role="table" aria-label="Aggregate capability proxy scores">
        <div className="final-table-row final-table-head" role="row"><span>Condition</span><span>G</span><span>A</span><span>Au</span><span>T</span><span>R</span><span>SAI proxy</span></div>
        {capability.map((row) => <div className="final-table-row" role="row" key={row.name}><strong>{row.name}</strong>{row.scores.map((score, index) => <span key={index}>{score.toFixed(3)}</span>)}</div>)}
      </div>
    </section>

    <section className="final-section seed-explorer">
      <div className="seed-heading"><div><p className="section-number">Inspect a matched run</p><h2>Seed-level details</h2></div><div className="seed-tabs" role="group" aria-label="Final run seed">{runs.map((run) => <button className={selected?.seed === run.seed ? "active" : ""} onClick={() => setSelectedSeed(run.seed)} key={run.seed}>Seed {run.seed}</button>)}</div></div>
      {selected && <>
        <p className="final-method">Condition order: {(selected.condition_order ?? []).map((item) => item.replace("qahs_qaoa", "Quantum").replace("qahs_classical", "Classical").replace("llm_only", "LLM")).join(" → ")}. Seed 8 is locked to its first completed run.</p>
        <div className="final-table seed-table" role="table" aria-label={`Seed ${selected.seed} details`}>
          <div className="final-table-row final-table-head" role="row"><span>Condition</span><span>Levels</span><span>First progress</span><span>Actions</span><span>Final state</span><span>Model calls</span><span>QAOA turns</span></div>
          {MODES.map(({ name, key }) => {
            const summary = episodeSummary(selected[key]);
            return <div className="final-table-row" role="row" key={name}><strong>{name}</strong><span>{summary.levels}</span><span>{summary.firstProgress ? `Action ${summary.firstProgress}` : "—"}</span><span>{summary.actions}</span><span>{summary.finalState.replace("_", " ")}</span><span>{summary.calls}</span><span>{summary.qaoaTurns || "—"}</span></div>;
          })}
        </div>
        <Link className="final-detail-link" href={`/results?run=${encodeURIComponent(selected.filename)}`}>Open this run’s full hypotheses, actions, and state changes <span aria-hidden="true">→</span></Link>
      </>}
    </section>
  </main>;
}

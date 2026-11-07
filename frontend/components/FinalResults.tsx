"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type Candidate = { candidate_id?: string; action_type?: string; x?: number | null; y?: number | null; next_action_type?: string | null; next_x?: number | null; next_y?: number | null };
type Observation = { frame?: number[][]; levels_completed?: number; win_levels?: number; state?: string };
type Hypothesis = { predictions?: Record<string, string> };
type PredictionRow = { candidate_id: string; outcome: string };
type Turn = {
  turn: number;
  observation_before: Observation;
  observation_after: Observation;
  selected_candidate: Candidate;
  proposal?: { recommended_candidate_id?: string; hypotheses?: Hypothesis[]; action_candidates?: Candidate[] };
  model_calls?: number;
  reflection?: { beliefs?: { status?: string }[] } | null;
  search_diagnostics?: {
    quantum_used?: boolean; backend?: string; override_permitted?: boolean; physical_outcome?: string; observed_outcome?: string; procedure_continued?: boolean;
    prediction_matrix?: { predictions: PredictionRow[] }[]; hypothesis_probabilities_before?: Record<string, number>; hypothesis_probabilities_after?: Record<string, number>;
    procedure_evidence_status?: string; retired_hypotheses?: string[];
  };
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
type ManifestRow = { filename: string; batch_id?: string | null; model?: string | null; seed?: number | null; game_id?: string | null };

const FINAL_FILES = [
  "20261103T195204Z__v12-structured-world-models__v12-final-lp85__lp85__seed-06__actions-40__gpt-6-luna__comparison.json",
  "20261103T205521Z__v12-structured-world-models__v12-final-lp85__lp85__seed-07__actions-40__gpt-6-luna__comparison.json",
  "20261103T231803Z__v12-structured-world-models__v12-final-lp85__lp85__seed-08__actions-40__gpt-6-luna__comparison.json",
  "20261104T022151Z__v12-structured-world-models__v12-final-lp85__lp85__seed-09__actions-40__gpt-6-luna__comparison.json",
  "20261104T035320Z__v12-structured-world-models__v12-final-lp85__lp85__seed-10__actions-40__gpt-6-luna__comparison.json",
] as const;

const FINAL_SEEDS = [6, 7, 8, 9, 10];
const HAIKU_SERIES = { id: "haiku", label: "Claude Haiku 5.5", batch: "claude-haiku-v12-final-lp85", model: "claude-haiku-5-5", missing: "pending" } as const;
const HAIKU_EXTENDED = { batch: "claude-haiku-v12-lp85-s11-15", model: "claude-haiku-5-5", seeds: [11, 12, 13, 14, 15] } as const;
const LUNA_EXTENDED = { batch: "v12-final2-lp85-retry", model: "gpt-6-luna", seeds: [11, 12, 13, 14, 15] } as const;
const INSPECTION_SEEDS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15] as const;
// Haiku seeds beyond the matched ten (no GPT-6 Luna runs), shown in their own section below.
const HAIKU_ADDITIONAL = { batch: "claude-haiku-v12-lp85-s16plus", model: "claude-haiku-5-5", seeds: Array.from({ length: 20 }, (_, index) => 16 + index) };
const PRIMARY_MODELS = [
  { id: "gpt", label: "GPT-6 Luna", missing: "pending" },
  HAIKU_SERIES,
] as const;
type PrimaryModelId = (typeof PRIMARY_MODELS)[number]["id"];

function ModelToggle({ selected, onSelect }: { selected: PrimaryModelId; onSelect: (model: PrimaryModelId) => void }) {
  return <div className="seed-tabs model-tabs" role="group" aria-label="Result model">
    {PRIMARY_MODELS.map((model) => <button type="button" className={selected === model.id ? "active" : ""} aria-pressed={selected === model.id} onClick={() => onSelect(model.id)} key={model.id}>{model.label}</button>)}
  </div>;
}

function seriesFiles(manifest: ManifestRow[], batch: string, model: string, seeds: readonly number[] = FINAL_SEEDS) {
  const metadata = (row: ManifestRow) => {
    const parts = row.filename.split("__");
    return {
      batch: row.batch_id ?? parts[2],
      game: row.game_id ?? parts[3],
      seed: row.seed ?? Number(parts[4]?.replace("seed-", "")),
      model: row.model ?? parts[6],
    };
  };
  const rows = manifest.filter((row) => {
    const parsed = metadata(row);
    return parsed.batch === batch && parsed.model === model && parsed.game === "lp85"
      && seeds.includes(Number(parsed.seed));
  }).sort((left, right) => left.filename.localeCompare(right.filename));
  const first = new Map<number, string>();
  rows.forEach((row) => {
    const seed = Number(metadata(row).seed);
    if (!first.has(seed)) first.set(seed, row.filename);
  });
  return Array.from(first.values());
}

function loadRun(filename: string) {
  return fetch(`/runs/${filename}`, { cache: "no-store" })
    .then((response) => { if (!response.ok) throw new Error(`Missing final run: ${filename}`); return response.json(); })
    .then((run: FinalRun) => ({ ...run, filename }));
}

function aggregateFor(runs: FinalRun[]) {
  return MODES.map(({ name, key }) => {
    const episodes = runs.map((run) => run[key]);
    const actions = episodes.reduce((sum, episode) => sum + turns(episode).length, 0);
    const completed = episodes.reduce((sum, episode) => sum + levels(episode), 0);
    return {
      name,
      seeds: runs.length,
      progressed: episodes.filter((episode) => levels(episode) > 0).length,
      levels: completed,
      actions,
      efficiency: actions ? 100 * completed / actions : 0,
      calls: episodes.reduce((sum, episode) => sum + modelCalls(episode), 0),
      exploration: episodes.length ? episodes.reduce((sum, episode) => sum + explorationCoverage(episode), 0) / episodes.length : 0,
      conversion: episodes.length ? episodes.reduce((sum, episode) => sum + explorationConversionRate(episode), 0) / episodes.length : 0,
      useful: episodes.length ? episodes.reduce((sum, episode) => sum + externallyUsefulRate(episode), 0) / episodes.length : 0,
      recovery: episodes.length ? episodes.reduce((sum, episode) => sum + adaptiveRecoveryStats(episode).rate, 0) / episodes.length : 0,
    };
  });
}

function capabilityFor(runs: FinalRun[]) {
  return [
    { name: "LLM only", episodes: runs.map((run) => run.without_qahs) },
    { name: "Quantum QAHS", episodes: runs.map((run) => run.with_qahs) },
  ].map(({ name, episodes }) => ({
    name,
    seeds: runs.length,
    scores: [0, 1, 2, 3, 4, 5].map((index) => episodes.length ? episodes.reduce((sum, episode) => sum + capabilityScores(episode)[index], 0) / episodes.length : 0),
  }));
}

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
function posteriorShift(turn: Turn) {
  const before = turn.search_diagnostics?.hypothesis_probabilities_before;
  const after = turn.search_diagnostics?.hypothesis_probabilities_after;
  if (!before || !after) return 0;
  const ids = new Set([...Object.keys(before), ...Object.keys(after)]);
  let distance = 0;
  ids.forEach((id) => { distance += Math.abs(Number(before[id] ?? 0) - Number(after[id] ?? 0)); });
  return distance / 2;
}
function normalizedOutcome(value?: string | null) {
  const outcome = String(value ?? "").trim().toUpperCase().replace(/[\s-]+/g, "_");
  if (["UNCHANGED", "NO_VISIBLE_CHANGE", "NONE"].includes(outcome)) return "NO_CHANGE";
  return outcome;
}
function observedOutcome(turn: Turn) {
  const classified = normalizedOutcome(turn.search_diagnostics?.physical_outcome ?? turn.search_diagnostics?.observed_outcome);
  if (classified && classified !== "UNKNOWN") return classified;
  if (Number(turn.observation_after.levels_completed ?? 0) > Number(turn.observation_before.levels_completed ?? 0)) return "PROGRESS";
  return changed(turn) > 0 ? "STATE_CHANGE" : "NO_CHANGE";
}
function selectedPredictions(turn: Turn) {
  const selectedId = turn.selected_candidate?.candidate_id;
  if (!selectedId) return [];
  const proposalPredictions = (turn.proposal?.hypotheses ?? []).map((hypothesis) => normalizedOutcome(hypothesis.predictions?.[selectedId])).filter(Boolean);
  const matrixPredictions = (turn.search_diagnostics?.prediction_matrix ?? []).map((row) => normalizedOutcome(row.predictions.find((prediction) => prediction.candidate_id === selectedId)?.outcome)).filter(Boolean);
  return proposalPredictions.length ? proposalPredictions : matrixPredictions;
}
function recordedBeliefRevision(turn: Turn) {
  const procedureEvidence = String(turn.search_diagnostics?.procedure_evidence_status ?? "").toLowerCase();
  const reflectedRefutation = turn.reflection?.beliefs?.some((belief) => ["refuted", "contradicted", "weakened"].includes(String(belief.status).toLowerCase()));
  return ["refuted", "contradicted", "weakened"].includes(procedureEvidence)
    || Boolean(turn.search_diagnostics?.retired_hypotheses?.length)
    || Boolean(reflectedRefutation)
    || posteriorShift(turn) >= .05;
}
function meaningfulRefutation(turn: Turn) {
  const predictions = selectedPredictions(turn);
  const distinct = new Set(predictions);
  return distinct.size >= 2 && distinct.has(observedOutcome(turn)) && recordedBeliefRevision(turn);
}
function progressOutcome(turn: Turn) {
  return Number(turn.observation_after.levels_completed ?? 0) > Number(turn.observation_before.levels_completed ?? 0)
    || String(turn.observation_after.state ?? "").toUpperCase() === "WIN";
}
function usefulActionCredit(turn: Turn) { return progressOutcome(turn) ? 1 : meaningfulRefutation(turn) ? .5 : 0; }
function explorationCoverage(episode: Episode) {
  const attempted = new Set(turns(episode).map((turn) => actionKey(turn.selected_candidate)));
  const offered = new Set(turns(episode).flatMap((turn) => turn.proposal?.action_candidates ?? []).map(actionKey));
  const denominator = Math.max(attempted.size, offered.size, 1);
  return attempted.size / denominator;
}
function explorationConversionRate(episode: Episode) {
  const converted = new Map<string, boolean>();
  turns(episode).forEach((turn) => {
    const key = actionKey(turn.selected_candidate);
    converted.set(key, Boolean(converted.get(key)) || usefulActionCredit(turn) > 0);
  });
  return converted.size ? Array.from(converted.values()).filter(Boolean).length / converted.size : 0;
}
function externallyUsefulRate(episode: Episode) {
  const allTurns = turns(episode);
  return allTurns.length ? allTurns.reduce((sum, turn) => sum + usefulActionCredit(turn), 0) / allTurns.length : 0;
}
function adaptiveRecoveryStats(episode: Episode) {
  const allTurns = turns(episode);
  const failures = allTurns.map((turn, index) => ({ turn, index })).filter(({ turn }) => meaningfulRefutation(turn));
  const recovered = failures.filter(({ turn, index }) => {
    const failedAction = actionKey(turn.selected_candidate);
    return allTurns.some((candidate, candidateIndex) => candidateIndex > index
      && actionKey(candidate.selected_candidate) !== failedAction
      && usefulActionCredit(candidate) > 0);
  }).length;
  return { opportunities: failures.length, recovered, rate: failures.length ? recovered / failures.length : 0 };
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
    const recoveredAt = allTurns.findIndex((turn, candidateIndex) => candidateIndex > index && usefulActionCredit(turn) > 0);
    return Math.max(0, 1 - (recoveredAt > index ? recoveredAt - index : budget) / budget);
  }).reduce((sum, score) => sum + score, 0) / transitions.length;
}
function autonomyScore(episode: Episode) {
  return adaptiveRecoveryStats(episode).rate;
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
  const [haikuRuns, setHaikuRuns] = useState<FinalRun[]>([]);
  const [haikuExtended, setHaikuExtended] = useState<FinalRun[]>([]);
  const [lunaExtended, setLunaExtended] = useState<FinalRun[]>([]);
  const [haikuAdditional, setHaikuAdditional] = useState<FinalRun[]>([]);
  const [selectedModel, setSelectedModel] = useState<PrimaryModelId>("gpt");
  const [selectedSeed, setSelectedSeed] = useState(6);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all(FINAL_FILES.map((filename) => fetch(`/runs/${filename}`, { cache: "no-store" })
      .then((response) => { if (!response.ok) throw new Error(`Missing final run: ${filename}`); return response.json(); })
      .then((run: FinalRun) => ({ ...run, filename }))))
      .then((loaded) => setRuns(loaded.sort((left, right) => left.seed - right.seed)))
      .catch((reason: Error) => setError(reason.message));
    fetch("/runs/index.json", { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : []))
      .then((manifest: ManifestRow[]) => Promise.all([
        Promise.all(seriesFiles(manifest, HAIKU_SERIES.batch, HAIKU_SERIES.model).map(loadRun)),
        Promise.all(seriesFiles(manifest, HAIKU_EXTENDED.batch, HAIKU_EXTENDED.model, HAIKU_EXTENDED.seeds).map(loadRun)),
        Promise.all(seriesFiles(manifest, LUNA_EXTENDED.batch, LUNA_EXTENDED.model, LUNA_EXTENDED.seeds).map(loadRun)),
      ]))
      .then(([haiku, extendedHaiku, extendedLuna]) => {
        setHaikuRuns(haiku.sort((left, right) => left.seed - right.seed));
        setHaikuExtended(extendedHaiku.sort((left, right) => left.seed - right.seed));
        setLunaExtended(extendedLuna.sort((left, right) => left.seed - right.seed));
      })
      .catch(() => { setHaikuRuns([]); setHaikuExtended([]); setLunaExtended([]); });
    fetch("/runs/index.json", { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : []))
      .then((manifest: ManifestRow[]) => Promise.all(seriesFiles(manifest, HAIKU_ADDITIONAL.batch, HAIKU_ADDITIONAL.model, HAIKU_ADDITIONAL.seeds).map(loadRun)))
      .then((loaded) => setHaikuAdditional(loaded.sort((left, right) => left.seed - right.seed)))
      .catch(() => setHaikuAdditional([]));
  }, []);

  const allLunaRuns = useMemo(() => [...runs, ...lunaExtended].sort((left, right) => left.seed - right.seed), [runs, lunaExtended]);
  const allHaikuRuns = useMemo(() => [...haikuRuns, ...haikuExtended].sort((left, right) => left.seed - right.seed), [haikuRuns, haikuExtended]);
  const runsByModel = useMemo(() => ({ gpt: allLunaRuns, haiku: allHaikuRuns }) as Record<PrimaryModelId, FinalRun[]>, [allLunaRuns, allHaikuRuns]);
  const haikuAllSeeds = useMemo(() => [...allHaikuRuns, ...haikuAdditional].sort((left, right) => left.seed - right.seed), [allHaikuRuns, haikuAdditional]);
  const additionalTotals = useMemo(() => [
    { label: "Seeds 6–15 (matched)", runs: allHaikuRuns },
    { label: `Seeds 16–${Math.max(16, ...haikuAdditional.map((run) => run.seed))}`, runs: haikuAdditional },
    { label: `All ${haikuAllSeeds.length} seeds`, runs: haikuAllSeeds },
  ].map(({ label, runs: group }) => ({ label, seeds: group.length, progressed: MODES.map(({ key }) => group.filter((run) => levels(run[key]) > 0).length) })), [allHaikuRuns, haikuAdditional, haikuAllSeeds]);
  const selectedAggregate = useMemo(() => aggregateFor(runsByModel[selectedModel]), [runsByModel, selectedModel]);
  const selectedModelDefinition = PRIMARY_MODELS.find((model) => model.id === selectedModel) ?? PRIMARY_MODELS[0];
  const capability = useMemo(() => PRIMARY_MODELS.flatMap(({ id, label, missing }) => capabilityFor(runsByModel[id]).map((row) => ({ ...row, model: label, missing }))), [runsByModel]);
  const inspectionRuns = (selectedModel === "gpt" ? allLunaRuns : allHaikuRuns).filter((run) => INSPECTION_SEEDS.includes(run.seed as (typeof INSPECTION_SEEDS)[number]));
  const selected = inspectionRuns.find((run) => run.seed === selectedSeed) ?? inspectionRuns[0];
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
      <h1>No selector consistently outperformed the model alone on LP85</h1>
      <blockquote>Across ten matched LP85 seeds, Quantum QAHS produced progress on two Luna seeds, compared with one seed each for Luna alone and Classical QAHS (sign test p = 1.0). Across 26 Haiku seeds, Haiku alone and Classical QAHS each progressed on 12 seeds and Quantum QAHS on 10.</blockquote>
      <div className="final-facts"><span><strong>10</strong> matched seeds</span><span><strong>40</strong> action maximum</span><span><strong>30</strong> episodes per model</span><span><strong>V12</strong> frozen harness</span></div>
    </header>

    <section className="final-section">
      <div className="seed-heading"><div><p className="section-number">Aggregate outcome</p><h2>Matched ten-seed comparison</h2></div><ModelToggle selected={selectedModel} onSelect={setSelectedModel} /></div>
      <div className="final-table" role="table" aria-label={`${selectedModelDefinition.label} aggregate final evaluation`}>
        <div className="final-table-row final-table-head" role="row"><span>Condition</span><span>Progressing seeds</span><span>Levels</span><span>Actions</span><span>Levels / 100 actions</span><span>Model calls</span></div>
        {selectedAggregate.map((row) => <div className={`final-table-row mode-${row.name.toLowerCase()}`} role="row" key={`${selectedModel}-${row.name}`}>
          <strong>{row.name}</strong><span>{row.progressed}/{row.seeds}</span><span>{row.levels}</span><span>{row.actions}</span><span>{row.efficiency.toFixed(3)}</span><span>{row.calls}</span>
        </div>)}
      </div>
      <p className="final-method">Showing {selectedModelDefinition.label} under the matched V12 LP85 configuration: seeds 6–15, 40 actions, and medium effort. Use the model toggle to view the other model without combining unlike rows. GPT-6 Luna seed 16 was also run; it completed no level in any condition and is not shown.</p>
      {selectedModel === "gpt" && directOverride && <div className="final-key-event"><small>Direct QAOA result · seed {directOverride.run.seed}, action {directOverride.turn.turn}</small><strong>Luna proposed {directOverride.turn.proposal?.recommended_candidate_id ?? "another action"}; QAOA selected {directOverride.turn.selected_candidate.candidate_id}; the game reported progress.</strong></div>}
    </section>

    <section className="final-section">
      <div className="seed-heading"><div><p className="section-number">Search behavior diagnostics</p><h2>Breadth is separated from productive evidence</h2></div><ModelToggle selected={selectedModel} onSelect={setSelectedModel} /></div>
      <div className="final-table behavior-single-model" role="table" aria-label={`${selectedModelDefinition.label} search behavior diagnostics`}>
        <div className="final-table-row final-table-head" role="row"><span>Condition</span><span>Exploration breadth</span><span>Evidence conversion</span><span>Externally useful</span><span>Adaptive recovery</span></div>
        {selectedAggregate.map((row) => <div className={`final-table-row mode-${row.name.toLowerCase()}`} role="row" key={`behavior-${selectedModel}-${row.name}`}>
          <strong>{row.name}</strong><span>{(100 * row.exploration).toFixed(1)}%</span><span>{(100 * row.conversion).toFixed(1)}%</span><span>{(100 * row.useful).toFixed(1)}%</span><span>{(100 * row.recovery).toFixed(1)}%</span>
        </div>)}
      </div>
      <p className="final-method">Exploration breadth is descriptive only. Evidence conversion counts distinct actions that produced progress or a correctly observed, discriminating refutation. Externally useful actions give full credit to progress and half credit to such refutations. Adaptive recovery measures productive strategy changes after refutation.</p>
    </section>

    <section className="final-section">
      <p className="section-number">Capability proxy averages</p>
      <h2>Run-level scores aggregated across ten seeds</h2>
      <div className="final-table capability-final-table with-model" role="table" aria-label="Aggregate capability proxy scores">
        <div className="final-table-row final-table-head" role="row"><span>Condition</span><span>Model</span><span>G</span><span>A</span><span>Au</span><span>T</span><span>R</span><span>SAI proxy</span></div>
        {capability.map((row) => <div className="final-table-row" role="row" key={`${row.model}-${row.name}`}><strong>{row.name}</strong><span>{row.model}</span>{row.seeds ? row.scores.map((score, index) => <span key={index}>{score.toFixed(3)}</span>) : [0, 1, 2, 3, 4, 5].map((index) => <span key={index}>{index ? "—" : row.missing}</span>)}</div>)}
      </div>
    </section>

    <section className="final-section seed-explorer">
      <div className="seed-heading"><div><p className="section-number">Inspect a matched run</p><h2>Seed-level details</h2></div><ModelToggle selected={selectedModel} onSelect={setSelectedModel} /></div>
      <div className="seed-tabs seed-run-tabs" role="group" aria-label={`${selectedModelDefinition.label} final run seed`}>{INSPECTION_SEEDS.map((seed) => {
        const available = inspectionRuns.some((run) => run.seed === seed);
        return <button type="button" className={selectedSeed === seed ? "active" : ""} disabled={!available} onClick={() => setSelectedSeed(seed)} key={seed}>Seed {seed}</button>;
      })}</div>
      {selected && <>
        <p className="final-method">{selectedModelDefinition.label} · Condition order: {(selected.condition_order ?? []).map((item) => item.replace("qahs_qaoa", "Quantum").replace("qahs_classical", "Classical").replace("llm_only", "LLM")).join(" → ")}.{selectedModel === "gpt" && selected.seed === 8 ? " Seed 8 is locked to its first completed run; a repeat run completed no level in any condition." : ""}</p>
        <div className="final-table seed-table" role="table" aria-label={`${selectedModelDefinition.label} seed ${selected.seed} details`}>
          <div className="final-table-row final-table-head" role="row"><span>Condition</span><span>Levels</span><span>First progress</span><span>Actions</span><span>Final state</span><span>Model calls</span><span>QAOA turns</span></div>
          {MODES.map(({ name, key }) => {
            const run = selected;
            const summary = episodeSummary(run[key]);
            return <div className="final-table-row" role="row" key={`${selectedModel}-${name}`}><strong>{name}</strong><span>{summary.levels}</span><span>{summary.firstProgress ? `Action ${summary.firstProgress}` : "—"}</span><span>{summary.actions}</span><span>{summary.finalState.replace("_", " ")}</span><span>{summary.calls}</span><span>{summary.qaoaTurns || "—"}</span></div>;
          })}
        </div>
        <Link className="final-detail-link" href={`/results?run=${encodeURIComponent(selected.filename)}`}>Open the {selectedModelDefinition.label} run’s full hypotheses, actions, and state changes <span aria-hidden="true">→</span></Link>
      </>}
    </section>

    {haikuAdditional.length > 0 && <section className="final-section">
      <p className="section-number">Additional Claude Haiku 5.5 seeds</p>
      <h2>Claude Haiku 5.5 beyond the matched ten seeds</h2>
      <p className="final-method">Seeds 16 onward use batch {HAIKU_ADDITIONAL.batch} with the same V12 harness, 40-action budget and medium effort. GPT-6 Luna was not run on these seeds, so they extend the Haiku sample only. Each cell is levels completed / actions taken; “lost” means the game ended before the action budget.</p>
      <div className="final-table with-model extended-table" role="table" aria-label="Additional Claude Haiku 5.5 seeds">
        <div className="final-table-row final-table-head" role="row"><span>Seed</span><span>Set</span>{MODES.map(({ name }) => <span key={name}>{name}</span>)}<span></span><span></span></div>
        {haikuAdditional.map((run) => <div className="final-table-row" role="row" key={run.filename}><strong>Seed {run.seed}</strong><span>additional (16+)</span>
          {MODES.map(({ key }) => { const summary = episodeSummary(run[key]); return <span key={key}>{summary.levels} / {summary.actions}{summary.finalState === "GAME_OVER" ? " lost" : ""}</span>; })}<span></span><span></span></div>)}
        {additionalTotals.map((row) => <div className="final-table-row final-table-total" role="row" key={row.label}><strong>{row.label}</strong><span>seeds with a level</span>
          {row.progressed.map((count, index) => <span key={index}>{count}/{row.seeds}</span>)}<span></span><span></span></div>)}
      </div>
    </section>}
  </main>;
}

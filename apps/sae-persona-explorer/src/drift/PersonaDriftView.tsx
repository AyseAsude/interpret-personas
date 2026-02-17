import { useEffect, useMemo, useState } from "react";

import type {
  ConversationMessage,
  DriftTransition,
  PersonaDriftBundle,
  TransitionFeature,
} from "./types";
import {
  clamp,
  formatPct,
  formatSigned,
  interpolateMultiStopHexColor,
  shortenText,
  textColorForBackground,
  tokenDisplay,
} from "./utils";
import { sanitizeExternalUrl } from "../utils";

const DEFAULT_RUNS_MANIFEST_URL = `${import.meta.env.BASE_URL}persona-drift/runs_manifest.json`;

const RUNS_MANIFEST_URL = import.meta.env.VITE_DRIFT_RUNS_MANIFEST_URL ?? DEFAULT_RUNS_MANIFEST_URL;
const PERSONA_EXPLORER_URL = import.meta.env.VITE_PERSONA_EXPLORER_URL ?? "#/";
const GITHUB_ALLOWED_HOSTS = new Set<string>(["github.com", "www.github.com"]);
const GITHUB_PROFILE_URL = sanitizeExternalUrl(
  import.meta.env.VITE_GITHUB_PROFILE_URL ?? "https://github.com/AyseAsude/interpret-personas",
  GITHUB_ALLOWED_HOSTS,
);
const DRIFT_BASE_URL = `${import.meta.env.BASE_URL}persona-drift/`;
const DRIFT_NEURONPEDIA_ID = import.meta.env.VITE_DRIFT_NEURONPEDIA_ID ?? "";
const NEURONPEDIA_PUBLIC_ROOT = "https://neuronpedia.org";
const MODEL_NAME_OVERRIDES: Record<string, string> = {
  "gemma-3-27b-it": "Gemma 3 27B IT",
  "qwen-3-32b": "Qwen 3 32B",
  "llama-3.3-70b": "Llama 3.3 70B",
};

type RunSource = {
  id: string;
  label: string;
  uiBundleUrl: string;
  conversationUrl: string | null;
  conversationName: string;
  isSensitive: boolean;
};

type AssistantTurnRow = {
  assistant_turn: number;
  message_index: number;
  content: string;
};

type TokenRow = {
  token_idx: number;
  token: string;
  u_k: number;
  u_k_norm: number;
};

// ColorBrewer-style RdBu diverging stops (red=away side, blue=toward side).
const TOKEN_DIVERGING_STOPS: Array<[number, number, number]> = [
  [103, 0, 31],
  [178, 24, 43],
  [214, 96, 77],
  [244, 165, 130],
  [253, 219, 199],
  [247, 247, 247],
  [209, 229, 240],
  [146, 197, 222],
  [67, 147, 195],
  [33, 102, 172],
  [5, 48, 97],
];

function normalizeConversationPayload(payload: unknown): ConversationMessage[] {
  if (Array.isArray(payload)) {
    return payload
      .filter((item) => typeof item === "object" && item !== null)
      .map((item) => {
        const row = item as Record<string, unknown>;
        return {
          role: String(row.role ?? "unknown"),
          content: String(row.content ?? ""),
        };
      });
  }

  if (typeof payload === "object" && payload !== null && "conversation" in payload) {
    const rows = (payload as { conversation?: unknown }).conversation;
    return normalizeConversationPayload(rows ?? []);
  }

  return [];
}

function validateBundle(payload: unknown): PersonaDriftBundle {
  const candidate = payload as PersonaDriftBundle;
  if (!candidate?.q1?.turn_scores || !candidate?.q2?.transitions || !candidate?.q3?.token_localization) {
    throw new Error("ui_bundle.json is missing q1/q2/q3 sections.");
  }
  return candidate;
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function resolveRunAssetUrl(rawUrl: string): string {
  const trimmed = rawUrl.trim();
  if (!trimmed) {
    return trimmed;
  }
  if (/^(https?:)?\/\//i.test(trimmed)) {
    return trimmed;
  }
  if (trimmed.startsWith("/runs/")) {
    return `${DRIFT_BASE_URL}${trimmed.slice(1)}`;
  }
  if (trimmed.startsWith("runs/")) {
    return `${DRIFT_BASE_URL}${trimmed}`;
  }
  if (trimmed.startsWith("/")) {
    return `${import.meta.env.BASE_URL}${trimmed.slice(1)}`;
  }
  return trimmed;
}

function normalizeNonEmptyString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function isSensitiveConversationName(name: string | null): boolean {
  if (!name) {
    return false;
  }
  const normalized = name.trim().toLowerCase();
  return normalized === "self-harm" || normalized === "self harm";
}

function prettifyModelSlug(slug: string): string {
  const override = MODEL_NAME_OVERRIDES[slug];
  if (override) {
    return override;
  }
  return slug
    .split("-")
    .map((part) => {
      if (/^\d+b$/i.test(part)) {
        return part.toUpperCase();
      }
      if (part === "it") {
        return "IT";
      }
      if (/^\d+(\.\d+)?$/.test(part)) {
        return part;
      }
      return `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`;
    })
    .join(" ");
}

function prettifyModelName(modelName: string): string {
  const slug = modelName.trim().split("/").filter(Boolean).pop() ?? modelName.trim();
  return prettifyModelSlug(slug);
}

function prettifySaeId(saeId: string): string {
  const match = saeId.match(/^layer_(\d+)_width_([^_]+)_l0_(.+)$/);
  if (!match) {
    return saeId.replace(/_/g, " ");
  }
  return `Layer ${match[1]} · Width ${match[2]} · L0 ${match[3]}`;
}

function deriveNeuronpediaId(runMeta: PersonaDriftBundle["run_meta"] | null): string | null {
  if (DRIFT_NEURONPEDIA_ID.trim()) {
    return DRIFT_NEURONPEDIA_ID.trim();
  }
  if (!runMeta) {
    return null;
  }
  const modelSlug = runMeta.model_name.trim().split("/").filter(Boolean).pop() ?? "";
  const saeMatch = runMeta.sae_id.match(/^layer_(\d+)_width_([^_]+)_l0_[^_]+$/);
  if (!modelSlug || !saeMatch) {
    return null;
  }
  const layer = Number.parseInt(saeMatch[1], 10);
  if (!Number.isFinite(layer) || layer < 0) {
    return null;
  }
  const width = saeMatch[2].toLowerCase();
  return `${modelSlug}/${layer}-gemmascope-2-res-${width}`;
}

function neuronpediaFeatureUrl(
  neuronpediaId: string | null,
  featureId: number,
): string | null {
  if (!neuronpediaId || !Number.isInteger(featureId) || featureId < 0) {
    return null;
  }
  const encodedId = neuronpediaId
    .split("/")
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  if (!encodedId) {
    return null;
  }
  return `${NEURONPEDIA_PUBLIC_ROOT}/${encodedId}/${featureId}`;
}

function normalizeRunsManifest(payload: unknown): { runs: RunSource[]; defaultRunId: string | null } {
  let items: unknown[] = [];
  let defaultRunId: string | null = null;

  if (Array.isArray(payload)) {
    items = payload;
  } else if (typeof payload === "object" && payload !== null) {
    const root = payload as { runs?: unknown; default_run_id?: unknown; defaultRunId?: unknown };
    if (Array.isArray(root.runs)) {
      items = root.runs;
    }
    if (typeof root.default_run_id === "string") {
      defaultRunId = root.default_run_id;
    } else if (typeof root.defaultRunId === "string") {
      defaultRunId = root.defaultRunId;
    }
  }

  const out: RunSource[] = [];
  items.forEach((item, index) => {
    if (typeof item !== "object" || item === null) {
      return;
    }

    const row = item as {
      id?: unknown;
      run_id?: unknown;
      runId?: unknown;
      label?: unknown;
      run_name?: unknown;
      runName?: unknown;
      conversation_name?: unknown;
      conversationName?: unknown;
      ui_bundle_url?: unknown;
      uiBundleUrl?: unknown;
      conversation_url?: unknown;
      conversationUrl?: unknown;
      sensitive?: unknown;
      is_sensitive?: unknown;
      isSensitive?: unknown;
    };

    const uiBundleUrlRaw = row.ui_bundle_url ?? row.uiBundleUrl;
    if (typeof uiBundleUrlRaw !== "string" || !uiBundleUrlRaw.trim()) {
      return;
    }

    const idRaw = row.id ?? row.run_id ?? row.runId;
    const id = typeof idRaw === "string" && idRaw.trim() ? idRaw : `run-${index}`;
    const labelRaw = row.label ?? row.run_name ?? row.runName;
    const label = typeof labelRaw === "string" && labelRaw.trim() ? labelRaw : id;
    const conversationNameRaw = row.conversation_name ?? row.conversationName;
    const conversationName = normalizeNonEmptyString(conversationNameRaw);
    if (!conversationName) {
      return;
    }

    const conversationUrlRaw = row.conversation_url ?? row.conversationUrl;
    const resolvedUiBundleUrl = resolveRunAssetUrl(uiBundleUrlRaw);
    const resolvedConversationUrl =
      typeof conversationUrlRaw === "string" && conversationUrlRaw.trim()
        ? resolveRunAssetUrl(conversationUrlRaw)
        : null;
    const isSensitiveFromManifest =
      row.sensitive === true ||
      row.is_sensitive === true ||
      row.isSensitive === true;

    out.push({
      id,
      label,
      uiBundleUrl: resolvedUiBundleUrl,
      conversationUrl: resolvedConversationUrl,
      conversationName,
      isSensitive: isSensitiveFromManifest || isSensitiveConversationName(conversationName),
    });
  });

  return { runs: out, defaultRunId };
}

function transitionIndexForTurn(transitions: DriftTransition[], turn: number): number {
  if (!transitions.length) {
    return -1;
  }

  const incoming = transitions.findIndex((row) => row.turn === turn);
  if (incoming >= 0) {
    return incoming;
  }

  const outgoing = transitions.findIndex((row) => row.prev_turn === turn);
  if (outgoing >= 0) {
    return outgoing;
  }

  return 0;
}

function featureTableTitle(direction: "activated" | "deactivated"): string {
  if (direction === "activated") {
    return "Activated features";
  }
  return "Deactivated features";
}

type FeatureTableProps = {
  direction: "activated" | "deactivated";
  features: TransitionFeature[];
  neuronpediaId: string | null;
};

function FeatureTable({ direction, features, neuronpediaId }: FeatureTableProps) {
  const shareColor = direction === "activated" ? "var(--accent-2)" : "var(--accent-1)";

  return (
    <section className="feature-table-card">
      <header>
        <h4>{featureTableTitle(direction)}</h4>
        <span>{features.length} rows</span>
      </header>
      <div className="feature-table-scroll">
        <table>
          <thead>
            <tr>
              <th>Feature</th>
              <th>Delta</th>
              <th>Contrib</th>
              <th>Share of change</th>
              <th>Share of away mass</th>
              <th>Role profile</th>
            </tr>
          </thead>
          <tbody>
            {features.map((row, idx) => {
              const featureLink = neuronpediaFeatureUrl(neuronpediaId, row.feature_idx);
              return (
                <tr key={`${direction}-${row.feature_idx}-${idx}`}>
                  <td>
                    {featureLink ? (
                      <a
                        className="feature-id feature-id-link"
                        href={featureLink}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        #{row.feature_idx}
                      </a>
                    ) : (
                      <div className="feature-id">#{row.feature_idx}</div>
                    )}
                    <div className={`effect-pill ${row.effect}`}>{row.effect === "away_from_assistant" ? "away" : "toward"}</div>
                  </td>
                  <td className="numeric">{formatSigned(row.delta_x)}</td>
                  <td className="numeric">{formatSigned(row.contrib)}</td>
                  <td>
                    <div className="share-cell">
                      <span>{formatPct(row.share_of_change)}</span>
                      <div className="share-bar-track">
                        <div
                          className="share-bar"
                          style={{ width: `${Math.max(row.share_of_change * 100, row.share_of_change > 0 ? 2 : 0)}%`, backgroundColor: shareColor }}
                        />
                      </div>
                    </div>
                  </td>
                  <td>
                    <div className="share-cell">
                      <span>{formatPct(row.share_of_away)}</span>
                      <div className="share-bar-track">
                        <div
                          className="share-bar"
                          style={{ width: `${Math.max(row.share_of_away * 100, row.share_of_away > 0 ? 2 : 0)}%` }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="role-profile">{row.role_profile}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function PersonaDriftView() {
  const [bundle, setBundle] = useState<PersonaDriftBundle | null>(null);
  const [conversation, setConversation] = useState<ConversationMessage[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [warning, setWarning] = useState<string>("");
  const [selectedTransitionIndex, setSelectedTransitionIndex] = useState<number>(0);
  const [selectedTurn, setSelectedTurn] = useState<number | null>(null);
  const [runOptions, setRunOptions] = useState<RunSource[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");

  useEffect(() => {
    let cancelled = false;

    async function loadRunsManifest() {
      setLoading(true);
      setError("");
      setWarning("");

      try {
        const manifestResponse = await fetch(RUNS_MANIFEST_URL);
        if (!manifestResponse.ok) {
          throw new Error(`Could not load runs_manifest.json (status ${manifestResponse.status}).`);
        }
        const parsed = normalizeRunsManifest(await manifestResponse.json());
        if (!parsed.runs.length) {
          throw new Error(
            "No valid runs found in runs_manifest.json. Each run must include conversation_name and ui_bundle_url.",
          );
        }

        if (cancelled) {
          return;
        }

        setRunOptions(parsed.runs);
        setSelectedRunId((prev) => {
          if (parsed.runs.some((run) => run.id === prev)) {
            return prev;
          }
          if (parsed.defaultRunId && parsed.runs.some((run) => run.id === parsed.defaultRunId)) {
            return parsed.defaultRunId;
          }
          return parsed.runs[0]?.id ?? "";
        });
      } catch (manifestError) {
        if (cancelled) {
          return;
        }
        setRunOptions([]);
        setSelectedRunId("");
        setBundle(null);
        setConversation([]);
        setLoading(false);
        setError(`Manifest load failed at ${RUNS_MANIFEST_URL}: ${toErrorMessage(manifestError)}`);
      }
    }

    void loadRunsManifest();

    return () => {
      cancelled = true;
    };
  }, []);

  const activeRun = useMemo(() => {
    if (!runOptions.length) {
      return null;
    }
    return runOptions.find((run) => run.id === selectedRunId) ?? runOptions[0] ?? null;
  }, [runOptions, selectedRunId]);
  const isSensitiveRun = activeRun?.isSensitive ?? false;
  const neuronpediaId = useMemo(
    () => deriveNeuronpediaId(bundle?.run_meta ?? null),
    [bundle?.run_meta],
  );
  const runMetaLabels = useMemo(
    () => ({
      model: bundle?.run_meta.model_name ? prettifyModelName(bundle.run_meta.model_name) : "-",
      sae: bundle?.run_meta.sae_id ? prettifySaeId(bundle.run_meta.sae_id) : "-",
    }),
    [bundle?.run_meta.model_name, bundle?.run_meta.sae_id],
  );

  useEffect(() => {
    let cancelled = false;

    async function loadRunData() {
      if (!activeRun) {
        return;
      }

      setLoading(true);
      setError("");
      setWarning("");

      const warnings: string[] = [];

      try {
        const uiResponse = await fetch(activeRun.uiBundleUrl);
        if (!uiResponse.ok) {
          throw new Error(`Could not load the selected run (status ${uiResponse.status}).`);
        }
        const bundlePayload = validateBundle(await uiResponse.json());

        let conversationPayload: ConversationMessage[] = [];
        if (activeRun.conversationUrl) {
          try {
            const conversationResponse = await fetch(activeRun.conversationUrl);
            if (!conversationResponse.ok) {
              throw new Error(`status ${conversationResponse.status}`);
            }
            conversationPayload = normalizeConversationPayload(await conversationResponse.json());
          } catch {
            warnings.push("Conversation text for this run could not be loaded.");
          }
        }

        if (cancelled) {
          return;
        }

        setBundle(bundlePayload);
        setConversation(conversationPayload);

        const firstTurn = bundlePayload.q1.turn_scores[0]?.assistant_turn ?? null;
        if (firstTurn !== null) {
          setSelectedTurn(firstTurn);
          setSelectedTransitionIndex(transitionIndexForTurn(bundlePayload.q2.transitions, firstTurn));
        } else {
          setSelectedTurn(null);
          setSelectedTransitionIndex(-1);
        }

        if (warnings.length > 0) {
          setWarning(warnings.join(" | "));
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(toErrorMessage(loadError));
          setBundle(null);
          setConversation([]);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadRunData();

    return () => {
      cancelled = true;
    };
  }, [activeRun]);

  const turnScores = bundle?.q1.turn_scores ?? [];
  const transitions = bundle?.q2.transitions ?? [];
  const tokenTurns = bundle?.q3.token_localization ?? [];

  const safeTransitionIndex = transitions.length
    ? clamp(selectedTransitionIndex, 0, transitions.length - 1)
    : -1;

  const selectedTransition: DriftTransition | null =
    safeTransitionIndex >= 0 ? transitions[safeTransitionIndex] : null;

  const selectedAssistantTurn = selectedTurn ?? turnScores[0]?.assistant_turn ?? null;

  const selectedTokenTurn = useMemo(() => {
    if (selectedAssistantTurn === null) {
      return null;
    }
    return tokenTurns.find((row) => row.assistant_turn === selectedAssistantTurn) ?? null;
  }, [selectedAssistantTurn, tokenTurns]);

  const assistantTurnRows = useMemo<AssistantTurnRow[]>(() => {
    const out: AssistantTurnRow[] = [];
    let assistantTurn = 0;
    conversation.forEach((message, messageIndex) => {
      if (message.role !== "assistant") {
        return;
      }
      out.push({
        assistant_turn: assistantTurn,
        message_index: messageIndex,
        content: message.content,
      });
      assistantTurn += 1;
    });
    return out;
  }, [conversation]);

  const scoreByTurn = useMemo(() => {
    const map = new Map<number, number>();
    turnScores.forEach((row) => map.set(row.assistant_turn, row.score));
    return map;
  }, [turnScores]);

  const tokenRows = useMemo<TokenRow[]>(() => {
    if (!selectedTokenTurn) {
      return [];
    }

    const n = Math.min(
      selectedTokenTurn.tokens.length,
      selectedTokenTurn.u_k.length,
      selectedTokenTurn.u_k_norm.length,
    );

    return Array.from({ length: n }, (_, tokenIdx) => ({
      token_idx: tokenIdx,
      token: selectedTokenTurn.tokens[tokenIdx],
      u_k: selectedTokenTurn.u_k[tokenIdx],
      u_k_norm: selectedTokenTurn.u_k_norm[tokenIdx],
    }));
  }, [selectedTokenTurn]);

  const tokenLegendGradient = useMemo(() => {
    if (!TOKEN_DIVERGING_STOPS.length) {
      return "linear-gradient(90deg, #f3f3f3, #f3f3f3)";
    }

    const last = Math.max(TOKEN_DIVERGING_STOPS.length - 1, 1);
    const stops = TOKEN_DIVERGING_STOPS.map(([r, g, b], idx) => {
      const pct = (idx / last) * 100;
      return `rgb(${r}, ${g}, ${b}) ${pct.toFixed(1)}%`;
    }).join(", ");

    return `linear-gradient(90deg, ${stops})`;
  }, []);

  const chart = useMemo(() => {
    if (!turnScores.length) {
      return null;
    }

    const width = 940;
    const height = 320;
    const padLeft = 56;
    const padRight = 20;
    const padTop = 20;
    const padBottom = 40;

    const scores = turnScores.map((row) => row.score);
    const minScore = Math.min(...scores, 0);
    const maxScore = Math.max(...scores, 0);
    const span = Math.max(maxScore - minScore, 1e-6);

    const minTurn = turnScores[0].assistant_turn;
    const maxTurn = turnScores[turnScores.length - 1].assistant_turn;
    const turnSpan = Math.max(maxTurn - minTurn, 1);

    const xForTurn = (turn: number) => {
      return padLeft + ((turn - minTurn) / turnSpan) * (width - padLeft - padRight);
    };

    const yForScore = (score: number) => {
      return padTop + ((maxScore - score) / span) * (height - padTop - padBottom);
    };

    const points = turnScores.map((row) => ({
      turn: row.assistant_turn,
      score: row.score,
      x: xForTurn(row.assistant_turn),
      y: yForScore(row.score),
    }));

    const path = points
      .map((point, idx) => `${idx === 0 ? "M" : "L"}${point.x.toFixed(2)},${point.y.toFixed(2)}`)
      .join(" ");

    const yTicks = Array.from({ length: 5 }, (_, idx) => {
      const score = minScore + ((maxScore - minScore) * idx) / 4;
      return {
        score,
        y: yForScore(score),
      };
    });

    return {
      width,
      height,
      padLeft,
      padRight,
      padTop,
      padBottom,
      points,
      path,
      yTicks,
      yForScore,
    };
  }, [turnScores]);

  function pickTransition(index: number) {
    if (index < 0 || index >= transitions.length) {
      return;
    }
    const transition = transitions[index];
    setSelectedTransitionIndex(index);
    setSelectedTurn(transition.turn);
  }

  function pickTurn(turn: number) {
    setSelectedTurn(turn);
    setSelectedTransitionIndex(transitionIndexForTurn(transitions, turn));
  }

  return (
    <div className="drift-view">
      <div className="drift-app-shell">
        <header className="topbar">
          <div className="topbar-main">
            <h1>Persona Drift Explorer</h1>
            <div className="topbar-link-row">
              <a className="topbar-nav-link" href={PERSONA_EXPLORER_URL}>
                Back to SAE Persona Explorer
              </a>
              {GITHUB_PROFILE_URL && (
                <a
                  className="topbar-icon-link"
                  href={GITHUB_PROFILE_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Open GitHub repository"
                  title="Open GitHub repository"
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 .5C5.73.5.75 5.48.75 11.75c0 5.02 3.24 9.28 7.74 10.78.57.1.78-.25.78-.56 0-.27-.01-1.17-.02-2.12-3.15.68-3.82-1.34-3.82-1.34-.51-1.3-1.26-1.64-1.26-1.64-1.03-.71.08-.7.08-.7 1.13.08 1.73 1.17 1.73 1.17 1.01 1.72 2.65 1.22 3.29.93.1-.73.39-1.22.71-1.5-2.51-.29-5.15-1.26-5.15-5.61 0-1.24.44-2.25 1.17-3.05-.12-.29-.51-1.45.11-3.02 0 0 .95-.3 3.1 1.16a10.8 10.8 0 0 1 5.64 0c2.15-1.46 3.1-1.16 3.1-1.16.62 1.57.23 2.73.11 3.02.73.8 1.17 1.81 1.17 3.05 0 4.36-2.65 5.31-5.17 5.59.4.35.76 1.02.76 2.07 0 1.5-.01 2.7-.01 3.06 0 .31.2.67.79.56 4.49-1.5 7.73-5.76 7.73-10.78C23.25 5.48 18.27.5 12 .5Z" />
                  </svg>
                </a>
              )}
            </div>
          </div>
          <div className="dataset-chip">
            <div className="dataset-item">
              <span>Model</span>
              <strong>{runMetaLabels.model}</strong>
            </div>
            <div className="dataset-item">
              <span>SAE</span>
              <strong>{runMetaLabels.sae}</strong>
            </div>
          </div>
        </header>

        {(warning || error) && (
          <section className="status-row">
            {warning && <p className="warning-text">{warning}</p>}
            {error && <p className="error-text">{error}</p>}
          </section>
        )}

        <main className="drift-layout">
          <aside className="left-column card">
            <h2>Conversation Selection</h2>
            <p>
              Choose a conversation run to explore.
            </p>
            {runOptions.length > 1 && (
              <div className="run-picker-row">
                <label htmlFor="run-selector">Conversation</label>
                <div className="run-selector-wrap">
                  <select
                    id="run-selector"
                    className={`run-selector ${isSensitiveRun ? "sensitive" : ""}`.trim()}
                    value={selectedRunId}
                    onChange={(event) => setSelectedRunId(event.target.value)}
                  >
                    {runOptions.map((run) => {
                      const displayName = run.conversationName;
                      return (
                        <option key={run.id} value={run.id}>
                          {run.isSensitive ? `! ${displayName}` : displayName}
                        </option>
                      );
                    })}
                  </select>
                  {isSensitiveRun && (
                    <span className="sensitive-flag-wrap" aria-hidden="true">
                      <span className="sensitive-flag">!</span>
                      <span className="sensitive-flag-tooltip">
                        Sensitive content, contains self-harm
                      </span>
                    </span>
                  )}
                </div>
              </div>
            )}

            <h3>Assistant turns</h3>
            <div className="turn-list">
              {loading && <p>Loading run data…</p>}
              {!loading && turnScores.length === 0 && <p>No turn scores found.</p>}
              {!loading &&
                turnScores.map((row) => {
                  const prevScore = scoreByTurn.get(row.assistant_turn - 1);
                  const delta = prevScore === undefined ? null : row.score - prevScore;
                  const snippet =
                    assistantTurnRows.find((entry) => entry.assistant_turn === row.assistant_turn)?.content ?? "";
                  return (
                    <button
                      key={`turn-${row.assistant_turn}`}
                      type="button"
                      className={`turn-button ${selectedAssistantTurn === row.assistant_turn ? "active" : ""}`.trim()}
                      onClick={() => pickTurn(row.assistant_turn)}
                    >
                      <div className="turn-header">
                        <strong>Turn {row.assistant_turn}</strong>
                        <span>{row.score.toFixed(3)}</span>
                      </div>
                      <div className="turn-subline">
                        <span>{delta === null ? "baseline" : `Δ ${formatSigned(delta)}`}</span>
                      </div>
                      <p>{snippet ? shortenText(snippet, 120) : "(no assistant text loaded)"}</p>
                    </button>
                  );
                })}
            </div>
          </aside>

          <section className="main-column">
            <section className="card q1-card">
              <div className="card-header">
                <div className="card-header-title">
                  <h2>Drift Over Time</h2>
                  <span className="drift-help-wrap">
                    <button
                      className="inline-help-btn"
                      type="button"
                      aria-label="How to read Drift Over Time"
                    >
                      <span aria-hidden="true">?</span>
                    </button>
                    <span className="drift-help-tooltip" role="tooltip">
                      <p>
                        Each point is one assistant turn. Higher points mean the response is
                        more like the default assistant persona, and lower points mean it is
                        drifting toward non-default role-playing like behavior.
                      </p>
                      <div className="drift-help-metric">
                        <p>
                          Technically: for each turn <code>t</code>, <code>x_t</code> is computed by aggregating SAE features at each token position, then a
                          projection score is computed by:
                        </p>
                        <div className="drift-help-eq-block">
                          <code>
                            s<sub>t</sub> = x<sub>t</sub> · a_hat
                          </code>
                        </div>
                      </div>
                      <p>
                        where <code>a_hat</code> is the assistant axis computed as "assistant vs non-assistant centroid" contrast.
                        Downward trend means movement away from default assistant behavior; stable/high trend means staying assistant-like.
                      </p>
                    </span>
                  </span>
                </div>
                <span>
                  selected change: {selectedTransition?.prev_turn ?? "-"} → {selectedTransition?.turn ?? "-"}
                </span>
              </div>

              {chart ? (
                <svg viewBox={`0 0 ${chart.width} ${chart.height}`} role="img" aria-label="Persona drift trajectory">
                  {chart.yTicks.map((tick) => (
                    <g key={`tick-${tick.score.toFixed(6)}`}>
                      <line
                        x1={chart.padLeft}
                        x2={chart.width - chart.padRight}
                        y1={tick.y}
                        y2={tick.y}
                        stroke="var(--line-soft)"
                        strokeWidth={1}
                      />
                      <text x={8} y={tick.y + 4} className="axis-label">
                        {tick.score.toFixed(2)}
                      </text>
                    </g>
                  ))}

                  <line
                    x1={chart.padLeft}
                    x2={chart.width - chart.padRight}
                    y1={chart.yForScore(0)}
                    y2={chart.yForScore(0)}
                    stroke="var(--line-strong)"
                    strokeDasharray="6 4"
                    strokeWidth={1.2}
                  />

                  {transitions.map((transition, index) => {
                    const p0 = chart.points[index];
                    const p1 = chart.points[index + 1];
                    if (!p0 || !p1) {
                      return null;
                    }
                    const downward = transition.delta_score < 0;
                    const isActive = index === safeTransitionIndex;
                    return (
                      <line
                        key={`segment-${transition.prev_turn}-${transition.turn}`}
                        x1={p0.x}
                        x2={p1.x}
                        y1={p0.y}
                        y2={p1.y}
                        className={`segment ${downward ? "drop" : "rise"} ${isActive ? "active" : ""}`.trim()}
                        onClick={() => pickTransition(index)}
                      />
                    );
                  })}

                  <path d={chart.path} fill="none" stroke="var(--line-strong)" strokeWidth={1.5} opacity={0.55} />

                  {chart.points.map((point) => {
                    const isActive = point.turn === selectedAssistantTurn;
                    return (
                      <g key={`point-${point.turn}`}>
                        <circle
                          cx={point.x}
                          cy={point.y}
                          r={isActive ? 6.2 : 4.6}
                          className={`score-dot ${isActive ? "active" : ""}`.trim()}
                          onClick={() => pickTurn(point.turn)}
                        />
                        <text x={point.x} y={chart.height - 12} className="axis-label x-axis-label" textAnchor="middle">
                          {point.turn}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              ) : (
                <p>No trajectory available.</p>
              )}
            </section>

            <section className="card q3-card">
              <div className="card-header">
                <h2>Token Highlights</h2>
                <span>
                  Turn {selectedAssistantTurn ?? "-"} · {tokenRows.length} tokens
                </span>
              </div>

              <div className="token-legend" aria-label="Token polarity legend">
                <span className="token-legend-chip token-legend-chip-left">Role-playing</span>
                <div className="token-legend-bar-wrap">
                  <div className="token-legend-bar" style={{ backgroundImage: tokenLegendGradient }} />
                </div>
                <span className="token-legend-chip token-legend-chip-right">Assistant-like</span>
              </div>

              {tokenRows.length > 0 ? (
                <div className="token-heat">
                  {tokenRows.map((row) => {
                    const bg = interpolateMultiStopHexColor(TOKEN_DIVERGING_STOPS, row.u_k_norm);
                    const textColor = textColorForBackground(bg);
                    return (
                      <span
                        key={`token-${row.token_idx}`}
                        className="token-pill"
                        style={{ backgroundColor: bg, color: textColor }}
                        title={`idx ${row.token_idx} | u_k ${row.u_k.toFixed(4)} | norm ${row.u_k_norm.toFixed(4)}`}
                      >
                        {tokenDisplay(row.token)}
                      </span>
                    );
                  })}
                </div>
              ) : (
                <p>No token localization payload for selected turn.</p>
              )}
            </section>

            <section className="card q2-card">
              <div className="card-header">
                <div className="card-header-title">
                  <h2>Feature Changes</h2>
                  <span className="drift-help-wrap">
                    <button
                      className="inline-help-btn"
                      type="button"
                      aria-label="How to read Feature Changes"
                    >
                      <span aria-hidden="true">?</span>
                    </button>
                    <span className="drift-help-tooltip" role="tooltip">
                      <div className="drift-help-metric">
                        <p>
                          <strong>Delta (delta_x):</strong> How much this feature changed this turn.
                        </p>
                        <div className="drift-help-eq-block">
                          <code>
                            delta_x[j] = x_now[j] - x_prev[j]
                          </code>
                        </div>
                        <p>
                          Positive means the feature activated more; negative means it deactivated.
                        </p>
                      </div>
                      <div className="drift-help-metric">
                        <p>
                          <strong>Contrib (contrib):</strong> How much this feature pushed the drift
                          score up or down.
                        </p>
                        <div className="drift-help-eq-block">
                          <code>
                            contrib[j] = delta_x[j] * axis_weight[j]
                          </code>
                        </div>
                        <p>
                          This is axis-aware. Negative contrib pushes away from assistant; positive
                          contrib pushes toward assistant.
                        </p>
                      </div>
                      <div className="drift-help-metric">
                        <p>
                          <strong>Share of change (share_of_change):</strong> Of all feature movement
                          this turn, what fraction came from this feature?
                        </p>
                        <div className="drift-help-eq-block">
                          <code>
                            share_of_change[j] = |delta_x[j]| / Σ<sub>k</sub> |delta_x[k]|
                          </code>
                        </div>
                        <p>
                          Ignores axis direction; it is movement-size share only.
                        </p>
                      </div>
                      <div className="drift-help-metric">
                        <p>
                          <strong>Share of away mass (share_of_away):</strong> Of all away-from-assistant
                          pressure this turn, how much came from this feature?
                        </p>
                        <div className="drift-help-eq-block">
                          <code>
                            share_of_away[j] = max(-contrib[j], 0) / Σ<sub>k</sub> max(-contrib[k], 0)
                          </code>
                        </div>
                        <p>
                          If a feature is toward-assistant (<code>contrib &gt;= 0</code>), this is <code>0</code>.
                        </p>
                      </div>
                    </span>
                  </span>
                </div>
                <span>
                  {selectedTransition ? `${selectedTransition.prev_turn} → ${selectedTransition.turn}` : "No transition"}
                </span>
              </div>

              {selectedTransition ? (
                <div className="feature-grid">
                  <FeatureTable
                    direction="activated"
                    features={selectedTransition.activated}
                    neuronpediaId={neuronpediaId}
                  />
                  <FeatureTable
                    direction="deactivated"
                    features={selectedTransition.deactivated}
                    neuronpediaId={neuronpediaId}
                  />
                </div>
              ) : (
                <p>No transition records available.</p>
              )}
            </section>
          </section>
        </main>
      </div>
    </div>
  );
}

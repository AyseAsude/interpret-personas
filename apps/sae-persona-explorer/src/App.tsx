import type { ChangeEvent } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { OrthographicView } from "@deck.gl/core";
import { LineLayer, ScatterplotLayer } from "@deck.gl/layers";

import type { BundlePayload, FeaturePoint, LayoutMode } from "./types";
import {
  colorForFeature,
  fitOrthographicView,
  sanitizeExternalUrl,
  type OrthoViewState,
} from "./utils";

type HoverInfo = {
  x: number;
  y: number;
  point: FeaturePoint;
};

type AxisSegment = {
  source: [number, number];
  target: [number, number];
  color: [number, number, number, number];
  width: number;
};

type OriginPoint = {
  x: number;
  y: number;
};

type PanelTab = "feature" | "role" | "settings";
type CenteringMode = "off" | "on";
type ByMode<T> = Record<CenteringMode, T>;

const LEGACY_BUNDLE_URL = import.meta.env.VITE_BUNDLE_URL ?? `${import.meta.env.BASE_URL}bundle.json`;
const NON_CENTERED_BUNDLE_URL = import.meta.env.VITE_BUNDLE_URL_NON_CENTERED ?? LEGACY_BUNDLE_URL;
const CENTERED_BUNDLE_URL = import.meta.env.VITE_BUNDLE_URL_CENTERED ?? "";
const BUNDLE_URLS: ByMode<string> = {
  off: NON_CENTERED_BUNDLE_URL,
  on: CENTERED_BUNDLE_URL,
};
const PERSONA_DRIFT_URL = import.meta.env.VITE_PERSONA_DRIFT_URL ?? "#/persona-drift";
const GITHUB_ALLOWED_HOSTS = new Set<string>(["github.com", "www.github.com"]);
const GITHUB_PROFILE_URL = sanitizeExternalUrl(
  import.meta.env.VITE_GITHUB_PROFILE_URL ?? "https://github.com/AyseAsude/interpret-personas",
  GITHUB_ALLOWED_HOSTS,
);

const MODEL_NAME_OVERRIDES: Record<string, string> = {
  "gemma-3-27b-it": "Gemma 3 27B IT",
  "qwen-3-32b": "Qwen 3 32B",
  "llama-3.3-70b": "Llama 3.3 70B",
};

function extractModelSlug(path: string): string | null {
  const match = path.match(/\/([^/]+)_layer_\d+_width_[^/_]+_l0_[^/_]+(?:\/|$)/);
  return match?.[1] ?? null;
}

function extractSaeId(path: string): string | null {
  const match = path.match(/(layer_\d+_width_[^/_]+_l0_[^/_]+)/);
  return match?.[1] ?? null;
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

function prettifySaeId(saeId: string): string {
  const match = saeId.match(/^layer_(\d+)_width_([^_]+)_l0_(.+)$/);
  if (!match) {
    return saeId.replace(/_/g, " ");
  }
  const l0Label = match[3]
    .replace(/_/g, " ")
    .replace(/\boutputs?\b/gi, "")
    .replace(/\s+/g, " ")
    .trim();
  return `Layer ${match[1]} · Width ${match[2]} · L0 ${l0Label}`;
}

function deriveModelAndSaeLabel(dataset: BundlePayload["dataset"] | null): {
  model: string;
  sae: string;
} {
  if (!dataset) {
    return { model: "Not loaded", sae: "Not loaded" };
  }
  const source = `${dataset.features_dir ?? ""} ${dataset.aggregated_file ?? ""}`;
  const modelSlug = extractModelSlug(source);
  const saeId = extractSaeId(source);
  return {
    model: modelSlug ? prettifyModelSlug(modelSlug) : dataset.name,
    sae: saeId ? prettifySaeId(saeId) : `${dataset.sae_dim.toLocaleString()} features`,
  };
}

const HOW_TO_READ_ITEMS: Array<{ title: string; lines: string[] }> = [
  {
    title: "What each dot means",
    lines: [
      "Each dot is one SAE feature kept after filtering.",
      "Selected features have relatively higher variance across roles (role-specific signal).",
      "Selected features are also consistent within each role (not just prompt noise).",
    ],
  },
  {
    title: "Map distance vs high-D truth",
    lines: [
      "The 2D map is a simplified view for exploration.",
      "Related features and role similarity are computed in high-dimensional feature space with cosine similarity.",
    ],
  },
  {
    title: "Question centering ON",
    lines: [
      "Technical: For each question index (across roles), subtracts the cross-role mean feature activation, then runs filtering/ranking on the residuals.",
      "Intuition: Highlights residual role differences after removing shared prompt/question structure.",
      "Use this when you want role-specific deviations from the common pattern.",
    ],
  },
  {
    title: "Question centering OFF",
    lines: [
      "Technical: No centering is applied; feature values are shown directly from log of role activations.",
      "Intuition: Shows absolute activation patterns, including signal shared across roles.",
      "Use this when you want overall activity by role.",
    ],
  },
  {
    title: "Settings options",
    lines: [
      "Layout: UMAP preserves local neighborhoods; PCA shows broad linear structure.",
      "Preferred role colors each feature by the role with the strongest value in the active mode.",
    ],
  },
  {
    title: "Related features, preferred role, missing descriptions",
    lines: [
      "Related features are nearest neighbors in high-D cosine space, not nearest dots on the 2D map.",
      "Preferred role depends on mode: OFF = highest absolute value, ON = largest positive residual.",
      "Missing description usually means no Neuronpedia text was found in the local cache.",
    ],
  },
];
function helpCardId(title: string): string {
  return `help-card-${title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "")}`;
}

export default function App() {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const selectedFeatureIdOnSwitchRef = useRef<number | null>(null);
  const helpGlowTimeoutRef = useRef<number | null>(null);
  const [bundleMode, setBundleMode] = useState<CenteringMode>("off");
  const [bundles, setBundles] = useState<ByMode<BundlePayload | null>>({
    off: null,
    on: null,
  });
  const [loadErrors, setLoadErrors] = useState<ByMode<string>>({
    off: "",
    on: "",
  });
  const [loadingStates, setLoadingStates] = useState<ByMode<boolean>>({
    off: false,
    on: false,
  });
  const [layoutMode, setLayoutMode] = useState<LayoutMode>("umap");
  const [activeTab, setActiveTab] = useState<PanelTab>("feature");
  const [selectedFeatureRow, setSelectedFeatureRow] = useState<number | null>(null);
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [roleFocus, setRoleFocus] = useState<string>("");
  const [highlightRole, setHighlightRole] = useState<string | null>(null);
  const [helpExpanded, setHelpExpanded] = useState<boolean>(true);
  const [highlightedHelpTitle, setHighlightedHelpTitle] = useState<string | null>(null);
  const [viewState, setViewState] = useState<OrthoViewState>({
    target: [0, 0, 0],
    zoom: 0,
  });
  const [mapSize, setMapSize] = useState({ width: 1000, height: 700 });

  const bundle = bundles[bundleMode];
  const loading = loadingStates[bundleMode];
  const loadError = loadErrors[bundleMode];
  const activeBundleUrl = BUNDLE_URLS[bundleMode];
  const centeredBundleFromUrlEnabled = Boolean(BUNDLE_URLS.on);
  const centeredModeAvailable = centeredBundleFromUrlEnabled || Boolean(bundles.on);
  const modeMismatch =
    bundle !== null
      ? (bundleMode === "on") !== Boolean(bundle.dataset.question_centering)
      : false;
  const headerLabels = useMemo(
    () => deriveModelAndSaeLabel(bundle?.dataset ?? null),
    [bundle?.dataset],
  );

  const loadBundle = useCallback(
    async (mode: CenteringMode, force = false) => {
      const url = BUNDLE_URLS[mode];
      if (!url) {
        setLoadErrors((prev) => ({
          ...prev,
          [mode]:
            "No bundle URL configured for centered mode. " +
            "Set VITE_BUNDLE_URL_CENTERED or upload a centered bundle JSON.",
        }));
        return;
      }
      if (!force && bundles[mode]) {
        return;
      }

      setLoadingStates((prev) => ({ ...prev, [mode]: true }));
      setLoadErrors((prev) => ({ ...prev, [mode]: "" }));

      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Failed to fetch bundle from ${url} (${response.status})`);
        }
        const payload = (await response.json()) as BundlePayload;
        setBundles((prev) => ({ ...prev, [mode]: payload }));
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setLoadErrors((prev) => ({ ...prev, [mode]: message }));
      } finally {
        setLoadingStates((prev) => ({ ...prev, [mode]: false }));
      }
    },
    [bundles],
  );

  useEffect(() => {
    if (!bundles[bundleMode] && !loadingStates[bundleMode]) {
      void loadBundle(bundleMode);
    }
  }, [bundleMode, bundles, loadingStates, loadBundle]);

  useEffect(() => {
    if (!helpExpanded || !highlightedHelpTitle) {
      return;
    }

    const frame = window.requestAnimationFrame(() => {
      const card = document.getElementById(helpCardId(highlightedHelpTitle));
      card?.scrollIntoView({ behavior: "smooth", block: "center" });
    });

    if (helpGlowTimeoutRef.current !== null) {
      window.clearTimeout(helpGlowTimeoutRef.current);
    }
    helpGlowTimeoutRef.current = window.setTimeout(() => {
      setHighlightedHelpTitle((current) =>
        current === highlightedHelpTitle ? null : current,
      );
      helpGlowTimeoutRef.current = null;
    }, 2200);

    return () => {
      window.cancelAnimationFrame(frame);
    };
  }, [helpExpanded, highlightedHelpTitle]);

  useEffect(() => {
    return () => {
      if (helpGlowTimeoutRef.current !== null) {
        window.clearTimeout(helpGlowTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!bundle) {
      return;
    }
    setHoverInfo(null);
    setHighlightRole((prev) => (prev && bundle.roles.includes(prev) ? prev : null));
    setRoleFocus((prev) => (prev && bundle.roles.includes(prev) ? prev : bundle.roles[0] ?? ""));
    setSelectedFeatureRow(() => {
      const preferredFeatureId = selectedFeatureIdOnSwitchRef.current;
      if (preferredFeatureId !== null) {
        const match = bundle.features.find((feature) => feature.feature_id === preferredFeatureId);
        selectedFeatureIdOnSwitchRef.current = null;
        if (match) {
          return match.feature_row;
        }
      }
      return bundle.features[0]?.feature_row ?? null;
    });
  }, [bundle]);

  useEffect(() => {
    if (!mapContainerRef.current) {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }
      setMapSize({
        width: Math.max(320, Math.floor(entry.contentRect.width)),
        height: Math.max(320, Math.floor(entry.contentRect.height)),
      });
    });
    observer.observe(mapContainerRef.current);
    return () => observer.disconnect();
  }, []);

  const points = useMemo<FeaturePoint[]>(() => {
    if (!bundle) {
      return [];
    }
    const coords = bundle.coords[layoutMode];
    return bundle.features.map((feature, idx) => ({
      ...feature,
      x: coords[idx]?.[0] ?? 0,
      y: coords[idx]?.[1] ?? 0,
    }));
  }, [bundle, layoutMode]);

  useEffect(() => {
    if (!points.length) {
      return;
    }
    const coords: [number, number][] = points.map((point) => [point.x, point.y]);
    if (layoutMode === "pca") {
      coords.push([0, 0]);
    }
    setViewState(fitOrthographicView(coords, mapSize.width, mapSize.height));
  }, [points, mapSize.width, mapSize.height, layoutMode]);

  const selectedFeature = useMemo(() => {
    if (!bundle || selectedFeatureRow === null) {
      return null;
    }
    return bundle.features[selectedFeatureRow] ?? null;
  }, [bundle, selectedFeatureRow]);

  const selectedFeatureNeuronpediaUrl = useMemo(
    () => sanitizeExternalUrl(selectedFeature?.neuronpedia_url),
    [selectedFeature?.neuronpedia_url],
  );

  const roleStats = useMemo(() => {
    if (!bundle || !roleFocus) {
      return {
        preferredCount: 0,
        topFeatures: [] as BundlePayload["features"],
        nearestRoles: [] as Array<{ role: string; similarity: number }>,
      };
    }
    const preferred = bundle.features
      .filter((feature) => feature.preferred_role === roleFocus)
      .sort(
        (a, b) =>
          (b.top_roles[0]?.share ?? 0) - (a.top_roles[0]?.share ?? 0),
      );
    const roleIdx = bundle.roles.indexOf(roleFocus);
    const nearestRoles =
      roleIdx >= 0
        ? bundle.roles
          .map((role, idx) => ({
            role,
            similarity: bundle.role_similarity[roleIdx]?.[idx] ?? 0,
          }))
          .filter((row) => row.role !== roleFocus)
          .sort((a, b) => b.similarity - a.similarity)
          .slice(0, 8)
        : [];
    return {
      preferredCount: preferred.length,
      topFeatures: preferred.slice(0, 20),
      nearestRoles,
    };
  }, [bundle, roleFocus]);

  const relatedFeatures = useMemo(() => {
    if (!bundle || selectedFeatureRow === null) {
      return [] as Array<{ feature: BundlePayload["features"][number]; sim: number }>;
    }
    const indices = bundle.neighbors.indices[selectedFeatureRow] ?? [];
    const sims = bundle.neighbors.similarities[selectedFeatureRow] ?? [];
    return indices
      .map((neighborRow, idx) => ({
        feature: bundle.features[neighborRow],
        sim: sims[idx] ?? 0,
      }))
      .filter((entry) => Boolean(entry.feature));
  }, [bundle, selectedFeatureRow]);

  const layers = useMemo(() => {
    const selectedNeighborRows =
      selectedFeatureRow !== null ? bundle?.neighbors.indices[selectedFeatureRow] ?? [] : [];
    const selectedNeighborSet = new Set<number>(
      selectedNeighborRows.filter((row) => row >= 0),
    );

    let axisLayers: Array<LineLayer<AxisSegment> | ScatterplotLayer<OriginPoint>> = [];
    if (layoutMode === "pca" && points.length) {
      let minX = Number.POSITIVE_INFINITY;
      let minY = Number.POSITIVE_INFINITY;
      let maxX = Number.NEGATIVE_INFINITY;
      let maxY = Number.NEGATIVE_INFINITY;
      for (const point of points) {
        minX = Math.min(minX, point.x);
        minY = Math.min(minY, point.y);
        maxX = Math.max(maxX, point.x);
        maxY = Math.max(maxY, point.y);
      }

      minX = Math.min(minX, 0);
      minY = Math.min(minY, 0);
      maxX = Math.max(maxX, 0);
      maxY = Math.max(maxY, 0);

      const spanX = Math.max(maxX - minX, 1e-6);
      const spanY = Math.max(maxY - minY, 1e-6);
      const padX = spanX * 0.08;
      const padY = spanY * 0.08;

      const axisData: AxisSegment[] = [
        {
          source: [minX - padX, 0],
          target: [maxX + padX, 0],
          color: [11, 110, 79, 160],
          width: 1.8,
        },
        {
          source: [0, minY - padY],
          target: [0, maxY + padY],
          color: [157, 43, 37, 150],
          width: 1.4,
        },
      ];

      const axisLayer = new LineLayer<AxisSegment>({
        id: "pca-axes",
        data: axisData,
        pickable: false,
        getSourcePosition: (d) => d.source,
        getTargetPosition: (d) => d.target,
        getColor: (d) => d.color,
        widthUnits: "pixels",
        getWidth: (d) => d.width,
      });

      const originLayer = new ScatterplotLayer<OriginPoint>({
        id: "pca-origin",
        data: [{ x: 0, y: 0 }],
        pickable: false,
        radiusUnits: "pixels",
        getPosition: (d) => [d.x, d.y],
        getRadius: 5.6,
        getFillColor: [18, 40, 29, 230],
        stroked: true,
        getLineColor: [250, 250, 245, 245],
        lineWidthUnits: "pixels",
        getLineWidth: 1.4,
      });

      axisLayers = [axisLayer, originLayer];
    }

    const baseLayer = new ScatterplotLayer<FeaturePoint>({
      id: "feature-points",
      data: points,
      pickable: true,
      radiusUnits: "pixels",
      getPosition: (d) => [d.x, d.y],
      getRadius: (d) => {
        if (selectedFeatureRow === null) {
          return 3.4;
        }
        if (d.feature_row === selectedFeatureRow) {
          return 8.6;
        }
        if (selectedNeighborSet.has(d.feature_row)) {
          return 5.8;
        }
        return 3.1;
      },
      getFillColor: (d) => {
        const [r, g, b] = colorForFeature(d);
        const isDimmed = highlightRole !== null && d.preferred_role !== highlightRole;
        return [r, g, b, isDimmed ? 35 : 210];
      },
      onHover: (info) => {
        if (!info.object) {
          setHoverInfo(null);
          return;
        }
        setHoverInfo({
          x: info.x,
          y: info.y,
          point: info.object as FeaturePoint,
        });
      },
      onClick: (info) => {
        if (info.object) {
          setSelectedFeatureRow((info.object as FeaturePoint).feature_row);
          setActiveTab("feature");
        }
      },
      updateTriggers: {
        getFillColor: [highlightRole],
        getRadius: [selectedFeatureRow],
      },
    });

    if (selectedFeatureRow === null) {
      return [...axisLayers, baseLayer];
    }

    const selectedPoint = points.find((point) => point.feature_row === selectedFeatureRow);
    if (!selectedPoint) {
      return [...axisLayers, baseLayer];
    }

    const [selectedR, selectedG, selectedB] = colorForFeature(selectedPoint);
    const neighborPoints = points.filter(
      (point) =>
        point.feature_row !== selectedFeatureRow &&
        selectedNeighborSet.has(point.feature_row),
    );

    const neighborHueLayer = new ScatterplotLayer<FeaturePoint>({
      id: "feature-neighbor-hue",
      data: neighborPoints,
      pickable: false,
      filled: true,
      stroked: false,
      radiusUnits: "pixels",
      getPosition: (d) => [d.x, d.y],
      getRadius: 8.4,
      getFillColor: (d) => {
        const [r, g, b] = colorForFeature(d);
        return [r, g, b, 66];
      },
      updateTriggers: {
        getFillColor: [selectedFeatureRow],
      },
    });

    const selectedHueLayer = new ScatterplotLayer<FeaturePoint>({
      id: "feature-selected-hue",
      data: [selectedPoint],
      pickable: false,
      filled: true,
      stroked: false,
      radiusUnits: "pixels",
      getPosition: (d) => [d.x, d.y],
      getRadius: 12.8,
      getFillColor: [selectedR, selectedG, selectedB, 102],
    });

    return [...axisLayers, neighborHueLayer, selectedHueLayer, baseLayer];
  }, [bundle, points, highlightRole, selectedFeatureRow, layoutMode]);

  function handleBundleModeChange(nextMode: CenteringMode) {
    if (nextMode === bundleMode) {
      return;
    }
    if (nextMode === "on" && !centeredModeAvailable) {
      return;
    }
    selectedFeatureIdOnSwitchRef.current = selectedFeature?.feature_id ?? null;
    setBundleMode(nextMode);
    if (!bundles[nextMode] && !loadingStates[nextMode]) {
      void loadBundle(nextMode);
    }
  }

  function jumpToCenteringHelp() {
    const targetTitle = bundleMode === "on" ? "Question centering ON" : "Question centering OFF";
    setHelpExpanded(true);
    setHighlightedHelpTitle(targetTitle);
  }

  function jumpToLayoutHelp() {
    setHelpExpanded(true);
    setHighlightedHelpTitle("Settings options");
  }

  function handleBundleFileUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const payload = JSON.parse(String(reader.result)) as BundlePayload;
        const inferredMode: CenteringMode = payload.dataset.question_centering ? "on" : "off";
        selectedFeatureIdOnSwitchRef.current = null;
        setBundles((prev) => ({ ...prev, [inferredMode]: payload }));
        setLoadErrors((prev) => ({ ...prev, [inferredMode]: "" }));
        setBundleMode(inferredMode);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setLoadErrors((prev) => ({
          ...prev,
          [bundleMode]: `Unable to parse uploaded JSON: ${message}`,
        }));
      }
    };
    reader.readAsText(file);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-main">
          <h1>SAE Persona Feature Explorer</h1>
          <div className="topbar-link-row">
            <a className="topbar-nav-link" href={PERSONA_DRIFT_URL}>
              Open Persona Drift Explorer
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
            <strong>{headerLabels.model}</strong>
          </div>
          <div className="dataset-item">
            <span>SAE</span>
            <strong>{headerLabels.sae}</strong>
          </div>
        </div>
      </header>

      <main className={`workspace ${helpExpanded ? "" : "help-collapsed"}`.trim()}>
        <aside className={`help-panel ${helpExpanded ? "expanded" : "collapsed"}`}>
          {helpExpanded ? (
            <>
              <div className="help-header">
                <h2>How to read this map</h2>
                <button
                  className="help-toggle-btn"
                  onClick={() => setHelpExpanded(false)}
                  type="button"
                  aria-expanded={true}
                  aria-controls="how-to-read-content"
                  aria-label="Collapse help panel"
                  title="Collapse help panel"
                >
                  <span aria-hidden="true">&lt;</span>
                </button>
              </div>
              <div className="help-content" id="how-to-read-content">
                {HOW_TO_READ_ITEMS.map((item) => (
                  <section
                    className={`help-card ${highlightedHelpTitle === item.title ? "glow" : ""}`.trim()}
                    id={helpCardId(item.title)}
                    key={item.title}
                  >
                    <h3>{item.title}</h3>
                    {item.lines.map((line) => (
                      <p key={line}>{line}</p>
                    ))}
                  </section>
                ))}
                <p className="help-more">
                  More details: <code>https://github.com/AyseAsude/interpret-personas/blob/master/README.md</code>.
                </p>
              </div>
            </>
          ) : (
            <button
              className="help-rail-btn"
              onClick={() => setHelpExpanded(true)}
              type="button"
              aria-expanded={false}
              aria-controls="how-to-read-content"
              aria-label="Expand help panel"
              title="Expand help panel"
            >
              <span aria-hidden="true">?</span>
            </button>
          )}
        </aside>

        <section className="map-section" ref={mapContainerRef}>
          {loading && (
            <div className="load-overlay">
              Loading {bundleMode === "on" ? "centered" : "non-centered"} bundle
              {activeBundleUrl ? ` from ${activeBundleUrl}` : "..."}
            </div>
          )}

          {!loading && !bundle && (
            <div className="load-overlay">
              <p>
                Could not load {bundleMode === "on" ? "centered" : "non-centered"} bundle.
              </p>
              {activeBundleUrl && <p>URL: {activeBundleUrl}</p>}
              {loadError && <p className="error-text">{loadError}</p>}
              <label className="upload-btn">
                Load bundle JSON
                <input type="file" accept=".json,application/json" onChange={handleBundleFileUpload} />
              </label>
            </div>
          )}

          {bundle && (
            <>
              <DeckGL
                views={new OrthographicView({ id: "ortho" })}
                controller={true}
                viewState={viewState}
                onViewStateChange={({ viewState: nextViewState }) => {
                  const target = (nextViewState.target ?? [0, 0, 0]) as [number, number, number];
                  const zoom = typeof nextViewState.zoom === "number" ? nextViewState.zoom : 0;
                  setViewState({
                    target,
                    zoom,
                  });
                }}
                layers={layers}
              />
              <div className="map-stats">
                <span>{bundle.features.length.toLocaleString()} selected features</span>
                <span>{bundle.roles.length} roles</span>
                {layoutMode === "pca" && <span>PCA axes shown: PC1 horizontal, origin at (0, 0)</span>}
              </div>
            </>
          )}

          {hoverInfo && (
            <div className="tooltip" style={{ left: hoverInfo.x + 16, top: hoverInfo.y + 16 }}>
              <div className="tooltip-title">Feature #{hoverInfo.point.feature_id}</div>
              <div>Preferred role: {hoverInfo.point.preferred_role}</div>
              <div className="tooltip-roles">
                {hoverInfo.point.top_roles.map((entry) => (
                  <span key={entry.role}>
                    {entry.role}: {(entry.share * 100).toFixed(0)}%
                  </span>
                ))}
              </div>
              {hoverInfo.point.description && (
                <p className="tooltip-desc">{hoverInfo.point.description}</p>
              )}
            </div>
          )}
        </section>

        <aside className="side-panel">
          <nav className="tab-row">
            <button
              className={activeTab === "feature" ? "active" : ""}
              onClick={() => setActiveTab("feature")}
            >
              Feature
            </button>
            <button
              className={activeTab === "role" ? "active" : ""}
              onClick={() => setActiveTab("role")}
            >
              Role
            </button>
            <button
              className={activeTab === "settings" ? "active" : ""}
              onClick={() => setActiveTab("settings")}
            >
              Settings
            </button>
          </nav>

          {activeTab === "feature" && (
            <div className="panel-content">
              {!selectedFeature && <p>Select a point to inspect its details.</p>}

              {selectedFeature && (
                <>
                  <h2>Feature #{selectedFeature.feature_id}</h2>
                  <p className="badge">{selectedFeature.preferred_role}</p>
                  <p>
                    {selectedFeature.description ??
                      "No cached description is available for this feature."}
                  </p>
                  {selectedFeatureNeuronpediaUrl && (
                    <p>
                      <a
                        href={selectedFeatureNeuronpediaUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Open Neuronpedia entry
                      </a>
                    </p>
                  )}

                  <h3>Top Role Activations</h3>
                  <div className="mini-bars">
                    {selectedFeature.top_roles.map((entry) => (
                      <div className="mini-bar-row" key={entry.role}>
                        <span>{entry.role}</span>
                        <div className="mini-bar">
                          <div
                            className="mini-bar-fill"
                            style={{ width: `${Math.max(4, entry.share * 100)}%` }}
                          />
                        </div>
                        <span>{(entry.share * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>

                  <h3>Related Features</h3>
                  <ul className="neighbor-list">
                    {relatedFeatures.map((entry) => (
                      <li key={entry.feature.feature_row}>
                        <button
                          onClick={() => setSelectedFeatureRow(entry.feature.feature_row)}
                        >
                          #{entry.feature.feature_id} · {entry.feature.preferred_role}
                          <span>{entry.sim.toFixed(3)}</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          )}

          {activeTab === "role" && bundle && (
            <div className="panel-content">
              <h2>Role explorer</h2>
              <label className="field">
                Role
                <select value={roleFocus} onChange={(e) => setRoleFocus(e.target.value)}>
                  {bundle.roles.map((role) => (
                    <option key={role} value={role}>
                      {role}
                    </option>
                  ))}
                </select>
              </label>

              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={highlightRole === roleFocus}
                  onChange={(e) =>
                    setHighlightRole(e.target.checked ? roleFocus : null)
                  }
                />
                Highlight this role on map
              </label>

              <p>
                Total features on the map: <strong>{roleStats.preferredCount}</strong>
              </p>

              <h3>Nearest roles</h3>
              <ul className="plain-list">
                {roleStats.nearestRoles.map((entry) => (
                  <li key={entry.role}>
                    <span>{entry.role}</span>
                    <span>{entry.similarity.toFixed(3)}</span>
                  </li>
                ))}
              </ul>

              <h3>Top features for {roleFocus}</h3>
              <ul className="neighbor-list">
                {roleStats.topFeatures.map((feature) => (
                  <li key={feature.feature_row}>
                    <button onClick={() => setSelectedFeatureRow(feature.feature_row)}>
                      #{feature.feature_id}
                      <span>{((feature.top_roles[0]?.share ?? 0) * 100).toFixed(0)}%</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {activeTab === "settings" && bundle && (
            <div className="panel-content">
              <h2>Layout & quality</h2>
              <label className="field">
                <span className="field-label with-help">
                  Question centering
                  <button
                    className="inline-help-btn"
                    type="button"
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      jumpToCenteringHelp();
                    }}
                    aria-label="Show question centering help"
                    title="Show question centering help"
                  >
                    <span aria-hidden="true">?</span>
                  </button>
                </span>
                <select
                  value={bundleMode}
                  onChange={(e) => handleBundleModeChange(e.target.value as CenteringMode)}
                >
                  <option value="off">Off (non-centered)</option>
                  <option value="on" disabled={!centeredModeAvailable}>
                    On (question-centered)
                  </option>
                </select>
              </label>
              {!centeredModeAvailable && (
                <p>
                  Centered bundle URL not configured. Set `VITE_BUNDLE_URL_CENTERED` or upload
                  a centered bundle JSON.
                </p>
              )}
              {modeMismatch && (
                <p className="error-text">
                  Active mode and bundle metadata disagree. Verify bundle URLs.
                </p>
              )}

              <label className="field">
                <span className="field-label with-help">
                  Layout
                  <button
                    className="inline-help-btn"
                    type="button"
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      jumpToLayoutHelp();
                    }}
                    aria-label="Show layout help"
                    title="Show layout help"
                  >
                    <span aria-hidden="true">?</span>
                  </button>
                </span>
                <select
                  value={layoutMode}
                  onChange={(e) => setLayoutMode(e.target.value as LayoutMode)}
                >
                  <option value="umap">UMAP</option>
                  <option value="pca">PCA</option>
                </select>
              </label>

              <div className="quality-card">
                <p>{bundle.guardrails.note}</p>
                <p className="quality-metric">
                  Neighborhood preservation at k={bundle.guardrails.knn_overlap_at_k}:{" "}
                  <strong>{bundle.guardrails.knn_overlap_score.toFixed(3)}</strong>
                </p>
              </div>
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}

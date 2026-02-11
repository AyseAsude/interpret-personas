import type { ChangeEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { OrthographicView } from "@deck.gl/core";
import { LineLayer, ScatterplotLayer } from "@deck.gl/layers";

import type { BundlePayload, ColorMode, FeaturePoint, LayoutMode } from "./types";
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

const BUNDLE_URL = import.meta.env.VITE_BUNDLE_URL ?? `${import.meta.env.BASE_URL}bundle.json`;

export default function App() {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const [bundle, setBundle] = useState<BundlePayload | null>(null);
  const [loadError, setLoadError] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [layoutMode, setLayoutMode] = useState<LayoutMode>("umap");
  const [colorMode, setColorMode] = useState<ColorMode>("preferred_role");
  const [activeTab, setActiveTab] = useState<PanelTab>("feature");
  const [selectedFeatureRow, setSelectedFeatureRow] = useState<number | null>(null);
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [roleFocus, setRoleFocus] = useState<string>("");
  const [highlightRole, setHighlightRole] = useState<string | null>(null);
  const [viewState, setViewState] = useState<OrthoViewState>({
    target: [0, 0, 0],
    zoom: 0,
  });
  const [mapSize, setMapSize] = useState({ width: 1000, height: 700 });

  useEffect(() => {
    let isCancelled = false;
    async function fetchBundle() {
      setLoading(true);
      setLoadError("");
      try {
        const response = await fetch(BUNDLE_URL);
        if (!response.ok) {
          throw new Error(
            `Failed to fetch bundle from ${BUNDLE_URL} (${response.status})`,
          );
        }
        const payload = (await response.json()) as BundlePayload;
        if (isCancelled) {
          return;
        }
        setBundle(payload);
        setRoleFocus(payload.roles[0] ?? "");
        setSelectedFeatureRow(payload.features[0]?.feature_row ?? null);
      } catch (error) {
        if (!isCancelled) {
          const message = error instanceof Error ? error.message : String(error);
          setLoadError(message);
        }
      } finally {
        if (!isCancelled) {
          setLoading(false);
        }
      }
    }
    fetchBundle();
    return () => {
      isCancelled = true;
    };
  }, []);

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
      .sort((a, b) => b.metrics.score - a.metrics.score);
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
        const [r, g, b] = colorForFeature(d, colorMode);
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
        getFillColor: [colorMode, highlightRole],
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

    const [selectedR, selectedG, selectedB] = colorForFeature(selectedPoint, colorMode);
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
        const [r, g, b] = colorForFeature(d, colorMode);
        return [r, g, b, 66];
      },
      updateTriggers: {
        getFillColor: [colorMode, selectedFeatureRow],
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
  }, [bundle, points, colorMode, highlightRole, selectedFeatureRow, layoutMode]);

  function handleBundleFileUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const payload = JSON.parse(String(reader.result)) as BundlePayload;
        setBundle(payload);
        setRoleFocus(payload.roles[0] ?? "");
        setSelectedFeatureRow(payload.features[0]?.feature_row ?? null);
        setLoadError("");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setLoadError(`Unable to parse uploaded JSON: ${message}`);
      }
    };
    reader.readAsText(file);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>SAE Persona Feature Explorer</h1>
          <p>
            Map is navigation only. Related features and role similarity are computed in
            high-dimensional role-profile space.
          </p>
        </div>
        <div className="dataset-chip">
          <span>Dataset</span>
          <strong>{bundle?.dataset.name ?? "Not loaded"}</strong>
        </div>
      </header>

      <main className="workspace">
        <section className="map-section" ref={mapContainerRef}>
          {loading && <div className="load-overlay">Loading bundle from {BUNDLE_URL}...</div>}

          {!loading && !bundle && (
            <div className="load-overlay">
              <p>Could not load bundle from default URL.</p>
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
                <span>kNN overlap@{bundle.guardrails.knn_overlap_at_k}: {bundle.guardrails.knn_overlap_score.toFixed(3)}</span>
                {layoutMode === "pca" && <span>PCA axes shown: PC1 horizontal, origin at (0, 0)</span>}
              </div>
            </>
          )}

          {hoverInfo && (
            <div className="tooltip" style={{ left: hoverInfo.x + 16, top: hoverInfo.y + 16 }}>
              <div className="tooltip-title">Feature #{hoverInfo.point.feature_id}</div>
              <div>Preferred role: {hoverInfo.point.preferred_role}</div>
              <div>Score: {hoverInfo.point.metrics.score.toFixed(3)}</div>
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

                      <h3>Metrics</h3>
                      <dl className="metrics-grid">
                        <dt>score</dt>
                        <dd>{selectedFeature.metrics.score.toFixed(4)}</dd>
                        <dt>stability</dt>
                        <dd>{selectedFeature.metrics.stability.toFixed(4)}</dd>
                        <dt>pref_ratio</dt>
                        <dd>{selectedFeature.metrics.pref_ratio.toFixed(3)}</dd>
                        <dt>active_frac</dt>
                        <dd>{selectedFeature.metrics.active_frac.toFixed(3)}</dd>
                        <dt>bridge_entropy</dt>
                        <dd>{selectedFeature.metrics.bridge_entropy.toFixed(3)}</dd>
                        <dt>mean_activation</dt>
                        <dd>{selectedFeature.metrics.mean_activation.toFixed(3)}</dd>
                      </dl>

                      <h3>Related Features (high-D cosine)</h3>
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
                    Preferred features: <strong>{roleStats.preferredCount}</strong>
                  </p>

                  <h3>Nearest roles (high-D cosine)</h3>
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
                          <span>{feature.metrics.score.toFixed(3)}</span>
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
                    Layout
                    <select
                      value={layoutMode}
                      onChange={(e) => setLayoutMode(e.target.value as LayoutMode)}
                    >
                      <option value="umap">UMAP</option>
                      <option value="pca">PCA</option>
                    </select>
                  </label>

                  <label className="field">
                    Color mode
                    <select
                      value={colorMode}
                      onChange={(e) => setColorMode(e.target.value as ColorMode)}
                    >
                      <option value="preferred_role">Preferred role</option>
                      <option value="bridge_entropy">Bridge entropy</option>
                      <option value="active_frac">Active fraction</option>
                    </select>
                  </label>

                  <div className="quality-card">
                    <p>{bundle.guardrails.note}</p>
                    <p>
                      Neighborhood preservation @k={bundle.guardrails.knn_overlap_at_k}:{" "}
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

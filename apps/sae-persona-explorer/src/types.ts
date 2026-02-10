export type RoleTopEntry = {
  role_idx: number;
  role: string;
  activation: number;
  share: number;
};

export type FeatureMetrics = {
  score: number;
  stability: number;
  sd: number;
  mu: number;
  pref_ratio: number;
  active_frac: number;
  cv: number;
  mean_activation: number;
  max_activation: number;
  bridge_entropy: number;
};

export type FeatureRow = {
  feature_row: number;
  feature_id: number;
  preferred_role_idx: number;
  preferred_role: string;
  top_roles: RoleTopEntry[];
  metrics: FeatureMetrics;
  description: string | null;
  neuronpedia_url: string | null;
};

export type DatasetMeta = {
  name: string;
  aggregated_file: string;
  features_dir: string;
  strategy: "mean" | "max";
  n_roles: number;
  sae_dim: number;
  top_k: number;
};

export type GuardrailMeta = {
  note: string;
  knn_overlap_at_k: number;
  knn_overlap_score: number;
};

export type BundlePayload = {
  dataset: DatasetMeta;
  guardrails: GuardrailMeta;
  roles: string[];
  feature_ids: number[];
  coords: {
    umap: [number, number][];
    pca: [number, number][];
  };
  neighbors: {
    k: number;
    indices: number[][];
    similarities: number[][];
  };
  role_similarity: number[][];
  features: FeatureRow[];
};

export type LayoutMode = "umap" | "pca";
export type ColorMode = "preferred_role" | "bridge_entropy" | "active_frac";

export type FeaturePoint = FeatureRow & {
  x: number;
  y: number;
};

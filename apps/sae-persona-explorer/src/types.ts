export type RoleTopEntry = {
  role_idx: number;
  role: string;
  activation: number;
  share: number;
};

export type FeatureRow = {
  feature_row: number;
  feature_id: number;
  preferred_role_idx: number;
  preferred_role: string;
  top_roles: RoleTopEntry[];
  description: string | null;
  neuronpedia_url: string | null;
};

export type DatasetMeta = {
  name: string;
  aggregated_file: string;
  features_dir: string;
  strategy: "mean" | "max";
  question_centering?: boolean;
  n_roles: number;
  sae_dim: number;
  top_k: number;
  selection?: {
    role_matrix_mode?: string;
  };
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

export type FeaturePoint = FeatureRow & {
  x: number;
  y: number;
};

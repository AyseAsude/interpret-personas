export type ConversationMessage = {
  role: string;
  content: string;
};

export type ConversationPayload = {
  conversation: ConversationMessage[];
};

export type TurnScore = {
  assistant_turn: number;
  score: number;
};

export type TransitionScore = {
  prev_turn: number;
  turn: number;
  delta_score: number;
};

export type TransitionFeature = {
  feature_idx: number;
  direction: "activated" | "deactivated";
  delta_x: number;
  axis_weight: number;
  contrib: number;
  axis_side: "assistant_like" | "non_assistant_like";
  effect: "away_from_assistant" | "toward_assistant";
  share_of_change: number;
  share_of_away: number;
  role_profile: string;
};

export type DriftTransition = TransitionScore & {
  away_mass: number;
  toward_mass: number;
  activated: TransitionFeature[];
  deactivated: TransitionFeature[];
};

export type TokenLocalizationTurn = {
  assistant_turn: number;
  tokens: string[];
  u_k: number[];
  u_k_norm: number[];
};

export type PersonaDriftBundle = {
  version: number;
  run_meta: {
    model_name: string;
    sae_id: string;
    assistant_role: string;
    score_method: string;
    conversation_hash: string;
  };
  q1: {
    turn_scores: TurnScore[];
    transitions: TransitionScore[];
  };
  q2: {
    top_features_per_direction: number;
    transitions: DriftTransition[];
  };
  q3: {
    token_localization: TokenLocalizationTurn[];
  };
};

export type SummaryPayload = {
  run_name?: string;
  conversation_name?: string;
  timestamp_utc?: string;
  config?: {
    top_features?: number;
    score_method?: string;
    [k: string]: unknown;
  };
  selection?: {
    selected_turn?: number;
    selected_prev_turn?: number;
    selected_delta_score?: number;
    [k: string]: unknown;
  };
  diagnostics?: {
    confound_verdict?: string;
    [k: string]: unknown;
  };
};

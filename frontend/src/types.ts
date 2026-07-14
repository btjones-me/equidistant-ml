export type CombineMode = "max" | "mean" | "fairness" | "balanced";
export type DetailMode = "fast" | "fine";
export type FocusMode = "central" | "inner" | "wide";
export type PaletteMode = "central" | "green-red" | "viridis" | "inferno" | "custom" | "error";
export type ViewMode = "model" | "graph" | "residual" | "reference" | "error";

export type ColorScale = {
  lowerPercentile: number;
  upperPercentile: number;
  contrast: number;
};

export type ColorMapStop = {
  position: number;
  color: string;
};

export type EditableColorMapStop = ColorMapStop & {
  id: string;
};

export type Friend = {
  id: string;
  name: string;
  lat: number;
  lng: number;
  locationLabel?: string;
};

export type ExperienceMode = "explore" | "developer";

export type ComparisonMetrics = {
  reference_origin_count?: number;
  included_friend_indexes?: number[];
  mae_minutes?: number | null;
  median_abs_error_minutes?: number | null;
  p90_abs_error_minutes?: number | null;
  within_5_min_pct?: number | null;
  within_10_min_pct?: number | null;
  mean_signed_error_minutes?: number | null;
  model_distance_correlation?: number | null;
  reference_distance_correlation?: number | null;
};

export type SurfaceCell = {
  destination_id: string;
  lat: number;
  lng: number;
  x_index: number;
  y_index: number;
  south?: number;
  north?: number;
  west?: number;
  east?: number;
  boundary?: [number, number][];
  h3_cell?: string;
  h3_resolution?: number;
  grid_band?: string;
  grid_priority?: number;
  cell_area_km2?: number;
  score_minutes?: number;
  model_score_minutes?: number;
  graph_score_minutes?: number;
  model_residual_minutes?: number;
  graph_modes?: string;
  nearest_corridors?: string;
  graph_interchanges?: number;
  graph_access_seconds?: number;
  graph_egress_seconds?: number;
  reference_score_minutes?: number;
  abs_error_minutes?: number;
  signed_error_minutes?: number;
  travel_time_minutes?: number;
  max_seconds?: number;
  max_minutes?: number;
  mean_seconds?: number;
  mean_minutes?: number;
  fairness_seconds?: number;
  fairness_minutes?: number;
  [key: string]: string | number | boolean | [number, number][] | undefined;
};

export type SurfaceResponse = {
  lats: number[];
  lngs: number[];
  Z: number[][];
  cells: SurfaceCell[];
  metadata?: {
    value_column: string;
    cell_count: number;
    grid_type: "uniform" | "h3";
    friend_columns?: string[];
    value_columns?: {
      model: string;
      graph?: string;
      residual?: string;
      reference: string;
      error: string;
    };
    comparison?: ComparisonMetrics;
    min?: number | null;
    max?: number | null;
    p10?: number | null;
    p50?: number | null;
    p90?: number | null;
    source?: "api" | "browser_atlas";
    model_type?: string;
    interpolation_mae_minutes?: number;
    coverage_notice?: string;
  };
};

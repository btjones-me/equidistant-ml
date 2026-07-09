export type CombineMode = "max" | "mean" | "fairness" | "balanced";

export type Friend = {
  name: string;
  lat: number;
  lng: number;
};

export type SurfaceCell = {
  destination_id: string;
  lat: number;
  lng: number;
  x_index: number;
  y_index: number;
  score_minutes?: number;
  travel_time_minutes?: number;
  max_seconds?: number;
  mean_seconds?: number;
  fairness_seconds?: number;
};

export type SurfaceResponse = {
  lats: number[];
  lngs: number[];
  Z: number[][];
  cells: SurfaceCell[];
};

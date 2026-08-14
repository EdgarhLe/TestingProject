export type TaskType = 'KIS' | 'Trake' | 'Q&A';

export interface SearchRequest {
  query: string;
  taskType: TaskType;
}

export interface KisResult {
  video_id: string;
  video_fps: number;
  frame_index: number;
  start_frame: number;
  end_frame: number;
  timestamp_seconds: number;
  score: number;
}

export interface KisResponse {
  results: KisResult[];
  total_results: number;
  has_next: boolean;
  query_time_ms: number;
}

export interface TrakeEvent {
  event_index: number;
  description: string;
  frame_index: number;
}

export interface TrakeResult {
  video_id: string;
  events: TrakeEvent[];
}

export interface QaResult {
  video_id: string;
  video_fps: number;
  frame_index: number;
  start_frame: number;
  end_frame: number;
  timestamp_seconds: number;
  answer: string;
  answer_source: 'ocr' | 'llm';
  score: number;
}

export interface QaResponse {
  results: QaResult[];
  total_results: number;
  has_next: boolean;
  query_time_ms: number;
}

export interface SearchResponse {
  taskType: TaskType;
  results: (KisResult | TrakeResult | QaResult)[];
}

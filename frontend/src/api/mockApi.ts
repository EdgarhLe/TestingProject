import { SearchRequest, SearchResponse, KisResult, TrakeResult, QaResult } from './types';

const MOCK_KIS_RESULTS: KisResult[] = [
  {
    video_id: 'L01_V001',
    video_fps: 25,
    frame_index: 235,
    start_frame: 220,
    end_frame: 250,
    timestamp_seconds: 9.4,
    score: 0.98,
  },
  {
    video_id: 'L01_V008',
    video_fps: 25,
    frame_index: 412,
    start_frame: 400,
    end_frame: 430,
    timestamp_seconds: 16.48,
    score: 0.95,
  },
  {
    video_id: 'L02_V001',
    video_fps: 25,
    frame_index: 112,
    start_frame: 100,
    end_frame: 125,
    timestamp_seconds: 4.48,
    score: 0.93,
  },
  {
    video_id: 'L03_V010',
    video_fps: 30,
    frame_index: 889,
    start_frame: 870,
    end_frame: 910,
    timestamp_seconds: 29.63,
    score: 0.91,
  },
  {
    video_id: 'L05_V003',
    video_fps: 25,
    frame_index: 420,
    start_frame: 400,
    end_frame: 440,
    timestamp_seconds: 16.8,
    score: 0.88,
  },
];

const MOCK_TRAKE_RESULTS: TrakeResult[] = [
  {
    video_id: 'V_TRAKE_101',
    events: [
      { event_index: 0, description: 'Person in black jacket enters the store', frame_index: 450 },
      { event_index: 1, description: 'Person picks up a red bottle from the shelf', frame_index: 680 },
      { event_index: 2, description: 'Person exits through the main sliding door', frame_index: 920 },
    ],
  },
  {
    video_id: 'V_TRAKE_102',
    events: [
      { event_index: 0, description: 'Yellow delivery van halts near intersection', frame_index: 120 },
      { event_index: 1, description: 'Driver unloads a medium brown parcel box', frame_index: 310 },
      { event_index: 2, description: 'Van drives away turning north-east', frame_index: 540 },
    ],
  },
];

const MOCK_QA_RESULTS: QaResult[] = [
  {
    video_id: 'V_QA_201',
    video_fps: 25,
    frame_index: 4320,
    start_frame: 4290,
    end_frame: 4360,
    timestamp_seconds: 172.8,
    answer: 'The visual text on the screen displays the license plate "29A-555.22" on the blue sedan.',
    answer_source: 'ocr',
    score: 0.91,
  },
  {
    video_id: 'V_QA_202',
    video_fps: 30,
    frame_index: 1040,
    start_frame: 1010,
    end_frame: 1080,
    timestamp_seconds: 34.67,
    answer: 'Yes, the shop assistant wearing the green shirt is holding a black clipboard.',
    answer_source: 'llm',
    score: 0.78,
  },
  {
    video_id: 'V_QA_203',
    video_fps: 25,
    frame_index: 7800,
    start_frame: 7760,
    end_frame: 7840,
    timestamp_seconds: 312,
    answer: 'There are a total of four people dining at the outdoor patio table.',
    answer_source: 'ocr',
    score: 0.67,
  },
];

/**
 * Simulates a backend search request with a delay between 500ms and 800ms
 */
export const mockSearch = (request: SearchRequest): Promise<SearchResponse> => {
  return new Promise((resolve) => {
    const delay = Math.floor(Math.random() * (800 - 500 + 1)) + 500;
    
    setTimeout(() => {
      let filteredResults: (KisResult | TrakeResult | QaResult)[] = [];
      const queryLower = request.query.toLowerCase().trim();

      if (request.taskType === 'KIS') {
        filteredResults = queryLower
          ? MOCK_KIS_RESULTS.filter(
              (r) =>
                r.video_id.toLowerCase().includes(queryLower) ||
                r.frame_index.toString().includes(queryLower)
            )
          : MOCK_KIS_RESULTS;
      } else if (request.taskType === 'Trake') {
        filteredResults = queryLower
          ? MOCK_TRAKE_RESULTS.filter(
              (r) =>
                r.video_id.toLowerCase().includes(queryLower) ||
                r.events.some((e) => e.description.toLowerCase().includes(queryLower))
            )
          : MOCK_TRAKE_RESULTS;
      } else if (request.taskType === 'Q&A') {
        filteredResults = queryLower
          ? MOCK_QA_RESULTS.filter(
              (r) =>
                r.video_id.toLowerCase().includes(queryLower) ||
                r.answer.toLowerCase().includes(queryLower)
            )
          : MOCK_QA_RESULTS;
      }

      resolve({
        taskType: request.taskType,
        results: filteredResults,
      });
    }, delay);
  });
};

export const TASK_TYPES = {
  KIS: 'KIS',
  TRAKE: 'Trake',
  QA: 'Q&A',
} as const;

export type TaskTypeValue = typeof TASK_TYPES[keyof typeof TASK_TYPES];

export const TASK_TYPE_LABELS = [
  { value: TASK_TYPES.KIS, label: 'KIS (Known-Item Search)', description: 'Find a specific video scene based on a text prompt.' },
  { value: TASK_TYPES.TRAKE, label: 'Trake (Temporal Event Tracking)', description: 'Track a sequence of sequential events in the timeline.' },
  { value: TASK_TYPES.QA, label: 'Q&A (Visual Question Answering)', description: 'Answer questions using text and OCR matching.' },
];

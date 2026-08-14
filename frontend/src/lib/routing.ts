import type { QaRequest, KisRequest, TrakeRequest } from '@/api/queryParser';
import type { TaskType } from '@/api/types';

export type ManualSearchPayload = KisRequest | TrakeRequest | QaRequest;

const RESULTS_ROUTES: Record<TaskType, string> = {
  KIS: '/results/kis',
  Trake: '/results/trake',
  'Q&A': '/results/qa',
};

export function buildResultsPath(
  taskType: TaskType,
  payload: ManualSearchPayload,
): string {
  const params = new URLSearchParams();

  if (payload.query_id.trim()) {
    params.set('query_id', payload.query_id);
  }

  params.set('top_k', String(payload.top_k));

  if (taskType === 'KIS') {
    params.set('query', (payload as KisRequest).query);
  } else if (taskType === 'Trake') {
    for (const event of (payload as TrakeRequest).events) {
      params.append('event', event);
    }
  } else {
    const request = payload as QaRequest;
    params.set('scene', request.scene);
    params.set('question', request.question);
  }

  return `${RESULTS_ROUTES[taskType]}?${params.toString()}`;
}

import type { QaRequest } from './queryParser';
import type { QaResponse } from './types';

interface ErrorResponseBody {
  code?: string;
  message?: string;
  detail?: string;
}

export class QaApiError extends Error {
  readonly status: number;
  readonly code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = 'QaApiError';
    this.status = status;
    this.code = code;
  }
}

function getApiBaseUrl() {
  if (import.meta.env.DEV) {
    return '';
  }

  return (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
}

export async function fetchQaResults(
  request: QaRequest,
  signal?: AbortSignal,
): Promise<QaResponse> {
  const response = await fetch(`${getApiBaseUrl()}/query/qa`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
    signal,
  });

  const body = (await response.json().catch(() => undefined)) as
    | (QaResponse & ErrorResponseBody)
    | undefined;

  if (!response.ok) {
    throw new QaApiError(
      body?.message || body?.detail || `Q&A request failed (${response.status})`,
      response.status,
      body?.code,
    );
  }

  if (!body || !Array.isArray(body.results)) {
    throw new QaApiError('The Q&A API returned an invalid response.', response.status);
  }

  return body;
}

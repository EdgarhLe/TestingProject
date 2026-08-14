import type { KisRequest } from './queryParser';
import type { KisResponse } from './types';

interface ErrorResponseBody {
  code?: string;
  message?: string;
  detail?: string;
}

export class KisApiError extends Error {
  readonly status: number;
  readonly code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = 'KisApiError';
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

export async function fetchKisResults(
  request: KisRequest,
  signal?: AbortSignal,
): Promise<KisResponse> {
  const response = await fetch(`${getApiBaseUrl()}/query/kis`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
    signal,
  });

  const body = (await response.json().catch(() => undefined)) as
    | (KisResponse & ErrorResponseBody)
    | undefined;

  if (!response.ok) {
    throw new KisApiError(
      body?.message || body?.detail || `KIS request failed (${response.status})`,
      response.status,
      body?.code,
    );
  }

  if (!body || !Array.isArray(body.results)) {
    throw new KisApiError('The KIS API returned an invalid response.', response.status);
  }

  return body;
}

import * as React from 'react';
import { ArrowLeft, Search } from 'lucide-react';
import { Link, useSearchParams } from 'react-router-dom';
import type { KisRequest } from '@/api/queryParser';
import type { KisResponse } from '@/api/types';
import { fetchKisResults } from '@/api/kisApi';
import { KisResultView } from '@/components/results/KisResultView';

export const KisResultsPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const query = searchParams.get('query')?.trim() ?? '';
  const queryId = searchParams.get('query_id') ?? '';
  const parsedTopK = Number.parseInt(searchParams.get('top_k') ?? '', 10);
  const topK = Number.isFinite(parsedTopK) && parsedTopK > 0 ? parsedTopK : 5;

  const [response, setResponse] = React.useState<KisResponse | null>(null);
  const [errorMessage, setErrorMessage] = React.useState<string | null>(() => (
    query ? null : 'Missing KIS query. Return to search and submit a query first.'
  ));
  const [isLoading, setIsLoading] = React.useState(Boolean(query));
  const [retryToken, setRetryToken] = React.useState(0);

  React.useEffect(() => {
    const controller = new AbortController();

    setResponse(null);
    setErrorMessage(null);

    if (!query) {
      setIsLoading(false);
      setErrorMessage('Missing KIS query. Return to search and submit a query first.');
      return () => controller.abort();
    }

    const request: KisRequest = {
      query_id: queryId,
      query,
      top_k: topK,
    };

    setIsLoading(true);
    fetchKisResults(request, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) {
          setResponse(result);
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setErrorMessage(error instanceof Error ? error.message : 'Unable to load KIS results.');
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setIsLoading(false);
        }
      });

    return () => controller.abort();
  }, [query, queryId, retryToken, topK]);

  return (
    <div className="flex w-full flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-300">
      <Link
        to="/"
        className="inline-flex w-fit items-center gap-2 text-sm font-medium text-slate-400 transition-colors hover:text-white"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to search
      </Link>

      <div className="max-w-3xl space-y-3">
        <div className="inline-flex items-center gap-1.5 rounded-full border border-indigo-500/20 bg-indigo-500/10 px-3 py-1 text-xs font-semibold text-indigo-400">
          <Search className="h-3.5 w-3.5" />
          <span>KIS result view</span>
        </div>
        <h1 className="text-3xl font-extrabold tracking-tight text-white md:text-4xl">
          Known-Item Search Results
        </h1>
        <p className="break-words text-sm leading-relaxed text-slate-400">
          Query: <span className="font-mono text-slate-300">{query || 'Missing query'}</span>
        </p>
      </div>

      <KisResultView
        response={response}
        isLoading={isLoading}
        error={errorMessage}
        onRetry={() => setRetryToken((value) => value + 1)}
      />
    </div>
  );
};

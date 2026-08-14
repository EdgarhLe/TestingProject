import * as React from 'react';
import { ArrowLeft, HelpCircle } from 'lucide-react';
import { Link, useSearchParams } from 'react-router-dom';
import type { QaRequest } from '@/api/queryParser';
import type { QaResponse } from '@/api/types';
import { fetchQaResults } from '@/api/qaApi';
import { QaResultView } from '@/components/results/QaResultView';

export const QaResultsPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const scene = searchParams.get('scene')?.trim() ?? '';
  const question = searchParams.get('question')?.trim() ?? '';
  const queryId = searchParams.get('query_id') ?? '';
  const parsedTopK = Number.parseInt(searchParams.get('top_k') ?? '', 10);
  const topK = Number.isFinite(parsedTopK) && parsedTopK > 0 ? parsedTopK : 3;
  const [response, setResponse] = React.useState<QaResponse | null>(null);
  const [errorMessage, setErrorMessage] = React.useState<string | null>(() => (
    scene && question ? null : 'Missing Q&A scene or question. Return to search and submit a query first.'
  ));
  const [isLoading, setIsLoading] = React.useState(Boolean(scene && question));
  const [retryToken, setRetryToken] = React.useState(0);

  React.useEffect(() => {
    const controller = new AbortController();

    setResponse(null);
    setErrorMessage(null);

    if (!scene || !question) {
      setIsLoading(false);
      setErrorMessage('Missing Q&A scene or question. Return to search and submit a query first.');
      return () => controller.abort();
    }

    const request: QaRequest = {
      query_id: queryId,
      scene,
      question,
      top_k: topK,
    };

    setIsLoading(true);
    fetchQaResults(request, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setResponse(result);
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setErrorMessage(error instanceof Error ? error.message : 'Unable to load Q&A results.');
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsLoading(false);
      });

    return () => controller.abort();
  }, [scene, question, queryId, retryToken, topK]);

  return (
    <div className="flex min-w-0 w-full flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-300">
      <Link to="/" className="inline-flex w-fit items-center gap-2 text-sm font-medium text-slate-400 transition-colors hover:text-white">
        <ArrowLeft className="h-4 w-4" /> Back to search
      </Link>

      <div className="min-w-0 max-w-3xl space-y-3">
        <div className="inline-flex items-center gap-1.5 rounded-full border border-indigo-500/20 bg-indigo-500/10 px-3 py-1 text-xs font-semibold text-indigo-400">
          <HelpCircle className="h-3.5 w-3.5" /> <span>Q&amp;A result view</span>
        </div>
        <h1 className="break-words text-2xl font-extrabold tracking-tight text-white sm:text-3xl md:text-4xl">Question &amp; Answer Results</h1>
        <div className="space-y-1 text-sm leading-relaxed text-slate-400">
          <p className="break-words">Scene: <span className="text-slate-300">{scene || 'Missing scene'}</span></p>
          <p className="break-words">Question: <span className="font-medium text-slate-300">{question || 'Missing question'}</span></p>
        </div>
      </div>

      <QaResultView
        response={response}
        isLoading={isLoading}
        error={errorMessage}
        onRetry={() => setRetryToken((value) => value + 1)}
      />
    </div>
  );
};

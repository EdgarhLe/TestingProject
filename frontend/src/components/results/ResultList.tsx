import * as React from 'react';
import { AlertTriangle, Inbox, RefreshCw } from 'lucide-react';

export interface ResultListProps<T> {
  isLoading: boolean;
  error: string | null;
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  totalResults: number;
  hasNext: boolean;
  onRetry: () => void;
  onLoadMore?: () => void;
  emptyMessage?: string;
  emptyHint?: string;
}

const skeletonKeys = ['one', 'two', 'three'];

export function ResultList<T>({
  isLoading,
  error,
  items,
  renderItem,
  totalResults,
  hasNext,
  onRetry,
  onLoadMore,
  emptyMessage = 'No results found',
  emptyHint = 'Try a different query or search task.',
}: ResultListProps<T>) {
  if (isLoading) {
    return (
      <div className="flex flex-col gap-3" role="status" aria-label="Loading results">
        {skeletonKeys.map((key) => (
          <div
            key={key}
            className="animate-pulse rounded-xl border border-slate-800/80 bg-slate-900/40 p-4"
          >
            <div className="flex items-center justify-between gap-4">
              <div className="flex min-w-0 flex-1 items-center gap-3">
                <div className="h-10 w-10 shrink-0 rounded-lg bg-slate-800" />
                <div className="flex min-w-0 flex-1 flex-col gap-2">
                  <div className="h-3 w-2/5 rounded bg-slate-800" />
                  <div className="h-2.5 w-3/5 rounded bg-slate-800/80" />
                </div>
              </div>
              <div className="flex w-28 flex-col items-end gap-2">
                <div className="h-5 w-20 rounded bg-slate-800" />
                <div className="h-1.5 w-28 rounded bg-slate-800/80" />
              </div>
            </div>
          </div>
        ))}
        <p className="text-center text-xs text-slate-500">Loading results...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex min-h-[260px] flex-col items-center justify-center rounded-xl border border-rose-500/20 bg-rose-500/5 p-12 text-center">
        <AlertTriangle className="mb-3 h-8 w-8 text-rose-400" />
        <h3 className="text-sm font-semibold text-slate-200">Could not load results</h3>
        <p className="mt-2 max-w-xl text-xs leading-relaxed text-rose-300/80">{error}</p>
        <button
          type="button"
          onClick={onRetry}
          className="mt-5 inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-4 py-2 text-xs font-semibold text-slate-200 transition-colors hover:border-indigo-500/50 hover:text-white"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Retry request
        </button>
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="flex min-h-[300px] flex-col items-center justify-center rounded-xl border border-slate-900 bg-slate-950/20 p-12 text-center">
        <Inbox className="mb-3 h-10 w-10 text-slate-700" />
        <h3 className="text-sm font-semibold text-slate-300">{emptyMessage}</h3>
        <p className="mt-1 max-w-md text-xs text-slate-500">{emptyHint}</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex max-h-[620px] flex-col gap-3 overflow-y-auto pr-1">
        {items.map((item, index) => (
          <React.Fragment key={index}>{renderItem(item, index)}</React.Fragment>
        ))}
      </div>

      <footer className="flex flex-col items-start justify-between gap-3 border-t border-slate-800/70 pt-3 text-xs text-slate-500 sm:flex-row sm:items-center">
        <span aria-live="polite">
          Showing {items.length} / {totalResults} results
        </span>
        {hasNext && (
          <button
            type="button"
            onClick={onLoadMore}
            disabled={!onLoadMore}
            className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-semibold text-slate-300 transition-colors hover:border-indigo-500/50 hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
          >
            Load More
          </button>
        )}
      </footer>
    </div>
  );
}

import * as React from 'react';
import { ArrowLeft, Braces, Route } from 'lucide-react';
import { Link, useLocation, useSearchParams } from 'react-router-dom';
import type { TaskType } from '@/api/types';

interface ResultsPlaceholderPageProps {
  taskType: TaskType;
}

export const ResultsPlaceholderPage: React.FC<ResultsPlaceholderPageProps> = ({
  taskType,
}) => {
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const queryEntries = Array.from(searchParams.entries()).map(([key, value]) => ({
    key,
    value,
  }));

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
          <Route className="h-3.5 w-3.5" />
          <span>{taskType} results route</span>
        </div>
        <h1 className="text-3xl font-extrabold tracking-tight text-white md:text-4xl">
          {taskType} Results
        </h1>
        <p className="text-sm leading-relaxed text-slate-400">
          This placeholder confirms that the search payload reached the correct
          route. The real result view will be implemented in the next milestone.
        </p>
      </div>

      <div className="grid max-w-4xl gap-6 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
        <div className="glass-panel rounded-2xl border border-slate-800/80 bg-slate-950/20 p-5">
          <div className="mb-4 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-400">
            <Route className="h-4 w-4 text-indigo-400" />
            Current route
          </div>
          <code className="block break-all rounded-lg border border-slate-800 bg-slate-950/70 px-3 py-3 text-sm text-indigo-300">
            {location.pathname}
          </code>
        </div>

        <div className="glass-panel rounded-2xl border border-slate-800/80 bg-slate-950/20 p-5">
          <div className="mb-4 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-400">
            <Braces className="h-4 w-4 text-indigo-400" />
            Query parameters
          </div>
          {queryEntries.length > 0 ? (
            <pre className="max-h-80 overflow-auto rounded-lg border border-slate-800 bg-slate-950/70 p-3 text-xs leading-relaxed text-slate-300">
              {JSON.stringify(queryEntries, null, 2)}
            </pre>
          ) : (
            <p className="rounded-lg border border-dashed border-slate-800 px-3 py-6 text-center text-sm text-slate-500">
              No query parameters received.
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

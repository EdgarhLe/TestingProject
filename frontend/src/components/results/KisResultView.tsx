import * as React from 'react';
import { BarChart3, CheckCircle2, Clock3, Video } from 'lucide-react';
import type { KisResponse } from '@/api/types';
import { KisResultCard } from './KisResultCard';
import { ResultList } from './ResultList';
import { VideoPlayer } from '@/components/player/VideoPlayer';
import { resolveVideoUrl } from '@/lib/video';

interface KisResultViewProps {
  response: KisResponse | null;
  isLoading: boolean;
  error: string | null;
  onRetry: () => void;
}

export const KisResultView: React.FC<KisResultViewProps> = ({
  response,
  isLoading,
  error,
  onRetry,
}) => {
  const [selectedIndex, setSelectedIndex] = React.useState(0);
  const results = response?.results ?? [];
  const selectedResult = results[selectedIndex];

  React.useEffect(() => {
    setSelectedIndex(0);
  }, [response]);

  return (
    <div className="flex w-full flex-col gap-6">
      {response && !isLoading && !error && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="glass-panel rounded-xl border border-slate-800/80 bg-slate-950/20 p-4">
            <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">
              <BarChart3 className="h-3.5 w-3.5 text-indigo-400" />
              Results
            </div>
            <p className="mt-2 text-xl font-bold text-white">{response.total_results}</p>
          </div>
          <div className="glass-panel rounded-xl border border-slate-800/80 bg-slate-950/20 p-4">
            <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">
              <Clock3 className="h-3.5 w-3.5 text-cyan-400" />
              Query time
            </div>
            <p className="mt-2 text-xl font-bold text-white">{response.query_time_ms}ms</p>
          </div>
          <div className="glass-panel rounded-xl border border-slate-800/80 bg-slate-950/20 p-4">
            <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">
              <Video className="h-3.5 w-3.5 text-emerald-400" />
              Selected
            </div>
            <p className="mt-2 truncate text-sm font-bold text-white">
              {selectedResult?.video_id ?? 'None'}
            </p>
          </div>
          <div className="glass-panel rounded-xl border border-slate-800/80 bg-slate-950/20 p-4">
            <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">
              <CheckCircle2 className="h-3.5 w-3.5 text-amber-400" />
              Pagination
            </div>
            <p className="mt-2 text-sm font-bold text-white">
              {response.has_next ? 'More available' : 'Complete'}
            </p>
          </div>
        </div>
      )}

      <div className={selectedResult ? 'grid grid-cols-1 items-start gap-6 lg:grid-cols-12' : 'w-full'}>
        <section className={selectedResult ? 'flex min-w-0 flex-col gap-4 lg:col-span-7' : 'w-full'}>
          <div className="flex items-center justify-between">
            <div>
              <h2 className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-slate-400">
                <BarChart3 className="h-3.5 w-3.5 text-indigo-400" />
                Retrieved KIS candidates
              </h2>
              <p className="mt-1 text-xs text-slate-500">
                Select a card to inspect its target segment.
              </p>
            </div>
            <span className="text-xs font-mono text-slate-500">
              {results.length} loaded
            </span>
          </div>

          <ResultList
            isLoading={isLoading}
            error={error}
            items={results}
            totalResults={response?.total_results ?? 0}
            hasNext={response?.has_next ?? false}
            onRetry={onRetry}
            emptyMessage="No KIS results found"
            emptyHint="Try a different query to find matching video segments."
            renderItem={(result, index) => (
              <KisResultCard
                result={result}
                isActive={index === selectedIndex}
                onClick={() => setSelectedIndex(index)}
              />
            )}
          />
        </section>

        {selectedResult && (
          <section className="min-w-0 lg:sticky lg:top-24 lg:col-span-5">
            <VideoPlayer
              videoUrl={resolveVideoUrl(selectedResult.video_id)}
              videoId={selectedResult.video_id}
              fps={selectedResult.video_fps}
              frameIndex={selectedResult.frame_index}
              startFrame={selectedResult.start_frame}
              endFrame={selectedResult.end_frame}
              timestampSeconds={selectedResult.timestamp_seconds}
            />
          </section>
        )}
      </div>
    </div>
  );
};

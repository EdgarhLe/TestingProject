import * as React from 'react';
import { HelpCircle, MessageSquare, Play, Target } from 'lucide-react';
import type { QaResult } from '@/api/types';
import { Badge } from '@/components/ui/Badge';

interface QaResultCardProps {
  result: QaResult;
  onClick: () => void;
  isActive: boolean;
}

export const QaResultCard: React.FC<QaResultCardProps> = ({
  result,
  onClick,
  isActive,
}) => {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={isActive}
      className={`glass-panel p-4 rounded-xl border cursor-pointer glass-card-hover group flex flex-col gap-3 ${
        isActive
          ? 'border-indigo-500 bg-indigo-950/20 shadow-md shadow-indigo-500/5'
          : 'border-slate-800/80 bg-slate-900/10 hover:border-slate-700/60'
      }`}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className={`h-9 w-9 rounded-lg flex items-center justify-center transition-colors ${
            isActive ? 'bg-indigo-500 text-white' : 'bg-slate-900 text-slate-400 group-hover:bg-slate-800'
          }`}>
            <HelpCircle className="h-4 w-4" />
          </div>
          <div className="min-w-0">
            <h4 className="text-sm font-semibold text-slate-200 group-hover:text-white transition-colors">
              {result.video_id}
            </h4>
            <p className="text-[10px] text-slate-500">Answer source</p>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Badge variant={result.answer_source === 'ocr' ? 'success' : 'warning'}>
            {result.answer_source.toUpperCase()}
          </Badge>
          <Play className="h-4 w-4 text-indigo-400 opacity-0 transition-opacity group-hover:opacity-100" />
        </div>
      </div>

      <div className="flex items-start gap-2.5 rounded-lg border border-slate-900 bg-slate-950/40 p-3 text-left">
        <MessageSquare className="h-4 w-4 text-indigo-400 mt-0.5 shrink-0" />
        <p className="text-sm font-semibold leading-relaxed text-slate-100">
          {result.answer}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-slate-900/80 pt-3 text-[10px] text-slate-500">
        <span className="inline-flex items-center gap-1">
          <Target className="h-3 w-3 text-indigo-400" />
          Frame <span className="font-mono text-slate-300">{result.frame_index}</span>
        </span>
        <span>Segment <span className="font-mono text-slate-300">{result.start_frame}-{result.end_frame}</span></span>
        <span>At <span className="font-mono text-slate-300">{result.timestamp_seconds.toFixed(2)}s</span></span>
        <span className="ml-auto inline-flex items-center gap-1">
          Score <span className="font-mono font-semibold text-emerald-300">{(result.score * 100).toFixed(0)}%</span>
        </span>
      </div>
    </button>
  );
};

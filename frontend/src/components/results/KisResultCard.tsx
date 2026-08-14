import * as React from 'react';
import { Play, Target } from 'lucide-react';
import { KisResult } from '@/api/types';
import { Badge } from '@/components/ui/Badge';

interface KisResultCardProps {
  result: KisResult;
  onClick: () => void;
  isActive: boolean;
}

export const KisResultCard: React.FC<KisResultCardProps> = ({
  result,
  onClick,
  isActive,
}) => {
  const scorePercent = Math.min(100, Math.max(0, result.score * 100));
  const percentage = scorePercent.toFixed(0);

  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={isActive}
      className={`glass-panel w-full p-4 rounded-xl flex items-center justify-between border text-left cursor-pointer glass-card-hover group ${
        isActive
          ? 'border-indigo-500 bg-indigo-950/20 shadow-md shadow-indigo-500/5'
          : 'border-slate-800/80 bg-slate-900/10 hover:border-slate-700/60'
      }`}
    >
      <div className="flex items-center gap-3">
        <div className={`h-10 w-10 rounded-lg flex items-center justify-center transition-colors ${
          isActive ? 'bg-indigo-500 text-white' : 'bg-slate-900 text-slate-400 group-hover:bg-slate-800'
        }`}>
          <Target className="h-5 w-5" />
        </div>
        <div>
          <h4 className="text-sm font-semibold text-slate-200 group-hover:text-white transition-colors">
            {result.video_id}
          </h4>
          <p className="text-xs text-slate-400 mt-1">
            Frame: <span className="font-mono text-slate-300">{result.frame_index}</span>
            <span className="mx-1 text-slate-600">|</span>
            Segment: <span className="font-mono text-slate-300">{result.start_frame}-{result.end_frame}</span>
          </p>
        </div>
      </div>

      <div className="flex min-w-32 flex-col items-end gap-1.5">
        <div className="flex items-center gap-2">
          <Badge variant={result.score > 0.8 ? 'success' : 'primary'}>
            {percentage}% Match
          </Badge>
          <span className="text-[10px] font-mono text-slate-500">
            {result.timestamp_seconds.toFixed(2)}s
          </span>
        </div>
        <div className="h-1.5 w-28 overflow-hidden rounded-full bg-slate-800">
          <div
            className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-cyan-400 transition-all"
            style={{ width: `${scorePercent}%` }}
          />
        </div>
        <span className="text-slate-500 group-hover:text-slate-300 transition-colors">
          <Play className="h-4 w-4 fill-current opacity-0 group-hover:opacity-100 transition-opacity" />
        </span>
      </div>
    </button>
  );
};

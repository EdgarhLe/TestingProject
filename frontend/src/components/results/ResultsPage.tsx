import * as React from 'react';
import { TaskType, KisResult, TrakeResult, QaResult } from '@/api/types';
import { KisResultCard } from './KisResultCard';
import { QaResultCard } from './QaResultCard';
import { TrakeResultTimeline } from './TrakeResultTimeline';
import { Inbox, Compass } from 'lucide-react';

interface ResultsPageProps {
  results: any[];
  taskType: TaskType;
  onSelectResult: (videoId: string, frameIndex: number) => void;
  selectedVideoId?: string;
  selectedFrameIndex?: number;
}

export const ResultsPage: React.FC<ResultsPageProps> = ({
  results,
  taskType,
  onSelectResult,
  selectedVideoId,
  selectedFrameIndex,
}) => {
  if (results.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center text-center p-12 border border-slate-900 bg-slate-950/20 rounded-xl min-h-[300px]">
        <Inbox className="h-10 w-10 text-slate-700 mb-3" />
        <h4 className="text-sm font-semibold text-slate-400">No results found</h4>
        <p className="text-xs text-slate-500 mt-1 max-w-[280px]">
          Try typing a different keyword or check alternative search tasks.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold tracking-widest text-slate-500 uppercase flex items-center gap-1.5">
          <Compass className="h-3.5 w-3.5" />
          Retrieved Candidates ({results.length})
        </h3>
      </div>

      <div className="space-y-3 max-h-[500px] overflow-y-auto pr-1">
        {taskType === 'KIS' &&
          (results as KisResult[]).map((res, idx) => (
            <KisResultCard
              key={`${res.video_id}-${res.frame_index}-${idx}`}
              result={res}
              isActive={selectedVideoId === res.video_id && selectedFrameIndex === res.frame_index}
              onClick={() => onSelectResult(res.video_id, res.frame_index)}
            />
          ))}

        {taskType === 'Trake' &&
          (results as TrakeResult[]).map((res, idx) => (
            <TrakeResultTimeline
              key={`${res.video_id}-${idx}`}
              result={res}
              isActive={selectedVideoId === res.video_id}
              activeFrameIndex={selectedVideoId === res.video_id ? selectedFrameIndex : undefined}
              onClick={(frameIndex) => onSelectResult(res.video_id, frameIndex ?? 0)}
            />
          ))}

        {taskType === 'Q&A' &&
          (results as QaResult[]).map((res, idx) => (
            <QaResultCard
              key={`${res.video_id}-${res.frame_index}-${idx}`}
              result={res}
              isActive={selectedVideoId === res.video_id && selectedFrameIndex === res.frame_index}
              onClick={() => onSelectResult(res.video_id, res.frame_index)}
            />
          ))}
      </div>
    </div>
  );
};

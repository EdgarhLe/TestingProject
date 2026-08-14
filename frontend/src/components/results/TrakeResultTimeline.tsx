import * as React from 'react';
import { Milestone } from 'lucide-react';
import { TrakeResult } from '@/api/types';
import { Badge } from '@/components/ui/Badge';

interface TrakeResultTimelineProps {
  result: TrakeResult;
  onClick: (frameIndex?: number) => void;
  isActive: boolean;
  activeFrameIndex?: number;
}

export const TrakeResultTimeline: React.FC<TrakeResultTimelineProps> = ({
  result,
  onClick,
  isActive,
  activeFrameIndex,
}) => {
  return (
    <div
      className={`glass-panel p-4 rounded-xl border flex flex-col gap-4 transition-all ${
        isActive
          ? 'border-indigo-500 bg-indigo-950/20 shadow-md'
          : 'border-slate-800/80 bg-slate-900/10 hover:border-slate-700/60'
      }`}
    >
      {/* Top Header */}
      <div 
        onClick={() => onClick(result.events[0]?.frame_index)}
        className="flex items-center justify-between cursor-pointer group"
      >
        <div className="flex items-center gap-3">
          <div className={`h-9 w-9 rounded-lg flex items-center justify-center transition-colors ${
            isActive ? 'bg-indigo-500 text-white' : 'bg-slate-900 text-slate-400 group-hover:bg-slate-800'
          }`}>
            <Milestone className="h-4 w-4" />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-slate-200 group-hover:text-white transition-colors">
              {result.video_id}
            </h4>
            <p className="text-[10px] text-slate-500">
              {result.events.length} temporal events tracked
            </p>
          </div>
        </div>
        <Badge variant="outline" className="text-[10px]">
          Timeline Sequence
        </Badge>
      </div>

      {/* Sequential Timeline List */}
      <div className="relative pl-4 border-l border-slate-800 space-y-4 ml-4.5 py-1">
        {result.events.map((event) => {
          const isEventActive = activeFrameIndex === event.frame_index;
          return (
            <div 
              key={event.event_index} 
              onClick={(e) => {
                e.stopPropagation();
                onClick(event.frame_index);
              }}
              className={`relative cursor-pointer transition-all hover:translate-x-0.5 group/item ${
                isEventActive ? 'text-indigo-400' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {/* Point Node on the line */}
              <span className={`absolute -left-[23px] top-1.5 h-3.5 w-3.5 rounded-full border flex items-center justify-center transition-all ${
                isEventActive 
                  ? 'bg-indigo-500 border-indigo-400 ring-4 ring-indigo-500/10 scale-110' 
                  : 'bg-slate-950 border-slate-700 group-hover/item:border-slate-500'
              }`}>
                <span className={`h-1.5 w-1.5 rounded-full ${isEventActive ? 'bg-white' : 'bg-transparent'}`} />
              </span>

              {/* Event contents */}
              <div className="flex flex-col gap-0.5 pl-1">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono font-semibold tracking-wider text-slate-500 uppercase">
                    Event {event.event_index}
                  </span>
                  <Badge variant="secondary" className="px-1.5 py-0 text-[8px] font-mono">
                    F: {event.frame_index}
                  </Badge>
                </div>
                <p className="text-xs mt-0.5 leading-relaxed">
                  {event.description}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

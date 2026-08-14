import * as React from 'react';
import { Play, Pause, Volume2, Maximize, Video, Compass } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';

interface VideoPlayerPlaceholderProps {
  videoId?: string;
  frameIndex?: number;
}

export const VideoPlayerPlaceholder: React.FC<VideoPlayerPlaceholderProps> = ({
  videoId,
  frameIndex,
}) => {
  const [isPlaying, setIsPlaying] = React.useState(false);

  // Auto reset playing state when video changes
  React.useEffect(() => {
    setIsPlaying(false);
  }, [videoId, frameIndex]);

  return (
    <Card className="w-full h-full flex flex-col">
      <CardHeader className="py-4">
        <CardTitle className="text-sm font-semibold flex items-center gap-2">
          <Compass className="h-4 w-4 text-indigo-400" />
          Scene Inspector
        </CardTitle>
        <CardDescription>
          {videoId ? `Analyzing Scene: ${videoId}` : 'Select a search result to inspect details'}
        </CardDescription>
      </CardHeader>
      
      <CardContent className="flex-1 flex flex-col justify-between p-4 bg-slate-950/20">
        {videoId ? (
          <div className="flex-1 flex flex-col gap-4">
            
            {/* Visual Screen Container (16:9) */}
            <div className="relative aspect-video w-full rounded-lg overflow-hidden bg-slate-900 border border-slate-800 flex items-center justify-center group shadow-2xl">
              
              {/* Animated abstract mesh overlay to look like AI processing */}
              <div className="absolute inset-0 bg-radial-gradient from-indigo-500/10 via-transparent to-transparent pointer-events-none opacity-60" />
              
              {/* Scanner Grid Lines */}
              <div className="absolute inset-0 bg-[linear-gradient(to_bottom,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:100%_8px] pointer-events-none" />
              
              {/* Playback overlay */}
              {isPlaying && (
                <div className="absolute inset-0 bg-slate-950/40 backdrop-blur-[1px] pointer-events-none transition-all duration-300" />
              )}

              {/* Information Overlay */}
              <div className="absolute top-4 left-4 z-10 bg-slate-950/80 backdrop-blur-md px-2.5 py-1 rounded-md border border-slate-800 text-[10px] font-mono text-slate-300 flex items-center gap-1.5 shadow-md">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                FRAME: {frameIndex ?? 'N/A'}
              </div>

              {/* Video Content Mock */}
              <div className="flex flex-col items-center gap-3 text-center z-10 px-4">
                <div className="flex h-14 w-14 items-center justify-center rounded-full bg-slate-950/80 border border-slate-800 text-indigo-400 group-hover:scale-105 transition-transform duration-300">
                  <Video className="h-6 w-6 animate-pulse" />
                </div>
                <div>
                  <h5 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">{videoId}</h5>
                  <p className="text-[10px] text-slate-400 mt-1">Simulated Frame View index: {frameIndex}</p>
                </div>
              </div>

              {/* Laser line effect if playing */}
              {isPlaying && (
                <div className="absolute left-0 right-0 h-0.5 bg-indigo-500/50 shadow-[0_0_10px_2px_rgba(99,102,241,0.5)] top-0 animate-[bounce_3s_infinite]" />
              )}
            </div>

            {/* Video Controls Bar */}
            <div className="flex flex-col gap-2">
              {/* Progress Slider (mock) */}
              <div className="relative w-full h-1 bg-slate-800 rounded-full overflow-hidden cursor-pointer group">
                <div 
                  className="absolute left-0 top-0 h-full bg-indigo-500 rounded-full transition-all"
                  style={{ width: isPlaying ? '65%' : '35%' }}
                />
              </div>

              {/* Action Buttons */}
              <div className="flex items-center justify-between text-slate-400 px-1 text-xs">
                <div className="flex items-center gap-3">
                  <button 
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="p-1 hover:text-white transition-colors cursor-pointer"
                  >
                    {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                  </button>
                  <span className="font-mono text-[10px]">
                    {isPlaying ? '00:15 / 00:24' : '00:08 / 00:24'}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Volume2 className="h-4 w-4 hover:text-white cursor-pointer" />
                  <Maximize className="h-4 w-4 hover:text-white cursor-pointer" />
                </div>
              </div>
            </div>

          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-center p-8 border border-dashed border-slate-800/80 rounded-lg">
            <Video className="h-8 w-8 text-slate-600 mb-3 animate-pulse" />
            <p className="text-xs text-slate-500 max-w-[200px]">
              Perform a search and select an item to preview and inspect frames.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

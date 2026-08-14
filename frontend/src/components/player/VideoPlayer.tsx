import * as React from 'react';
import { Play, Pause, Volume2, Maximize, Video, Compass, AlertTriangle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';

interface VideoPlayerProps {
  videoUrl?: string;   
  videoId?: string;  
  fps?: number;          
  frameIndex?: number;   
  startFrame?: number;   
  endFrame?: number;     
  timestampSeconds?: number;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  videoUrl,
  videoId,
  fps = 25,
  frameIndex,
  startFrame,
  endFrame,
  timestampSeconds,
}) => {
  const activeVideoUrl = videoUrl;
  const videoRef = React.useRef<HTMLVideoElement>(null);
  
  // 1. Khai báo state đúng vị trí cấu trúc Component
  const [isPlaying, setIsPlaying] = React.useState(false);
  const [duration, setDuration] = React.useState(0);
  const [currentTime, setCurrentTime] = React.useState(0);
  const [isError, setIsError] = React.useState(false);

  const seekToTimestamp = React.useCallback((video: HTMLVideoElement) => {
    if (timestampSeconds === undefined || !Number.isFinite(timestampSeconds)) return;

    const safeTimestamp = Math.max(0, timestampSeconds);
    const targetTime = Number.isFinite(video.duration) && video.duration > 0
      ? Math.min(safeTimestamp, video.duration)
      : safeTimestamp;

    video.currentTime = targetTime;
  }, [timestampSeconds]);

  // Reset trạng thái lỗi khi đổi video URL mới
  React.useEffect(() => {
    setIsError(false);
  }, [activeVideoUrl]);

  // 2. Logic tự động SEEK khi timestampSeconds từ backend v1 thay đổi
  React.useEffect(() => {
    const video = videoRef.current;
    if (!video || timestampSeconds === undefined || isError) return;

    if (video.readyState >= 1) {
      seekToTimestamp(video);
    }
  }, [timestampSeconds, activeVideoUrl, isError, seekToTimestamp]);

  // 3. Dự phòng logic seek theo frameIndex cũ (nếu timestampSeconds không được truyền)
  React.useEffect(() => {
    const video = videoRef.current;
    if (!video || timestampSeconds !== undefined || frameIndex === undefined || !fps || isError) return;

    const handleFrameSeek = () => {
      const calculatedSeconds = frameIndex / fps;
      if (video.duration && calculatedSeconds > video.duration) {
        video.currentTime = video.duration;
      } else {
        video.currentTime = calculatedSeconds;
      }
    };

    if (video.readyState >= 1) {
      handleFrameSeek();
    } else {
      video.addEventListener('loadedmetadata', handleFrameSeek);
    }

    return () => video.removeEventListener('loadedmetadata', handleFrameSeek);
  }, [frameIndex, fps, timestampSeconds, activeVideoUrl, isError]);

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video) return;

    setDuration(video.duration);
    seekToTimestamp(video);
  };

  const togglePlay = () => {
    if (!videoRef.current || isError) return;
    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play();
    }
  };

  const formatTime = (timeInSeconds: number) => {
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };

  // Tính toán tỷ lệ phần trăm vùng Highlight chuẩn xác từ startFrame -> endFrame
  const getSegmentStyles = () => {
    const effectiveDuration = duration || (endFrame && fps ? (endFrame / fps) * 1.2 : 0); 

    if (!effectiveDuration || !fps || startFrame === undefined || endFrame === undefined) {
      return { left: '0%', width: '0%', display: 'none' };
    }
    const startTimeSec = startFrame / fps;
    const endTimeSec = endFrame / fps;

    const leftPercent = (startTimeSec / effectiveDuration) * 100;
    const widthPercent = ((endTimeSec - startTimeSec) / effectiveDuration) * 100;

    return {
      left: `${Math.min(100, Math.max(0, leftPercent))}%`,
      width: `${Math.min(100, Math.max(0, widthPercent))}%`,
      display: 'block'
    };
  };

  const currentProgressPercent = duration ? (currentTime / duration) * 100 : 0;

  return (
    <Card className="w-full bg-slate-900 border-slate-800 text-slate-100 shadow-xl overflow-hidden">
      <CardHeader className="py-3 px-4 border-b border-slate-800 bg-slate-950/40">
        <CardTitle className="text-xs font-bold uppercase tracking-wider flex items-center gap-2 text-indigo-400">
          <Compass className="h-3.5 w-3.5" />
          Standalone Video Inspector
        </CardTitle>
        <CardDescription className="text-[11px] text-slate-400 mt-0.5">
          {activeVideoUrl ? `Streaming local asset source...` : 'Waiting for video file assignment'}
        </CardDescription>
      </CardHeader>
      
      <CardContent className="p-4 bg-slate-950/20 flex flex-col gap-4">
        {activeVideoUrl ? (
          <div className="w-full flex flex-col gap-4">
            
            {/* THÈ VIDEO CHUẨN - FIX CỐ ĐỊNH TỶ LỆ ASPECT-VIDEO TRÁNH TRÀN KHUNG */}
            <div className="relative w-full aspect-video rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center shadow-inner">
              
              {/* Xử lý Edge Case: Hiện Error Placeholder thay vì crash khi video sập */}
              {isError ? (
                <div className="absolute inset-0 flex flex-col items-center justify-center p-4 bg-slate-950/80 text-center gap-2 z-30">
                  <AlertTriangle className="h-8 w-8 text-rose-500 animate-bounce" />
                  <p className="text-xs font-semibold text-slate-200">Video load failed</p>
                  <p className="text-[10px] text-slate-500 max-w-[260px] font-mono break-all">
                    {activeVideoUrl}
                  </p>
                </div>
              ) : (  
                <video
                  key={activeVideoUrl}
                  ref={videoRef}
                  src={activeVideoUrl}
                  preload="metadata"
                  className="w-full h-full object-contain"
                  onTimeUpdate={handleTimeUpdate}
                  onLoadedMetadata={handleLoadedMetadata}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onError={() => setIsError(true)}
                  controls={false}
                />
              )}

              <div className="absolute top-3 left-3 z-10 bg-slate-950/90 backdrop-blur-md px-2 py-1 rounded border border-slate-800 text-[9px] font-mono text-slate-300 flex items-center gap-1.5 shadow">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                TARGET FRAME: #{frameIndex ?? 0} ({fps} FPS)
              </div>
            </div>

            {/* THANH ĐIỀU KHIỂN ĐƯỢC TÁCH BIỆT RÕ RÀNG KHÔNG THỂ BỊ ĐÈ */}
            <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-800/60 flex flex-col gap-3">
              
              {/* THANH SCRUBBER TIẾN TRÌNH VÀ HIGHLIGHT SEGMENT */}
              <div className="relative w-full h-2 bg-slate-950 rounded-full cursor-pointer group border border-slate-800/40">
                {/* 1. Lớp phủ Highlight khoảng phân đoạn từ startFrame đến endFrame */}
                <div 
                  className="absolute top-0 h-full bg-amber-500/30 border-x border-amber-400/60 z-10"
                  style={getSegmentStyles()}
                  title={`Segment Frame: #${startFrame} -> #${endFrame}`}
                />
                {/* 2. Thanh chạy tiến trình hiện tại */}
                <div 
                  className="absolute left-0 top-0 h-full bg-indigo-500 rounded-full z-20 shadow-sm shadow-indigo-500/50"
                  style={{ width: `${currentProgressPercent}%` }}
                />
              </div>

              {/* HÀNG NÚT THAO TÁC CUSTOM */}
              <div className="flex flex-wrap items-center justify-between gap-3 text-slate-400 text-xs">
                <div className="flex min-w-0 flex-wrap items-center gap-3">
                  <button 
                    onClick={togglePlay}
                    disabled={isError}
                    className="p-1 hover:text-slate-100 transition-colors cursor-pointer bg-slate-800 rounded hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    {isPlaying ? <Pause className="h-3.5 w-3.5 text-indigo-400" /> : <Play className="h-3.5 w-3.5 text-indigo-400" />}
                  </button>
                  <span className="font-mono text-[10px] text-slate-300 bg-slate-950 px-2 py-0.5 rounded border border-slate-800/60">
                    {formatTime(currentTime)} / {formatTime(duration)}
                  </span>
                  {startFrame !== undefined && (
                    <span className="text-[9px] font-mono bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20 text-amber-400 font-bold">
                      Active Window: #{startFrame}-#{endFrame}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3 bg-slate-950/40 px-2 py-1 rounded border border-slate-800/30">
                  <Volume2 className="h-3.5 w-3.5 hover:text-slate-100 cursor-pointer" />
                  <Maximize className="h-3.5 w-3.5 hover:text-slate-100 cursor-pointer" />
                </div>
              </div>

            </div>

          </div>
        ) : (
          <div className="w-full flex flex-col items-center justify-center text-center p-8 border border-dashed border-slate-800 rounded-xl bg-slate-900/20">
            <Video className="h-8 w-8 text-slate-600 mb-2 animate-pulse" />
            <p className="text-xs text-slate-500 max-w-[200px]">
              Missing video source for {videoId ?? 'selected result'}. Configure VITE_VIDEO_BASE_URL to load this asset.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

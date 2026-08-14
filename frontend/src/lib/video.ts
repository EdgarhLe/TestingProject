const videoBaseUrl = import.meta.env.VITE_VIDEO_BASE_URL?.replace(/\/$/, '');

export function resolveVideoUrl(videoId: string): string | undefined {
  if (!videoBaseUrl) {
    return undefined;
  }

  const group = videoId.split('_')[0];
  return `${videoBaseUrl}/${group}/video/${encodeURIComponent(videoId)}.mp4`;
}

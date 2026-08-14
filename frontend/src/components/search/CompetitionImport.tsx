import * as React from 'react';
import { Upload, Trash2, CheckCircle2, XCircle, Play, Code } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { parseQueryFile, ParsedQuery, KisRequest, TrakeRequest, QaRequest } from '@/api/queryParser';

interface QueueItem {
  filename: string;
  query_id: string;
  task_type: 'kis' | 'trake' | 'qa' | 'unknown';
  preview: string;
  status: 'success' | 'error';
  errorMsg?: string;
  request?: any;
}

interface CompetitionImportProps {
  onRunAll: (batchRequests: {
    kis?: any;
    trake?: any;
    qa?: any;
  }) => void;
  isLoading: boolean;
}

export const CompetitionImport: React.FC<CompetitionImportProps> = ({
  onRunAll,
  isLoading,
}) => {
  const [queue, setQueue] = React.useState<QueueItem[]>([]);
  const [isReadingFiles, setIsReadingFiles] = React.useState(false);
  const [batchPreview, setBatchPreview] = React.useState<any | null>(null);

  // Helper to extract basic metadata from filename on parse failure
  const getFallbackMeta = (filename: string) => {
    const cleanName = filename.replace(/\.txt$/, '');
    const parts = cleanName.split('-');
    let task_type: QueueItem['task_type'] = 'unknown';
    let query_id = 'Unknown';
    if (parts.length >= 4) {
      query_id = `${parts[1]}-${parts[2]}`;
      const type = parts[parts.length - 1].toLowerCase();
      if (type === 'kis' || type === 'trake' || type === 'qa') {
        task_type = type;
      }
    }
    return { query_id, task_type };
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsReadingFiles(true);
    setBatchPreview(null);

    const newItems: QueueItem[] = [];

    const fileReadPromises = Array.from(files).map((file) => {
      return new Promise<void>((resolve) => {
        const reader = new FileReader();
        reader.onload = (event) => {
          const content = event.target?.result as string;
          try {
            const parsed: ParsedQuery = parseQueryFile(file.name, content);
            
            // Build simple preview text
            let preview = '';
            if (parsed.task_type === 'kis') {
              preview = (parsed.request as KisRequest).query;
            } else if (parsed.task_type === 'qa') {
              preview = `[Scene] ${(parsed.request as QaRequest).scene} [Q] ${(parsed.request as QaRequest).question}`;
            } else if (parsed.task_type === 'trake') {
              preview = (parsed.request as TrakeRequest).events.join(' | ');
            }

            newItems.push({
              filename: file.name,
              query_id: parsed.query_id,
              task_type: parsed.task_type,
              preview,
              status: 'success',
              request: parsed.request,
            });
          } catch (err: any) {
            const fallback = getFallbackMeta(file.name);
            newItems.push({
              filename: file.name,
              query_id: fallback.query_id,
              task_type: fallback.task_type,
              preview: content ? (content.substring(0, 100) + (content.length > 100 ? '...' : '')) : '',
              status: 'error',
              errorMsg: err.message || 'Unknown parsing error',
            });
          }
          resolve();
        };

        reader.onerror = () => {
          const fallback = getFallbackMeta(file.name);
          newItems.push({
            filename: file.name,
            query_id: fallback.query_id,
            task_type: fallback.task_type,
            preview: '',
            status: 'error',
            errorMsg: 'Could not read file from disk',
          });
          resolve();
        };

        reader.readAsText(file);
      });
    });

    await Promise.all(fileReadPromises);

    // Sort queue items to maintain order by filename / ID if desired
    newItems.sort((a, b) => a.filename.localeCompare(b.filename));

    setQueue((prev) => {
      const merged = [...prev, ...newItems];
      // Deduplicate by filename to prevent duplicates
      const seen = new Set();
      return merged.filter((item) => {
        const duplicate = seen.has(item.filename);
        seen.add(item.filename);
        return !duplicate;
      });
    });

    setIsReadingFiles(false);
    // Reset file input value so same files can be imported again
    e.target.value = '';
  };

  const handleClearQueue = () => {
    setQueue([]);
    setBatchPreview(null);
  };

  const handleRemoveItem = (index: number) => {
    const updated = queue.filter((_, i) => i !== index);
    setQueue(updated);
    setBatchPreview(null);
  };

  const handleRunAll = () => {
    const successItems = queue.filter((item) => item.status === 'success');
    if (successItems.length === 0) return;

    // Group requests by task type
    const kisQueries = successItems
      .filter((item) => item.task_type === 'kis')
      .map((item) => item.request);
      
    const trakeQueries = successItems
      .filter((item) => item.task_type === 'trake')
      .map((item) => item.request);

    const qaQueries = successItems
      .filter((item) => item.task_type === 'qa')
      .map((item) => item.request);

    const batchPayloads: {
      kis?: { queries: any[] };
      trake?: { queries: any[] };
      qa?: { queries: any[] };
    } = {};

    if (kisQueries.length > 0) {
      batchPayloads.kis = { queries: kisQueries };
    }
    if (trakeQueries.length > 0) {
      batchPayloads.trake = { queries: trakeQueries };
    }
    if (qaQueries.length > 0) {
      batchPayloads.qa = { queries: qaQueries };
    }

    console.log('Batch Request Payloads (API Contract v1):', batchPayloads);
    setBatchPreview(batchPayloads);
    onRunAll(batchPayloads);
  };

  const successCount = queue.filter((i) => i.status === 'success').length;
  const errorCount = queue.filter((i) => i.status === 'error').length;

  return (
    <div className="flex flex-col gap-6 w-full">
      {/* Upload Console Controls */}
      <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
        <div className="flex flex-wrap gap-3 items-center w-full sm:w-auto">
          <input
            type="file"
            multiple
            accept=".txt"
            id="query-pack-uploader"
            onChange={handleFileChange}
            className="hidden"
            disabled={isLoading || isReadingFiles}
          />
          <label htmlFor="query-pack-uploader">
            <span className="inline-flex h-11 px-5 items-center justify-center gap-2 rounded-lg font-medium text-sm text-white bg-indigo-600 hover:bg-indigo-500 shadow-lg shadow-indigo-600/10 cursor-pointer transition-colors select-none disabled:pointer-events-none disabled:opacity-50">
              <Upload className="h-4 w-4" />
              Import Query Pack
            </span>
          </label>

          {queue.length > 0 && (
            <Button
              variant="outline"
              size="md"
              onClick={handleClearQueue}
              disabled={isLoading || isReadingFiles}
              className="gap-2 text-slate-400 border-slate-800 hover:border-red-500/20 hover:text-red-400"
            >
              <Trash2 className="h-4 w-4" />
              Clear Queue
            </Button>
          )}
        </div>

        {queue.length > 0 && (
          <Button
            onClick={handleRunAll}
            disabled={successCount === 0 || isLoading || isReadingFiles}
            className="w-full sm:w-auto gap-2 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 border-0 shadow-lg shadow-emerald-600/10 h-11 font-semibold"
          >
            <Play className="h-4 w-4 fill-white" />
            Run All ({successCount} Queries)
          </Button>
        )}
      </div>

      {/* Queue Listing */}
      {queue.length === 0 ? (
        // Empty State
        <div className="flex flex-col items-center justify-center text-center p-16 border border-dashed border-slate-800 bg-slate-950/10 rounded-2xl min-h-[260px]">
          <div className="h-12 w-12 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-400 mb-4 animate-pulse">
            <Upload className="h-6 w-6" />
          </div>
          <h4 className="text-sm font-bold text-slate-200">No Query Files Loaded</h4>
          <p className="text-xs text-slate-400 mt-2 max-w-sm leading-relaxed">
            Click &ldquo;Import Query Pack&rdquo; to load theorganiser&rsquo;s `.txt` query files. Filenames should follow the standard naming structure.
          </p>
        </div>
      ) : (
        // Queue Table
        <div className="flex flex-col gap-4">
          <div className="flex justify-between items-center text-xs text-slate-400 font-semibold px-2">
            <span>
              Queue Status: {successCount} loaded successfully
              {errorCount > 0 && <span className="text-red-400">, {errorCount} failed</span>}
            </span>
          </div>

          <div className="border border-slate-800/80 bg-slate-950/20 rounded-xl overflow-hidden shadow-lg">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm border-collapse">
                <thead>
                  <tr className="border-b border-slate-900 bg-slate-900/40 text-slate-400 text-xs font-bold uppercase tracking-wider">
                    <th className="px-4 py-3">Filename</th>
                    <th className="px-4 py-3">Query ID</th>
                    <th className="px-4 py-3">Task Type</th>
                    <th className="px-4 py-3">Content Preview</th>
                    <th className="px-4 py-3">Status</th>
                    <th className="px-4 py-3 w-12"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-900/60">
                  {queue.map((item, idx) => (
                    <tr key={item.filename} className="hover:bg-slate-900/10 transition-colors group">
                      <td className="px-4 py-3.5 font-medium text-slate-200 text-xs truncate max-w-[160px]">
                        {item.filename}
                      </td>
                      <td className="px-4 py-3.5 text-slate-300 font-mono text-xs">
                        {item.query_id}
                      </td>
                      <td className="px-4 py-3.5 text-xs text-slate-400 uppercase font-semibold">
                        {item.task_type === 'unknown' ? 'Unknown' : item.task_type}
                      </td>
                      <td className="px-4 py-3.5 text-xs text-slate-400 max-w-xs truncate">
                        {item.preview}
                      </td>
                      <td className="px-4 py-3.5">
                        {item.status === 'success' ? (
                          <span className="inline-flex items-center gap-1 text-xs text-emerald-400 font-medium">
                            <CheckCircle2 className="h-3.5 w-3.5" />
                            Success
                          </span>
                        ) : (
                          <div className="flex flex-col gap-1">
                            <span className="inline-flex items-center gap-1 text-xs text-red-400 font-medium">
                              <XCircle className="h-3.5 w-3.5" />
                              Error
                            </span>
                            <span className="text-[10px] text-red-500 leading-tight block max-w-[200px]">
                              {item.errorMsg}
                            </span>
                          </div>
                        )}
                      </td>
                      <td className="px-4 py-3.5">
                        <button
                          type="button"
                          onClick={() => handleRemoveItem(idx)}
                          className="text-slate-500 hover:text-red-400 p-1 rounded hover:bg-slate-900/40 opacity-0 group-hover:opacity-100 transition-all duration-200 cursor-pointer"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Batch JSON Payload Preview */}
      {batchPreview && (
        <div className="rounded-xl border border-slate-900 bg-slate-950/60 p-4 font-mono text-xs overflow-hidden flex flex-col gap-3 animate-in fade-in duration-300">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5 text-emerald-400 font-semibold">
              <Code className="h-4 w-4" />
              <span>V1 Batch Requests Payload Preview (Grouped by Task Endpoint)</span>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {batchPreview.kis && (
              <div className="flex flex-col gap-1.5">
                <span className="text-slate-400 font-bold uppercase text-[10px] tracking-wider flex items-center gap-1">
                  <Play className="h-2.5 w-2.5 fill-slate-400" />
                  POST /query/kis/batch
                </span>
                <pre className="text-slate-300 overflow-x-auto max-h-48 p-2 rounded bg-slate-950/40 border border-slate-900/60">
                  {JSON.stringify(batchPreview.kis, null, 2)}
                </pre>
              </div>
            )}
            {batchPreview.trake && (
              <div className="flex flex-col gap-1.5">
                <span className="text-slate-400 font-bold uppercase text-[10px] tracking-wider flex items-center gap-1">
                  <Play className="h-2.5 w-2.5 fill-slate-400" />
                  POST /query/trake/batch
                </span>
                <pre className="text-slate-300 overflow-x-auto max-h-48 p-2 rounded bg-slate-950/40 border border-slate-900/60">
                  {JSON.stringify(batchPreview.trake, null, 2)}
                </pre>
              </div>
            )}
            {batchPreview.qa && (
              <div className="flex flex-col gap-1.5">
                <span className="text-slate-400 font-bold uppercase text-[10px] tracking-wider flex items-center gap-1">
                  <Play className="h-2.5 w-2.5 fill-slate-400" />
                  POST /query/qa/batch
                </span>
                <pre className="text-slate-300 overflow-x-auto max-h-48 p-2 rounded bg-slate-950/40 border border-slate-900/60">
                  {JSON.stringify(batchPreview.qa, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

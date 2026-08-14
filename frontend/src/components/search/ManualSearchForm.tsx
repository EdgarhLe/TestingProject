import * as React from 'react';
import { Plus, Trash2, Code, AlertCircle } from 'lucide-react';
import { TaskType } from '@/api/types';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';

interface ManualSearchFormProps {
  taskType: TaskType;
  onSearch: (requestBody: any) => void;
  isLoading: boolean;
}

export const ManualSearchForm: React.FC<ManualSearchFormProps> = ({
  taskType,
  onSearch,
  isLoading,
}) => {
  // State for KIS
  const [kisQuery, setKisQuery] = React.useState('');

  // State for Q&A
  const [qaScene, setQaScene] = React.useState('');
  const [qaQuestion, setQaQuestion] = React.useState('');

  // State for Trake
  const [trakeEvents, setTrakeEvents] = React.useState<string[]>(['']);

  // Error messages
  const [errorMsg, setErrorMsg] = React.useState('');

  // Auto-generate request payload preview
  const getRequestPayload = React.useCallback(() => {
    if (taskType === 'KIS') {
      return {
        query_id: '',
        query: kisQuery,
        top_k: 5,
      };
    } else if (taskType === 'Q&A') {
      return {
        query_id: '',
        scene: qaScene,
        question: qaQuestion,
        top_k: 3,
      };
    } else {
      return {
        query_id: '',
        events: trakeEvents.filter(e => e.trim() !== ''),
        top_k: 3,
      };
    }
  }, [taskType, kisQuery, qaScene, qaQuestion, trakeEvents]);

  // Handle dynamic event changes for Trake
  const handleAddEvent = () => {
    setTrakeEvents([...trakeEvents, '']);
  };

  const handleRemoveEvent = (index: number) => {
    if (trakeEvents.length > 1) {
      const updated = trakeEvents.filter((_, i) => i !== index);
      setTrakeEvents(updated);
    }
  };

  const handleUpdateEvent = (index: number, value: string) => {
    const updated = [...trakeEvents];
    updated[index] = value;
    setTrakeEvents(updated);
  };

  // Reset errors when inputs or taskType changes
  React.useEffect(() => {
    setErrorMsg('');
  }, [taskType, kisQuery, qaScene, qaQuestion, trakeEvents]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg('');

    const payload = getRequestPayload();

    if (taskType === 'KIS') {
      if (!kisQuery.trim()) {
        setErrorMsg('Please enter a KIS query.');
        return;
      }
    } else if (taskType === 'Q&A') {
      if (!qaScene.trim()) {
        setErrorMsg('Please enter the scene description.');
        return;
      }
      if (!qaQuestion.trim()) {
        setErrorMsg('Please enter the question.');
        return;
      }
    } else if (taskType === 'Trake') {
      const filledEvents = trakeEvents.filter(e => e.trim() !== '');
      if (filledEvents.length === 0) {
        setErrorMsg('Please fill in at least one event description.');
        return;
      }
      if (trakeEvents.some(e => !e.trim())) {
        setErrorMsg('All event fields must be filled.');
        return;
      }
    }

    console.log('Submitted Request Payload (API Contract v1):', payload);
    onSearch(payload);
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-6 w-full">
      {errorMsg && (
        <div className="flex items-center gap-2 rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3 text-sm text-red-400">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Render input elements based on taskType */}
      {taskType === 'KIS' && (
        <div className="flex flex-col gap-2">
          <label htmlFor="kis-query" className="text-xs font-bold text-slate-400 uppercase tracking-wider">
            Query Text Description
          </label>
          <Input
            id="kis-query"
            value={kisQuery}
            onChange={(e) => setKisQuery(e.target.value)}
            placeholder="Enter scene description (e.g. 'Cảnh quay bằng flycam một cây cầu...')"
            className="h-12"
            disabled={isLoading}
          />
        </div>
      )}

      {taskType === 'Q&A' && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <label htmlFor="qa-scene" className="text-xs font-bold text-slate-400 uppercase tracking-wider">
              Scene Description
            </label>
            <textarea
              id="qa-scene"
              value={qaScene}
              onChange={(e) => setQaScene(e.target.value)}
              placeholder="Describe the overall scene..."
              className="flex min-h-[100px] w-full rounded-lg border border-slate-800 bg-slate-900/50 px-4 py-3 text-sm text-slate-100 placeholder:text-slate-500 focus:border-indigo-500 focus:bg-slate-900 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:cursor-not-allowed disabled:opacity-50 transition-all duration-200 resize-y"
              disabled={isLoading}
            />
          </div>
          <div className="flex flex-col gap-2">
            <label htmlFor="qa-question" className="text-xs font-bold text-slate-400 uppercase tracking-wider">
              Question
            </label>
            <Input
              id="qa-question"
              value={qaQuestion}
              onChange={(e) => setQaQuestion(e.target.value)}
              placeholder="What is the visual detail you want to ask?"
              className="h-12"
              disabled={isLoading}
            />
          </div>
        </div>
      )}

      {taskType === 'Trake' && (
        <div className="flex flex-col gap-4">
          <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">
            Temporal Events Sequence
          </label>
          <div className="flex flex-col gap-3">
            {trakeEvents.map((evt, idx) => (
              <div key={idx} className="flex gap-2 items-center">
                <span className="text-xs font-bold text-slate-500 shrink-0 w-8">
                  E{idx + 1}
                </span>
                <Input
                  value={evt}
                  onChange={(e) => handleUpdateEvent(idx, e.target.value)}
                  placeholder={`Describe event E${idx + 1}...`}
                  className="h-11 flex-1"
                  disabled={isLoading}
                />
                {trakeEvents.length > 1 && (
                  <button
                    type="button"
                    onClick={() => handleRemoveEvent(idx)}
                    disabled={isLoading}
                    className="flex h-11 w-11 items-center justify-center rounded-lg border border-slate-800 bg-slate-950/20 text-slate-400 hover:text-red-400 hover:border-red-500/30 hover:bg-red-500/5 transition-all duration-200 cursor-pointer disabled:opacity-50"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                )}
              </div>
            ))}
          </div>

          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={handleAddEvent}
            disabled={isLoading}
            className="w-fit self-start gap-1.5 border-dashed border-slate-700 text-slate-400 hover:text-white"
          >
            <Plus className="h-3.5 w-3.5" />
            <span>Add Event</span>
          </Button>
        </div>
      )}

      <div className="flex flex-col md:flex-row gap-4 justify-between items-start md:items-stretch">
        <Button
          type="submit"
          isLoading={isLoading}
          className="w-full md:w-40 font-semibold bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 border-0 h-12"
        >
          Submit Search
        </Button>
      </div>

      {/* JSON Payload Preview */}
      <div className="rounded-xl border border-slate-900 bg-slate-950/60 p-4 font-mono text-xs overflow-hidden flex flex-col gap-2">
        <div className="flex items-center gap-1.5 text-indigo-400 font-semibold">
          <Code className="h-4 w-4" />
          <span>V1 Request Payload Preview</span>
        </div>
        <pre className="text-slate-300 overflow-x-auto max-h-48 p-2 rounded bg-slate-950/40 border border-slate-900/60">
          {JSON.stringify(getRequestPayload(), null, 2)}
        </pre>
      </div>
    </form>
  );
};

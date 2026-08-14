import * as React from 'react';
import { useNavigate } from 'react-router-dom';
import type { KisRequest, QaRequest, TrakeRequest } from '@/api/queryParser';
import { TaskTypeSelector } from '@/components/search/TaskTypeSelector';
import { ManualSearchForm } from '@/components/search/ManualSearchForm';
import { CompetitionImport } from '@/components/search/CompetitionImport';
import { useSearchContext } from '@/context/SearchContext';
import { buildResultsPath } from '@/lib/routing';
import { Sparkles, Terminal, FileText, Search } from 'lucide-react';

export const SearchPage: React.FC = () => {
  const navigate = useNavigate();
  const { state, setTaskType, setSearchMode, setCurrentQuery } = useSearchContext();
  const { taskType, searchMode } = state;

  // Quick search suggestions
  const getSuggestions = () => {
    switch (taskType) {
      case 'KIS':
        return ['V_KIS_001', 'V_KIS_002', 'frame: 1250'];
      case 'Trake':
        return ['person black jacket', 'yellow delivery van', 'V_TRAKE_101'];
      case 'Q&A':
        return ['license plate', 'green shirt', 'four people'];
    }
  };

  const handleManualSearch = (payload: KisRequest | TrakeRequest | QaRequest) => {
    setCurrentQuery(payload);
    navigate(buildResultsPath(taskType, payload));
  };

  const handleBatchRunAll = (batchPayloads: any) => {
    console.log('Batch run all queries submitted:', batchPayloads);
  };

  return (
    <div className="flex flex-col gap-8 w-full animate-in fade-in slide-in-from-bottom-4 duration-300">
      
      {/* Title Header Hero */}
      <div className="text-center md:text-left space-y-2 max-w-2xl">
        <div className="inline-flex items-center gap-1.5 rounded-full bg-indigo-500/10 px-3 py-1 text-xs font-semibold text-indigo-400 border border-indigo-500/20">
          <Sparkles className="h-3.5 w-3.5" />
          <span>Week 2 Milestone - Search &amp; Import</span>
        </div>
        <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-white">
          Multimedia Retrieval System
        </h1>
        <p className="text-sm text-slate-400 leading-relaxed">
          Verify and retrieve scene events, temporal keyframes, and QA from video indexes. Swap between Manual verification and organiser file Competition modes.
        </p>
      </div>

      {/* Mode Selector Tabs */}
      <div className="flex border-b border-slate-900/60 pb-px gap-6">
        <button
          onClick={() => setSearchMode('manual')}
          className={`flex items-center gap-2 pb-3 font-semibold text-sm transition-all relative border-b-2 cursor-pointer ${
            searchMode === 'manual'
              ? 'text-indigo-400 border-indigo-500 font-bold'
              : 'text-slate-400 border-transparent hover:text-slate-200'
          }`}
        >
          <Search className="h-4 w-4" />
          <span>Manual Verification Mode</span>
        </button>
        <button
          onClick={() => setSearchMode('competition')}
          className={`flex items-center gap-2 pb-3 font-semibold text-sm transition-all relative border-b-2 cursor-pointer ${
            searchMode === 'competition'
              ? 'text-indigo-400 border-indigo-500 font-bold'
              : 'text-slate-400 border-transparent hover:text-slate-200'
          }`}
        >
          <FileText className="h-4 w-4" />
          <span>Competition Mode (File Import)</span>
        </button>
      </div>

      {/* Control Console */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800/80 flex flex-col gap-6 shadow-xl bg-slate-950/20">
        {searchMode === 'manual' ? (
          <>
            {/* Task Type selector tabs */}
            <div>
              <label className="text-xs font-bold text-slate-400 tracking-wider uppercase mb-3 block">
                Select Retrieval Method
              </label>
              <TaskTypeSelector
                selectedType={taskType}
                onChange={setTaskType}
                disabled={false}
              />
            </div>

            {/* Manual input form */}
            <div className="pt-2">
              <ManualSearchForm
                taskType={taskType}
                onSearch={handleManualSearch}
                isLoading={false}
              />
            </div>

            {/* Suggestions tags */}
            <div className="flex flex-wrap items-center gap-2 text-xs border-t border-slate-900/40 pt-4">
              <span className="text-slate-500 flex items-center gap-1">
                <Terminal className="h-3 w-3" />
                Useful search hints:
              </span>
              {getSuggestions().map((sug) => (
                <span
                  key={sug}
                  className="px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-slate-400"
                >
                  {sug}
                </span>
              ))}
            </div>
          </>
        ) : (
          <>
            {/* Competition Import component */}
            <div className="flex flex-col gap-2">
              <h3 className="text-sm font-bold text-slate-200">Organiser Pack Batch Importer</h3>
              <p className="text-xs text-slate-400 leading-relaxed max-w-xl mb-2">
                Import queries in bulk from organiser text files. The parser reads filenames to route queries and processes events, scenes, and questions dynamically.
              </p>
              <CompetitionImport
                onRunAll={handleBatchRunAll}
                isLoading={false}
              />
            </div>
          </>
        )}
      </div>

    </div>
  );
};

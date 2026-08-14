import * as React from 'react';
import type { TaskType, KisResponse, QaResponse } from '@/api/types';
import type { KisRequest, TrakeRequest, QaRequest } from '@/api/queryParser';

export type SearchMode = 'manual' | 'competition';

export interface SelectedMediaItem {
  videoId: string;
  frameIdx?: number;
  timestampSeconds?: number;
}

export type BatchQueueStatus = 'idle' | 'running' | 'paused' | 'completed';

/**
 * Shared State interface for SIU Collective Frontend (Issue #97).
 * Agreed approach: React Context + useReducer for FE1 & FE2 alignment.
 */
export interface SearchState {
  /** Currently selected task type: 'KIS' | 'Trake' | 'Q&A' */
  taskType: TaskType;
  /** Search mode: 'manual' (UI verification) | 'competition' (batch file import) */
  searchMode: SearchMode;
  /** Active query payload (KisRequest, TrakeRequest, or QaRequest) */
  currentQuery: KisRequest | TrakeRequest | QaRequest | null;
  /** Active query results (KisResponse, QaResponse, or TrakeResponse) */
  currentResults: KisResponse | QaResponse | unknown | null;
  /** Loading indicator during search operations */
  isLoading: boolean;
  /** Error message if search operation fails */
  error: string | null;
  /** Selected video item/frame for inspector/player components */
  selectedItem: SelectedMediaItem | null;
  /** Batch queue status for Competition Mode: 'idle' | 'running' | 'paused' | 'completed' */
  batchQueueStatus: BatchQueueStatus;
}

export type SearchAction =
  | { type: 'SET_TASK_TYPE'; payload: TaskType }
  | { type: 'SET_SEARCH_MODE'; payload: SearchMode }
  | { type: 'SET_CURRENT_QUERY'; payload: KisRequest | TrakeRequest | QaRequest | null }
  | { type: 'SET_CURRENT_RESULTS'; payload: KisResponse | QaResponse | unknown | null }
  | { type: 'SET_IS_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_SELECTED_ITEM'; payload: SelectedMediaItem | null }
  | { type: 'SET_BATCH_QUEUE_STATUS'; payload: BatchQueueStatus }
  | { type: 'RESET_SEARCH' };

const initialState: SearchState = {
  taskType: 'KIS',
  searchMode: 'manual',
  currentQuery: null,
  currentResults: null,
  isLoading: false,
  error: null,
  selectedItem: null,
  batchQueueStatus: 'idle',
};

function searchReducer(state: SearchState, action: SearchAction): SearchState {
  switch (action.type) {
    case 'SET_TASK_TYPE':
      return { ...state, taskType: action.payload };
    case 'SET_SEARCH_MODE':
      return { ...state, searchMode: action.payload };
    case 'SET_CURRENT_QUERY':
      return { ...state, currentQuery: action.payload };
    case 'SET_CURRENT_RESULTS':
      return { ...state, currentResults: action.payload, isLoading: false, error: null };
    case 'SET_IS_LOADING':
      return { ...state, isLoading: action.payload, error: action.payload ? null : state.error };
    case 'SET_ERROR':
      return { ...state, error: action.payload, isLoading: false };
    case 'SET_SELECTED_ITEM':
      return { ...state, selectedItem: action.payload };
    case 'SET_BATCH_QUEUE_STATUS':
      return { ...state, batchQueueStatus: action.payload };
    case 'RESET_SEARCH':
      return { ...initialState, taskType: state.taskType, searchMode: state.searchMode };
    default:
      return state;
  }
}

export interface SearchContextValue {
  state: SearchState;
  dispatch: React.Dispatch<SearchAction>;
  setTaskType: (taskType: TaskType) => void;
  setSearchMode: (mode: SearchMode) => void;
  setCurrentQuery: (query: KisRequest | TrakeRequest | QaRequest | null) => void;
  setCurrentResults: (results: KisResponse | QaResponse | unknown | null) => void;
  setIsLoading: (isLoading: boolean) => void;
  setError: (error: string | null) => void;
  setSelectedItem: (item: SelectedMediaItem | null) => void;
  setBatchQueueStatus: (status: BatchQueueStatus) => void;
  resetSearch: () => void;
}

const SearchContext = React.createContext<SearchContextValue | undefined>(undefined);

export const SearchProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = React.useReducer(searchReducer, initialState);

  const setTaskType = React.useCallback((taskType: TaskType) => {
    dispatch({ type: 'SET_TASK_TYPE', payload: taskType });
  }, []);

  const setSearchMode = React.useCallback((mode: SearchMode) => {
    dispatch({ type: 'SET_SEARCH_MODE', payload: mode });
  }, []);

  const setCurrentQuery = React.useCallback((query: KisRequest | TrakeRequest | QaRequest | null) => {
    dispatch({ type: 'SET_CURRENT_QUERY', payload: query });
  }, []);

  const setCurrentResults = React.useCallback((results: KisResponse | QaResponse | unknown | null) => {
    dispatch({ type: 'SET_CURRENT_RESULTS', payload: results });
  }, []);

  const setIsLoading = React.useCallback((isLoading: boolean) => {
    dispatch({ type: 'SET_IS_LOADING', payload: isLoading });
  }, []);

  const setError = React.useCallback((error: string | null) => {
    dispatch({ type: 'SET_ERROR', payload: error });
  }, []);

  const setSelectedItem = React.useCallback((item: SelectedMediaItem | null) => {
    dispatch({ type: 'SET_SELECTED_ITEM', payload: item });
  }, []);

  const setBatchQueueStatus = React.useCallback((status: BatchQueueStatus) => {
    dispatch({ type: 'SET_BATCH_QUEUE_STATUS', payload: status });
  }, []);

  const resetSearch = React.useCallback(() => {
    dispatch({ type: 'RESET_SEARCH' });
  }, []);

  const value = React.useMemo<SearchContextValue>(
    () => ({
      state,
      dispatch,
      setTaskType,
      setSearchMode,
      setCurrentQuery,
      setCurrentResults,
      setIsLoading,
      setError,
      setSelectedItem,
      setBatchQueueStatus,
      resetSearch,
    }),
    [
      state,
      setTaskType,
      setSearchMode,
      setCurrentQuery,
      setCurrentResults,
      setIsLoading,
      setError,
      setSelectedItem,
      setBatchQueueStatus,
      resetSearch,
    ]
  );

  return <SearchContext.Provider value={value}>{children}</SearchContext.Provider>;
};

export function useSearchContext(): SearchContextValue {
  const context = React.useContext(SearchContext);
  if (!context) {
    throw new Error('useSearchContext must be used within a SearchProvider');
  }
  return context;
}

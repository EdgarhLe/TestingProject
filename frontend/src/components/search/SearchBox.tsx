import * as React from 'react';
import { Search } from 'lucide-react';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';

interface SearchBoxProps {
  query: string;
  onChange: (value: string) => void;
  onSearch: () => void;
  isLoading: boolean;
}

export const SearchBox: React.FC<SearchBoxProps> = ({
  query,
  onChange,
  onSearch,
  isLoading,
}) => {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      onSearch();
    }
  };

  return (
    <div className="flex w-full flex-col sm:flex-row gap-3">
      <div className="relative flex-1">
        <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-4 text-slate-400">
          <Search className="h-5 w-5" />
        </div>
        <Input
          type="text"
          placeholder="Enter search queries (e.g. 'V_KIS_001', 'person black jacket', '29A-555.22')..."
          value={query}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          className="pl-11 pr-4 h-12"
          disabled={isLoading}
        />
      </div>
      <Button
        onClick={onSearch}
        isLoading={isLoading}
        size="lg"
        className="w-full sm:w-32 h-12 font-semibold bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 border-0"
      >
        Search
      </Button>
    </div>
  );
};

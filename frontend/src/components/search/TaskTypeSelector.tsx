import * as React from 'react';
import { Target, Milestone, HelpCircle } from 'lucide-react';
import { TaskType } from '@/api/types';
import { cn } from '@/lib/cn';
import { TASK_TYPE_LABELS } from '@/lib/constants';

interface TaskTypeSelectorProps {
  selectedType: TaskType;
  onChange: (type: TaskType) => void;
  disabled?: boolean;
}

export const TaskTypeSelector: React.FC<TaskTypeSelectorProps> = ({
  selectedType,
  onChange,
  disabled,
}) => {
  const getIcon = (value: TaskType) => {
    switch (value) {
      case 'KIS':
        return <Target className="h-5 w-5" />;
      case 'Trake':
        return <Milestone className="h-5 w-5" />;
      case 'Q&A':
        return <HelpCircle className="h-5 w-5" />;
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full">
      {TASK_TYPE_LABELS.map((item) => {
        const isSelected = selectedType === item.value;
        return (
          <button
            key={item.value}
            type="button"
            disabled={disabled}
            onClick={() => onChange(item.value as TaskType)}
            className={cn(
              'flex flex-col items-start p-4 rounded-xl text-left border transition-all duration-300 relative overflow-hidden group cursor-pointer disabled:pointer-events-none disabled:opacity-50',
              isSelected
                ? 'border-indigo-500/50 bg-slate-900/60 shadow-lg shadow-indigo-500/5 ring-1 ring-indigo-500/20'
                : 'border-slate-800 bg-slate-950/40 hover:border-slate-700 hover:bg-slate-900/20'
            )}
          >
            {/* Background Glow */}
            {isSelected && (
              <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500/10 rounded-full blur-2xl -mr-6 -mt-6 pointer-events-none" />
            )}

            <div className="flex items-center gap-3">
              <div
                className={cn(
                  'flex h-9 w-9 items-center justify-center rounded-lg transition-colors',
                  isSelected
                    ? 'bg-indigo-500 text-white shadow-md shadow-indigo-500/20'
                    : 'bg-slate-900 text-slate-400 group-hover:text-slate-200'
                )}
              >
                {getIcon(item.value as TaskType)}
              </div>
              <div>
                <h4
                  className={cn(
                    'text-sm font-semibold transition-colors',
                    isSelected ? 'text-white' : 'text-slate-300 group-hover:text-slate-100'
                  )}
                >
                  {item.label}
                </h4>
              </div>
            </div>
            <p className="mt-2 text-xs text-slate-400 leading-relaxed pl-12">
              {item.description}
            </p>
          </button>
        );
      })}
    </div>
  );
};

import * as React from 'react';
import { cn } from '@/lib/cn';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'primary' | 'secondary' | 'success' | 'warning' | 'destructive' | 'outline';
}

export function Badge({ className, variant = 'primary', ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-slate-950',
        {
          'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20': variant === 'primary',
          'bg-slate-800 text-slate-300 border border-slate-700': variant === 'secondary',
          'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20': variant === 'success',
          'bg-amber-500/10 text-amber-400 border border-amber-500/20': variant === 'warning',
          'bg-rose-500/10 text-rose-400 border border-rose-500/20': variant === 'destructive',
          'border border-slate-700 text-slate-400 bg-transparent': variant === 'outline',
        },
        className
      )}
      {...props}
    />
  );
}

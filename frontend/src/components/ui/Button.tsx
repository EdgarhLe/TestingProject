import * as React from 'react';
import { cn } from '@/lib/cn';
import { Loader2 } from 'lucide-react';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, children, variant = 'primary', size = 'md', isLoading, disabled, ...props }, ref) => {
    return (
      <button
        className={cn(
          'inline-flex items-center justify-center rounded-lg font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950 disabled:pointer-events-none disabled:opacity-50 active:scale-98 transform duration-100 cursor-pointer',
          {
            // Variants
            'bg-indigo-600 text-white hover:bg-indigo-500 shadow-lg shadow-indigo-600/10': variant === 'primary',
            'bg-slate-800 text-slate-100 hover:bg-slate-700 hover:text-white': variant === 'secondary',
            'border border-slate-700 bg-transparent text-slate-300 hover:bg-slate-850 hover:text-white': variant === 'outline',
            'text-slate-400 hover:bg-slate-900 hover:text-slate-100': variant === 'ghost',
            // Sizes
            'h-9 px-3 text-xs': size === 'sm',
            'h-11 px-5 text-sm': size === 'md',
            'h-13 px-7 text-base': size === 'lg',
          },
          className
        )}
        disabled={disabled || isLoading}
        ref={ref}
        {...props}
      >
        {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';

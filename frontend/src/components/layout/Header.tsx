import * as React from 'react';
import { Search, Github, Video, Database } from 'lucide-react';

export const Header: React.FC = () => {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-900 bg-slate-950/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 min-w-0 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        
        {/* Logo and Brand */}
        <div className="flex min-w-0 items-center space-x-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-600 text-white shadow-lg shadow-indigo-500/20">
            <Video className="h-5 w-5" />
          </div>
          <div className="flex min-w-0 items-center whitespace-nowrap">
            <span className="bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-lg font-bold tracking-tight text-transparent">
              SIU Collective
            </span>
            <span className="ml-1.5 rounded-md bg-indigo-500/10 px-2 py-0.5 text-[10px] font-semibold text-indigo-400 border border-indigo-500/20">
              Retrieval
            </span>
          </div>
        </div>

        {/* Navigation Items (Placeholders/Visual-only) */}
        <nav className="hidden md:flex items-center space-x-1">
          <span className="flex items-center space-x-2 rounded-lg bg-slate-900 px-3 py-1.5 text-sm font-medium text-slate-100 border border-slate-800">
            <Search className="h-4 w-4 text-indigo-400" />
            <span>Search Console</span>
          </span>
          <span className="flex items-center space-x-2 rounded-lg px-3 py-1.5 text-sm font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-900/50 transition-colors cursor-not-allowed">
            <Database className="h-4 w-4" />
            <span>Index Status</span>
          </span>
        </nav>

        {/* Extra Action Buttons */}
        <div className="flex shrink-0 items-center space-x-4">
          <a
            href="https://github.com/Pham-Hoang-Nhat-Thanh/SIU_Collective"
            target="_blank"
            rel="noreferrer"
            className="rounded-lg p-2 text-slate-400 hover:text-slate-100 hover:bg-slate-900 transition-colors"
            title="GitHub Repository"
          >
            <Github className="h-5 w-5" />
          </a>
        </div>

      </div>
    </header>
  );
};

import * as React from 'react';
import { Header } from './Header';

interface AppShellProps {
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({ children }) => {
  return (
    <div className="relative flex min-h-screen min-w-0 flex-col overflow-x-hidden bg-slate-950">
      {/* Background radial effects */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-indigo-900/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute top-1/3 right-1/4 w-[400px] h-[400px] bg-purple-900/5 rounded-full blur-3xl pointer-events-none" />

      {/* Main Header */}
      <Header />

      {/* Core Main Content Area */}
      <main className="z-10 flex min-w-0 w-full max-w-7xl flex-1 flex-col px-4 py-8 sm:px-6 lg:px-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="w-full border-t border-slate-900/80 bg-slate-950/40 py-6 text-center text-xs text-slate-500 z-10">
        <div className="max-w-7xl mx-auto px-4">
          <p>© {new Date().getFullYear()} SIU Collective. Built for Multimedia Retrieval System.</p>
        </div>
      </footer>
    </div>
  );
};

import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { AppShell } from '@/components/layout/AppShell';
import { KisResultsPage } from '@/pages/KisResultsPage';
import { QaResultsPage } from '@/pages/QaResultsPage';
import { SearchPage } from '@/pages/SearchPage';
import { ResultsPlaceholderPage } from '@/pages/ResultsPlaceholderPage';

function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<SearchPage />} />
          <Route path="/results/kis" element={<KisResultsPage />} />
          <Route
            path="/results/trake"
            element={<ResultsPlaceholderPage taskType="Trake" />}
          />
          <Route
            path="/results/qa"
            element={<QaResultsPage />}
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}

export default App;

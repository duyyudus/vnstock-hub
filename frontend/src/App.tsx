import Dashboard from './features/dashboard/Dashboard';
import AdminPage from './features/admin/AdminPage';
import './index.css';

function App() {
  const rawPath = typeof window !== 'undefined' ? window.location.pathname : '/';
  const path = rawPath.endsWith('/') && rawPath.length > 1 ? rawPath.slice(0, -1) : rawPath;
  if (path === '/admin') {
    return <AdminPage />;
  }
  return <Dashboard />;
}

export default App;

import React, { useState, useEffect } from 'react';
import { LayoutDashboard, Upload, FileText, CheckCircle, Activity } from 'lucide-react';
import DataUploader from './components/DataUploader';
import Dashboard from './components/Dashboard';
import AuditView from './components/AuditView';

const API_BASE = 'http://localhost:8000';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [serverStatus, setServerStatus] = useState(null);
  const [selectedApplication, setSelectedApplication] = useState(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/status`);
      if (res.ok) setServerStatus(await res.json());
    } catch (err) {
      console.error("Backend not reachable", err);
      setServerStatus(null);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const navigateToAudit = (appId) => {
    setSelectedApplication(appId);
    setActiveTab('audit');
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'upload':
        return <DataUploader onUploadComplete={() => {
          fetchStatus();
          setActiveTab('dashboard');
        }} />;
      case 'dashboard':
        return <Dashboard 
          serverStatus={serverStatus} 
          onViewAudit={navigateToAudit}
          onRefreshStatus={fetchStatus}
        />;
      case 'audit':
        return <AuditView 
          applicationId={selectedApplication} 
          onBack={() => setActiveTab('dashboard')} 
        />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar glass-panel" style={{ borderLeft: 'none', borderTop: 'none', borderBottom: 'none', borderRadius: 0 }}>
        <div style={{ paddingBottom: '20px', borderBottom: '1px solid var(--border-color)' }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'white', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity color="var(--accent-blue)" />
            Orchestrator
          </h2>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px' }}>Autonomous Credit System</p>
        </div>

        <nav style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <button 
            className="btn-primary" 
            style={{ 
              background: activeTab === 'dashboard' ? 'var(--accent-blue)' : 'transparent',
              color: activeTab === 'dashboard' ? 'white' : 'var(--text-secondary)',
              boxShadow: activeTab === 'dashboard' ? '' : 'none',
              display: 'flex', alignItems: 'center', gap: '12px', justifyContent: 'flex-start',
              padding: '12px 16px'
            }}
            onClick={() => setActiveTab('dashboard')}
          >
            <LayoutDashboard size={18} />
            Dashboard
          </button>
          
          <button 
            className="btn-primary" 
            style={{ 
              background: activeTab === 'upload' ? 'var(--accent-blue)' : 'transparent',
              color: activeTab === 'upload' ? 'white' : 'var(--text-secondary)',
              boxShadow: activeTab === 'upload' ? '' : 'none',
              display: 'flex', alignItems: 'center', gap: '12px', justifyContent: 'flex-start',
              padding: '12px 16px'
            }}
            onClick={() => setActiveTab('upload')}
          >
            <Upload size={18} />
            Data Center
          </button>
        </nav>

        <div style={{ marginTop: 'auto', padding: '16px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px' }}>
          <h4 style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '8px', textTransform: 'uppercase' }}>System Status</h4>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.9rem' }}>
            {serverStatus ? (
              <><CheckCircle size={14} color="var(--status-approve)" /> <span style={{color: 'var(--status-approve)'}}>Backend Online</span></>
            ) : (
              <><div style={{width: 14, height: 14, borderRadius: '50%', background: 'var(--status-refer)'}}></div> <span style={{color: 'var(--status-refer)'}}>Backend Offline</span></>
            )}
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="main-content">
        {renderContent()}
      </div>
    </div>
  );
}

export default App;

import React, { useState, useEffect } from 'react';
import { Play, Loader2, Search, ArrowRight, RefreshCw } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

export default function Dashboard({ serverStatus, onViewAudit, onRefreshStatus }) {
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState([]);
  const [loadingResults, setLoadingResults] = useState(false);
  const [jobId, setJobId] = useState(null);

  const fetchResults = async () => {
    setLoadingResults(true);
    try {
      const res = await fetch(`${API_BASE}/results?limit=50`);
      if (res.ok) {
        const data = await res.json();
        setResults(data.results);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingResults(false);
    }
  };

  useEffect(() => {
    fetchResults();
  }, [serverStatus]);

  const handleRunBatch = async () => {
    setProcessing(true);
    try {
      const res = await fetch(`${API_BASE}/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: 100 }) // Demo limit
      });
      const data = await res.json();
      if (res.ok) {
        setJobId(data.job_id);
        // Start polling results
        const poll = setInterval(() => {
          fetchResults();
          onRefreshStatus();
        }, 3000);
        setTimeout(() => {
          clearInterval(poll);
          setProcessing(false);
        }, 15000); // 15 seconds polling for demo
      }
    } catch (e) {
      console.error(e);
      setProcessing(false);
    }
  };

  const getDecisionBadge = (decision) => {
    if (!decision) return <span className="badge">PENDING</span>;
    switch(decision) {
      case 'APPROVE': return <span className="badge badge-approve">APPROVE</span>;
      case 'CONDITIONAL': return <span className="badge badge-conditional">CONDITIONAL</span>;
      case 'REFER': return <span className="badge badge-refer">REFER</span>;
      case 'REJECT': return <span className="badge" style={{backgroundColor:'rgba(255,0,0,0.1)', color: 'red'}}>REJECT</span>;
      default: return <span className="badge">{decision}</span>;
    }
  };

  const formatCurrency = (val) => {
    if (!val) return 'N/A';
    return `NRs ${val.toLocaleString()}`;
  };

  return (
    <div className="animate-fade-in">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '32px' }}>
        <div>
          <h1 style={{ fontSize: '2rem', fontWeight: 600, marginBottom: '8px' }}>Orchestrator Dashboard</h1>
          <p style={{ color: 'var(--text-secondary)' }}>
            Monitor automated credit decisions and pipeline throughput.
          </p>
        </div>
        
        <div style={{ display: 'flex', gap: '16px' }}>
          <button className="glass-panel" style={{ padding: '10px 16px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)' }} onClick={fetchResults}>
            <RefreshCw size={16} /> Refresh
          </button>
          <button 
            className="btn-primary" 
            onClick={handleRunBatch}
            disabled={processing || !serverStatus?.loaded_tables?.profiles}
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            {processing ? <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <Play size={16} />}
            {processing ? 'Processing Batch...' : 'Run Pipeline'}
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '24px', marginBottom: '32px' }}>
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '8px' }}>Applications Processed</h4>
          <p style={{ fontSize: '2rem', fontWeight: 700 }}>{serverStatus?.results_count || 0}</p>
        </div>
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '8px' }}>Total Approvals</h4>
          <p style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--status-approve)' }}>
            {results.filter(r => r.final_decision === 'APPROVE' || r.final_decision === 'CONDITIONAL').length}
          </p>
        </div>
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '8px' }}>Data Rows Loaded</h4>
          <p style={{ fontSize: '2rem', fontWeight: 700 }}>
            {serverStatus?.loaded_tables?.transactions ? (serverStatus.loaded_tables.transactions / 1000000).toFixed(1) + 'M' : '0'}
          </p>
        </div>
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '8px' }}>Avg Processing Time</h4>
          <p style={{ fontSize: '2rem', fontWeight: 700 }}>
            ~1.2s
          </p>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Recent Decisions</h3>
          <div style={{ position: 'relative' }}>
            <Search size={16} style={{ position: 'absolute', left: '12px', top: '10px', color: 'var(--text-secondary)' }} />
            <input 
              type="text" 
              placeholder="Search ID..." 
              style={{ 
                background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border-color)', borderRadius: '8px',
                padding: '8px 12px 8px 36px', color: 'white', outline: 'none'
              }}
            />
          </div>
        </div>

        {results.length === 0 ? (
          <div style={{ padding: '48px', textAlign: 'center', color: 'var(--text-secondary)' }}>
            <p>No applications processed yet.</p>
            <p style={{ fontSize: '0.9rem', marginTop: '8px' }}>Click "Run Pipeline" to start processing.</p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Application ID</th>
                  <th>Applicant</th>
                  <th>Est. Income</th>
                  <th>Credit Score</th>
                  <th>Decision</th>
                  <th>Approved Amt</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {results.map((res, i) => (
                  <tr key={res.application_id || i} onClick={() => onViewAudit(res.application_id)}>
                    <td style={{ fontWeight: 500 }}>{res.application_id}</td>
                    <td style={{ color: 'var(--text-secondary)' }}>{res.applicant_id}</td>
                    <td>{formatCurrency(res.income_estimate_monthly)}</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span>{res.credit_score}</span>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'capitalize' }}>
                          {res.score_band?.replace('_', ' ')}
                        </span>
                      </div>
                    </td>
                    <td>{getDecisionBadge(res.final_decision)}</td>
                    <td style={{ fontWeight: 500 }}>{formatCurrency(res.approved_amount_nrs)}</td>
                    <td style={{ textAlign: 'right', color: 'var(--accent-blue)' }}>
                      <ArrowRight size={18} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

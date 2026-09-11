import React, { useState, useEffect } from 'react';
import { ArrowLeft, CheckCircle, AlertTriangle, XCircle, FileText, BarChart2 } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

export default function AuditView({ applicationId, onBack }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAudit = async () => {
      try {
        const res = await fetch(`${API_BASE}/audit/${applicationId}`);
        if (res.ok) {
          const json = await res.json();
          setData(json);
        }
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
    if (applicationId) fetchAudit();
  }, [applicationId]);

  if (loading) {
    return <div style={{ display: 'flex', justifyContent: 'center', padding: '64px' }}><div className="loader">Loading...</div></div>;
  }

  if (!data) {
    return <div>Failed to load audit trail for {applicationId}</div>;
  }

  const { audit_trail, shap_explanation, compliance_audit_trail } = data;
  
  const decisionLog = (audit_trail && audit_trail.length > 0 && audit_trail[0].decision) ? audit_trail[0].decision : {};
  const { 
    final_decision, 
    approved_amount_nrs, 
    interest_rate_pct,
    rationale: decision_rationale 
  } = decisionLog;

  // Render SHAP Bars
  const renderShapBars = () => {
    if (!shap_explanation || Object.keys(shap_explanation).length === 0) {
      return <p style={{ color: 'var(--text-secondary)' }}>No SHAP explanation available.</p>;
    }
    
    // Find max absolute value to scale the bars
    const maxVal = Math.max(...Object.values(shap_explanation).map(Math.abs));
    
    return Object.entries(shap_explanation).map(([feature, impact], i) => {
      const isPositive = impact >= 0;
      const widthPct = (Math.abs(impact) / (maxVal || 1)) * 50; // max 50% width from center
      
      return (
        <div key={i} className="shap-bar-container">
          <div className="shap-label" title={feature}>
            {feature.length > 30 ? feature.substring(0, 30) + '...' : feature}
          </div>
          <div className="shap-bar-wrapper">
            <div className="shap-bar-center"></div>
            <div 
              className={`shap-bar ${isPositive ? 'shap-bar-positive' : 'shap-bar-negative'}`}
              style={{ width: `${widthPct}%`, [isPositive ? 'marginLeft' : 'marginRight']: '50%' }}
            ></div>
          </div>
          <div style={{ width: '60px', textAlign: 'left', fontSize: '0.85rem', color: isPositive ? 'var(--status-approve)' : 'var(--status-refer)', fontWeight: 500 }}>
            {isPositive ? '+' : ''}{impact.toFixed(1)}
          </div>
        </div>
      );
    });
  };

  const getDecisionIcon = (decision) => {
    switch(decision) {
      case 'APPROVE': return <CheckCircle size={32} color="var(--status-approve)" />;
      case 'CONDITIONAL': return <AlertTriangle size={32} color="var(--status-conditional)" />;
      case 'REFER': return <AlertTriangle size={32} color="var(--status-refer)" />;
      case 'REJECT': return <XCircle size={32} color="red" />;
      default: return <FileText size={32} />;
    }
  };

  return (
    <div className="animate-fade-in">
      <button 
        style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', marginBottom: '24px' }}
        onClick={onBack}
      >
        <ArrowLeft size={16} /> Back to Dashboard
      </button>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '32px' }}>
        <div>
          <h1 style={{ fontSize: '2rem', fontWeight: 600, marginBottom: '8px' }}>Application Audit</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>
            ID: <span style={{ color: 'white' }}>{applicationId}</span>
          </p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '24px' }}>
        {/* Decision Summary Panel */}
        <div className="glass-panel" style={{ padding: '32px' }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px', marginBottom: '24px' }}>
            {getDecisionIcon(final_decision)}
            <div>
              <h3 style={{ fontSize: '0.9rem', textTransform: 'uppercase', color: 'var(--text-secondary)', letterSpacing: '1px' }}>Final Decision</h3>
              <p style={{ fontSize: '2.5rem', fontWeight: 700, lineHeight: 1 }}>{final_decision}</p>
            </div>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '24px', background: 'rgba(0,0,0,0.2)', padding: '16px', borderRadius: '8px' }}>
            <div>
              <h4 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>Approved Amount</h4>
              <p style={{ fontSize: '1.25rem', fontWeight: 600 }}>{approved_amount_nrs ? `NRs ${approved_amount_nrs.toLocaleString()}` : 'N/A'}</p>
            </div>
            <div>
              <h4 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>Interest Rate</h4>
              <p style={{ fontSize: '1.25rem', fontWeight: 600 }}>{interest_rate_pct ? `${interest_rate_pct.toFixed(2)}%` : 'N/A'}</p>
            </div>
          </div>

          <h4 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>Decision Rationale</h4>
          <p style={{ fontSize: '0.95rem', lineHeight: 1.6, color: 'var(--text-primary)' }}>
            {decision_rationale || "No rationale provided."}
          </p>
        </div>

        {/* SHAP Explanation Panel */}
        <div className="glass-panel" style={{ padding: '32px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
            <BarChart2 color="var(--accent-blue)" />
            <h3 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Credit Score Drivers</h3>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '24px' }}>
            Visualizing the top features impacting the XGBoost credit score output (SHAP).
          </p>
          
          <div style={{ padding: '16px 0' }}>
            {renderShapBars()}
          </div>
        </div>
      </div>

      {/* Compliance Log */}
      <div className="glass-panel" style={{ padding: '32px' }}>
        <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '24px' }}>Compliance Audit Trail</h3>
        {compliance_audit_trail && compliance_audit_trail.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {compliance_audit_trail.map((flagObj, i) => {
              const isFail = flagObj.status === 'FAIL' || flagObj.status === 'WARN' || flagObj.status === 'REJECT';
              return (
                <div key={i} style={{ 
                  padding: '16px', 
                  borderLeft: `4px solid ${isFail ? 'var(--status-refer)' : 'var(--status-approve)'}`,
                  background: 'rgba(255,255,255,0.02)',
                  borderRadius: '0 8px 8px 0'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <strong style={{ color: isFail ? 'var(--status-refer)' : 'var(--status-approve)' }}>
                      {flagObj.rule}
                    </strong>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{flagObj.status}</span>
                  </div>
                  <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                    {flagObj.reason}
                  </p>
                </div>
              );
            })}
          </div>
        ) : (
          <p style={{ color: 'var(--text-secondary)' }}>No compliance flags triggered. Clean record.</p>
        )}
      </div>

    </div>
  );
}

import React, { useState, useRef } from 'react';
import { UploadCloud, File as FileIcon, Loader2, CheckCircle, X } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const UPLOAD_SLOTS = [
  { id: 'profiles', label: 'Applicant Profiles', expectedBase: 'applicant_profiles' },
  { id: 'loans', label: 'Loan Applications', expectedBase: 'loan_applications' },
  { id: 'coop_members', label: 'Cooperative Members', expectedBase: 'cooperative_members' },
  { id: 'coop_sales', label: 'Cooperative Sales', expectedBase: 'cooperative_sales' },
  { id: 'utilities', label: 'Utility Payments', expectedBase: 'utility_payments' },
  { id: 'transactions', label: 'Mobile Transactions', expectedBase: 'mobile_money_transactions' },
  { id: 'remittances', label: 'Remittance Records', expectedBase: 'remittance_records' }
];

export default function DataUploader({ onUploadComplete }) {
  const [slotFiles, setSlotFiles] = useState({});
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState(null);
  
  // Create refs for each hidden input
  const fileInputRefs = useRef({});

  const handleFileSelect = (slotId, e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSlotFiles(prev => ({ ...prev, [slotId]: file }));
    }
  };

  const removeFile = (slotId, e) => {
    e.stopPropagation();
    setSlotFiles(prev => {
      const newFiles = { ...prev };
      delete newFiles[slotId];
      return newFiles;
    });
    // Reset the input value so the same file can be selected again if needed
    if (fileInputRefs.current[slotId]) {
      fileInputRefs.current[slotId].value = '';
    }
  };

  const handleUpload = async () => {
    const selectedSlotIds = Object.keys(slotFiles);
    if (selectedSlotIds.length === 0) return;
    
    setUploading(true);
    setStatus(null);
    
    const formData = new FormData();
    
    selectedSlotIds.forEach(slotId => {
      const file = slotFiles[slotId];
      const slotDef = UPLOAD_SLOTS.find(s => s.id === slotId);
      
      // Get the extension of the original file (e.g., .csv, .parquet)
      const extension = file.name.substring(file.name.lastIndexOf('.'));
      
      // Construct the new filename required by the backend
      const newFileName = `${slotDef.expectedBase}${extension}`;
      
      // Append with the new filename
      formData.append('files', file, newFileName);
    });

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      if (response.ok) {
        setStatus({ type: 'success', msg: `Successfully processed ${data.saved_files.length} tables to memory.` });
        setTimeout(() => {
          onUploadComplete();
        }, 1500);
      } else {
        setStatus({ type: 'error', msg: data.detail || 'Upload failed' });
      }
    } catch (err) {
      setStatus({ type: 'error', msg: 'Backend connection error' });
    } finally {
      setUploading(false);
    }
  };

  const getSlotStatus = (slotId) => {
    return !!slotFiles[slotId];
  };

  return (
    <div className="animate-fade-in" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <h1 style={{ fontSize: '2rem', fontWeight: 600, marginBottom: '8px' }}>Data Center</h1>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '32px' }}>
        Assign your dataset files to the correct tables below. They will be automatically mapped to the correct cleaning pipelines.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px', marginBottom: '32px' }}>
        {UPLOAD_SLOTS.map((slot) => {
          const hasFile = getSlotStatus(slot.id);
          const file = slotFiles[slot.id];
          
          return (
            <div 
              key={slot.id}
              className="glass-panel"
              style={{ 
                padding: '24px', 
                border: hasFile ? '1px solid var(--accent-blue)' : '1px dashed var(--border-color)',
                backgroundColor: hasFile ? 'rgba(59, 130, 246, 0.05)' : 'rgba(26, 29, 36, 0.3)',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                position: 'relative'
              }}
              onClick={() => !hasFile && fileInputRefs.current[slot.id]?.click()}
            >
              <input 
                ref={el => fileInputRefs.current[slot.id] = el}
                type="file" 
                accept=".csv,.parquet" 
                style={{ display: 'none' }} 
                onChange={(e) => handleFileSelect(slot.id, e)}
              />
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: hasFile ? 'white' : 'var(--text-secondary)' }}>
                  {slot.label}
                </h3>
                {hasFile && <CheckCircle size={20} color="var(--accent-blue)" />}
              </div>

              {hasFile ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(0,0,0,0.2)', padding: '8px 12px', borderRadius: '6px' }}>
                  <FileIcon size={16} color="var(--text-secondary)" />
                  <span style={{ fontSize: '0.85rem', flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {file.name}
                  </span>
                  <button 
                    onClick={(e) => removeFile(slot.id, e)}
                    style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '4px' }}
                  >
                    <X size={14} color="var(--text-secondary)" />
                  </button>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', padding: '12px 0' }}>
                  <UploadCloud size={24} color="var(--border-color)" />
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Click to assign file</span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {Object.keys(slotFiles).length > 0 && (
        <div className="glass-panel animate-fade-in" style={{ padding: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h4 style={{ fontWeight: 500, marginBottom: '4px' }}>Ready to process {Object.keys(slotFiles).length} tables</h4>
            {status && (
              <p style={{ 
                fontSize: '0.9rem',
                color: status.type === 'success' ? 'var(--status-approve)' : 'var(--status-refer)'
              }}>
                {status.msg}
              </p>
            )}
          </div>
          
          <div style={{ display: 'flex', gap: '12px' }}>
            <button 
              style={{ padding: '10px 20px', color: 'var(--text-secondary)' }}
              onClick={() => { setSlotFiles({}); setStatus(null); }}
              disabled={uploading}
            >
              Clear All
            </button>
            <button 
              className="btn-primary" 
              onClick={handleUpload}
              disabled={uploading}
              style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
            >
              {uploading && <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />}
              {uploading ? 'Processing Data...' : 'Upload & Route Tables'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

import React from 'react';
import { ExportResult } from '../../types';
import './ExportProgress.css';

interface ExportProgressProps {
  result: ExportResult;
  isProcessing: boolean;
}

const ExportProgress: React.FC<ExportProgressProps> = ({
  result,
  isProcessing,
}) => {
  const formatSize = (bytes?: number): string => {
    if (!bytes) return 'Unknown';
    const mb = bytes / (1024 * 1024);
    if (mb < 1) {
      return `${(bytes / 1024).toFixed(2)} KB`;
    }
    return `${mb.toFixed(2)} MB`;
  };

  if (isProcessing) {
    return (
      <div className="export-progress">
        <div className="progress-spinner" />
        <p>Exporting model...</p>
      </div>
    );
  }

  return (
    <div className={`export-result ${result.success ? 'success' : 'error'}`}>
      <div className="result-header">
        <span className="result-icon">{result.success ? '✓' : '✗'}</span>
        <h4>{result.success ? 'Export Successful' : 'Export Failed'}</h4>
      </div>

      {result.success && (
        <div className="result-details">
          {result.exportPath && (
            <div className="detail-row">
              <span className="detail-label">Export Path:</span>
              <code className="detail-value">{result.exportPath}</code>
            </div>
          )}

          <div className="detail-row">
            <span className="detail-label">Format:</span>
            <span className="detail-value format-badge">{result.format.toUpperCase()}</span>
          </div>

          {result.size && (
            <div className="detail-row">
              <span className="detail-label">Size:</span>
              <span className="detail-value">{formatSize(result.size)}</span>
            </div>
          )}

          {result.message && (
            <div className="detail-row">
              <span className="detail-label">Info:</span>
              <span className="detail-value">{result.message}</span>
            </div>
          )}
        </div>
      )}

      {!result.success && result.error && (
        <div className="error-details">
          <p className="error-message">{result.error}</p>
        </div>
      )}
    </div>
  );
};

export default ExportProgress;

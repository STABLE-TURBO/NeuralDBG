import React from 'react';
import { ExportFormat } from '../../types';
import './ExportFormatSelector.css';

interface ExportFormatSelectorProps {
  selectedFormat: ExportFormat;
  backend: 'tensorflow' | 'pytorch';
  onFormatChange: (format: ExportFormat) => void;
}

const ExportFormatSelector: React.FC<ExportFormatSelectorProps> = ({
  selectedFormat,
  backend,
  onFormatChange,
}) => {
  const formats: { value: ExportFormat; label: string; description: string; supportedBackends: string[] }[] = [
    {
      value: 'onnx',
      label: 'ONNX',
      description: 'Universal format for cross-platform deployment',
      supportedBackends: ['tensorflow', 'pytorch'],
    },
    {
      value: 'tflite',
      label: 'TensorFlow Lite',
      description: 'Optimized for mobile and edge devices',
      supportedBackends: ['tensorflow'],
    },
    {
      value: 'torchscript',
      label: 'TorchScript',
      description: 'Optimized PyTorch format for production',
      supportedBackends: ['pytorch'],
    },
    {
      value: 'savedmodel',
      label: 'SavedModel',
      description: 'TensorFlow Serving compatible format',
      supportedBackends: ['tensorflow'],
    },
  ];

  const isFormatSupported = (format: { supportedBackends: string[] }) => {
    return format.supportedBackends.includes(backend);
  };

  return (
    <div className="export-format-selector">
      <h3>Export Format</h3>
      <div className="format-grid">
        {formats.map((format) => {
          const supported = isFormatSupported(format);
          return (
            <div
              key={format.value}
              className={`format-card ${
                selectedFormat === format.value ? 'selected' : ''
              } ${!supported ? 'disabled' : ''}`}
              onClick={() => supported && onFormatChange(format.value)}
            >
              <div className="format-header">
                <input
                  type="radio"
                  checked={selectedFormat === format.value}
                  onChange={() => supported && onFormatChange(format.value)}
                  disabled={!supported}
                />
                <span className="format-label">{format.label}</span>
              </div>
              <p className="format-description">{format.description}</p>
              {!supported && (
                <div className="format-warning">
                  Requires {format.supportedBackends.join(' or ')} backend
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ExportFormatSelector;

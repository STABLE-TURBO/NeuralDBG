import React from 'react';
import { DeploymentTarget, ServingPlatform, ExportFormat } from '../../types';
import './DeploymentTargetSelector.css';

interface DeploymentTargetSelectorProps {
  selectedTarget: DeploymentTarget;
  selectedPlatform: ServingPlatform;
  exportFormat: ExportFormat;
  onTargetChange: (target: DeploymentTarget) => void;
  onPlatformChange: (platform: ServingPlatform) => void;
}

const DeploymentTargetSelector: React.FC<DeploymentTargetSelectorProps> = ({
  selectedTarget,
  selectedPlatform,
  exportFormat,
  onTargetChange,
  onPlatformChange,
}) => {
  const targets: { value: DeploymentTarget; label: string; description: string; icon: string }[] = [
    {
      value: 'cloud',
      label: 'Cloud',
      description: 'Deploy to cloud platforms (AWS, GCP, Azure)',
      icon: '☁️',
    },
    {
      value: 'edge',
      label: 'Edge',
      description: 'Deploy to edge devices (IoT, embedded systems)',
      icon: '📡',
    },
    {
      value: 'mobile',
      label: 'Mobile',
      description: 'Deploy to mobile devices (iOS, Android)',
      icon: '📱',
    },
    {
      value: 'server',
      label: 'Server',
      description: 'Deploy to on-premise servers',
      icon: '🖥️',
    },
  ];

  const platforms: { value: ServingPlatform; label: string; supportedFormats: ExportFormat[] }[] = [
    {
      value: 'torchserve',
      label: 'TorchServe',
      supportedFormats: ['torchscript', 'onnx'],
    },
    {
      value: 'tfserving',
      label: 'TensorFlow Serving',
      supportedFormats: ['savedmodel', 'tflite', 'onnx'],
    },
    {
      value: 'onnxruntime',
      label: 'ONNX Runtime',
      supportedFormats: ['onnx'],
    },
    {
      value: 'triton',
      label: 'NVIDIA Triton',
      supportedFormats: ['onnx', 'torchscript', 'savedmodel'],
    },
  ];

  const getAvailablePlatforms = () => {
    return platforms.filter((platform) =>
      platform.supportedFormats.includes(exportFormat)
    );
  };

  return (
    <div className="deployment-target-selector">
      <div className="target-section">
        <h3>Deployment Target</h3>
        <div className="target-grid">
          {targets.map((target) => (
            <div
              key={target.value}
              className={`target-card ${
                selectedTarget === target.value ? 'selected' : ''
              }`}
              onClick={() => onTargetChange(target.value)}
            >
              <div className="target-icon">{target.icon}</div>
              <div className="target-content">
                <h4>{target.label}</h4>
                <p>{target.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="platform-section">
        <h3>Serving Platform</h3>
        <div className="platform-grid">
          {getAvailablePlatforms().map((platform) => (
            <div
              key={platform.value}
              className={`platform-card ${
                selectedPlatform === platform.value ? 'selected' : ''
              }`}
              onClick={() => onPlatformChange(platform.value)}
            >
              <input
                type="radio"
                checked={selectedPlatform === platform.value}
                onChange={() => onPlatformChange(platform.value)}
              />
              <span className="platform-label">{platform.label}</span>
              <div className="supported-formats">
                {platform.supportedFormats.map((format) => (
                  <span
                    key={format}
                    className={`format-badge ${
                      format === exportFormat ? 'active' : ''
                    }`}
                  >
                    {format}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>

        {getAvailablePlatforms().length === 0 && (
          <div className="no-platforms-warning">
            No serving platforms available for {exportFormat} format.
            Please select a different export format.
          </div>
        )}
      </div>
    </div>
  );
};

export default DeploymentTargetSelector;

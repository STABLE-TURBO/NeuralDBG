import React, { useState, useEffect } from 'react';
import { ExportService } from '../../services';
import { DeploymentConfig, ServingConfig } from '../../types';
import './ServingConfigGenerator.css';

interface ServingConfigGeneratorProps {
  exportPath: string;
  platform: string;
  modelName: string;
  config: DeploymentConfig;
}

const ServingConfigGenerator: React.FC<ServingConfigGeneratorProps> = ({
  exportPath,
  platform,
  modelName,
  config,
}) => {
  const [servingConfig, setServingConfig] = useState<ServingConfig | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedSection, setExpandedSection] = useState<string | null>(null);

  const exportService = new ExportService();

  const handleGenerate = async () => {
    if (!exportPath) {
      setError('Export the model first');
      return;
    }

    setIsGenerating(true);
    setError(null);

    try {
      const result = await exportService.generateServingConfig(
        exportPath,
        platform,
        modelName,
        config
      );
      setServingConfig(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate config');
      setServingConfig(null);
    } finally {
      setIsGenerating(false);
    }
  };

  const toggleSection = (section: string) => {
    setExpandedSection(expandedSection === section ? null : section);
  };

  const getPlatformInstructions = () => {
    switch (platform) {
      case 'torchserve':
        return [
          'Install TorchServe: pip install torchserve torch-model-archiver',
          'Archive the model using torch-model-archiver',
          'Start TorchServe with the generated configuration',
          'Access the model at http://localhost:8080/predictions/{model_name}',
        ];
      case 'tfserving':
        return [
          'Install TensorFlow Serving via Docker or package manager',
          'Place the SavedModel in the model directory',
          'Start TF Serving with docker-compose or the generated script',
          'Access REST API at http://localhost:8501/v1/models/{model_name}:predict',
        ];
      case 'onnxruntime':
        return [
          'Install ONNX Runtime: pip install onnxruntime',
          'Load the ONNX model in your application',
          'Create inference session and run predictions',
          'Optionally wrap with FastAPI or Flask for API serving',
        ];
      case 'triton':
        return [
          'Install NVIDIA Triton Inference Server',
          'Structure model repository with model configuration',
          'Start Triton server with docker or native installation',
          'Access HTTP/gRPC endpoints for inference',
        ];
      default:
        return ['Generate configuration to see deployment instructions'];
    }
  };

  return (
    <div className="serving-config-generator">
      <h3>Serving Configuration</h3>

      <div className="generator-actions">
        <button
          className="btn-generate"
          onClick={handleGenerate}
          disabled={isGenerating || !exportPath}
        >
          {isGenerating ? 'Generating...' : 'Generate Serving Config'}
        </button>
      </div>

      {error && <div className="config-error">{error}</div>}

      {servingConfig && (
        <div className="config-results">
          <div className="config-section">
            <div
              className="config-header"
              onClick={() => toggleSection('config')}
            >
              <span>📄 Configuration File</span>
              <span className="expand-icon">
                {expandedSection === 'config' ? '▼' : '▶'}
              </span>
            </div>
            {expandedSection === 'config' && (
              <div className="config-content">
                <code className="config-path">{servingConfig.configPath}</code>
                <p>Configuration file generated and ready to use</p>
              </div>
            )}
          </div>

          {servingConfig.modelStorePath && (
            <div className="config-section">
              <div
                className="config-header"
                onClick={() => toggleSection('store')}
              >
                <span>📦 Model Store</span>
                <span className="expand-icon">
                  {expandedSection === 'store' ? '▼' : '▶'}
                </span>
              </div>
              {expandedSection === 'store' && (
                <div className="config-content">
                  <code className="config-path">{servingConfig.modelStorePath}</code>
                  <p>Model files organized for serving</p>
                </div>
              )}
            </div>
          )}

          <div className="config-section">
            <div
              className="config-header"
              onClick={() => toggleSection('scripts')}
            >
              <span>🚀 Deployment Scripts</span>
              <span className="expand-icon">
                {expandedSection === 'scripts' ? '▼' : '▶'}
              </span>
            </div>
            {expandedSection === 'scripts' && (
              <div className="config-content">
                <ul className="scripts-list">
                  {servingConfig.scripts.map((script, index) => (
                    <li key={index}>
                      <code>{script}</code>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <div className="config-section">
            <div
              className="config-header"
              onClick={() => toggleSection('instructions')}
            >
              <span>📋 Deployment Instructions</span>
              <span className="expand-icon">
                {expandedSection === 'instructions' ? '▼' : '▶'}
              </span>
            </div>
            {expandedSection === 'instructions' && (
              <div className="config-content">
                <ol className="instructions-list">
                  {servingConfig.instructions.map((instruction, index) => (
                    <li key={index}>{instruction}</li>
                  ))}
                </ol>
              </div>
            )}
          </div>
        </div>
      )}

      {!servingConfig && (
        <div className="platform-info">
          <h4>{platform.toUpperCase()} Deployment</h4>
          <ol className="instructions-list">
            {getPlatformInstructions().map((instruction, index) => (
              <li key={index}>{instruction}</li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
};

export default ServingConfigGenerator;

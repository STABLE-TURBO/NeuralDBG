import React, { useState, useEffect } from 'react';
import ExportFormatSelector from './ExportFormatSelector';
import OptimizationOptions from './OptimizationOptions';
import DeploymentTargetSelector from './DeploymentTargetSelector';
import ServingConfigGenerator from './ServingConfigGenerator';
import ExportProgress from './ExportProgress';
import { ExportService } from '../../services';
import {
  ExportOptions,
  ExportResult,
  DeploymentConfig,
  DeploymentResult,
  ExportFormat,
  QuantizationType,
  DeploymentTarget,
  ServingPlatform,
} from '../../types';
import './ExportPanel.css';

interface ExportPanelProps {
  modelData: any;
  backend: 'tensorflow' | 'pytorch';
  onExportComplete?: (result: ExportResult) => void;
  onDeploymentComplete?: (result: DeploymentResult) => void;
}

const ExportPanel: React.FC<ExportPanelProps> = ({
  modelData,
  backend,
  onExportComplete,
  onDeploymentComplete,
}) => {
  const [exportService] = useState(() => new ExportService());
  const [activeTab, setActiveTab] = useState<'export' | 'deploy'>('export');

  const [exportOptions, setExportOptions] = useState<ExportOptions>({
    format: 'onnx',
    outputPath: './exported_model',
    backend: backend,
    optimize: true,
    quantization: {
      enabled: false,
      type: 'none',
    },
    pruning: {
      enabled: false,
      sparsity: 0,
    },
    opsetVersion: 13,
  });

  const [deploymentConfig, setDeploymentConfig] = useState<DeploymentConfig>({
    target: 'cloud',
    servingPlatform: 'torchserve',
    modelName: 'neural_model',
    version: '1.0',
    resources: {
      gpuEnabled: false,
      replicas: 1,
      batchSize: 1,
      maxBatchDelay: 100,
    },
    networking: {
      port: 8080,
      enableMetrics: true,
      enableHealthCheck: true,
    },
  });

  const [exportResult, setExportResult] = useState<ExportResult | null>(null);
  const [deploymentResult, setDeploymentResult] = useState<DeploymentResult | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [isDeploying, setIsDeploying] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  useEffect(() => {
    setExportOptions((prev) => ({ ...prev, backend }));
  }, [backend]);

  useEffect(() => {
    const format = exportOptions.format;
    if (format === 'torchscript' && backend !== 'pytorch') {
      setExportOptions((prev) => ({ ...prev, format: 'onnx' }));
    } else if (
      (format === 'tflite' || format === 'savedmodel') &&
      backend !== 'tensorflow'
    ) {
      setExportOptions((prev) => ({ ...prev, format: 'onnx' }));
    }
  }, [backend, exportOptions.format]);

  const handleFormatChange = (format: ExportFormat) => {
    setExportOptions((prev) => ({ ...prev, format }));
    setValidationErrors([]);
  };

  const handleQuantizationChange = (
    enabled: boolean,
    type: QuantizationType
  ) => {
    setExportOptions((prev) => ({
      ...prev,
      quantization: { enabled, type },
    }));
  };

  const handlePruningChange = (enabled: boolean, sparsity: number) => {
    setExportOptions((prev) => ({
      ...prev,
      pruning: { enabled, sparsity },
    }));
  };

  const handleOptimizeChange = (optimize: boolean) => {
    setExportOptions((prev) => ({ ...prev, optimize }));
  };

  const handleOutputPathChange = (path: string) => {
    setExportOptions((prev) => ({ ...prev, outputPath: path }));
  };

  const handleTargetChange = (target: DeploymentTarget) => {
    setDeploymentConfig((prev) => ({ ...prev, target }));
  };

  const handlePlatformChange = (platform: ServingPlatform) => {
    setDeploymentConfig((prev) => ({ ...prev, servingPlatform: platform }));
  };

  const handleResourcesChange = (resources: any) => {
    setDeploymentConfig((prev) => ({
      ...prev,
      resources: { ...prev.resources, ...resources },
    }));
  };

  const handleNetworkingChange = (networking: any) => {
    setDeploymentConfig((prev) => ({
      ...prev,
      networking: { ...prev.networking, ...networking },
    }));
  };

  const handleModelNameChange = (modelName: string) => {
    setDeploymentConfig((prev) => ({ ...prev, modelName }));
  };

  const handleVersionChange = (version: string) => {
    setDeploymentConfig((prev) => ({ ...prev, version }));
  };

  const handleExport = async () => {
    const validation = await exportService.validateExportOptions(exportOptions);
    if (!validation.valid) {
      setValidationErrors(validation.errors);
      return;
    }

    setIsExporting(true);
    setValidationErrors([]);
    setExportResult(null);

    try {
      const result = await exportService.exportModel(modelData, exportOptions);
      setExportResult(result);
      if (onExportComplete) {
        onExportComplete(result);
      }
    } catch (error) {
      setExportResult({
        success: false,
        format: exportOptions.format,
        error: error instanceof Error ? error.message : 'Export failed',
      });
    } finally {
      setIsExporting(false);
    }
  };

  const handleDeploy = async () => {
    if (!exportResult || !exportResult.success || !exportResult.exportPath) {
      setValidationErrors(['Please export the model first']);
      return;
    }

    setIsDeploying(true);
    setDeploymentResult(null);

    try {
      const result = await exportService.deployModel(
        exportResult.exportPath,
        deploymentConfig
      );
      setDeploymentResult(result);
      if (onDeploymentComplete) {
        onDeploymentComplete(result);
      }
    } catch (error) {
      setDeploymentResult({
        success: false,
        error: error instanceof Error ? error.message : 'Deployment failed',
      });
    } finally {
      setIsDeploying(false);
    }
  };

  return (
    <div className="export-panel">
      <div className="export-panel-header">
        <h2>Model Export & Deployment</h2>
        <div className="tab-selector">
          <button
            className={activeTab === 'export' ? 'active' : ''}
            onClick={() => setActiveTab('export')}
          >
            Export
          </button>
          <button
            className={activeTab === 'deploy' ? 'active' : ''}
            onClick={() => setActiveTab('deploy')}
          >
            Deploy
          </button>
        </div>
      </div>

      <div className="export-panel-content">
        {validationErrors.length > 0 && (
          <div className="validation-errors">
            {validationErrors.map((error, index) => (
              <div key={index} className="error-message">
                {error}
              </div>
            ))}
          </div>
        )}

        {activeTab === 'export' && (
          <div className="export-section">
            <ExportFormatSelector
              selectedFormat={exportOptions.format}
              backend={backend}
              onFormatChange={handleFormatChange}
            />

            <div className="output-path-section">
              <label>Output Path:</label>
              <input
                type="text"
                value={exportOptions.outputPath}
                onChange={(e) => handleOutputPathChange(e.target.value)}
                className="output-path-input"
              />
            </div>

            <OptimizationOptions
              optimize={exportOptions.optimize}
              quantization={exportOptions.quantization}
              pruning={exportOptions.pruning}
              onOptimizeChange={handleOptimizeChange}
              onQuantizationChange={handleQuantizationChange}
              onPruningChange={handlePruningChange}
            />

            <div className="export-actions">
              <button
                className="btn-export"
                onClick={handleExport}
                disabled={isExporting}
              >
                {isExporting ? 'Exporting...' : 'Export Model'}
              </button>
            </div>

            {exportResult && (
              <ExportProgress
                result={exportResult}
                isProcessing={isExporting}
              />
            )}
          </div>
        )}

        {activeTab === 'deploy' && (
          <div className="deployment-section">
            <div className="deployment-config">
              <div className="config-group">
                <label>Model Name:</label>
                <input
                  type="text"
                  value={deploymentConfig.modelName}
                  onChange={(e) => handleModelNameChange(e.target.value)}
                  className="config-input"
                />
              </div>

              <div className="config-group">
                <label>Version:</label>
                <input
                  type="text"
                  value={deploymentConfig.version}
                  onChange={(e) => handleVersionChange(e.target.value)}
                  className="config-input"
                />
              </div>
            </div>

            <DeploymentTargetSelector
              selectedTarget={deploymentConfig.target}
              selectedPlatform={deploymentConfig.servingPlatform}
              exportFormat={exportOptions.format}
              onTargetChange={handleTargetChange}
              onPlatformChange={handlePlatformChange}
            />

            <div className="resource-config">
              <h3>Resource Configuration</h3>
              <div className="config-row">
                <div className="config-group">
                  <label>
                    <input
                      type="checkbox"
                      checked={deploymentConfig.resources.gpuEnabled}
                      onChange={(e) =>
                        handleResourcesChange({ gpuEnabled: e.target.checked })
                      }
                    />
                    GPU Enabled
                  </label>
                </div>

                <div className="config-group">
                  <label>Replicas:</label>
                  <input
                    type="number"
                    min="1"
                    value={deploymentConfig.resources.replicas}
                    onChange={(e) =>
                      handleResourcesChange({ replicas: parseInt(e.target.value) })
                    }
                    className="config-input-small"
                  />
                </div>

                <div className="config-group">
                  <label>Batch Size:</label>
                  <input
                    type="number"
                    min="1"
                    value={deploymentConfig.resources.batchSize}
                    onChange={(e) =>
                      handleResourcesChange({ batchSize: parseInt(e.target.value) })
                    }
                    className="config-input-small"
                  />
                </div>

                <div className="config-group">
                  <label>Max Batch Delay (ms):</label>
                  <input
                    type="number"
                    min="0"
                    value={deploymentConfig.resources.maxBatchDelay}
                    onChange={(e) =>
                      handleResourcesChange({
                        maxBatchDelay: parseInt(e.target.value),
                      })
                    }
                    className="config-input-small"
                  />
                </div>
              </div>
            </div>

            <div className="networking-config">
              <h3>Networking Configuration</h3>
              <div className="config-row">
                <div className="config-group">
                  <label>Port:</label>
                  <input
                    type="number"
                    min="1"
                    max="65535"
                    value={deploymentConfig.networking.port}
                    onChange={(e) =>
                      handleNetworkingChange({ port: parseInt(e.target.value) })
                    }
                    className="config-input-small"
                  />
                </div>

                <div className="config-group">
                  <label>
                    <input
                      type="checkbox"
                      checked={deploymentConfig.networking.enableMetrics}
                      onChange={(e) =>
                        handleNetworkingChange({ enableMetrics: e.target.checked })
                      }
                    />
                    Enable Metrics
                  </label>
                </div>

                <div className="config-group">
                  <label>
                    <input
                      type="checkbox"
                      checked={deploymentConfig.networking.enableHealthCheck}
                      onChange={(e) =>
                        handleNetworkingChange({
                          enableHealthCheck: e.target.checked,
                        })
                      }
                    />
                    Enable Health Check
                  </label>
                </div>
              </div>
            </div>

            <ServingConfigGenerator
              exportPath={exportResult?.exportPath || ''}
              platform={deploymentConfig.servingPlatform}
              modelName={deploymentConfig.modelName}
              config={deploymentConfig}
            />

            <div className="deployment-actions">
              <button
                className="btn-deploy"
                onClick={handleDeploy}
                disabled={isDeploying || !exportResult?.success}
              >
                {isDeploying ? 'Deploying...' : 'Deploy Model'}
              </button>
            </div>

            {deploymentResult && (
              <div
                className={`deployment-result ${
                  deploymentResult.success ? 'success' : 'error'
                }`}
              >
                <h4>{deploymentResult.success ? 'Success!' : 'Error'}</h4>
                {deploymentResult.message && <p>{deploymentResult.message}</p>}
                {deploymentResult.error && <p>{deploymentResult.error}</p>}
                {deploymentResult.endpoint && (
                  <div className="endpoint-info">
                    <strong>Endpoint:</strong> {deploymentResult.endpoint}
                  </div>
                )}
                {deploymentResult.deploymentId && (
                  <div className="deployment-id">
                    <strong>Deployment ID:</strong> {deploymentResult.deploymentId}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ExportPanel;

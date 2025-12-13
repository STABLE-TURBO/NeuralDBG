import React, { useState } from 'react';
import { ExportPanel } from './index';
import { ExportResult, DeploymentResult } from '../../types';

const ExportPanelExample: React.FC = () => {
  const [exportResult, setExportResult] = useState<ExportResult | null>(null);
  const [deploymentResult, setDeploymentResult] = useState<DeploymentResult | null>(null);

  const sampleModelData = {
    name: 'mnist_classifier',
    input: {
      shape: [28, 28, 1],
    },
    layers: [
      {
        type: 'Conv2D',
        filters: 32,
        kernel_size: [3, 3],
        activation: 'relu',
      },
      {
        type: 'MaxPooling2D',
        pool_size: [2, 2],
      },
      {
        type: 'Conv2D',
        filters: 64,
        kernel_size: [3, 3],
        activation: 'relu',
      },
      {
        type: 'MaxPooling2D',
        pool_size: [2, 2],
      },
      {
        type: 'Flatten',
      },
      {
        type: 'Dense',
        units: 128,
        activation: 'relu',
      },
      {
        type: 'Dropout',
        rate: 0.5,
      },
      {
        type: 'Dense',
        units: 10,
        activation: 'softmax',
      },
    ],
  };

  const handleExportComplete = (result: ExportResult) => {
    console.log('Export completed:', result);
    setExportResult(result);
    
    if (result.success) {
      alert(`Model successfully exported to ${result.exportPath}`);
    } else {
      alert(`Export failed: ${result.error}`);
    }
  };

  const handleDeploymentComplete = (result: DeploymentResult) => {
    console.log('Deployment completed:', result);
    setDeploymentResult(result);
    
    if (result.success) {
      alert(
        `Model deployed successfully!\nEndpoint: ${result.endpoint}\nDeployment ID: ${result.deploymentId}`
      );
    } else {
      alert(`Deployment failed: ${result.error}`);
    }
  };

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '20px', background: '#252525', borderBottom: '1px solid #333' }}>
        <h1 style={{ margin: 0, color: '#fff' }}>Export Panel Example</h1>
        <p style={{ margin: '10px 0 0 0', color: '#b0b0b0' }}>
          Demonstrate model export and deployment functionality
        </p>
      </div>

      <div style={{ flex: 1, overflow: 'hidden' }}>
        <ExportPanel
          modelData={sampleModelData}
          backend="tensorflow"
          onExportComplete={handleExportComplete}
          onDeploymentComplete={handleDeploymentComplete}
        />
      </div>

      {(exportResult || deploymentResult) && (
        <div style={{ padding: '20px', background: '#252525', borderTop: '1px solid #333' }}>
          <h3 style={{ margin: '0 0 10px 0', color: '#fff' }}>Results</h3>
          
          {exportResult && (
            <div style={{ marginBottom: '15px' }}>
              <h4 style={{ margin: '0 0 5px 0', color: '#4fc3f7' }}>Export Result:</h4>
              <pre style={{ 
                background: '#1e1e1e', 
                padding: '10px', 
                borderRadius: '4px',
                overflow: 'auto',
                color: '#e0e0e0'
              }}>
                {JSON.stringify(exportResult, null, 2)}
              </pre>
            </div>
          )}

          {deploymentResult && (
            <div>
              <h4 style={{ margin: '0 0 5px 0', color: '#4fc3f7' }}>Deployment Result:</h4>
              <pre style={{ 
                background: '#1e1e1e', 
                padding: '10px', 
                borderRadius: '4px',
                overflow: 'auto',
                color: '#e0e0e0'
              }}>
                {JSON.stringify(deploymentResult, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ExportPanelExample;

export type ExportFormat = 'onnx' | 'tflite' | 'torchscript' | 'savedmodel';
export type QuantizationType = 'none' | 'int8' | 'float16' | 'dynamic';
export type DeploymentTarget = 'cloud' | 'edge' | 'mobile' | 'server';
export type ServingPlatform = 'torchserve' | 'tfserving' | 'onnxruntime' | 'triton';

export interface ExportOptions {
  format: ExportFormat;
  outputPath: string;
  backend: 'tensorflow' | 'pytorch';
  optimize: boolean;
  quantization: {
    enabled: boolean;
    type: QuantizationType;
  };
  pruning: {
    enabled: boolean;
    sparsity: number;
  };
  opsetVersion?: number;
  dynamicAxes?: Record<string, Record<number, string>>;
}

export interface DeploymentConfig {
  target: DeploymentTarget;
  servingPlatform: ServingPlatform;
  modelName: string;
  version: string;
  resources: {
    instanceType?: string;
    gpuEnabled: boolean;
    replicas: number;
    batchSize: number;
    maxBatchDelay: number;
  };
  networking: {
    port: number;
    enableMetrics: boolean;
    enableHealthCheck: boolean;
  };
  autoscaling?: {
    enabled: boolean;
    minReplicas: number;
    maxReplicas: number;
    targetCPU: number;
  };
}

export interface ExportResult {
  success: boolean;
  exportPath?: string;
  format: ExportFormat;
  size?: number;
  message?: string;
  error?: string;
}

export interface DeploymentResult {
  success: boolean;
  deploymentId?: string;
  endpoint?: string;
  configFiles?: string[];
  message?: string;
  error?: string;
}

export interface ServingConfig {
  configPath: string;
  modelStorePath?: string;
  scripts: string[];
  instructions: string[];
}

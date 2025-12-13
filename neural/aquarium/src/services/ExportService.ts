import axios from 'axios';
import {
  ExportOptions,
  ExportResult,
  DeploymentConfig,
  DeploymentResult,
  ServingConfig,
} from '../types/export';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export class ExportService {
  private apiUrl: string;

  constructor(apiUrl: string = API_BASE_URL) {
    this.apiUrl = apiUrl;
  }

  async exportModel(
    modelData: any,
    options: ExportOptions
  ): Promise<ExportResult> {
    try {
      const response = await axios.post(`${this.apiUrl}/api/export/model`, {
        model_data: modelData,
        options: options,
      });

      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        return {
          success: false,
          format: options.format,
          error: error.response.data.error || 'Export failed',
        };
      }
      return {
        success: false,
        format: options.format,
        error: 'Failed to communicate with export service',
      };
    }
  }

  async deployModel(
    exportPath: string,
    config: DeploymentConfig
  ): Promise<DeploymentResult> {
    try {
      const response = await axios.post(`${this.apiUrl}/api/deployment/deploy`, {
        export_path: exportPath,
        config: config,
      });

      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        return {
          success: false,
          error: error.response.data.error || 'Deployment failed',
        };
      }
      return {
        success: false,
        error: 'Failed to communicate with deployment service',
      };
    }
  }

  async generateServingConfig(
    modelPath: string,
    platform: string,
    modelName: string,
    config: any
  ): Promise<ServingConfig> {
    try {
      const response = await axios.post(
        `${this.apiUrl}/api/deployment/serving-config`,
        {
          model_path: modelPath,
          platform: platform,
          model_name: modelName,
          config: config,
        }
      );

      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw new Error(
          error.response.data.error || 'Failed to generate serving config'
        );
      }
      throw new Error('Failed to communicate with deployment service');
    }
  }

  async getDeploymentStatus(deploymentId: string): Promise<any> {
    try {
      const response = await axios.get(
        `${this.apiUrl}/api/deployment/${deploymentId}/status`
      );
      return response.data;
    } catch (error) {
      throw new Error('Failed to get deployment status');
    }
  }

  async listDeployments(): Promise<any[]> {
    try {
      const response = await axios.get(`${this.apiUrl}/api/deployment/list`);
      return response.data.deployments || [];
    } catch (error) {
      console.error('Failed to list deployments:', error);
      return [];
    }
  }

  async validateExportOptions(options: ExportOptions): Promise<{
    valid: boolean;
    errors: string[];
  }> {
    const errors: string[] = [];

    if (!options.outputPath || options.outputPath.trim() === '') {
      errors.push('Output path is required');
    }

    if (options.format === 'tflite' && options.backend !== 'tensorflow') {
      errors.push('TFLite export requires TensorFlow backend');
    }

    if (
      options.format === 'torchscript' &&
      options.backend !== 'pytorch'
    ) {
      errors.push('TorchScript export requires PyTorch backend');
    }

    if (
      options.format === 'savedmodel' &&
      options.backend !== 'tensorflow'
    ) {
      errors.push('SavedModel export requires TensorFlow backend');
    }

    if (options.pruning.enabled && options.pruning.sparsity < 0) {
      errors.push('Pruning sparsity must be non-negative');
    }

    if (options.pruning.enabled && options.pruning.sparsity > 1) {
      errors.push('Pruning sparsity must be <= 1.0');
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }
}

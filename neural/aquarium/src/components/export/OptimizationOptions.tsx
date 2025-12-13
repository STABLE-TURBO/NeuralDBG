import React from 'react';
import { QuantizationType } from '../../types';
import './OptimizationOptions.css';

interface OptimizationOptionsProps {
  optimize: boolean;
  quantization: {
    enabled: boolean;
    type: QuantizationType;
  };
  pruning: {
    enabled: boolean;
    sparsity: number;
  };
  onOptimizeChange: (optimize: boolean) => void;
  onQuantizationChange: (enabled: boolean, type: QuantizationType) => void;
  onPruningChange: (enabled: boolean, sparsity: number) => void;
}

const OptimizationOptions: React.FC<OptimizationOptionsProps> = ({
  optimize,
  quantization,
  pruning,
  onOptimizeChange,
  onQuantizationChange,
  onPruningChange,
}) => {
  return (
    <div className="optimization-options">
      <h3>Optimization Options</h3>

      <div className="option-section">
        <label className="option-checkbox">
          <input
            type="checkbox"
            checked={optimize}
            onChange={(e) => onOptimizeChange(e.target.checked)}
          />
          <span>Enable General Optimizations</span>
        </label>
        <p className="option-description">
          Apply standard optimization passes (constant folding, layer fusion, etc.)
        </p>
      </div>

      <div className="option-section">
        <label className="option-checkbox">
          <input
            type="checkbox"
            checked={quantization.enabled}
            onChange={(e) =>
              onQuantizationChange(e.target.checked, quantization.type)
            }
          />
          <span>Enable Quantization</span>
        </label>
        <p className="option-description">
          Reduce model size and improve inference speed by using lower precision
        </p>

        {quantization.enabled && (
          <div className="option-details">
            <label>Quantization Type:</label>
            <select
              value={quantization.type}
              onChange={(e) =>
                onQuantizationChange(true, e.target.value as QuantizationType)
              }
              className="quantization-select"
            >
              <option value="none">None</option>
              <option value="int8">INT8 (Full Integer Quantization)</option>
              <option value="float16">Float16 (Half Precision)</option>
              <option value="dynamic">Dynamic Range Quantization</option>
            </select>

            <div className="quantization-info">
              {quantization.type === 'int8' && (
                <p>
                  <strong>INT8:</strong> Maximum compression, requires
                  representative dataset. Best for edge devices.
                </p>
              )}
              {quantization.type === 'float16' && (
                <p>
                  <strong>Float16:</strong> Good balance between size and
                  accuracy. Supported on GPUs with Tensor Cores.
                </p>
              )}
              {quantization.type === 'dynamic' && (
                <p>
                  <strong>Dynamic:</strong> Quantizes weights but keeps
                  activations in float. Easy to use, no dataset required.
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="option-section">
        <label className="option-checkbox">
          <input
            type="checkbox"
            checked={pruning.enabled}
            onChange={(e) => onPruningChange(e.target.checked, pruning.sparsity)}
          />
          <span>Enable Pruning</span>
        </label>
        <p className="option-description">
          Remove less important weights to reduce model size
        </p>

        {pruning.enabled && (
          <div className="option-details">
            <label>Sparsity Level: {(pruning.sparsity * 100).toFixed(0)}%</label>
            <input
              type="range"
              min="0"
              max="0.9"
              step="0.1"
              value={pruning.sparsity}
              onChange={(e) =>
                onPruningChange(true, parseFloat(e.target.value))
              }
              className="sparsity-slider"
            />
            <div className="sparsity-labels">
              <span>0%</span>
              <span>90%</span>
            </div>
            <p className="pruning-warning">
              Note: Higher sparsity may reduce accuracy. Test thoroughly.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default OptimizationOptions;

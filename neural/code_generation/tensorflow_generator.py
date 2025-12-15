import logging
import warnings
from typing import Any, Dict
from neural.code_generation.base_generator import BaseCodeGenerator
from neural.code_generation.shape_policy_helpers import ensure_2d_before_dense_tf, get_rank_non_batch

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class TensorFlowGenerator(BaseCodeGenerator):
    def generate(self) -> str:
        expanded_layers = self.expand_layers()
        optimizer_config = self.model_data.get('optimizer', {'type': 'Adam'})
        optimizer_type = optimizer_config['type'] if isinstance(optimizer_config, dict) else optimizer_config

        code = "import tensorflow as tf\nfrom tensorflow.keras import layers\n"
        code += f"from tensorflow.keras.optimizers import {optimizer_type}\n"
        code += "from neural.tracking.experiment_tracker import ExperimentManager\n\n"

        code += "# Initialize Experiment Tracking\n"
        code += "experiment_manager = ExperimentManager()\n"
        code += "experiment = experiment_manager.create_experiment()\n"
        code += "experiment.log_hyperparameters({'optimizer': '" + optimizer_type + "', 'backend': 'tensorflow'})\n\n"

        code += "# Custom Callback for Tracking\n"
        code += "class NeuralTrackingCallback(tf.keras.callbacks.Callback):\n"
        code += "    def __init__(self, experiment):\n"
        code += "        super().__init__()\n"
        code += "        self.experiment = experiment\n\n"
        code += "    def on_epoch_end(self, epoch, logs=None):\n"
        code += "        if logs:\n"
        code += "            self.experiment.log_metrics(logs, step=epoch)\n\n"

        input_shape = tuple(self.model_data['input']['shape'])
        code += f"# Input layer with shape {input_shape}\n"
        code += f"inputs = layers.Input(shape={input_shape})\n"
        code += "x = inputs\n\n"

        for layer in expanded_layers:
            layer_type = layer['type']
            params = layer.get('params', {})

            rank_non_batch = get_rank_non_batch(self.current_input_shape)
            if layer_type in ("Dense", "Output"):
                insert_code, self.current_input_shape = ensure_2d_before_dense_tf(
                    rank_non_batch, self.auto_flatten_output, self.propagator, self.current_input_shape
                )
                code += insert_code

            if layer_type == "Residual":
                code += "# Residual block\n"
                code += "residual_input = x\n"
                for sub_layer in layer.get('sub_layers', []):
                    sub_type = sub_layer['type']
                    sub_params = sub_layer.get('params', {})
                    layer_code = self.generate_layer(sub_type, sub_params)
                    if layer_code:
                        if ('\n' in layer_code) or ('x =' in layer_code):
                            code += layer_code + "\n"
                        else:
                            code += f"x = {layer_code}(x)\n"
                code += "x = layers.Add()([x, residual_input])\n"
            else:
                layer_code = self.generate_layer(layer_type, params)
                if layer_code:
                    if ('\n' in layer_code) or ('x =' in layer_code):
                        code += layer_code + "\n"
                    else:
                        code += f"x = {layer_code}(x)\n"
            try:
                self.current_input_shape = self.propagator.propagate(self.current_input_shape, layer)
            except Exception as e:
                logger.warning(f"Shape propagation warning: {e}")

        code += "\n# Build model\n"
        code += "model = tf.keras.Model(inputs=inputs, outputs=x)\n"

        opt_params = []
        if isinstance(optimizer_config, dict):
            for k, v in optimizer_config.get('params', {}).items():
                opt_params.append(f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}")
        loss_entry = self.model_data.get('loss', {'value': 'categorical_crossentropy'})
        if loss_entry is None or not isinstance(loss_entry, (str, dict)):
            loss_value = 'categorical_crossentropy'
        elif isinstance(loss_entry, str):
            loss_value = loss_entry
        else:
            loss_value = loss_entry.get('value', 'categorical_crossentropy')
        code += f"# Compile model with {optimizer_type} optimizer and {loss_value} loss\n"
        code += f"model.compile(loss='{loss_value}', optimizer={optimizer_type}({', '.join(opt_params)}))\n"

        if 'training_config' in self.model_data:
            tc = self.model_data['training_config']
            code += "# Training configuration\n"
            code += (
                f"model.fit(\n    x_train, y_train,\n"
                f"    epochs={tc.get('epochs', 10)},\n"
                f"    batch_size={tc.get('batch_size', 32)},\n"
                f"    validation_split={tc.get('validation_split', 0.2)},\n"
                f"    callbacks=[NeuralTrackingCallback(experiment)],\n"
                f"    verbose=1\n)\n"
            )
            if tc.get('mixed_precision', False):
                code = "from tensorflow.keras.mixed_precision import set_global_policy\n" + code
                code += "set_global_policy('mixed_float16')\n"
            if 'save_path' in tc:
                code += f"model.save('{tc['save_path']}')\n"
        return code

    def generate_layer(self, layer_type: str, params: Dict[str, Any]) -> str:
        if layer_type == "MultiHeadAttention":
            num_heads = params.get("num_heads", 8)
            key_dim = params.get("key_dim", 64)
            value_dim = params.get("value_dim", None)
            dropout = params.get("dropout", 0.0)
            use_bias = params.get("use_bias", True)
            mode = params.get("mode", "self")

            value_dim_str = f", value_dim={value_dim}" if value_dim else ""
            dropout_str = f", dropout={dropout}" if dropout > 0 else ""
            use_bias_str = f", use_bias={use_bias}" if not use_bias else ""

            if mode == "cross":
                return f"layers.MultiHeadAttention(num_heads={num_heads}, key_dim={key_dim}{value_dim_str}{dropout_str}{use_bias_str})(x, context)"
            else:
                return f"layers.MultiHeadAttention(num_heads={num_heads}, key_dim={key_dim}{value_dim_str}{dropout_str}{use_bias_str})(x, x)"
        elif layer_type == "TransformerEncoder":
            num_heads = params.get("num_heads", 8)
            ff_dim = params.get("ff_dim", 512)
            dropout = params.get("dropout", 0.1)
            num_layers = params.get("num_layers", 1)
            activation = params.get("activation", "relu")
            use_attention_mask = params.get("use_attention_mask", False)
            # d_model is used for key_dim in MultiHeadAttention and can be inferred from input
            d_model = params.get("d_model", None)

            code = ["# TransformerEncoder block"]

            if use_attention_mask:
                code.append("# Attention mask should be provided as input")
                code.append("attention_mask = None  # Set this to your mask tensor")

            for layer_idx in range(num_layers):
                code.append(f"# Encoder Layer {layer_idx + 1}")
                code.append("x = layers.LayerNormalization(epsilon=1e-6)(x)")

                # Use d_model if provided, otherwise use ff_dim for key_dim
                key_dim = d_model if d_model else ff_dim

                if use_attention_mask:
                    code.append(f"attn_output = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={key_dim})(x, x, attention_mask=attention_mask)")
                else:
                    code.append(f"attn_output = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={key_dim})(x, x)")

                code.append(f"attn_output = layers.Dropout({dropout})(attn_output)")
                code.append("x = layers.Add()([x, attn_output])")
                code.append("x = layers.LayerNormalization(epsilon=1e-6)(x)")
                code.append(f"ffn_output = layers.Dense({ff_dim}, activation='{activation}')(x)")
                code.append(f"ffn_output = layers.Dense({key_dim})(ffn_output)")
                code.append(f"ffn_output = layers.Dropout({dropout})(ffn_output)")
                code.append("x = layers.Add()([x, ffn_output])")

            return "\n".join(code)
        elif layer_type == "TransformerDecoder":
            num_heads = params.get("num_heads", 8)
            ff_dim = params.get("ff_dim", 512)
            dropout = params.get("dropout", 0.1)
            d_model = params.get("d_model", ff_dim)
            use_causal_mask = params.get("use_causal_mask", True)
            code = [
                "# TransformerDecoder block with cross-attention",
                "# Self-attention with causal masking",
                "decoder_norm1 = layers.LayerNormalization(epsilon=1e-6)(x)",
            ]
            if use_causal_mask:
                code.append("# Apply causal mask for autoregressive decoding")
                code.append(f"self_attn_output = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={d_model}, use_causal_mask=True)(decoder_norm1, decoder_norm1)")
            else:
                code.append(f"self_attn_output = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={d_model})(decoder_norm1, decoder_norm1)")
            code.extend([
                f"x = layers.Add()([x, layers.Dropout({dropout})(self_attn_output)])",
                "# Cross-attention with encoder output (assume encoder_output available)",
                "decoder_norm2 = layers.LayerNormalization(epsilon=1e-6)(x)",
                f"cross_attn_output = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={d_model})(decoder_norm2, encoder_output, encoder_output)",
                f"x = layers.Add()([x, layers.Dropout({dropout})(cross_attn_output)])",
                "# Feed-forward network",
                "decoder_norm3 = layers.LayerNormalization(epsilon=1e-6)(x)",
                f"ff_output = layers.Dense({ff_dim}, activation='relu')(decoder_norm3)",
                f"ff_output = layers.Dense({d_model})(ff_output)",
                f"x = layers.Add()([x, layers.Dropout({dropout})(ff_output)])"
            ])
            return "\n".join(code)
        elif layer_type == "BatchNormalization":
            momentum = params.get("momentum", 0.99)
            epsilon = params.get("epsilon", 0.001)
            if momentum == 0.99 and epsilon == 0.001:
                return "layers.BatchNormalization()"
            return f"layers.BatchNormalization(momentum={momentum}, epsilon={epsilon})"
        elif layer_type == "Conv2D":
            filters = params.get("filters", 32)
            kernel_size = params.get("kernel_size", (3, 3))
            if isinstance(kernel_size, (tuple, list)):
                kernel_size = kernel_size[0]
            padding = params.get("padding", "same")
            activation = params.get("activation", None)
            code = f"layers.Conv2D(filters={filters}, kernel_size={kernel_size}, padding='{padding}'"
            if activation:
                code += f", activation='{activation}'"
            code += ")"
            return code
        elif layer_type == "Dense":
            units = params.get("units", 64)
            activation = params.get("activation", None)
            code = f"layers.Dense(units={units}"
            if activation:
                code += f", activation='{activation}'"
            code += ")"
            return code
        elif layer_type == "MaxPooling2D":
            pool_size = params.get("pool_size", (2, 2))
            if isinstance(pool_size, (tuple, list)):
                pool_size = pool_size
            strides = params.get("strides", None)
            if strides:
                return f"layers.MaxPooling2D(pool_size={pool_size}, strides={strides})"
            return f"layers.MaxPooling2D(pool_size={pool_size})"
        elif layer_type == "AveragePooling2D":
            pool_size = params.get("pool_size", (2, 2))
            if isinstance(pool_size, (tuple, list)):
                pool_size = pool_size[0] if isinstance(pool_size[0], int) else pool_size
            return f"layers.AveragePooling2D(pool_size={pool_size})"
        elif layer_type == "Flatten":
            return "layers.Flatten()"
        elif layer_type == "Reshape":
            target_shape = params.get("target_shape")
            if target_shape:
                return f"layers.Reshape(target_shape={target_shape})"
            return "layers.Reshape()"
        elif layer_type == "LSTM":
            units = params.get("units", 128)
            return_sequences = params.get("return_sequences", False)
            return f"layers.LSTM(units={units}, return_sequences={str(return_sequences)})"
        elif layer_type == "GRU":
            units = params.get("units", 64)
            return_sequences = params.get("return_sequences", False)
            return f"layers.GRU(units={units}, return_sequences={str(return_sequences)})"
        elif layer_type == "Dropout":
            rate = params.get("rate", 0.5)
            return f"layers.Dropout(rate={rate})"
        elif layer_type == "Embedding":
            input_dim = params.get("input_dim", 10000)
            output_dim = params.get("output_dim", 128)
            mask_zero = params.get("mask_zero", False)
            input_length = params.get("input_length", None)
            code = f"layers.Embedding(input_dim={input_dim}, output_dim={output_dim}"
            if mask_zero:
                code += f", mask_zero={mask_zero}"
            if input_length:
                code += f", input_length={input_length}"
            code += ")"
            return code
        elif layer_type == "GlobalAveragePooling1D":
            return "layers.GlobalAveragePooling1D()"
        elif layer_type == "GlobalAveragePooling2D":
            return "layers.GlobalAveragePooling2D()"
        elif layer_type == "GlobalAveragePooling3D":
            return "layers.GlobalAveragePooling3D()"
        elif layer_type == "GlobalMaxPooling1D":
            return "layers.GlobalMaxPooling1D()"
        elif layer_type == "GlobalMaxPooling2D":
            return "layers.GlobalMaxPooling2D()"
        elif layer_type == "GlobalMaxPooling3D":
            return "layers.GlobalMaxPooling3D()"
        elif layer_type == "Reshape":
            target_shape = params.get("target_shape", None)
            if target_shape:
                return f"layers.Reshape({target_shape})"
            else:
                return None
        elif layer_type == "LayerNormalization":
            axis = params.get("axis", -1)
            epsilon = params.get("epsilon", 1e-6)
            return f"layers.LayerNormalization(axis={axis}, epsilon={epsilon})"
        elif layer_type == "Output":
            units = params.get("units", 10)
            activation = params.get("activation", "softmax")
            return f"layers.Dense(units={units}, activation='{activation}')"
        elif layer_type == "PositionalEncoding":
            max_len = params.get("max_len", 5000)
            encoding_type = params.get("encoding_type", "sinusoidal")
            if encoding_type == "sinusoidal":
                code = [
                    "# Sinusoidal Positional Encoding",
                    "import numpy as np",
                    f"def get_positional_encoding(seq_len, d_model, max_len={max_len}):",
                    "    position = np.arange(seq_len)[:, np.newaxis]",
                    "    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))",
                    "    pos_encoding = np.zeros((seq_len, d_model))",
                    "    pos_encoding[:, 0::2] = np.sin(position * div_term)",
                    "    pos_encoding[:, 1::2] = np.cos(position * div_term)",
                    "    return tf.constant(pos_encoding, dtype=tf.float32)",
                    "seq_len = tf.shape(x)[1]",
                    "d_model = tf.shape(x)[2]",
                    "pos_encoding = get_positional_encoding(seq_len, d_model)",
                    "x = x + pos_encoding"
                ]
                return "\n".join(code)
            else:
                code = [
                    "# Learnable Positional Encoding",
                    f"pos_embedding = layers.Embedding(input_dim={max_len}, output_dim=tf.shape(x)[2])",
                    "seq_len = tf.shape(x)[1]",
                    "positions = tf.range(start=0, limit=seq_len, delta=1)",
                    "x = x + pos_embedding(positions)"
                ]
                return "\n".join(code)
        else:
            warnings.warn(f"Unsupported layer type '{layer_type}' for tensorflow. Skipping.", UserWarning)
            return None

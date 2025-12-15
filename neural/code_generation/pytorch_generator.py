import logging
import warnings
import numpy as np
from typing import Any, Dict, List, Optional
from neural.code_generation.base_generator import BaseCodeGenerator
from neural.code_generation.shape_policy_helpers import ensure_2d_before_dense_pt, get_rank_non_batch

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class PyTorchGenerator(BaseCodeGenerator):
    def generate(self) -> str:
        expanded_layers = self.expand_layers()
        optimizer_config = self.model_data.get('optimizer', {'type': 'Adam'})
        optimizer_type = optimizer_config['type'] if isinstance(optimizer_config, dict) else optimizer_config
        
        code = "import logging\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nimport torchvision.transforms as transforms\nimport math\n"
        code += "from torchvision import datasets\n"
        code += "from torch.utils.data import DataLoader\n"
        code += "from neural.tracking.experiment_tracker import ExperimentManager\n\n"
        code += "logger = logging.getLogger(__name__)\n\n"

        code += "# Initialize Experiment Tracking\n"
        code += "experiment_manager = ExperimentManager()\n"
        code += "experiment = experiment_manager.create_experiment()\n"
        code += "experiment.log_hyperparameters({'optimizer': '" + optimizer_type + "', 'backend': 'pytorch'})\n\n"
        
        needs_positional_encoding = any(layer.get('type') == 'PositionalEncoding' for layer in expanded_layers)
        if needs_positional_encoding:
            code += "# Sinusoidal Positional Encoding\n"
            code += "class SinusoidalPositionalEncoding(nn.Module):\n"
            code += "    def __init__(self, max_len=5000):\n"
            code += "        super(SinusoidalPositionalEncoding, self).__init__()\n"
            code += "        self.max_len = max_len\n\n"
            code += "    def forward(self, x):\n"
            code += "        batch_size, seq_len, d_model = x.size()\n"
            code += "        position = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)\n"
            code += "        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=x.device) * -(math.log(10000.0) / d_model))\n"
            code += "        pos_encoding = torch.zeros(seq_len, d_model, device=x.device)\n"
            code += "        pos_encoding[:, 0::2] = torch.sin(position * div_term)\n"
            code += "        pos_encoding[:, 1::2] = torch.cos(position * div_term)\n"
            code += "        return x + pos_encoding.unsqueeze(0)\n\n"
            code += "# Learnable Positional Encoding\n"
            code += "class LearnablePositionalEncoding(nn.Module):\n"
            code += "    def __init__(self, max_len=5000, d_model=512):\n"
            code += "        super(LearnablePositionalEncoding, self).__init__()\n"
            code += "        self.max_len = max_len\n"
            code += "        self.d_model = d_model\n"
            code += "        self.pos_embedding = None\n\n"
            code += "    def forward(self, x):\n"
            code += "        batch_size, seq_len, d_model = x.size()\n"
            code += "        if self.pos_embedding is None or self.pos_embedding.size(0) != self.max_len or self.pos_embedding.size(1) != d_model:\n"
            code += "            self.pos_embedding = nn.Parameter(torch.randn(self.max_len, d_model, device=x.device))\n"
            code += "        positions = self.pos_embedding[:seq_len, :].unsqueeze(0)\n"
            code += "        return x + positions\n\n"
        
        code += "# Neural network model definition\n"
        code += "class NeuralNetworkModel(nn.Module):\n"
        code += f"{self.indent}def __init__(self):\n"
        code += f"{self.indent}{self.indent}super(NeuralNetworkModel, self).__init__()\n"

        layers_code = []
        forward_code_body: List[str] = []
        layer_counts = {}

        for i, layer in enumerate(expanded_layers):
            layer_type = layer['type']
            params = layer.get('params', {})
            if params is not None:
                params = params.copy()
            else:
                params = {}

            rank_non_batch = get_rank_non_batch(self.current_input_shape)
            if layer_type in ("Dense", "Output"):
                self.current_input_shape = ensure_2d_before_dense_pt(
                    rank_non_batch, self.auto_flatten_output, forward_code_body, self.propagator, self.current_input_shape
                )

            if layer_type not in layer_counts:
                layer_counts[layer_type] = 0

            layer_name = f"layer{i}_{layer_type.lower()}"
            layer_counts[layer_type] += 1

            if layer_type == "MultiHeadAttention":
                layer_code = self.generate_layer(layer_type, params)
                layers_code.append(f"self.{layer_name} = {layer_code}")
                mode = params.get("mode", "self")
                if mode == "cross":
                    forward_code_body.append(f"x, _ = self.{layer_name}(x, context, context)")
                else:
                    forward_code_body.append(f"x, _ = self.{layer_name}(x, x, x)")
            elif layer_type == "Dense":
                if i == 0 or expanded_layers[i-1]['type'] in ["Input", "Flatten"]:
                    dims = []
                    logger.warning(f"Current input shape: {self.current_input_shape}")
                    for dim in self.current_input_shape[1:]:
                        if dim is not None:
                            if isinstance(dim, dict):
                                if 'value' in dim:
                                    dims.append(dim['value'])
                                else:
                                    logger.warning(f"Dictionary dimension without 'value' key: {dim}, using default")
                                    dims.append(64)
                            elif isinstance(dim, (int, float)):
                                dims.append(dim)
                            else:
                                logger.warning(f"Unexpected dimension type: {type(dim)}, value: {dim}, using default")
                                dims.append(64)
                        else:
                            logger.warning("None dimension found, skipping")
                    logger.warning(f"Dimensions after processing: {dims}")
                    in_features = np.prod(dims) if dims else 64
                else:
                    in_features = self.current_input_shape[-1]
                    if isinstance(in_features, dict):
                        if 'value' in in_features:
                            in_features = in_features['value']
                        else:
                            logger.warning(f"Dictionary dimension without 'value' key: {in_features}, using default")
                            in_features = 64
                out_features = params.get("units", 64)
                if isinstance(out_features, dict):
                    if 'value' in out_features:
                        out_features = out_features['value']
                    else:
                        logger.warning(f"Dictionary parameter without 'value' key: {out_features}, using default")
                        out_features = 64
                layer_code = f"nn.Linear(in_features={in_features}, out_features={out_features})"
                layers_code.append(f"self.{layer_name} = {layer_code}")
                forward_code_body.append(f"x = self.{layer_name}(x)")
            elif layer_type == "Dropout":
                rate = params.get("rate", 0.5)
                if isinstance(rate, dict):
                    if 'value' in rate:
                        rate = rate['value']
                    else:
                        logger.warning(f"Dictionary parameter without 'value' key: {rate}, using default")
                        rate = 0.5
                layer_code = f"nn.Dropout(p={rate})"
                layers_code.append(f"self.{layer_name} = {layer_code}")
                forward_code_body.append(f"x = self.{layer_name}(x)")
            elif layer_type == "Embedding":
                num_embeddings = params.get("input_dim", 1000)
                if isinstance(num_embeddings, dict):
                    if 'value' in num_embeddings:
                        num_embeddings = num_embeddings['value']
                    else:
                        logger.warning(f"Dictionary parameter without 'value' key: {num_embeddings}, using default")
                        num_embeddings = 1000
                embedding_dim = params.get("output_dim", 128)
                if isinstance(embedding_dim, dict):
                    if 'value' in embedding_dim:
                        embedding_dim = embedding_dim['value']
                    else:
                        logger.warning(f"Dictionary parameter without 'value' key: {embedding_dim}, using default")
                        embedding_dim = 128
                layer_code = f"nn.Embedding(num_embeddings={num_embeddings}, embedding_dim={embedding_dim})"
                layers_code.append(f"self.{layer_name} = {layer_code}")
                forward_code_body.append(f"x = self.{layer_name}(x)")
            elif layer_type == "PositionalEncoding":
                layer_code = self.generate_layer(layer_type, params)
                if layer_code:
                    layers_code.append(f"self.{layer_name} = {layer_code}")
                    forward_code_body.append(f"x = self.{layer_name}(x)")
            elif layer_type == "TransformerEncoder":
                layer_code = self.generate_layer(layer_type, params)
                if layer_code:
                    layers_code.append(f"self.{layer_name} = {layer_code}")
                    use_attention_mask = params.get("use_attention_mask", False)
                    if isinstance(use_attention_mask, dict):
                        if 'value' in use_attention_mask:
                            use_attention_mask = use_attention_mask['value']
                    
                    if use_attention_mask:
                        forward_code_body.append(f"# Pass attention mask if available (set src_key_padding_mask for padding)")
                        forward_code_body.append(f"x = self.{layer_name}(x, src_key_padding_mask=None)")
                    else:
                        forward_code_body.append(f"x = self.{layer_name}(x)")
            elif layer_type == "TransformerDecoder":
                layer_code = self.generate_layer(layer_type, params)
                if layer_code:
                    layers_code.append(f"self.{layer_name} = {layer_code}")
                    forward_code_body.append(f"# TransformerDecoder requires memory from encoder")
                    forward_code_body.append(f"# x = self.{layer_name}(x, memory)")
                    forward_code_body.append(f"x = self.{layer_name}(x, x)  # Self-attention only for now")
            elif layer_type == "Flatten":
                # Flatten is handled inline, no layer needed
                forward_code_body.append(f"x = x.view(x.size(0), -1)  # Flatten")
            elif layer_type == "GlobalAveragePooling1D":
                layer_code = self.generate_layer(layer_type, params)
                if layer_code:
                    layers_code.append(f"self.{layer_name} = {layer_code}")
                    forward_code_body.append(f"x = self.{layer_name}(x).squeeze(-1)  # Remove last dim after pooling")
            elif layer_type == "Reshape":
                target_shape = params.get("target_shape", None)
                if target_shape:
                    forward_code_body.append(f"x = x.view(x.size(0), {', '.join(map(str, target_shape))})")
            elif layer_type == "LayerNormalization":
                # Get the normalized shape from input
                if self.current_input_shape and len(self.current_input_shape) >= 2:
                    normalized_shape = self.current_input_shape[1:]
                    layers_code.append(f"self.{layer_name} = nn.LayerNorm({normalized_shape})")
                    forward_code_body.append(f"x = self.{layer_name}(x)")
            elif layer_type == "Output":
                in_features = self.current_input_shape[-1]
                if isinstance(in_features, dict):
                    if 'value' in in_features:
                        in_features = in_features['value']
                    else:
                        logger.warning(f"Dictionary dimension without 'value' key: {in_features}, using default")
                        in_features = 64
                out_features = params.get("units", 10)
                if isinstance(out_features, dict):
                    if 'value' in out_features:
                        out_features = out_features['value']
                    else:
                        logger.warning(f"Dictionary parameter without 'value' key: {out_features}, using default")
                        out_features = 10
                activation = params.get("activation", "softmax")
                if isinstance(activation, dict):
                    if 'value' in activation:
                        activation = activation['value']
                    else:
                        logger.warning(f"Dictionary parameter without 'value' key: {activation}, using default")
                        activation = "softmax"
                if activation == "softmax":
                    layer_code = f"nn.Sequential(nn.Linear(in_features={in_features}, out_features={out_features}), nn.Softmax(dim=1))"
                else:
                    layer_code = f"nn.Linear(in_features={in_features}, out_features={out_features})"
                layers_code.append(f"self.{layer_name} = {layer_code}")
                forward_code_body.append(f"x = self.{layer_name}(x)")

            try:
                self.current_input_shape = self.propagator.propagate(self.current_input_shape, layer)
            except Exception as e:
                logger.warning(f"Shape propagation warning: {e}")

        for line in layers_code:
            code += f"{self.indent}{self.indent}{line}\n"
        code += f"\n{self.indent}# Forward pass\n"
        code += f"{self.indent}def forward(self, x):\n"
        if expanded_layers and expanded_layers[0]['type'] == 'Dense':
            code += f"{self.indent}{self.indent}x = x.view(x.size(0), -1)  # Flatten input\n"
        for line in forward_code_body:
            code += f"{self.indent}{self.indent}{line}\n"
        code += f"{self.indent}{self.indent}return x\n\n"

        code += "# Model instantiation\n"
        code += "model = NeuralNetworkModel()\n"
        code += "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        code += "model.to(device)\n\n"

        code += "# MNIST dataset\n"
        code += "transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])\n"
        code += "train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)\n"
        batch_size = self.model_data.get('training_config', {}).get('batch_size', 64)
        if self.best_params and 'batch_size' in self.best_params:
            batch_size = self.best_params['batch_size']
        code += f"train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)\n\n"

        loss_entry = self.model_data.get('loss', {'value': 'crossentropy'})
        if loss_entry is None or not isinstance(loss_entry, (str, dict)):
            loss_value = 'crossentropy'
        elif isinstance(loss_entry, str):
            loss_value = loss_entry
        else:
            loss_value = loss_entry.get('value', 'crossentropy')
        loss_fn = "nn.CrossEntropyLoss()" if "crossentropy" in loss_value.lower() else "nn.MSELoss()"
        code += f"# Loss function\nloss_fn = {loss_fn}\n"

        opt_params = []
        if isinstance(optimizer_config, dict):
            for k, v in optimizer_config.get('params', {'lr': 0.001}).items():
                param_name = 'lr' if k == 'learning_rate' else k
                if isinstance(v, dict):
                    if 'hpo' in v and self.best_params and 'learning_rate' in self.best_params:
                        v = self.best_params['learning_rate']
                    elif 'value' in v:
                        v = v['value']
                    else:
                        logger.warning(f"Dictionary parameter without 'value' key: {v}, using default")
                        v = 0.001
                opt_params.append(f"{param_name}={repr(v)}")
        code += f"# Optimizer\noptimizer = optim.{optimizer_type}(model.parameters(), {', '.join(opt_params)})\n"

        if 'training_config' in self.model_data:
            tc = self.model_data['training_config']
            code += "\n# Mixed precision training setup\n"
            code += "scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')\n"
            code += f"for epoch in range({tc.get('epochs', 10)}):\n"
            code += f"{self.indent}running_loss = 0.0\n"
            code += f"{self.indent}for batch_idx, (data, target) in enumerate(train_loader):\n"
            code += f"{self.indent}{self.indent}data, target = data.to(device), target.to(device)\n"
            code += f"{self.indent}{self.indent}optimizer.zero_grad()\n"
            code += f"{self.indent}{self.indent}with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):\n"
            code += f"{self.indent}{self.indent}{self.indent}output = model(data)\n"
            code += f"{self.indent}{self.indent}{self.indent}loss = loss_fn(output, target)\n"
            code += f"{self.indent}{self.indent}scaler.scale(loss).backward()\n"
            code += f"{self.indent}{self.indent}scaler.step(optimizer)\n"
            code += f"{self.indent}{self.indent}scaler.update()\n"
            code += f"{self.indent}{self.indent}running_loss += loss.item()  # Accumulate loss\n"
            code += f"{self.indent}avg_loss = running_loss / len(train_loader)\n"
            code += f"{self.indent}print(f'Epoch {{epoch+1}}/{{{tc.get('epochs', 10)}}} - Loss: {{avg_loss:.4f}}')\n"
            code += "\n# Evaluate model\n"
            code += "model.eval()\n"
            code += "correct = 0\n"
            code += "total = 0\n"
            code += "with torch.no_grad():\n"
            code += f"{self.indent}for data, target in train_loader:\n"
            code += f"{self.indent}{self.indent}data, target = data.to(device), target.to(device)\n"
            code += f"{self.indent}{self.indent}outputs = model(data)\n"
            code += f"{self.indent}{self.indent}_, predicted = torch.max(outputs.data, 1)\n"
            code += f"{self.indent}{self.indent}total += target.size(0)\n"
            code += f"{self.indent}{self.indent}correct += (predicted == target).sum().item()\n"
            code += f"{self.indent}accuracy = 100 * correct / total\n"
            code += "logger.info(f'Accuracy: {accuracy:.2f}%')\n"
            code += "experiment.log_metrics({'loss': avg_loss, 'accuracy': accuracy}, step=epoch)\n"
            if 'save_path' in tc:
                code += f"{self.indent}{self.indent}torch.save(model.state_dict(), '{tc['save_path']}')\n"

        return code

    def generate_layer(self, layer_type: str, params: Dict[str, Any]) -> str:
        return generate_pytorch_layer(layer_type, params, self.current_input_shape)


def generate_pytorch_layer(layer_type: str, params: Dict[str, Any], input_shape: Optional[tuple] = None) -> str:
    if layer_type == "MultiHeadAttention":
        embed_dim = params.get("embed_dim", None)
        num_heads = params.get("num_heads", 8)
        dropout = params.get("dropout", 0.0)
        batch_first = params.get("batch_first", True)
        
        if embed_dim is None and input_shape is not None and len(input_shape) >= 2:
            embed_dim = input_shape[-1]
            if isinstance(embed_dim, dict):
                if 'value' in embed_dim:
                    embed_dim = embed_dim['value']
                else:
                    logger.warning(f"Dictionary dimension without 'value' key: {embed_dim}, using default")
                    embed_dim = 512
        elif embed_dim is None:
            embed_dim = 512
        
        if isinstance(embed_dim, dict):
            if 'value' in embed_dim:
                embed_dim = embed_dim['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {embed_dim}, using default")
                embed_dim = 512
        
        if isinstance(num_heads, dict):
            if 'value' in num_heads:
                num_heads = num_heads['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {num_heads}, using default")
                num_heads = 8
        
        if isinstance(dropout, dict):
            if 'value' in dropout:
                dropout = dropout['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {dropout}, using default")
                dropout = 0.0
        
        return f"nn.MultiheadAttention(embed_dim={embed_dim}, num_heads={num_heads}, dropout={dropout}, batch_first={batch_first})"
    elif layer_type == "Conv2D":
        data_format = params.get("data_format", "channels_last")
        in_channels = 3
        if input_shape is not None:
            in_channels = input_shape[1] if data_format == "channels_first" else input_shape[3]
            in_channels = in_channels if len(input_shape) > 3 else 3
        out_channels = params.get("filters", 32)
        if isinstance(out_channels, dict):
            if 'value' in out_channels:
                out_channels = out_channels['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {out_channels}, using default")
                out_channels = 32
        kernel_size = params.get("kernel_size", 3)
        if isinstance(kernel_size, dict):
            if 'value' in kernel_size:
                kernel_size = kernel_size['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {kernel_size}, using default")
                kernel_size = 3
        if isinstance(kernel_size, (tuple, list)):
            kernel_size = kernel_size[0]
        return f"nn.Conv2d(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size})"
    elif layer_type == "BatchNormalization":
        data_format = params.get("data_format", "channels_last")
        if input_shape and len(input_shape) > 3:
            num_features = input_shape[1] if data_format == "channels_first" else input_shape[3]
        else:
            num_features = params.get("filters", 64)
        momentum = params.get("momentum", 0.9)
        eps = params.get("epsilon", 0.001)
        if momentum == 0.9 and eps == 0.001:
            return f"nn.BatchNorm2d(num_features={num_features})"
        return f"nn.BatchNorm2d(num_features={num_features}, momentum={momentum}, eps={eps})"
    elif layer_type == "Dense":
        if input_shape:
            dims = []
            for dim in input_shape[1:]:
                if dim is not None:
                    if isinstance(dim, dict):
                        if 'value' in dim:
                            dims.append(dim['value'])
                        else:
                            logger.warning(f"Dictionary dimension without 'value' key: {dim}, using default")
                            dims.append(64)
                    else:
                        dims.append(dim)
            in_features = np.prod(dims) if dims else 64
        else:
            in_features = 64
        out_features = params.get("units", 64)
        if isinstance(out_features, dict):
            if 'value' in out_features:
                out_features = out_features['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {out_features}, using default")
                out_features = 64
        activation = params.get("activation", None)
        if isinstance(activation, dict):
            if 'value' in activation:
                activation = activation['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {activation}, using default")
                activation = None
        layers = [f"nn.Linear(in_features={in_features}, out_features={out_features})"]
        if activation:
            if activation == "relu":
                layers.append("nn.ReLU()")
            elif activation == "tanh":
                layers.append("nn.Tanh()")
            elif activation == "softmax":
                layers.append("nn.Softmax(dim=1)")
            elif activation == "invalid":
                layers.append("nn.Identity()")
        return "nn.Sequential(" + ", ".join(layers) + ")"
    elif layer_type == "MaxPooling2D":
        pool_size = params.get("pool_size", 2)
        if isinstance(pool_size, (tuple, list)):
            pool_size = pool_size if len(pool_size) == 2 else (pool_size[0], pool_size[0])
        strides = params.get("strides", None)
        if strides:
            return f"nn.MaxPool2d(kernel_size={pool_size}, stride={strides})"
        return f"nn.MaxPool2d(kernel_size={pool_size})"
    elif layer_type == "AveragePooling2D":
        pool_size = params.get("pool_size", 2)
        if isinstance(pool_size, (tuple, list)):
            pool_size = pool_size if len(pool_size) == 2 else (pool_size[0], pool_size[0])
        return f"nn.AvgPool2d(kernel_size={pool_size})"
    elif layer_type == "Flatten":
        return "nn.Flatten()"
    elif layer_type == "Reshape":
        # PyTorch doesn't have a Reshape layer, we'll use view in forward pass
        # For now, just note it requires special handling
        target_shape = params.get("target_shape", (-1,))
        return f"# Reshape to {target_shape} (handled in forward with view/reshape)"
    elif layer_type == "Dropout":
        rate = params.get("rate", 0.5)
        if isinstance(rate, dict):
            if 'value' in rate:
                rate = rate['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {rate}, using default")
                rate = 0.5
        return f"nn.Dropout(p={rate})"
    elif layer_type == "Embedding":
        num_embeddings = params.get("input_dim", 1000)
        if isinstance(num_embeddings, dict):
            if 'value' in num_embeddings:
                num_embeddings = num_embeddings['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {num_embeddings}, using default")
                num_embeddings = 1000
        embedding_dim = params.get("output_dim", 128)
        if isinstance(embedding_dim, dict):
            if 'value' in embedding_dim:
                embedding_dim = embedding_dim['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {embedding_dim}, using default")
                embedding_dim = 128
        return f"nn.Embedding(num_embeddings={num_embeddings}, embedding_dim={embedding_dim})"
    elif layer_type == "Output":
        if input_shape:
            dims = []
            for dim in input_shape[1:]:
                if dim is not None:
                    if isinstance(dim, dict):
                        if 'value' in dim:
                            dims.append(dim['value'])
                        else:
                            logger.warning(f"Dictionary dimension without 'value' key: {dim}, using default")
                            dims.append(64)
                    else:
                        dims.append(dim)
            in_features = np.prod(dims) if dims else 64
        else:
            in_features = 64
        out_features = params.get("units", 10)
        if isinstance(out_features, dict):
            if 'value' in out_features:
                out_features = out_features['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {out_features}, using default")
                out_features = 10
        activation = params.get("activation", "softmax")
        if isinstance(activation, dict):
            if 'value' in activation:
                activation = activation['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {activation}, using default")
                activation = "softmax"
        layers = [f"nn.Linear(in_features={in_features}, out_features={out_features})"]
        if activation == "softmax":
            layers.append("nn.Softmax(dim=1)")
        return "nn.Sequential(" + ", ".join(layers) + ")"
    elif layer_type == "LSTM":
        if input_shape:
            input_size = input_shape[-1]
            if isinstance(input_size, dict):
                if 'value' in input_size:
                    input_size = input_size['value']
                else:
                    logger.warning(f"Dictionary dimension without 'value' key: {input_size}, using default")
                    input_size = 32
        else:
            input_size = 32

        hidden_size = params.get("units", 128)
        if isinstance(hidden_size, dict):
            if 'value' in hidden_size:
                hidden_size = hidden_size['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {hidden_size}, using default")
                hidden_size = 128
        return f"nn.LSTM(input_size={input_size}, hidden_size={hidden_size}, batch_first=True)"
    elif layer_type == "GRU":
        input_size = params.get("input_size", 128)
        if isinstance(input_size, dict):
            if 'value' in input_size:
                input_size = input_size['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {input_size}, using default")
                input_size = 128

        hidden_size = params.get("units", 64)
        if isinstance(hidden_size, dict):
            if 'value' in hidden_size:
                hidden_size = hidden_size['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {hidden_size}, using default")
                hidden_size = 64
        return f"nn.GRU(input_size={input_size}, hidden_size={hidden_size}, batch_first=True)"
    elif layer_type == "TransformerEncoder":
        # Infer d_model from input_shape if not explicitly provided
        d_model = params.get("d_model", None)
        if d_model is None and input_shape is not None and len(input_shape) >= 3:
            # For transformer, input shape is (batch, seq_len, d_model)
            d_model = input_shape[-1]
            if isinstance(d_model, dict):
                if 'value' in d_model:
                    d_model = d_model['value']
                else:
                    logger.warning(f"Dictionary dimension without 'value' key: {d_model}, using default")
                    d_model = 512
        elif d_model is None:
            d_model = 512
        
        if isinstance(d_model, dict):
            if 'value' in d_model:
                d_model = d_model['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {d_model}, using default")
                d_model = 512

        nhead = params.get("num_heads", 8)
        if isinstance(nhead, dict):
            if 'value' in nhead:
                nhead = nhead['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {nhead}, using default")
                nhead = 8

        dim_feedforward = params.get("ff_dim", 2048)
        if isinstance(dim_feedforward, dict):
            if 'value' in dim_feedforward:
                dim_feedforward = dim_feedforward['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {dim_feedforward}, using default")
                dim_feedforward = 2048

        dropout = params.get("dropout", 0.1)
        if isinstance(dropout, dict):
            if 'value' in dropout:
                dropout = dropout['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {dropout}, using default")
                dropout = 0.1
        
        num_layers = params.get("num_layers", 1)
        if isinstance(num_layers, dict):
            if 'value' in num_layers:
                num_layers = num_layers['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {num_layers}, using default")
                num_layers = 1
        
        activation = params.get("activation", "relu")
        if isinstance(activation, dict):
            if 'value' in activation:
                activation = activation['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {activation}, using default")
                activation = "relu"
        
        use_attention_mask = params.get("use_attention_mask", False)
        if isinstance(use_attention_mask, dict):
            if 'value' in use_attention_mask:
                use_attention_mask = use_attention_mask['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {use_attention_mask}, using default")
                use_attention_mask = False
        
        if num_layers > 1:
            return f"nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout}, activation='{activation}'), num_layers={num_layers})"
        else:
            return f"nn.TransformerEncoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout}, activation='{activation}')"
    elif layer_type == "TransformerDecoder":
        # Infer d_model from input_shape if not explicitly provided
        d_model = params.get("d_model", None)
        if d_model is None and input_shape is not None and len(input_shape) >= 3:
            # For transformer, input shape is (batch, seq_len, d_model)
            d_model = input_shape[-1]
            if isinstance(d_model, dict):
                if 'value' in d_model:
                    d_model = d_model['value']
                else:
                    logger.warning(f"Dictionary dimension without 'value' key: {d_model}, using default")
                    d_model = 512
        elif d_model is None:
            d_model = 512
        
        if isinstance(d_model, dict):
            if 'value' in d_model:
                d_model = d_model['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {d_model}, using default")
                d_model = 512

        nhead = params.get("num_heads", 8)
        if isinstance(nhead, dict):
            if 'value' in nhead:
                nhead = nhead['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {nhead}, using default")
                nhead = 8

        dim_feedforward = params.get("ff_dim", 2048)
        if isinstance(dim_feedforward, dict):
            if 'value' in dim_feedforward:
                dim_feedforward = dim_feedforward['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {dim_feedforward}, using default")
                dim_feedforward = 2048

        dropout = params.get("dropout", 0.1)
        if isinstance(dropout, dict):
            if 'value' in dropout:
                dropout = dropout['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {dropout}, using default")
                dropout = 0.1
        return f"nn.TransformerDecoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout})"
    elif layer_type == "GlobalAveragePooling1D":
        return "nn.AdaptiveAvgPool1d(1)"
    elif layer_type == "GlobalAveragePooling2D":
        return "nn.AdaptiveAvgPool2d(1)"
    elif layer_type == "GlobalAveragePooling3D":
        return "nn.AdaptiveAvgPool3d(1)"
    elif layer_type == "GlobalMaxPooling1D":
        return "nn.AdaptiveMaxPool1d(1)"
    elif layer_type == "GlobalMaxPooling2D":
        return "nn.AdaptiveMaxPool2d(1)"
    elif layer_type == "GlobalMaxPooling3D":
        return "nn.AdaptiveMaxPool3d(1)"
    elif layer_type == "PositionalEncoding":
        max_len = params.get("max_len", 5000)
        if isinstance(max_len, dict):
            if 'value' in max_len:
                max_len = max_len['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {max_len}, using default")
                max_len = 5000
        
        encoding_type = params.get("encoding_type", "sinusoidal")
        if isinstance(encoding_type, dict):
            if 'value' in encoding_type:
                encoding_type = encoding_type['value']
            else:
                logger.warning(f"Dictionary parameter without 'value' key: {encoding_type}, using default")
                encoding_type = "sinusoidal"
        
        if encoding_type == "sinusoidal":
            return f"SinusoidalPositionalEncoding(max_len={max_len})"
        else:
            return f"LearnablePositionalEncoding(max_len={max_len})"
    else:
        warnings.warn(f"Unsupported layer type '{layer_type}' for pytorch. Skipping.", UserWarning)
        return None

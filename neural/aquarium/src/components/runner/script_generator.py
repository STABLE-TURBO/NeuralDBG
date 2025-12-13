from typing import Any, Dict, Optional


class ScriptGenerator:
    @staticmethod
    def generate_training_script(
        model_code: str,
        dataset: str,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2,
        backend: str = "tensorflow",
        save_weights: bool = False
    ) -> str:
        if backend == "tensorflow":
            return ScriptGenerator._generate_tensorflow_script(
                model_code, dataset, epochs, batch_size, validation_split, save_weights
            )
        elif backend == "pytorch":
            return ScriptGenerator._generate_pytorch_script(
                model_code, dataset, epochs, batch_size, validation_split, save_weights
            )
        else:
            return model_code
    
    @staticmethod
    def _generate_tensorflow_script(
        model_code: str,
        dataset: str,
        epochs: int,
        batch_size: int,
        validation_split: float,
        save_weights: bool
    ) -> str:
        script = f'''#!/usr/bin/env python
"""
Auto-generated training script by Neural Aquarium
Dataset: {dataset}
Backend: TensorFlow
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load dataset
def load_dataset():
    print(f"Loading {{'{dataset}'}} dataset...")
    
    if '{dataset}' == 'MNIST':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        
    elif '{dataset}' == 'CIFAR10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
    elif '{dataset}' == 'CIFAR100':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
    else:
        raise ValueError(f"Unsupported dataset: {{'{dataset}'}}")
    
    print(f"Dataset loaded: {{x_train.shape[0]}} training samples, {{x_test.shape[0]}} test samples")
    return (x_train, y_train), (x_test, y_test)

# Build model
{model_code}

# Main training function
def main():
    print("="*60)
    print("Neural Aquarium - Model Training")
    print("="*60)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_dataset()
    
    # Build and compile model
    print("Building model...")
    model = build_model()
    
    print("\\nModel Summary:")
    model.summary()
    
    # Train model
    print(f"\\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Validation split: {validation_split}")
    print("="*60)
    
    history = model.fit(
        x_train, y_train,
        batch_size={batch_size},
        epochs={epochs},
        validation_split={validation_split},
        verbose=1
    )
    
    # Evaluate on test set
    print("\\n" + "="*60)
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {{test_loss:.4f}}")
    print(f"Test Accuracy: {{test_acc:.4f}}")
    
    {"# Save model weights" if save_weights else ""}
    {"if True:" if save_weights else "# Weights saving disabled"}
    {"    print('\\nSaving model weights...')" if save_weights else ""}
    {"    model.save_weights('model_weights.h5')" if save_weights else ""}
    {"    print('Weights saved to model_weights.h5')" if save_weights else ""}
    
    print("="*60)
    print("Training completed successfully!")
    
if __name__ == "__main__":
    main()
'''
        return script
    
    @staticmethod
    def _generate_pytorch_script(
        model_code: str,
        dataset: str,
        epochs: int,
        batch_size: int,
        validation_split: float,
        save_weights: bool
    ) -> str:
        script = f'''#!/usr/bin/env python
"""
Auto-generated training script by Neural Aquarium
Dataset: {dataset}
Backend: PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# Load dataset
def load_dataset():
    print(f"Loading {{'{dataset}'}} dataset...")
    
    if '{dataset}' == 'MNIST':
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
    elif '{dataset}' == 'CIFAR10':
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
        
    elif '{dataset}' == 'CIFAR100':
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100('./data', train=False, transform=transform)
        
    else:
        raise ValueError(f"Unsupported dataset: {{'{dataset}'}}")
    
    print(f"Dataset loaded: {{len(train_dataset)}} training samples, {{len(test_dataset)}} test samples")
    return train_dataset, test_dataset

# Build model
{model_code}

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {{batch_idx}}/{{len(loader)}}, Loss: {{loss.item():.4f}}, Acc: {{100.*correct/total:.2f}}%')
    
    return total_loss / len(loader), 100. * correct / total

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

# Main training function
def main():
    print("="*60)
    print("Neural Aquarium - Model Training (PyTorch)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {{device}}")
    
    # Load data
    train_dataset, test_dataset = load_dataset()
    
    # Split training data for validation
    val_size = int({validation_split} * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size={batch_size}, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)
    
    # Build model
    print("Building model...")
    model = build_model().to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    print(f"\\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Validation split: {validation_split}")
    print("="*60)
    
    for epoch in range({epochs}):
        print(f"\\nEpoch {{epoch+1}}/{epochs}")
        print("-"*60)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {{train_loss:.4f}}, Train Acc: {{train_acc:.2f}}%")
        print(f"Val Loss: {{val_loss:.4f}}, Val Acc: {{val_acc:.2f}}%")
    
    # Evaluate on test set
    print("\\n" + "="*60)
    print("Evaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {{test_loss:.4f}}")
    print(f"Test Accuracy: {{test_acc:.2f}}%")
    
    {"# Save model weights" if save_weights else ""}
    {"if True:" if save_weights else "# Weights saving disabled"}
    {"    print('\\nSaving model weights...')" if save_weights else ""}
    {"    torch.save(model.state_dict(), 'model_weights.pth')" if save_weights else ""}
    {"    print('Weights saved to model_weights.pth')" if save_weights else ""}
    
    print("="*60)
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
'''
        return script
    
    @staticmethod
    def wrap_model_code_tensorflow(model_code: str) -> str:
        return f'''
def build_model():
    """Build and compile the model"""
{chr(10).join("    " + line for line in model_code.split(chr(10)))}
    return model
'''
    
    @staticmethod
    def wrap_model_code_pytorch(model_code: str) -> str:
        return f'''
def build_model():
    """Build the model"""
{chr(10).join("    " + line for line in model_code.split(chr(10)))}
    return model
'''

---
sidebar_position: 3
---

# Your First Model

A step-by-step guide to building your first neural network with Neural DSL.

## What We'll Build

We'll create an image classifier for the MNIST dataset - handwritten digit recognition.

## Step 1: Define the Model

Create `mnist.neural`:

```yaml
network MNISTClassifier {
  input: (28, 28, 1)
  
  layers:
    Conv2D(filters=32, kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))
    Flatten()
    Dense(units=128, activation="relu")
    Dropout(rate=0.5)
    Output(units=10, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  metrics: ["accuracy"]
  
  train {
    epochs: 15
    batch_size: 64
    validation_split: 0.2
  }
}
```

## Step 2: Validate

```bash
neural validate mnist.neural
```

## Step 3: Visualize

```bash
neural visualize mnist.neural --format png
```

## Step 4: Train

```bash
neural run mnist.neural --backend tensorflow
```

## Step 5: Export

```bash
neural export mnist.neural --format onnx --optimize
```

## Next Steps

- [Tutorial: Working with Layers](/docs/tutorial/layers)
- [Guide: MNIST Classifier](/docs/guides/mnist)

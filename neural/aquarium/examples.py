"""
Example Neural DSL models for use with Aquarium IDE
"""

EXAMPLE_MODELS = {
    "MNIST Classifier": """network MNISTClassifier {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        Conv2D(filters=64, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        Flatten()
        Dense(units=128, activation=relu)
        Dropout(rate=0.5)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}""",

    "CIFAR10 CNN": """network CIFAR10CNN {
    input: (None, 32, 32, 3)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu, padding=same)
        BatchNormalization()
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu, padding=same)
        MaxPooling2D(pool_size=(2, 2))
        Dropout(rate=0.25)
        
        Conv2D(filters=64, kernel_size=(3, 3), activation=relu, padding=same)
        BatchNormalization()
        Conv2D(filters=64, kernel_size=(3, 3), activation=relu, padding=same)
        MaxPooling2D(pool_size=(2, 2))
        Dropout(rate=0.25)
        
        Flatten()
        Dense(units=512, activation=relu)
        BatchNormalization()
        Dropout(rate=0.5)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
}""",

    "Simple Dense Network": """network SimpleDense {
    input: (None, 784)
    layers:
        Dense(units=256, activation=relu)
        Dropout(rate=0.3)
        Dense(units=128, activation=relu)
        Dropout(rate=0.3)
        Dense(units=64, activation=relu)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: SGD(learning_rate=0.01, momentum=0.9)
}""",

    "LSTM Text Classifier": """network TextLSTM {
    input: (None, 100)
    layers:
        Embedding(input_dim=10000, output_dim=128)
        LSTM(units=64, return_sequences=True)
        LSTM(units=64)
        Dense(units=64, activation=relu)
        Dropout(rate=0.5)
        Output(units=1, activation=sigmoid)
    loss: binary_crossentropy
    optimizer: Adam(learning_rate=0.001)
}""",

    "VGG-style Network": """network VGGStyle {
    input: (None, 32, 32, 3)
    layers:
        Conv2D(filters=64, kernel_size=(3, 3), activation=relu, padding=same)
        Conv2D(filters=64, kernel_size=(3, 3), activation=relu, padding=same)
        MaxPooling2D(pool_size=(2, 2))
        
        Conv2D(filters=128, kernel_size=(3, 3), activation=relu, padding=same)
        Conv2D(filters=128, kernel_size=(3, 3), activation=relu, padding=same)
        MaxPooling2D(pool_size=(2, 2))
        
        Conv2D(filters=256, kernel_size=(3, 3), activation=relu, padding=same)
        Conv2D(filters=256, kernel_size=(3, 3), activation=relu, padding=same)
        MaxPooling2D(pool_size=(2, 2))
        
        Flatten()
        Dense(units=1024, activation=relu)
        Dropout(rate=0.5)
        Dense(units=512, activation=relu)
        Dropout(rate=0.5)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.0001)
}""",

    "ResNet-style Block": """network ResNetStyle {
    input: (None, 32, 32, 3)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu, padding=same)
        BatchNormalization()
        
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu, padding=same)
        BatchNormalization()
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu, padding=same)
        BatchNormalization()
        
        MaxPooling2D(pool_size=(2, 2))
        
        Conv2D(filters=64, kernel_size=(3, 3), activation=relu, padding=same)
        BatchNormalization()
        Conv2D(filters=64, kernel_size=(3, 3), activation=relu, padding=same)
        BatchNormalization()
        
        GlobalAveragePooling2D()
        Dense(units=128, activation=relu)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}""",

    "Autoencoder": """network SimpleAutoencoder {
    input: (None, 784)
    layers:
        Dense(units=128, activation=relu)
        Dense(units=64, activation=relu)
        Dense(units=32, activation=relu)
        Dense(units=64, activation=relu)
        Dense(units=128, activation=relu)
        Output(units=784, activation=sigmoid)
    loss: mse
    optimizer: Adam(learning_rate=0.001)
}""",

    "Transformer Encoder": """network TransformerEncoder {
    input: (None, 512)
    layers:
        Embedding(input_dim=10000, output_dim=256)
        MultiHeadAttention(num_heads=8, key_dim=32)
        LayerNormalization()
        Dense(units=512, activation=relu)
        Dense(units=256)
        LayerNormalization()
        GlobalAveragePooling1D()
        Dense(units=64, activation=relu)
        Output(units=2, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}""",

    "Transformer Encoder-Decoder": """network TransformerSeq2Seq {
    input: (None, 100, 512)
    layers:
        TransformerEncoder(num_heads=8, d_model=512, ff_dim=2048, dropout=0.1)
        TransformerDecoder(num_heads=8, d_model=512, ff_dim=2048, dropout=0.1, use_causal_mask=true)
        Dense(units=10000, activation=softmax)
    loss: sparse_categorical_crossentropy
    optimizer: Adam(learning_rate=0.0001)
}""",

    "Machine Translation Transformer": """network NMT {
    input: (None, 50, 256)
    layers:
        TransformerEncoder(num_heads=4, d_model=256, ff_dim=1024, dropout=0.2)
        TransformerEncoder(num_heads=4, d_model=256, ff_dim=1024, dropout=0.2)
        TransformerDecoder(num_heads=4, d_model=256, ff_dim=1024, dropout=0.2)
        TransformerDecoder(num_heads=4, d_model=256, ff_dim=1024, dropout=0.2)
        Dense(units=8000, activation=softmax)
    loss: sparse_categorical_crossentropy
    optimizer: Adam(learning_rate=0.0001)
}"""
}


def get_example(name: str) -> str:
    """Get an example model by name"""
    return EXAMPLE_MODELS.get(name, "")


def list_examples() -> list:
    """List all available example names"""
    return list(EXAMPLE_MODELS.keys())


def get_random_example() -> str:
    """Get a random example model"""
    import random
    return random.choice(list(EXAMPLE_MODELS.values()))

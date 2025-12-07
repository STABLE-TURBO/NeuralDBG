
from neural.parser.parser import NeuralParser
import sys

def debug_parser():
    parser = NeuralParser()
    text = """
    network Test {
        input: (10,)
        layers: Dense(10)
        optimizer: SGD(learning_rate=ExponentialDecay(
            HPO(range(0.05, 0.2, step=0.05)),
            1000,
            HPO(range(0.9, 0.99, step=0.01))
        ))
    }
    """
    try:
        print("Parsing network...")
        result = parser.parse(text)
        print("Parse successful")
        print(result)
    except Exception as e:
        print(f"Parse failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_parser()

from neural.parser.parser import create_parser, ModelTransformer

def debug_layer(layer_string):
    print(f"\nScanning: {layer_string}")
    parser = create_parser('layer')
    transformer = ModelTransformer()
    try:
        tree = parser.parse(layer_string)
        result = transformer.transform(tree)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_layer("ResidualConnection()")
    debug_layer("GRU(units=32)")
    debug_layer("Inception()")

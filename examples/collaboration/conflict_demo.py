"""
Conflict Resolution Example

Demonstrates operational transformation for concurrent edits.
"""

from neural.collaboration import ConflictResolver
from neural.collaboration.conflict_resolution import EditOperation, OperationType

def main():
    resolver = ConflictResolver()
    
    # Scenario: Two users editing the same document
    original_text = "The quick brown fox jumps over the lazy dog."
    print(f"Original text: {original_text}\n")
    
    # User 1: Inserts "red " before "fox"
    op1 = EditOperation(
        type=OperationType.INSERT,
        position=16,  # Position before "fox"
        length=4,
        content="red ",
        user_id="user1"
    )
    print(f"User 1 operation: Insert 'red ' at position 16")
    
    # User 2: Deletes "lazy "
    op2 = EditOperation(
        type=OperationType.DELETE,
        position=40,  # Position of "lazy "
        length=5,
        user_id="user2"
    )
    print(f"User 2 operation: Delete 'lazy ' at position 40\n")
    
    # Transform operations
    print("Transforming operations...")
    op1_prime, op2_prime = resolver.transform_operations(op1, op2)
    
    # Apply operations
    text = original_text
    text = resolver.apply_operation(text, op1_prime)
    print(f"After User 1's edit: {text}")
    
    text = resolver.apply_operation(text, op2_prime)
    print(f"After User 2's edit: {text}\n")
    
    # Demonstrate three-way merge
    print("=" * 60)
    print("Three-Way Merge Example")
    print("=" * 60)
    
    base_text = """network MNIST {
    input: shape=(1, 28, 28, 1)
    
    Conv2D(32, 3, activation='relu')
    MaxPool2D(2)
}"""
    
    local_text = """network MNIST {
    input: shape=(1, 28, 28, 1)
    
    Conv2D(64, 3, activation='relu')  // Changed to 64 filters
    MaxPool2D(2)
}"""
    
    remote_text = """network MNIST {
    input: shape=(1, 28, 28, 1)
    
    Conv2D(32, 3, activation='relu')
    MaxPool2D(2)
    Dropout(0.25)  // Added dropout
}"""
    
    print("\nBase version:")
    print(base_text)
    
    print("\nLocal changes (by User 1):")
    print(local_text)
    
    print("\nRemote changes (by User 2):")
    print(remote_text)
    
    merged_text, conflicts = resolver.three_way_merge(base_text, local_text, remote_text)
    
    print("\nMerged result:")
    print(merged_text)
    
    if conflicts:
        print(f"\n⚠ Found {len(conflicts)} conflict(s)")
        for i, conflict in enumerate(conflicts, 1):
            print(f"\nConflict {i} at line {conflict['line']}:")
            print(f"  Local:  {conflict['local'].strip()}")
            print(f"  Remote: {conflict['remote'].strip()}")
    else:
        print("\n✓ No conflicts detected!")

if __name__ == '__main__':
    main()

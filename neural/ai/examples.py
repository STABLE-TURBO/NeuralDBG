"""
Examples demonstrating Enhanced AI Assistant features.

This module provides usage examples for all new AI assistant capabilities.
"""

from __future__ import annotations

from typing import Any, Dict


def example_optimization_suggestions() -> None:
    """Example: Get optimization suggestions from training metrics."""
    from neural.ai import EnhancedAIAssistant
    
    # Initialize assistant
    assistant = EnhancedAIAssistant()
    
    # Simulate training metrics showing overfitting
    metrics = {
        'train_loss': [2.3, 1.8, 1.2, 0.8, 0.5, 0.3, 0.2],
        'val_loss': [2.2, 1.7, 1.3, 1.2, 1.3, 1.4, 1.5],
        'train_acc': [0.3, 0.5, 0.7, 0.85, 0.92, 0.95, 0.97],
        'val_acc': [0.35, 0.52, 0.68, 0.72, 0.70, 0.68, 0.65]
    }
    
    # Get optimization suggestions
    analysis = assistant.analyze_training_metrics(metrics)
    
    print("Optimization Analysis:")
    print("=" * 50)
    print(analysis['overall_assessment'])
    print("\nSuggestions:")
    for i, sugg in enumerate(analysis['optimization_suggestions'], 1):
        print(f"\n{i}. {sugg['title']} (Priority: {sugg['priority']})")
        print(f"   {sugg['description']}")
        print(f"   Example: {sugg['code_example']}")


def example_transfer_learning_recommendations() -> None:
    """Example: Get transfer learning recommendations."""
    from neural.ai import EnhancedAIAssistant
    
    assistant = EnhancedAIAssistant()
    
    # Get recommendations for small image classification dataset
    recommendations = assistant.recommend_transfer_learning(
        task_type='image_classification',
        dataset_size=5000,
        constraints={'max_params': 50e6}  # Max 50M parameters
    )
    
    print("\nTransfer Learning Recommendations:")
    print("=" * 50)
    for i, rec in enumerate(recommendations[:2], 1):
        print(f"\n{i}. {rec['model_name']}")
        print(f"   Strategy: {rec['strategy']}")
        print(f"   Expected accuracy: {rec['expected_accuracy']:.1%}")
        print(f"   Training time: {rec['training_time']}")
        print(f"\n   Code example:")
        print(rec['code_example'][:200] + "...")


def example_data_augmentation_pipeline() -> None:
    """Example: Generate data augmentation pipeline."""
    from neural.ai import EnhancedAIAssistant
    
    assistant = EnhancedAIAssistant()
    
    # Generate pipeline for small image dataset
    pipeline = assistant.generate_augmentation_pipeline(
        data_type='image',
        dataset_size=3000,
        level='moderate'
    )
    
    print("\nData Augmentation Pipeline:")
    print("=" * 50)
    print(f"Level: {pipeline['level']}")
    print(f"Expected improvement: {pipeline['expected_improvement']}")
    print("\nAugmentations:")
    for aug in pipeline['augmentations']:
        print(f"  - {aug['name']}: {aug['parameters']}")
    print("\nGenerated code:")
    print(pipeline['code'])


def example_debugging_assistance() -> None:
    """Example: Get debugging assistance."""
    from neural.ai import EnhancedAIAssistant
    
    assistant = EnhancedAIAssistant()
    
    # Diagnose NaN loss issue
    diagnosis = assistant.diagnose_issue(
        error_message="Loss is NaN after 5 epochs",
        metrics={
            'train_loss': [2.3, 1.8, 1.2, 0.8, float('nan')]
        }
    )
    
    print("\nDebugging Diagnosis:")
    print("=" * 50)
    if diagnosis['primary_issue']:
        issue = diagnosis['primary_issue']
        print(f"Issue: {issue['type'].value}")
        print(f"Confidence: {issue['confidence']:.0%}")
        print("\nRecommended actions:")
        for i, action in enumerate(diagnosis['recommended_actions'][:3], 1):
            print(f"{i}. {action['action']}")
    
    # Get debugging code
    debug_code = assistant.get_debugging_code('loss_nan')
    print("\nDebugging code:")
    print(debug_code['code'][:300] + "...")


def example_conversational_assistance() -> None:
    """Example: Conversational assistance with context."""
    from neural.ai import EnhancedAIAssistant
    
    assistant = EnhancedAIAssistant(persistence_dir='./.neural_sessions')
    
    # Start a session
    session_id = assistant.start_session()
    print(f"Started session: {session_id}")
    
    # Ask questions and get contextual responses
    queries = [
        "Why is my model overfitting?",
        "What augmentations should I use?",
        "My loss is stuck, help!"
    ]
    
    print("\nConversational Assistance:")
    print("=" * 50)
    
    for query in queries:
        print(f"\nUser: {query}")
        response = assistant.chat(query)
        print(f"Assistant: {response['response'][:200]}...")
        print(f"Category: {response.get('category', 'unknown')}")
    
    # Get session summary
    summary = assistant.get_session_summary()
    print(f"\n{summary}")
    
    # Save session
    assistant.save_session()
    print("\nSession saved!")


def example_comprehensive_advice() -> None:
    """Example: Get comprehensive advice for a task."""
    from neural.ai import EnhancedAIAssistant
    
    assistant = EnhancedAIAssistant()
    
    # Define task and dataset
    task = "Image classification for medical X-rays"
    dataset_info = {
        'type': 'image',
        'size': 2000,
        'task_type': 'image_classification',
        'description': 'Chest X-ray classification (normal vs. pneumonia)'
    }
    
    # Get comprehensive advice
    advice = assistant.get_comprehensive_advice(
        task_description=task,
        dataset_info=dataset_info
    )
    
    print("\nComprehensive Advice:")
    print("=" * 50)
    print(f"Task: {advice['task']}")
    
    print("\nTransfer Learning:")
    if advice['transfer_learning']:
        for rec in advice['transfer_learning'][:1]:
            print(f"  Recommended: {rec['model_name']}")
            print(f"  Strategy: {rec['strategy']}")
    
    print("\nData Augmentation:")
    if advice['data_augmentation']:
        pipeline = advice['data_augmentation']
        print(f"  Level: {pipeline['level']}")
        print(f"  Techniques: {len(pipeline['augmentations'])}")
    
    print("\nArchitecture Tips:")
    for tip in advice['architecture_tips'][:3]:
        print(f"  • {tip}")
    
    print("\nTraining Tips:")
    for tip in advice['training_tips'][:3]:
        print(f"  • {tip}")


def example_metric_based_optimization() -> None:
    """Example: Optimization based on real-time metrics."""
    from neural.ai import ModelOptimizer
    
    optimizer = ModelOptimizer()
    
    # Simulate training progress
    print("\nMetric-based Optimization:")
    print("=" * 50)
    
    epochs = [
        {'epoch': 5, 'train_loss': [2.3, 1.8, 1.2, 0.8, 0.5], 
         'val_loss': [2.2, 1.7, 1.3, 1.2, 1.3]},
        {'epoch': 10, 'train_loss': [2.3, 1.8, 1.2, 0.8, 0.5, 0.3, 0.2, 0.15, 0.12, 0.10],
         'val_loss': [2.2, 1.7, 1.3, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]},
    ]
    
    for metrics in epochs:
        print(f"\nAfter epoch {metrics['epoch']}:")
        suggestions = optimizer.analyze_metrics(metrics)
        
        if suggestions:
            top_suggestion = suggestions[0]
            print(f"  Issue detected: {top_suggestion['title']}")
            print(f"  Action: {top_suggestion['description']}")
        else:
            print("  Training looks good!")


def example_architecture_refinement() -> None:
    """Example: Automatic architecture refinement."""
    from neural.ai import ModelOptimizer
    
    optimizer = ModelOptimizer()
    
    # Current model configuration
    model_config = {
        'name': 'MyModel',
        'layers': [
            {'type': 'Conv2D', 'params': {'filters': 32}},
            {'type': 'Conv2D', 'params': {'filters': 64}},
            {'type': 'Flatten', 'params': {}},
            {'type': 'Dense', 'params': {'units': 128}},
            {'type': 'Output', 'params': {'units': 10}}
        ]
    }
    
    # Get architecture refinement suggestions
    refinement = optimizer.suggest_architecture_refinement(model_config)
    
    print("\nArchitecture Refinement:")
    print("=" * 50)
    print(f"Reason: {refinement['reason']}")
    print("\nChanges made:")
    for change in refinement['changes']:
        print(f"  • {change}")
    
    print("\nRefined architecture has", 
          len(refinement['refined_config']['layers']), "layers")


def run_all_examples() -> None:
    """Run all examples."""
    print("\n" + "=" * 70)
    print("ENHANCED AI ASSISTANT EXAMPLES")
    print("=" * 70)
    
    try:
        example_optimization_suggestions()
    except Exception as e:
        print(f"Error in optimization example: {e}")
    
    try:
        example_transfer_learning_recommendations()
    except Exception as e:
        print(f"Error in transfer learning example: {e}")
    
    try:
        example_data_augmentation_pipeline()
    except Exception as e:
        print(f"Error in augmentation example: {e}")
    
    try:
        example_debugging_assistance()
    except Exception as e:
        print(f"Error in debugging example: {e}")
    
    try:
        example_conversational_assistance()
    except Exception as e:
        print(f"Error in conversational example: {e}")
    
    try:
        example_comprehensive_advice()
    except Exception as e:
        print(f"Error in comprehensive advice example: {e}")
    
    try:
        example_metric_based_optimization()
    except Exception as e:
        print(f"Error in metric-based optimization example: {e}")
    
    try:
        example_architecture_refinement()
    except Exception as e:
        print(f"Error in architecture refinement example: {e}")
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == '__main__':
    run_all_examples()

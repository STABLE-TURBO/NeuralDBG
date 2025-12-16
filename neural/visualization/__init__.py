from neural.visualization.static_visualizer.visualizer import NeuralVisualizer


try:
    from neural.visualization.component_interface import (
        ComponentInterface,
        GalleryMetadata,
        VisualizationMetadata,
        create_component_interface,
        get_visualization_preview_html,
    )
    from neural.visualization.gallery import (
        ArchitectureVisualizer,
        ExportHandler,
        FlopsMemoryVisualizer,
        ShapePropagationVisualizer,
        TimelineVisualizer,
        VisualizationGallery,
    )
    
    GALLERY_AVAILABLE = True
except ImportError as e:
    GALLERY_AVAILABLE = False
    print(f"Gallery components not available: {e}")

__all__ = ['NeuralVisualizer']

if GALLERY_AVAILABLE:
    __all__.extend([
        'VisualizationGallery',
        'ArchitectureVisualizer',
        'ShapePropagationVisualizer',
        'FlopsMemoryVisualizer',
        'TimelineVisualizer',
        'ExportHandler',
        'ComponentInterface',
        'VisualizationMetadata',
        'GalleryMetadata',
        'create_component_interface',
        'get_visualization_preview_html',
    ])

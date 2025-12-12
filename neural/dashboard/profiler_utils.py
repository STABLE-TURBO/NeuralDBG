import functools
import time
from contextlib import contextmanager


class LayerProfiler:
    """
    Utility class for profiling layer execution in neural networks.
    
    Usage:
        from neural.dashboard.profiler_utils import layer_profiler
        
        @layer_profiler.profile("Conv2D_1")
        def my_layer_function():
            # layer computation
            pass
            
        # Or use as context manager:
        with layer_profiler.profile_context("Conv2D_1"):
            # layer computation
            pass
    """
    
    def __init__(self, profiler_instance=None):
        self.profiler = profiler_instance
        self._enabled = True
    
    def set_profiler(self, profiler_instance):
        """Set the profiler instance to use."""
        self.profiler = profiler_instance
    
    def enable(self):
        """Enable profiling."""
        self._enabled = True
    
    def disable(self):
        """Disable profiling."""
        self._enabled = False
    
    def profile(self, layer_name):
        """
        Decorator to profile a function as a layer.
        
        Args:
            layer_name: Name of the layer being profiled
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self._enabled or self.profiler is None:
                    return func(*args, **kwargs)
                
                self.profiler.start_layer(layer_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.profiler.end_layer(layer_name)
            
            return wrapper
        return decorator
    
    @contextmanager
    def profile_context(self, layer_name):
        """
        Context manager for profiling a code block.
        
        Args:
            layer_name: Name of the layer being profiled
            
        Usage:
            with layer_profiler.profile_context("Conv2D_1"):
                # layer computation
                pass
        """
        if self._enabled and self.profiler is not None:
            self.profiler.start_layer(layer_name)
        try:
            yield
        finally:
            if self._enabled and self.profiler is not None:
                self.profiler.end_layer(layer_name)


layer_profiler = LayerProfiler()


class BreakpointHelper:
    """
    Helper class for working with breakpoints in layer execution.
    
    Usage:
        from neural.dashboard.profiler_utils import breakpoint_helper
        
        breakpoint_helper.check("Conv2D_1", layer_data)
    """
    
    def __init__(self, breakpoint_manager=None):
        self.manager = breakpoint_manager
        self._enabled = True
    
    def set_manager(self, breakpoint_manager):
        """Set the breakpoint manager instance."""
        self.manager = breakpoint_manager
    
    def enable(self):
        """Enable breakpoint checking."""
        self._enabled = True
    
    def disable(self):
        """Disable breakpoint checking."""
        self._enabled = False
    
    def check(self, layer_name, layer_data):
        """
        Check if execution should break at this layer.
        
        Args:
            layer_name: Name of the layer
            layer_data: Dictionary of layer metrics
            
        Returns:
            True if should break, False otherwise
        """
        if not self._enabled or self.manager is None:
            return False
        
        should_break = self.manager.should_break(layer_name, layer_data)
        
        if should_break:
            self.manager.paused = True
            self.manager.current_layer = layer_name
            print(f"\n{'='*60}")
            print(f"BREAKPOINT HIT: {layer_name}")
            print(f"{'='*60}")
            print(f"Layer Data:")
            for key, value in layer_data.items():
                print(f"  {key}: {value}")
            print(f"{'='*60}\n")
            
            while self.manager.paused:
                time.sleep(0.1)
        
        return should_break
    
    @contextmanager
    def check_context(self, layer_name, layer_data):
        """
        Context manager for breakpoint checking.
        
        Args:
            layer_name: Name of the layer
            layer_data: Dictionary of layer metrics
            
        Usage:
            with breakpoint_helper.check_context("Conv2D_1", data):
                # layer computation
                pass
        """
        self.check(layer_name, layer_data)
        yield


breakpoint_helper = BreakpointHelper()


class AnomalyMonitor:
    """
    Helper class for monitoring anomalies during layer execution.
    
    Usage:
        from neural.dashboard.profiler_utils import anomaly_monitor
        
        anomaly_monitor.check("Conv2D_1", layer_metrics)
    """
    
    def __init__(self, anomaly_detector=None):
        self.detector = anomaly_detector
        self._enabled = True
        self._alert_callback = None
    
    def set_detector(self, anomaly_detector):
        """Set the anomaly detector instance."""
        self.detector = anomaly_detector
    
    def enable(self):
        """Enable anomaly monitoring."""
        self._enabled = True
    
    def disable(self):
        """Disable anomaly monitoring."""
        self._enabled = False
    
    def set_alert_callback(self, callback):
        """
        Set a callback function to be called when anomalies are detected.
        
        Args:
            callback: Function with signature callback(layer_name, metrics, result)
        """
        self._alert_callback = callback
    
    def check(self, layer_name, metrics):
        """
        Check for anomalies in layer execution.
        
        Args:
            layer_name: Name of the layer
            metrics: Dictionary of layer metrics
            
        Returns:
            Anomaly detection result dictionary
        """
        if not self._enabled or self.detector is None:
            return {"is_anomaly": False, "scores": {}}
        
        self.detector.add_sample(layer_name, metrics)
        
        result = self.detector.detect_anomalies(layer_name, metrics)
        
        if result["is_anomaly"] and self._alert_callback:
            self._alert_callback(layer_name, metrics, result)
        
        return result
    
    @contextmanager
    def monitor_context(self, layer_name, metrics):
        """
        Context manager for anomaly monitoring.
        
        Args:
            layer_name: Name of the layer
            metrics: Dictionary of layer metrics
            
        Usage:
            with anomaly_monitor.monitor_context("Conv2D_1", metrics):
                # layer computation
                pass
        """
        yield
        self.check(layer_name, metrics)


anomaly_monitor = AnomalyMonitor()


def initialize_helpers(profiler_instance=None, breakpoint_manager=None, anomaly_detector=None):
    """
    Initialize all helper utilities with dashboard instances.
    
    Args:
        profiler_instance: PerformanceProfiler instance
        breakpoint_manager: BreakpointManager instance
        anomaly_detector: AnomalyDetector instance
    """
    if profiler_instance:
        layer_profiler.set_profiler(profiler_instance)
    
    if breakpoint_manager:
        breakpoint_helper.set_manager(breakpoint_manager)
    
    if anomaly_detector:
        anomaly_monitor.set_detector(anomaly_detector)


def auto_initialize():
    """
    Automatically initialize helpers with dashboard instances.
    Attempts to import from dashboard module.
    """
    try:
        from neural.dashboard.dashboard import profiler, breakpoint_manager, anomaly_detector
        initialize_helpers(profiler, breakpoint_manager, anomaly_detector)
        return True
    except ImportError:
        return False


class LayerExecutionWrapper:
    """
    Comprehensive wrapper that combines profiling, breakpoints, and anomaly detection.
    
    Usage:
        from neural.dashboard.profiler_utils import LayerExecutionWrapper
        
        wrapper = LayerExecutionWrapper()
        
        @wrapper.wrap("Conv2D_1")
        def my_layer(x):
            return conv2d(x)
            
        # Or with metrics:
        with wrapper.wrap_context("Conv2D_1") as ctx:
            output = conv2d(input)
            ctx.set_metrics({
                "execution_time": 0.045,
                "memory": 256.5,
                "mean_activation": 0.35
            })
    """
    
    def __init__(self, profiler=None, breakpoint_mgr=None, anomaly_det=None):
        self.profiler = profiler or layer_profiler
        self.breakpoint_mgr = breakpoint_mgr or breakpoint_helper
        self.anomaly_det = anomaly_det or anomaly_monitor
    
    def wrap(self, layer_name, collect_metrics=None):
        """
        Decorator that wraps a layer function with profiling, breakpoints, and anomaly detection.
        
        Args:
            layer_name: Name of the layer
            collect_metrics: Optional function to collect metrics from the layer output
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                if hasattr(self.profiler, 'profiler') and self.profiler.profiler:
                    self.profiler.profiler.start_layer(layer_name)
                
                result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                
                metrics = {
                    "execution_time": execution_time,
                    "layer": layer_name
                }
                
                if collect_metrics:
                    custom_metrics = collect_metrics(result)
                    metrics.update(custom_metrics)
                
                if hasattr(self.breakpoint_mgr, 'manager') and self.breakpoint_mgr.manager:
                    self.breakpoint_mgr.check(layer_name, metrics)
                
                if hasattr(self.anomaly_det, 'detector') and self.anomaly_det.detector:
                    self.anomaly_det.check(layer_name, metrics)
                
                if hasattr(self.profiler, 'profiler') and self.profiler.profiler:
                    self.profiler.profiler.end_layer(layer_name)
                
                return result
            
            return wrapper
        return decorator
    
    @contextmanager
    def wrap_context(self, layer_name):
        """
        Context manager for comprehensive layer execution monitoring.
        
        Args:
            layer_name: Name of the layer
            
        Yields:
            Context object with set_metrics method
            
        Usage:
            with wrapper.wrap_context("Conv2D_1") as ctx:
                output = my_layer(input)
                ctx.set_metrics({"memory": 256.5})
        """
        class ExecutionContext:
            def __init__(self, wrapper, layer_name):
                self.wrapper = wrapper
                self.layer_name = layer_name
                self.metrics = {"layer": layer_name}
                self.start_time = time.time()
            
            def set_metrics(self, metrics):
                self.metrics.update(metrics)
            
            def finalize(self):
                self.metrics["execution_time"] = time.time() - self.start_time
                
                if hasattr(self.wrapper.breakpoint_mgr, 'manager') and self.wrapper.breakpoint_mgr.manager:
                    self.wrapper.breakpoint_mgr.check(self.layer_name, self.metrics)
                
                if hasattr(self.wrapper.anomaly_det, 'detector') and self.wrapper.anomaly_det.detector:
                    self.wrapper.anomaly_det.check(self.layer_name, self.metrics)
        
        ctx = ExecutionContext(self, layer_name)
        
        if hasattr(self.profiler, 'profiler') and self.profiler.profiler:
            self.profiler.profiler.start_layer(layer_name)
        
        try:
            yield ctx
        finally:
            ctx.finalize()
            if hasattr(self.profiler, 'profiler') and self.profiler.profiler:
                self.profiler.profiler.end_layer(layer_name)


auto_initialize()

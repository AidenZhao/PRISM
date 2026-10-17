"""PRISM: geodesic-guided representation learning for 3D geometric data."""
__version__ = "2.0.0"
__all__ = ["GeodesicTransformer", "PRESETS"]


def __getattr__(name):  # lazy: prism.preprocess must not require torch
    if name in __all__:
        from . import model

        return getattr(model, name)
    raise AttributeError(name)

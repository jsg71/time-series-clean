"""Convenience wrapper for :mod:`leela_ml.models.ncd` without requiring torch."""
import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location(
    'leela_ml.models.ncd',
    pathlib.Path(__file__).with_name('models').joinpath('ncd.py')
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ncd_adjacent = _mod.ncd_adjacent
ncd_first = _mod.ncd_first
__all__ = ['ncd_adjacent', 'ncd_first']

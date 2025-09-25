# leela_ml/demo_docs.py
from __future__ import annotations

"""
leela_ml.demo_docs
==================

Minimal, well-documented example that shows exactly how we want modules to look
for clean MkDocs + mkdocstrings rendering:

- Module docstring (Markdown allowed)
- `__all__` to curate the public API
- Type hints everywhere (shown in signatures)
- Google-style docstrings (Args / Returns / Raises / Examples / Notes)
- Small type aliases to keep signatures readable
- Dataclass config, a custom exception, doctest examples

!!! tip
    Docstrings are parsed as Markdown. Lists, code blocks, and admonitions like
    this **tip** render nicely thanks to the Material theme + pymdown-extensions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypedDict, Protocol, Literal
import json
import numpy as np

# ── Type aliases (keep signatures clean) ───────────────────────────────────────

PathLike = str | Path
ArrayLike = Sequence[float] | np.ndarray
DType = Literal["float32", "float64"]

# ── Public API surface (mkdocstrings hides names not in __all__ or starting "_" )

__all__ = [
    "DatasetMeta",
    "DetectorConfig",
    "ModelNotFittedError",
    "load_signal",
    "rolling_mad",
    "SimpleMADDetector",
]

# ── Small typed records / protocols ───────────────────────────────────────────


class DatasetMeta(TypedDict, total=False):
    """Minimal metadata for a 1D signal dataset."""

    name: str
    fs: float  # sampling frequency in Hz
    n_samples: int


class Fittable(Protocol):
    """Anything that supports `fit(X) -> self`."""

    def fit(self, X: np.ndarray) -> "Fittable":
        ...


# ── Exceptions ────────────────────────────────────────────────────────────────


class ModelNotFittedError(RuntimeError):
    """Raised when calling a method that requires a fitted model."""


# ── Config object ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DetectorConfig:
    """Configuration for :class:`SimpleMADDetector`.

    Attributes:
        window: Rolling window length (samples) for MAD.
        k: Threshold multiplier. Roughly, ``median + k * MAD`` flags an anomaly.
    """

    window: int = 256
    k: float = 4.0


# ── Public functions ──────────────────────────────────────────────────────────


def load_signal(path: PathLike, *, dtype: DType = "float32") -> np.ndarray:
    """Load a 1D signal from a ``.npy`` file.

    Args:
        path: Path to a NumPy ``.npy`` array.
        dtype: Output dtype (``"float32"`` or ``"float64"``).

    Returns:
        Array of shape ``(T,)``.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the array is not 1-D.

    Examples:
        >>> import numpy as _np, tempfile as _tf
        >>> with _tf.TemporaryDirectory() as _d:
        ...     p = Path(_d) / "x.npy"
        ...     _np.save(p, _np.arange(5, dtype=_np.float32))
        ...     load_signal(p).shape
        (5,)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    x = np.load(p)
    if x.ndim != 1:
        raise ValueError("expected a 1-D array")
    return x.astype(np.float32 if dtype == "float32" else np.float64, copy=False)


def rolling_mad(x: ArrayLike, window: int) -> np.ndarray:
    """Compute a simple rolling **median absolute deviation** (MAD).

    Args:
        x: 1-D sequence or array.
        window: Rolling window length (``>= 3`` sensible).

    Returns:
        Array of MAD values, same length as ``x``.

    Notes:
        This is intentionally simple/clear for docs. For speed, vectorize or
        use numba.
    """
    xx = np.asarray(x, dtype=float)
    n = len(xx)
    out = np.empty(n, dtype=float)
    half = max(1, window // 2)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w = xx[lo:hi]
        med = float(np.median(w))
        out[i] = float(np.median(np.abs(w - med)))
    return out


# ── Tiny model: fits, scores, predicts, save/load ─────────────────────────────


class SimpleMADDetector:
    """Rolling-MAD anomaly detector.

    The detector computes ``score = |x - global_median| / MAD_window`` and
    labels points above ``k`` as anomalies.

    Args:
        cfg: Configuration object. If omitted, sensible defaults are used.

    Attributes:
        cfg: The configuration in use.
        fitted_: Whether :meth:`fit` has been called.
        global_median_: Median of the training signal (if fitted).

    Examples:
        >>> m = SimpleMADDetector(DetectorConfig(window=5, k=3.0))
        >>> m.fit([0, 0, 0, 0, 0])
        SimpleMADDetector(...)
        >>> m.predict([0, 0, 10, 0, 0]).tolist()
        [0, 0, 1, 0, 0]
    """

    def __init__(self, cfg: DetectorConfig | None = None) -> None:
        self.cfg = cfg or DetectorConfig()
        self.fitted_: bool = False
        self.global_median_: float | None = None

    def fit(self, x: ArrayLike) -> "SimpleMADDetector":
        """Estimate a global median from training data and mark as fitted."""
        xx = np.asarray(x, dtype=float)
        self.global_median_ = float(np.median(xx))
        self.fitted_ = True
        return self

    def score(self, x: ArrayLike) -> np.ndarray:
        """Compute anomaly **scores**.

        Args:
            x: 1-D signal.

        Returns:
            Scores, same length as ``x``. Higher = more anomalous.

        Raises:
            ModelNotFittedError: If called before :meth:`fit`.
        """
        if not self.fitted_:
            raise ModelNotFittedError("call fit(...) first")
        xx = np.asarray(x, dtype=float)
        mad = rolling_mad(xx, self.cfg.window)
        med = float(self.global_median_)
        return np.abs(xx - med) / (mad + 1e-9)

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Binary labels using ``k`` from :class:`DetectorConfig`."""
        return (self.score(x) > self.cfg.k).astype(np.int8)

    def save(self, path: PathLike) -> None:
        """Save the model as JSON (human-readable, small state only)."""
        payload = {
            "cfg": {"window": self.cfg.window, "k": self.cfg.k},
            "fitted": self.fitted_,
            "global_median": self.global_median_,
        }
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: PathLike) -> "SimpleMADDetector":
        """Load a model previously written by :meth:`save`."""
        payload = json.loads(Path(path).read_text())
        m = cls(DetectorConfig(**payload["cfg"]))
        m.fitted_ = bool(payload["fitted"])
        m.global_median_ = (
            float(payload["global_median"])
            if payload["global_median"] is not None
            else None
        )
        return m

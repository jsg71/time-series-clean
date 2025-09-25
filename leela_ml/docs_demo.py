# src/demo/google_style_explanatory_showcase.py
"""Google-Style Documentation — Explanatory Showcase

This module is a **teaching artifact** designed to show how rich, explanatory
Google-style docstrings render in your current MkDocs setup (Material theme,
mkdocstrings with `docstring_section_style: table`, your existing `gen_pages.py`).

The goal is twofold:

1. **Demonstrate content patterns** (what to write) for beautiful, scannable docs.
2. **Demonstrate rendering behavior** (how it looks) for module text, admonitions,
   tables, code samples, cross-references, and API items.

---

## How this renders in your site

Your `gen_pages.py` writes a page like:

- Title: ``demo.google_style_explanatory_showcase``
- Heading: ``# `demo.google_style_explanatory_showcase` ``
- Then **this module docstring** (all of this prose), rendered as Markdown
  by Material for MkDocs.
- Then mkdocstrings renders each public symbol (functions/classes/etc.) with:
  - A clean **signature line** (types pulled from annotations).
  - A table for **Args / Returns / Raises** (thanks to the table section style).
  - The docstring body (Examples, Notes, Warnings, See Also…).

!!! tip
    You don't need to switch to NumPy style. With your config, Google style
    already renders **Args/Returns/Raises** as tidy tables. Keep it consistent.

!!! warning
    Keep **Examples** fast and pure (no network/disk). That makes doctests viable.

---

## What to include at the *module* level (this block)

- A short overview (what & why).
- A quick "How this renders" section for context.
- Usage guidance & design notes (keep it concise but helpful).
- Optional: a small table of exports (below) to orient readers.
- Optional: project-wide conventions that apply to everything in the module.

### Exported API (overview)

| Symbol                | Kind        | Purpose                                       |
|---------------------- |------------ |-----------------------------------------------|
| `PI`, `DEFAULT_CHUNK` | constants   | Reusable numeric defaults for examples        |
| `StationName`         | NewType     | Distinct station identifier type              |
| `FloatVec`, `BoolVec` | type alias  | Canonical array types used across functions   |
| `Level`               | Enum        | Severity levels (LOW/MEDIUM/HIGH)             |
| `EvalSummary`         | dataclass   | Structured return for metrics                 |
| `Record`              | TypedDict   | Minimal example of typed dict config          |
| `Readable`            | Protocol    | Demonstrates Protocols in signatures          |
| `zscore`              | function    | Core numeric example with Args/Returns/Raises |
| `moving_average`      | function    | Simple numeric helper                         |
| `batched`             | generator   | Shows `Yields` semantics                      |
| `evaluate_binary`     | function    | Returns `EvalSummary`                         |
| `temporary_seed`      | context mgr | Shows function-level context manager docs     |
| `SafeFile`, `open_safely` | ctx mgr | Shows class-based context manager docs        |
| `coerce_bool`         | function    | Parameter validation + `Raises`               |
| `async_ping`          | async func  | How async functions render                    |
| `BaseDetector`        | ABC         | Abstract callable interface                   |
| `ZScoreDetector`      | class       | Properties, `@classmethod`, `@staticmethod`   |
| `deprecated_score`    | function    | How to mark deprecations                      |

---

## Writing style (house rules)

- **Public objects only.** Keep internals prefixed with `_` (they won't render).
- **Always type-annotate** parameters and returns. Doc tables stay clean.
- Use **Args / Returns / Raises / Examples / Notes / Warnings / See Also**.
- Prefer **structured returns** (dataclasses or NamedTuple) over dicts.
- Add **one doctestable example** per public function/method.

Examples:
    >>> import numpy as np
    >>> z = zscore(np.array([1., 2., 3.], dtype=np.float32))
    >>> z.round(6)
    array([-1.224745,  0.      ,  1.224745], dtype=float32)

See Also:
    - :mod:`typing` for typing primitives you'll see below.
    - :mod:`contextlib` for context manager helpers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Iterator,
    Literal,
    NewType,
    Optional,
    Protocol,
    Sequence,
    TypedDict,
    overload,
)

import numpy as np
import numpy.typing as npt


# ─────────────────────────────────────────────────────────────────────────────
# Constants & public surface
# ─────────────────────────────────────────────────────────────────────────────

PI: float = 3.141592653589793
"""π to double precision (for examples)."""

DEFAULT_CHUNK: int = 1024
"""Default chunk size used by :func:`batched`."""

__all__ = [
    # constants & aliases
    "PI",
    "DEFAULT_CHUNK",
    "StationName",
    "FloatVec",
    "BoolVec",
    # simple types / records
    "Level",
    "EvalSummary",
    "Record",
    "Readable",
    # core API
    "zscore",
    "moving_average",
    "batched",
    "evaluate_binary",
    # contexts & helpers
    "temporary_seed",
    "SafeFile",
    "open_safely",
    "coerce_bool",
    # OO API
    "BaseDetector",
    "ZScoreDetector",
    # async & deprecations
    "async_ping",
    "deprecated_score",
]


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

StationName = NewType("StationName", str)
"""Distinct type for station identifiers (prevents mixing with plain str)."""

FloatVec = npt.NDArray[np.float32]
"""1-D float32 vector of shape ``(n,)``."""

BoolVec = npt.NDArray[np.bool_]
"""1-D boolean vector of shape ``(n,)``."""


class Level(Enum):
    """Severity levels (used as friendly metadata)."""

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


class Readable(Protocol):
    """Protocol for objects with a ``.read() -> str`` method (duck typing)."""

    def read(self) -> str: ...


class Record(TypedDict, total=False):
    """Typed dict example showing how config blobs can be documented.

    Attributes:
        station: Station name.
        level: Severity level as text.
        note: Optional free text.
    """

    station: StationName
    level: str
    note: str


@dataclass(frozen=True, slots=True)
class EvalSummary:
    """Binary metrics container (structured return for clarity & stability).

    Attributes:
        tp: True positives.
        fp: False positives.
        fn: False negatives.
        precision: TP / (TP + FP), or 0.0 if undefined.
        recall: TP / (TP + FN), or 0.0 if undefined.
        f1: Harmonic mean of precision and recall.
    """

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


class DetectorError(RuntimeError):
    """Errors raised by detectors.

    You can subclass standard exceptions to create domain-specific families.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Core functions (showing Args/Returns/Raises/Examples)
# ─────────────────────────────────────────────────────────────────────────────

def zscore(x: FloatVec, eps: float = 1e-8) -> FloatVec:
    """Return standardized vector.

    Args:
        x: 1-D vector of shape ``(n,)`` with dtype float32.
        eps: Small constant to avoid division by zero when ``std == 0``.

    Returns:
        Vector with mean approximately 0 and unit variance.

    Raises:
        ValueError: If ``x`` is not 1-D.

    Notes:
        This is intentionally tiny to keep doctests fast and deterministic.

    Examples:
        >>> import numpy as np
        >>> zscore(np.array([1., 2., 3.], dtype=np.float32)).round(6)
        array([-1.224745,  0.      ,  1.224745], dtype=float32)
    """
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    mu = float(x.mean())
    sd = float(x.std())
    den = sd if sd > eps else eps
    return (x - mu) / den


def moving_average(x: FloatVec, window: int) -> FloatVec:
    """Compute a simple moving average (SMA).

    Args:
        x: 1-D float32 vector.
        window: Window length (``>= 1``).

    Returns:
        SMA vector of length ``len(x) - window + 1``.

    Raises:
        ValueError: If ``window < 1`` or ``x`` is not 1-D.

    Examples:
        >>> import numpy as np
        >>> moving_average(np.array([1,2,3,4], dtype=np.float32), 2)
        array([1.5, 2.5, 3.5], dtype=float32)
    """
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    if window < 1:
        raise ValueError("window must be >= 1")
    c = np.cumsum(x, dtype=np.float64)
    c[window:] = c[window:] - c[:-window]
    out = c[window - 1 :] / window
    return out.astype(np.float32, copy=False)


@overload
def batched(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]: ...
@overload
def batched(seq: np.ndarray, size: int) -> Iterator[np.ndarray]: ...
def batched(seq, size: int):
    """Yield consecutive chunks of length ``size`` (generator example).

    Args:
        seq: Sequence-like (e.g., list, tuple, numpy array).
        size: Chunk size (must be positive).

    Yields:
        Consecutive chunks of ``seq``. The final chunk may be shorter.

    Raises:
        ValueError: If ``size <= 0``.

    Examples:
        >>> list(batched([1,2,3,4,5], 2))
        [[1, 2], [3, 4], [5]]
    """
    if size <= 0:
        raise ValueError("size must be positive")
    n = len(seq)
    for i in range(0, n, size):
        yield seq[i : i + size]


def evaluate_binary(pred: BoolVec, truth: BoolVec) -> EvalSummary:
    """Compute binary metrics and return a structured record.

    Args:
        pred: Boolean predictions, shape ``(n,)``.
        truth: Boolean ground truth, shape ``(n,)``.

    Returns:
        EvalSummary: Dataclass with ``tp, fp, fn, precision, recall, f1``.

    Examples:
        >>> import numpy as np
        >>> s = evaluate_binary(np.array([True, False, True]), np.array([True, True, False]))
        >>> (s.tp, s.fp, s.fn, round(s.f1, 3))
        (1, 1, 1, 0.5)
    """
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return EvalSummary(tp, fp, fn, precision, recall, f1)


def coerce_bool(x: Any, strict: bool = False) -> bool:
    """Coerce a value to boolean with optional strictness (validation demo).

    Args:
        x: Input value.
        strict: If True, only accepts ``{True, False, 0, 1, 'true', 'false'}`` (case-insensitive).
            Otherwise Python's built-in truthiness rules apply.

    Returns:
        Coerced boolean.

    Raises:
        ValueError: If ``strict`` and the value cannot be coerced.

    Examples:
        >>> coerce_bool("TRUE", strict=True)
        True
        >>> coerce_bool(0, strict=True)
        False
    """
    if not strict:
        return bool(x)
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)) and x in (0, 1):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "yes", "1"}:
            return True
        if s in {"false", "f", "no", "0"}:
            return False
    raise ValueError(f"cannot strictly coerce to bool: {x!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Context managers (function + class style)
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Temporarily set the NumPy RNG seed (function context manager demo).

    Args:
        seed: Seed value.

    Yields:
        Nothing. Restores RNG state on exit.

    Examples:
        >>> import numpy as np
        >>> with temporary_seed(123):
        ...     a = np.random.rand(2)
        >>> with temporary_seed(123):
        ...     b = np.random.rand(2)
        >>> np.allclose(a, b)
        True
    """
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


class SafeFile:
    """Simple context manager wrapping a UTF-8 text file (class context demo).

    Args:
        path: File path.
        mode: Open mode (``'r'`` or ``'w'``).

    Attributes:
        path: Original path.
        mode: Open mode.
        fh: The underlying file handle.

    Examples:
        >>> from tempfile import NamedTemporaryFile
        >>> with NamedTemporaryFile(delete=True) as tf:
        ...     with SafeFile(tf.name, "w") as sf:
        ...         _ = sf.fh.write("ok")  # doctest: +ELLIPSIS
    """

    path: str
    mode: str

    def __init__(self, path: str, mode: str = "r") -> None:
        self.path = path
        self.mode = mode
        self.fh: Optional[Any] = None

    def __enter__(self) -> "SafeFile":
        self.fh = open(self.path, self.mode, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fh is not None:
            self.fh.close()
        return None


def open_safely(path: str, mode: Literal["r", "w"] = "r") -> SafeFile:
    """Factory returning :class:`SafeFile`.

    Args:
        path: File path.
        mode: Either ``"r"`` or ``"w"``.

    Returns:
        SafeFile: Context manager instance.

    Raises:
        ValueError: If ``mode`` is not ``"r"`` or ``"w"``.
    """
    if mode not in ("r", "w"):
        raise ValueError('mode must be "r" or "w"')
    return SafeFile(path, mode)


# ─────────────────────────────────────────────────────────────────────────────
# Async function (rendering demo)
# ─────────────────────────────────────────────────────────────────────────────

async def async_ping(payload: str) -> str:
    """Asynchronously return a pong message (async rendering demo).

    Args:
        payload: Message payload.

    Returns:
        Response string of the form ``"pong:<payload>"``.

    Examples:
        # doctest: +SKIP
        >>> import asyncio
        >>> asyncio.run(async_ping("hi"))
        'pong:hi'
    """
    return f"pong:{payload}"


# ─────────────────────────────────────────────────────────────────────────────
# Object-oriented API (properties, classmethod, staticmethod)
# ─────────────────────────────────────────────────────────────────────────────

class BaseDetector(ABC):
    """Abstract base class for detectors."""

    @abstractmethod
    def __call__(self, x: FloatVec) -> BoolVec:
        """Return a boolean mask indicating detections."""
        raise NotImplementedError

    @property
    def receptive_field(self) -> int:
        """Effective receptive field in samples (override if not 1)."""
        return 1


class ZScoreDetector(BaseDetector):
    """Z-score threshold detector (complete OO example).

    Args:
        thr: Absolute z-score threshold.
        level: Severity level (metadata only).

    Attributes:
        thr: Current threshold.
        level: Severity level.

    Warnings:
        Assumes the signal is approximately stationary over the window.

    Examples:
        >>> import numpy as np
        >>> d = ZScoreDetector(thr=2.0, level=Level.HIGH)
        >>> x = np.array([0., 10., 0.], dtype=np.float32)
        >>> d(x).tolist()
        [False, True, False]
    """

    thr: float
    level: Level

    def __init__(self, thr: float = 3.0, level: Level = Level.MEDIUM) -> None:
        self.thr = float(thr)
        self.level = Level(level)

    def __call__(self, x: FloatVec) -> BoolVec:
        """Return mask where ``|zscore(x)| >= thr`` (documents __call__)."""
        z = zscore(x)
        return np.abs(z) >= self.thr

    @classmethod
    def from_level(cls, level: Level) -> "ZScoreDetector":
        """Construct a detector with a level-based default threshold.

        Args:
            level: Desired severity level.

        Returns:
            ZScoreDetector: Instance with pre-set threshold.

        Examples:
            >>> ZScoreDetector.from_level(Level.LOW).thr
            1.5
        """
        mapping = {Level.LOW: 1.5, Level.MEDIUM: 3.0, Level.HIGH: 4.0}
        return cls(mapping[level], level=level)

    @staticmethod
    def is_stationary(x: FloatVec, tol: float = 0.05) -> bool:
        """Heuristic stationarity check.

        Args:
            x: Input vector.
            tol: Max relative std difference between halves.

        Returns:
            True if first/second half std devs differ by at most ``tol``.
        """
        n = len(x)
        a, b = float(np.std(x[: n // 2])), float(np.std(x[n // 2 :]))
        denom = max(a + b, 1e-8)
        return abs(a - b) / denom <= tol


# ─────────────────────────────────────────────────────────────────────────────
# Deprecations (how to communicate migration)
# ─────────────────────────────────────────────────────────────────────────────

def deprecated_score(x: FloatVec, thr: float = 3.0) -> BoolVec:
    """DEPRECATED: use :class:`ZScoreDetector` instead (deprecation pattern).

    Args:
        x: Input vector.
        thr: Threshold.

    Returns:
        Boolean mask indicating detections.

    Deprecated:
        This function will be removed in a future release. Use
        ``ZScoreDetector(thr).__call__(x)`` or simply ``ZScoreDetector(thr)(x)``.
    """
    return ZScoreDetector(thr)(x)

"""4D-Humans (Goel et al., ICCV 2023) — modern HMR successor.

The Colab/Linux pipeline calls 4D-Humans directly through the official
`hmr2.models.load_hmr2` API. Detector + dataloader wiring lives in
`demo/scripts/run_pipeline.py` (the canonical execution path); this file is
intentionally thin so importing it on Mac (where detectron2 is absent) does not
fail.

Use `run_pipeline.py` for the actual pipeline run. This module is kept as a
documentation anchor to the v3 plan §C.6 architecture.
"""
from __future__ import annotations

__all__: list[str] = []

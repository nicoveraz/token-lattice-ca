"""Pytest bootstrap: put src/ and experiments/ on sys.path so tests can import
the library (model, ca) and pipeline modules the same way the scripts do.
Tests are run from the repo root so the hardcoded data/ ckpt/ results/ paths resolve."""
import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]

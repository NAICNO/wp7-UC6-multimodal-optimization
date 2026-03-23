"""
test_verbose.py — Unit tests for mmo.verbose.Verbose

Covers:
- Construction stores level
- __call__ prints when message level <= instance level
- __call__ does not print when message level > instance level
- Level boundary conditions
- Zero level (silent by default for any level > 0)
"""

import pytest
from io import StringIO
import sys

from mmo.verbose import Verbose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_output(verbose_obj, level, message):
    """Call verbose_obj(level, message) and capture stdout."""
    captured = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = captured
        verbose_obj(level, message)
    finally:
        sys.stdout = old_stdout
    return captured.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVerboseConstruction:

    def test_level_stored(self):
        v = Verbose(3)
        assert v.level == 3

    def test_level_zero_stored(self):
        v = Verbose(0)
        assert v.level == 0


class TestVerboseCall:

    def test_prints_when_level_at_or_below_threshold(self):
        v = Verbose(2)
        output = _capture_output(v, 1, 'hello')
        assert 'hello' in output

    def test_prints_when_level_equals_threshold(self):
        v = Verbose(2)
        output = _capture_output(v, 2, 'exact')
        assert 'exact' in output

    def test_suppresses_when_level_above_threshold(self):
        v = Verbose(1)
        output = _capture_output(v, 2, 'suppressed')
        assert output == ''

    def test_zero_level_suppresses_all_nonzero(self):
        v = Verbose(0)
        output = _capture_output(v, 1, 'silent')
        assert output == ''

    def test_zero_level_prints_at_zero(self):
        v = Verbose(0)
        output = _capture_output(v, 0, 'zero')
        assert 'zero' in output

    def test_high_threshold_prints_everything(self):
        v = Verbose(100)
        output = _capture_output(v, 50, 'loud')
        assert 'loud' in output

    def test_negative_message_level_always_prints(self):
        v = Verbose(0)
        output = _capture_output(v, -1, 'forced')
        assert 'forced' in output

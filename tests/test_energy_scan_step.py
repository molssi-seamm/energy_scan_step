#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `energy_scan_step` package."""

import pytest  # noqa: F401
import energy_scan_step  # noqa: F401


def test_construction():
    """Just create an object and test its type."""
    result = energy_scan_step.EnergyScan()
    assert str(type(result)) == "<class 'energy_scan_step.energy_scan.EnergyScan'>"

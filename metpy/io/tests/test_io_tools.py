# Copyright (c) 2016 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from metpy.io.cdm import Dataset
from metpy.io.tools import UnitLinker
from metpy.testing import assert_array_equal
from metpy.units import units


@pytest.fixture()
def test_var():
    r'Fixture to create a dataset and variable for tests'
    ds = Dataset()
    ds.createDimension('x', 5)
    var = ds.createVariable('data', 'f4', ('x',), 5)
    var[:] = np.arange(5)
    return var


def test_unit_linker(test_var):
    r'Test that UnitLinker successfully adds units'
    test_var.units = 'meters'
    new_var = UnitLinker(test_var)
    assert_array_equal(new_var[:], np.arange(5) * units.m)


def test_unit_linker_get_units(test_var):
    r'Test that we can get the units from UnitLinker'
    test_var.units = 'knots'
    new_var = UnitLinker(test_var)
    assert new_var.units == units('knots')


def test_unit_linker_missing(test_var):
    r'Test that UnitLinker works with missing units'
    new_var = UnitLinker(test_var)
    assert_array_equal(new_var[:], np.arange(5))


def test_unit_linker_bad(test_var):
    r'Test that UnitLinker ignores bad unit strings'
    test_var.units = 'badunit'
    new_var = UnitLinker(test_var)
    assert_array_equal(new_var[:], np.arange(5))


def test_unit_override(test_var):
    r"Test that we can override a variable's bad unit string"
    test_var.units = 'C'
    new_var = UnitLinker(test_var)
    new_var.units = 'degC'
    assert_array_equal(new_var[:], np.arange(5) * units.degC)


def test_unit_override_obj(test_var):
    r'Test that we can override with an object'
    test_var.units = 'C'
    new_var = UnitLinker(test_var)
    new_var.units = units.degC
    assert_array_equal(new_var[:], np.arange(5) * units.degC)


def test_attribute_forwarding(test_var):
    r'Test that we are properly able to access attributes from the variable'
    test_var.att = 'abc'
    new_var = UnitLinker(test_var)
    assert new_var.att == test_var.att

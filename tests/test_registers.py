#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mmu
----------------------------------
"""

import unittest

from py65emu.cpu import FlagBit, Registers


class TestRegisters(unittest.TestCase):
    def setUp(self):
        pass

    def test_flags_enum(self):
        r = Registers()
        r.setFlag(FlagBit.N)
        self.assertEqual(r.p, 0b10100100)
        r.setFlag(FlagBit.Z)
        self.assertEqual(r.p, 0b10100110)
        r.clearFlag(FlagBit.N)
        self.assertEqual(r.p, 0b00100110)
        self.assertTrue(r.getFlag(FlagBit.Z))
        r.setFlag(FlagBit.Z, False)
        self.assertFalse(r.getFlag(FlagBit.Z))
        r.setFlag(FlagBit.Z, False)
        self.assertFalse(r.getFlag(FlagBit.Z))

    def test_flags_string(self):
        r = Registers()
        r.setFlag("N")
        self.assertEqual(r.p, 0b10100100)
        r.setFlag("Z")
        self.assertEqual(r.p, 0b10100110)
        r.clearFlag("N")
        self.assertEqual(r.p, 0b00100110)
        self.assertTrue(r.getFlag("Z"))
        r.setFlag("Z", False)
        self.assertFalse(r.getFlag("Z"))
        r.setFlag("Z", False)
        self.assertFalse(r.getFlag("Z"))

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()

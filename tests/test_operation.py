#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_suites
----------------------------------

Tests for `py65emu` module.
"""


import unittest

from py65emu.cpu import CPU
from py65emu.mmu import MMU


class TestOperation(unittest.TestCase):
    def setUp(self):
        pass

    def test_something(self):
        mmu = MMU(
            [
                (0x0000, 0x800),  # RAM
                (0x2000, 0x8),  # PPU
                (0x4000, 0x18),
                (0x8000, 0xC000, True, [], 0x3FF0),  # ROM
            ]
        )
        c = CPU(mmu, 0xC000)
        c.r.s = 0xFD  # Not sure why the stack starts here.

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()

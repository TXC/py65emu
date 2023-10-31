#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_suites
----------------------------------

Tests for `py65emu` module.
"""


import unittest
import unittest.mock

from py65emu.cpu import CPU
from py65emu.mmu import MMU
from py65emu.operation import OpCodes, Operation, UndefinedOperation


class TestOperation(unittest.TestCase):
    def setUp(self):
        pass

    def _cpu(self, program=None, pc=0x0000) -> CPU:
        self.c = CPU(
            mmu=MMU([(0x0, 0x10000, False, program, pc)]),
            pc=pc,
        )
        mmu = MMU(
            [
                (0x0000, 0x800),  # RAM
                (0x2000, 0x8),  # PPU
                (0x4000, 0x18),
                (0x8000, 0xC000, True, [], pc),  # ROM
            ]
        )
        self.c = CPU(mmu, pc)
        self.c.r.s = 0xFD  # Not sure why the stack starts here.
        return self.c

    def test_unknown_operation(self):
        self._cpu()
        opc = OpCodes(self.c)
        with self.assertRaises(UndefinedOperation):
            opc[0x125]

    def test_repr_on_operation(self):
        subtests = [
            # OPC   LO    HI
            (0xEA, 'EA: NOP '),
            (0x0A, '0A: ASL A '),
            (0xA9, 'A9: LDA #$BB '),
            (0xAD, 'AD: LDA $LLHH '),
            (0xAD, 'AD: LDA $LLHH '),
            (0xA5, 'A5: LDA $LL '),
            (0xB5, 'B5: LDA $LL, X '),
            (0xB6, 'B6: LDX $LL, Y '),
            (0xAD, 'AD: LDA $LLHH '),
            (0xBD, 'BD: LDA $LLHH, X '),
            (0xB9, 'B9: LDA $LLHH, Y '),
            (0xA1, 'A1: LDA ($LL, X) '),
            (0xB1, 'B1: LDA ($LL), Y '),
            (0x6C, '6C: JMP ($LLHH) '),
            (0x90, '90: BCC $BB [PC + $BB] '),
            (0xB0, 'B0: BCS $BB [PC + $BB] '),
        ]

        self._cpu()
        for data in subtests:
            with self.subTest(data=data):
                c = self.c.opcodes[data[0]]

                self.assertIsInstance(c, Operation)
                self.assertEqual(repr(c), data[1])

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()

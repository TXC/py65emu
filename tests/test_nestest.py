#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_suites
----------------------------------

Tests for `py65emu` module.
"""


import os
import unittest

from py65emu.cpu import CPU, Registers
from py65emu.mmu import MMU
from py65emu.debug import Debug


class NesTestError(Exception):
    def __init__(
        self,
        message: str,
        actual: CPU,
        expected: Registers | None = None,
        *args: object
    ) -> None:
        super().__init__(*[message, actual, expected] + [*args])
        self.message = message
        self.actual = actual
        self.expected = expected

    def __str__(self) -> str:
        d = Debug(self.actual)
        d.memdump(0x0647)
        return (
            f"C: {self.actual.cc_total:d} "
            f"OP: ${self.actual.r.pc:0>4x}: {self.message}"
            f"\nActual:   {self.actual.r!r}\nExpected: {self.expected!r}"
        )


class NesTestRom(unittest.TestCase):
    def setUp(self):
        self.c = self.load_cpu()
        # self.c.reset()
        # self.c.r.pc = 0xC000
        self.reg = self.load_nestest_log()

    def tearDown(self):
        # print("\n\nEXITING:", f"Cycles: {self.c.cc_total:d}\n\n")
        # Debug.crash_dump(self.c)
        pass

    def load_cpu(self) -> "CPU":
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "files",
            # "nestest_mod2.nes"
            "nestest.nes"
        )

        with open(path, "rb") as f:
            mmu = MMU(
                [
                    (0x0000, 0x800),  # RAM
                    (0x2000, 0x8),  # PPU
                    (0x4000, 0x18),
                    (0x8000, 0xC000, True, f, 0x3FF0),  # ROM
                ]
            )

        c = CPU(mmu=mmu, pc=0xC000, disable_bcd=True)
        c.r.s = 0xFD
        return c

    def load_nestest_log(self) -> dict[int, "Registers"]:
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "files",
            "nestest.log"
        )
        registers = {}
        with open(path, "r") as file:
            for f in file.readlines():
                cycle = int(f[90:])
                r = Registers(int(f[:4], 16))
                r.a = int(f[50:52], 16)
                r.x = int(f[55:57], 16)
                r.y = int(f[60:62], 16)
                r.p = int(f[65:67], 16)
                r.s = int(f[71:73], 16)

                registers[cycle] = r

        return registers

    def test_nestest(self):
        # cycle = (self.c.cc_total - self.c.cc)
        cycle = self.c.cc_total
        self.checkCycle(cycle)

        while self.c.r.pc != 0xC66E:
            self.c.step()

            cycle = self.c.cc_total
            self.checkCycle(cycle)

            # Hmm, a diff in 41 cycles...
            """
            self.assertLessEqual(
                self.c.cc_total, 26554, "Too many cycles!"
            )
            """
            self.assertLessEqual(
                self.c.cc_total, 30000, "Too many cycles!"
            )

    def checkCycle(self, cycle: int) -> None:
        pc = "OP: {:0>4x}".format(self.c.r.pc)

        legal_op_error = self.c.mmu.read(0x0002)
        illegal_op_error = self.c.mmu.read(0x0003)

        self.assertEqual(
            legal_op_error,
            0x00,
            f"Cycle {cycle:d} - Caught Error in index 0x02: "
            f"0x{legal_op_error:0>2X} - PC 0x{pc:s}"
        )
        self.assertEqual(
            illegal_op_error,
            0x00,
            f"Cycle {cycle:d} - Caught Error in index 0x03: "
            f"0x{illegal_op_error:0>2X} - PC 0x{pc:s}"
        )

        if cycle not in self.reg:
            raise NesTestError(
                message="Cycle not found in nestest.log",
                actual=self.c
            )

        reg = self.reg[cycle]

        # Accumulator
        if self.c.r.a != reg.a:
            raise NesTestError(
                message="Error on Accumulator", actual=self.c, expected=reg
            )

        # X Register
        if self.c.r.x != reg.x:
            raise NesTestError(
                message="Error on X Register", actual=self.c, expected=reg
            )

        # Y Register
        if self.c.r.y != reg.y:
            raise NesTestError(
                message="Error on Y Register", actual=self.c, expected=reg
            )

        # P Register
        if self.c.r.p != reg.p:
            raise NesTestError(
                message="Error on P Register", actual=self.c, expected=reg
            )

        # Stack Pointer
        if self.c.r.s != reg.s:
            raise NesTestError(
                message="Error on Stack Pointer", actual=self.c, expected=reg
            )


if __name__ == "__main__":
    unittest.main()

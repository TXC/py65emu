#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_klaus_dormann
----------------------------------

Klaus Dormann's tests used for `py65emu` module.
"""


import os
import unittest
import copy

from py65emu.cpu import CPU, FlagBit
from py65emu.mmu import MMU
from py65emu.debug import Debug


class KlausDormann(unittest.TestCase):
    c: CPU

    def setUp(self) -> None:
        pass

    def _cpu(self, path: str, pc: int = 0x400) -> CPU:
        mmu = [
            (0, pc, False),
        ]
        self.c = CPU(MMU(mmu), pc)

        with open(path, "rb") as fp:
            self.c.mmu.addBlock(pc, 0x10000 - pc, False, fp)

        self.c.mmu.write(0xFFFC, (pc >> 8) & 0xFF)
        self.c.mmu.write(0xFFFD, pc & 0xFF)

        return self.c

    def tearDown(self) -> None:
        Debug.crash_dump(self.c)


class Decimal:
    cpu: CPU | None = None

    N1: int | None = None
    N2: int | None = None
    HA: int | None = None
    HNVZC: int | None = None
    DA: int | None = None
    DNVZC: int | None = None
    AR: int | None = None
    NF: int | None = None
    VF: int | None = None
    ZF: int | None = None
    CF: int | None = None
    ERROR: int | None = None
    N1L: int | None = None
    N1H: int | None = None
    N2L: int | None = None
    N2H: int | None = None

    def __init__(self, cpu: CPU | None = None):
        self.cpu = cpu

    def update(self) -> None:
        if self.cpu is None:
            return None

        # operands - register Y = carry in
        # 0x0000 1 byte
        self.N1 = self.cpu.mmu.read(0x0000)
        # 0x0001 1 byte
        self.N2 = self.cpu.mmu.read(0x0001)
        # binary result
        # 0x0002 1 byte
        self.HA = self.cpu.mmu.read(0x0002)
        # 0x0003 1 byte
        self.HNVZC = self.cpu.mmu.read(0x0003)
        # decimal result
        # 0x0004 1 byte
        self.DA = self.cpu.mmu.read(0x0004)
        # 0x0005 1 byte
        self.DNVZC = self.cpu.mmu.read(0x0005)
        # predicted results
        # 0x0006 1 byte
        self.AR = self.cpu.mmu.read(0x0006)
        # 0x0007 1 byte
        self.NF = self.cpu.mmu.read(0x0007)

        # 0x0008 1 byte
        self.VF = self.cpu.mmu.read(0x0008)
        # 0x0009 1 byte
        self.ZF = self.cpu.mmu.read(0x0009)
        # 0x000a 1 byte
        self.CF = self.cpu.mmu.read(0x000a)
        # 0x000b 1 byte
        self.ERROR = self.cpu.mmu.read(0x000b)
        # workspace
        # 0x000c 1 byte
        self.N1L = self.cpu.mmu.read(0x000c)
        # 0x000d 1 byte
        self.N1H = self.cpu.mmu.read(0x000d)
        # 0x000e 1 byte
        self.N2L = self.cpu.mmu.read(0x000e)
        # 0x000f 2 byte
        self.N2H = (
            self.cpu.mmu.read(0x000f) | (self.cpu.mmu.read(0x0010) << 8)
        )

    def __eq__(self, other):
        if not isinstance(other, Decimal):
            # don't attempt to compare against unrelated types
            return NotImplemented

        attr = [
            "N1", "N2",
            "HA", "HNVZC",
            "DA", "DNVZC",
            "AR", "NF", "VF", "ZF", "CF", "ERROR",
            "N1L", "N1H", "N2L", "N2H"
        ]
        for a in attr:
            self_value = getattr(self, a)
            other_value = getattr(other, a)

            if self_value is None:
                raise ValueError(f"L: '{a}' is None")

            if other_value is None:
                raise ValueError(f"R: '{a}' is None")

            if self_value != other_value:
                return False
        return True

    def __xor__(self, other):
        if not isinstance(other, Decimal):
            # don't attempt to compare against unrelated types
            return NotImplemented

        new = Decimal()

        attr = [
            "N1", "N2",
            "HA", "HNVZC",
            "DA", "DNVZC",
            "AR", "NF", "VF", "ZF", "CF", "ERROR",
            "N1L", "N1H", "N2L", "N2H"
        ]
        for a in attr:
            self_value = getattr(self, a)
            other_value = getattr(other, a)
            if self_value == other_value:
                continue
            result = self_value ^ other_value
            setattr(new, a, result)

        return new


@unittest.skip('Runs, but doesn\'t break')
class KlausDormannDecimal(KlausDormann):
    def setUp(self):
        super().setUp()
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "files",
            "6502_decimal_test.bin",
        )
        self.c = self._cpu(path, 0x200)

    def test(self):
        """
        Bruce Clark - Verify decimal mode behavior, modified by Klaus Dormann
        See https://github.com/Klaus2m5/6502_65C02_functional_tests
        """

        total_no_of_cycles = 0
        self.c.running = True

        comparison_value = Decimal(self.c)
        comparison_value.update()
        new_value = copy.copy(comparison_value)

        while self.c.running:
            self.c.step()

            new_value.update()
            if comparison_value != new_value:
                # res = comparison_value ^ new_value
                comparison_value = new_value

            # total_no_of_cycles += self.c.cc
            total_no_of_cycles += 1

            if self.c.running is not True:
                print("No. of cycles {}".format(total_no_of_cycles))
                self.assertEqual(
                    self.c.mmu.read(0x000b),
                    0,
                    "Test reported error"
                )
                break

            if self.c.r.pc == 0x0257:
                break

            self.assertLess(
                total_no_of_cycles,
                0x2500000,
                "Maximum number of loops exceeded"
            )

        print("No. of cycles {}".format(total_no_of_cycles))


@unittest.skip('Takes a long time to run!')
class KlausDormannFunctional(KlausDormann):
    def setUp(self):
        super().setUp()
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "files",
            "6502_functional_test.bin",
        )
        self.c = self._cpu(path)

    def test(self):
        """
        Klaus Dormann's Functional Test Program.
        See https://github.com/Klaus2m5/6502_65C02_functional_tests

        If a test is failing find the first test that fails. The tests are
        dumb, they do not catch error traps correctly.
        """

        data = {
            # Load Data
            0x0461: 0x01,
            # testing relative addressing with BEQ
            0x05AA: 0x02,
            # partial test BNE & CMP, CPX, CPY immediate
            0x05F1: 0x03,
            # testing stack operations PHA PHP PLA PLP
            0x0625: 0x04,
            # testing branch decisions BPL BMI BVC BVS BCC BCS BNE BEQ
            0x079F: 0x05,
            # test PHA does not alter flags or accumulator but PLA does
            0x089B: 0x06,
            # partial pretest EOR #
            0x08CF: 0x07,
            # PC modifying instructions except branches
            # (NOP, JMP, JSR, RTS, BRK, RTI)
            0x0919: 0x08,
            # Jump absolute
            0x096F: 0x09,
            # Jump indirect
            0x09AB: 0x0A,
            # Jump subroutine & return from subroutine
            0x09E2: 0x0B,
            # break & return from interrupt
            0x0A3A: 0x0C,
            # test set and clear flags CLC CLI CLD CLV SEC SEI SED
            0x0AE0: 0x0D,
            # testing index register increment/decrement and transfer
            # INX INY DEX DEY TAX TXA TAY TYA
            0x0DA6: 0x0E,
            # TSX sets NZ - TXS does not
            # This section also tests for proper stack wrap around.
            0x0E6F: 0x0F,
            # testing index register load & store LDY LDX STY STX all
            # addressing modes
            # LDX / STX - zp,y / abs,y
            0x0F2A: 0x10,
            # Indexed wraparound test (only zp should wrap)
            0x0F6C: 0x11,
            # LDY / STY - zp,x / abs,x
            0x1023: 0x12,
            # Indexed wraparound test (only zp should wrap)
            0x1063: 0x13,
            # LDX / STX - zp / abs / #
            0x1359: 0x14,
            # LDY / STY - zp / abs / #
            0x1653: 0x15,
            # Testing load / store accumulator LDA / STA all addressing modes
            # LDA / STA - zp,x / abs,x
            0x1704: 0x16,
            # LDA / STA - (zp),y / abs,y / (zp,x)
            0x181F: 0x17,
            # Indexed wraparound test (only zp should wrap)
            0x18C2: 0x18,
            # LDA / STA - zp / abs / #
            0x1B8C: 0x19,
            # testing bit test & compares BIT CPX CPY CMP all addressing modes
            # BIT - zp / abs
            0x1CE0: 0x1A,
            # CPX - zp / abs / #
            0x1DEE: 0x1B,
            # CPY - zp / abs / #
            0x1EFC: 0x1C,
            # CMP - zp / abs / #
            0x22E0: 0x1D,
            # testing shifts - ASL LSR ROL ROR all addressing modes
            # shifts - accumulator
            0x2424: 0x1E,
            # Shifts - zeropage
            0x25A4: 0x1F,
            # Shifts - absolute
            0x2748: 0x20,
            # Shifts - zp indexed
            0x28C8: 0x21,
            # Shifts - abs indexed
            0x2A6C: 0x22,
            # testing memory increment/decrement - INC DEC all addressing modes
            # zeropage
            0x2B16: 0x23,
            # absolute memory
            0x2BD0: 0x24,
            # zeropage indexed
            0x2C7E: 0x25,
            # memory indexed
            0x2D3C: 0x26,
            # Testing logical instructions - AND EOR ORA all addressing modes
            # AND
            0x2F34: 0x27,
            # EOR
            0x312C: 0x28,
            # OR
            0x3325: 0x29,
            # full binary add/subtract test
            # iterates through all combinations of operands and carry input
            # uses increments/decrements to predict result & result flags
            0x338A: 0x2A,
            # decimal add/subtract test
            # *** WARNING - tests documented behavior only! ***
            #   only valid BCD operands are tested, N V Z flags are ignored
            # iterates through all valid combinations of operands and carry
            # input uses increments/decrements to predict result & carry flag
            0x342E: 0x2B,
            # decimal/binary switch test
            # tests CLD, SED, PLP, RTI to properly switch between decimal &
            # binary opcode
            0x3486: 0xF0,
        }

        self.c.debug = True
        while 1:
            old_pc = self.c.r.pc
            self.c.step()

            if self.c.r.pc in data:
                self.assertEqual(self.c.r.a, data[self.c.r.pc])

            self.assertNotEqual(
                self.c.r.pc,
                old_pc,
                f'Catched Trap {old_pc:0<2x}'
            )

            self.assertLess(
                self.c.cc_total,
                0x262EE18,  # 40037912
                "Maximum number of loops exceeded"
            )


@unittest.expectedFailure
@unittest.skip('Not working without a IRQ/NMI generated')
class KlausDormannInterrupt(KlausDormann):
    def setUp(self):
        super().setUp()
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "files",
            "6502_interrupt_test.bin",
        )
        self.c = self._cpu(path)

    def test(self):
        """
        Klaus_Dormann's Interrupt Test Program.
        See https://github.com/Klaus2m5/6502_65C02_functional_tests

        This tests that the IRQ BRK and NMI all function correctly.
        """
        previous_interrupt_watch_value = 0
        total_no_of_cycles = 0

        """
        {
            # Load Data
            0x0424: True,
            # IRQ integrity test
            # test for clear flags seen in IRQ vector
            0x0501: True,
            # BRK integrity test
            # test for clear flags seen in IRQ vector
            0x05BF: True,
            # NMI integrity test
            # test for clear flags seen in NMI vector
            0x069D: True,
            # test IRQ & NMI with interrupts disabled
            0x06CF: True,
            # test overlapping NMI, IRQ & BRK
            0x0700: True,
            # 65C02 - WAI with interrupts disabled
            0x071a: False,
            # 65C02 - WAI with interrupts enabled
            0x0737: False,
            # 65C02 - manual test for the STP opcode of the 65c02
            0x073a: False,
        }
        """

        while 1:
            interrupt_watch = self.c.mmu.read(0xBFFC)
            # This is used to simulate the edge triggering of an NMI.
            # If we didn't do this we would get stuck in a loop forever
            if interrupt_watch != previous_interrupt_watch_value:
                previous_interrupt_watch_value = interrupt_watch
                if (interrupt_watch & 2) != 0:
                    self.c.trigger_nmi = True

            if (
                (
                    self.c.r.getFlag(FlagBit.I) is False
                ) and (
                    (interrupt_watch & 1) != 0
                )
            ):
                self.c.interruptRequest()

            old_pc = self.c.r.pc
            self.c.step()

            if self.c.r.pc == 0x0700:
                # Success
                break

            self.assertNotEqual(
                self.c.r.pc,
                old_pc,
                f'Catched Trap {old_pc:0<2x}'
            )

            self.assertLess(
                total_no_of_cycles,
                0x186A0,
                "Maximum number of loops exceeded"
            )


if __name__ == "__main__":
    unittest.main()

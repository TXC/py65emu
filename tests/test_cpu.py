#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_cpu
----------------------------------

Tests for `py65emu` module.
"""

import os
import unittest

from py65emu.cpu import CPU, FlagBit
from py65emu.mmu import MMU


class TestCPU(unittest.TestCase):
    def _cpu(
        self,
        ram=(0, 0x200, False),
        rom=(0x1000, 0x100),
        romInit=None,
        pc=0x1000,
    ):
        return CPU(MMU([ram, rom + (True, romInit)]), pc)

    def setUp(self):
        pass

    def test_fromBCD(self):
        c = self._cpu()
        self.assertEqual(c.fromBCD(0), 0)
        self.assertEqual(c.fromBCD(0x05), 5)
        self.assertEqual(c.fromBCD(0x11), 11)
        self.assertEqual(c.fromBCD(0x99), 99)

    def test_toBCD(self):
        c = self._cpu()
        self.assertEqual(c.toBCD(0), 0)
        self.assertEqual(c.toBCD(5), 0x05)
        self.assertEqual(c.toBCD(11), 0x11)
        self.assertEqual(c.toBCD(99), 0x99)

    def test_fromTwosCom(self):
        c = self._cpu()
        self.assertEqual(c.fromTwosCom(0x00), 0)
        self.assertEqual(c.fromTwosCom(0x01), 1)
        self.assertEqual(c.fromTwosCom(0x7F), 127)
        self.assertEqual(c.fromTwosCom(0xFF), -1)
        self.assertEqual(c.fromTwosCom(0x80), -128)

    def test_nextByte(self):
        c = self._cpu(romInit=[1, 2, 3])
        self.assertEqual(c.nextByte(), 1)
        self.assertEqual(c.nextByte(), 2)
        self.assertEqual(c.nextByte(), 3)
        self.assertEqual(c.nextByte(), 0)

    def test_nextWord(self):
        c = self._cpu(romInit=[1, 2, 3, 4, 5, 9, 10])
        self.assertEqual(c.nextWord(), 0x0201)
        c.nextByte()
        self.assertEqual(c.nextWord(), 0x0504)
        self.assertEqual(c.nextWord(), 0x0A09)

    def test_zeropage_addressing(self):
        c = self._cpu(romInit=[1, 2, 3, 4, 5])
        self.assertEqual(c.z_a(), 1)
        c.r.x = 0
        self.assertEqual(c.zx_a(), 2)
        c.r.x = 1
        self.assertEqual(c.zx_a(), 4)
        c.r.y = 0
        self.assertEqual(c.zy_a(), 4)
        c.r.y = 1
        self.assertEqual(c.zy_a(), 6)

    def test_absolute_addressing(self):
        c = self._cpu(romInit=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(c.a_a(), 0x0201)

        c.r.x = 0
        c.cc = 0
        self.assertEqual(c.ax_a(), 0x0403)
        self.assertEqual(c.cc, 2)
        c.r.x = 0xFF
        c.cc = 0
        self.assertEqual(c.ax_a(), 0x0605 + 0xFF)
        self.assertEqual(c.cc, 2)

        c.r.y = 0
        c.cc = 0
        self.assertEqual(c.ay_a(), 0x0807)
        self.assertEqual(c.cc, 2)
        c.r.y = 0xFF
        c.cc = 0
        self.assertEqual(c.ay_a(), 0x0A09 + 0xFF)
        self.assertEqual(c.cc, 2)

    def test_indirect_addressing(self):
        c = self._cpu(
            romInit=[
                0x06,
                0x10,
                0xFF,
                0x10,
                0x00,
                0x00,
                0xF0,
                0x00,
            ]
        )

        self.assertEqual(c.i_a(), 0x00F0)
        self.assertEqual(c.i_a(), 0x0600)

        c.r.y = 0x05
        c.mmu.write(0x00, 0x21)
        c.mmu.write(0x01, 0x43)
        self.assertEqual(c.iy_a(), 0x4326)

        c.r.x = 0x02
        c.mmu.write(0x02, 0x34)
        c.mmu.write(0x03, 0x12)
        self.assertEqual(c.ix_a(), 0x1234)

    def test_stack(self):
        c = self._cpu()
        c.stackPush(0x10)
        self.assertEqual(c.stackPop(), 0x10)
        c.stackPushWord(0x0510)
        self.assertEqual(c.stackPopWord(), 0x0510)
        self.assertEqual(c.stackPop(), 0x00)
        c.stackPush(0x00)
        c.stackPushWord(0x0510)
        self.assertEqual(c.stackPop(), 0x10)
        self.assertEqual(c.stackPop(), 0x05)

    def test_adc(self):
        c = self._cpu(romInit=[1, 2, 250, 3, 100, 100])
        # immediate
        c.execute([0x69])
        self.assertEqual(c.r.a, 1)
        c.execute([0x69])
        self.assertEqual(c.r.a, 3)
        c.execute([0x69])
        self.assertEqual(c.r.a, 253)
        self.assertTrue(c.r.getFlag(FlagBit.N))
        c.r.clearFlags()
        c.execute([0x69])
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertTrue(c.r.getFlag(FlagBit.Z))
        c.r.clearFlags()
        c.execute([0x69, 0x69])
        self.assertTrue(c.r.getFlag(FlagBit.V))

    def test_adc_decimal(self):
        c = self._cpu(romInit=[0x01, 0x55, 0x50])
        c.r.setFlag(FlagBit.D)

        c.execute([0x69])
        self.assertEqual(c.r.a, 0x01)
        c.execute([0x69])
        self.assertEqual(c.r.a, 0x56)
        c.execute([0x69])
        self.assertEqual(c.r.a, 0x06)
        self.assertTrue(c.r.getFlag(FlagBit.C))

    def test_and(self):
        c = self._cpu(romInit=[0xFF, 0xFF, 0x01, 0x2])

        c.r.a = 0x00
        c.execute([0x29])
        self.assertEqual(c.r.a, 0)

        c.r.a = 0xFF
        c.execute([0x29])
        self.assertEqual(c.r.a, 0xFF)

        c.r.a = 0x01
        c.execute([0x29])
        self.assertEqual(c.r.a, 0x01)

        c.r.a = 0x01
        c.execute([0x29])
        self.assertEqual(c.r.a, 0x00)

    def test_asl(self):
        c = self._cpu(romInit=[0x00])

        c.r.a = 1
        c.execute([0x0A])
        self.assertEqual(c.r.a, 2)

        c.mmu.write(0, 4)
        c.execute([0x06])
        self.assertEqual(c.mmu.read(0), 8)

    def test_bit(self):
        c = self._cpu(romInit=[0x00, 0x00, 0x10])
        c.mmu.write(0, 0xFF)
        c.r.a = 1

        c.execute([0x24])  # Zero page
        self.assertFalse(c.r.getFlag(FlagBit.Z))
        self.assertTrue(c.r.getFlag(FlagBit.N))
        self.assertTrue(c.r.getFlag(FlagBit.V))

        c.execute([0x2C])  # Absolute
        self.assertTrue(c.r.getFlag(FlagBit.Z))
        self.assertFalse(c.r.getFlag(FlagBit.N))
        self.assertFalse(c.r.getFlag(FlagBit.V))

    def test_brk(self):
        c = self._cpu()
        c.mmu.addBlock(0xFFFE, 0x2, True, [0x34, 0x12])
        c.r.p = 0b11101111
        c.execute([0x00])
        # self.assertTrue(c.r.getFlag(FlagBit.B))
        self.assertTrue(c.r.getFlag(FlagBit.I))
        self.assertEqual(c.r.pc, 0x1234)
        self.assertEqual(c.stackPop(), 255)
        self.assertEqual(c.stackPopWord(), 0x1001)

    def test_branching(self):
        c = self._cpu(romInit=[0x01, 0x00, 0x00, 0xFC])
        c.execute([0x10])
        self.assertEqual(c.r.pc, 0x1002)
        c.execute([0x70])
        self.assertEqual(c.r.pc, 0x1003)
        c.r.setFlag(FlagBit.C)
        c.execute([0xB0])
        self.assertEqual(c.r.pc, 0x1000)
        c.execute([0xD0])
        self.assertEqual(c.r.pc, 0x1002)

    def test_cmp(self):
        c = self._cpu(romInit=[0x0F, 0x10, 0x11, 0xFE, 0xFF, 0x00, 0x7F])

        c.r.a = 0x10
        c.execute([0xC9])
        self.assertFalse(c.r.getFlag(FlagBit.Z))
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertFalse(c.r.getFlag(FlagBit.N))
        c.execute([0xC9])
        self.assertTrue(c.r.getFlag(FlagBit.Z))
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertFalse(c.r.getFlag(FlagBit.N))
        c.execute([0xC9])
        self.assertFalse(c.r.getFlag(FlagBit.Z))
        self.assertFalse(c.r.getFlag(FlagBit.C))
        self.assertTrue(c.r.getFlag(FlagBit.N))

        c.r.a = 0xFF
        c.execute([0xC9])
        self.assertFalse(c.r.getFlag(FlagBit.Z))
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertFalse(c.r.getFlag(FlagBit.N))
        c.execute([0xC9])
        self.assertTrue(c.r.getFlag(FlagBit.Z))
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertFalse(c.r.getFlag(FlagBit.N))
        c.execute([0xC9])
        self.assertFalse(c.r.getFlag(FlagBit.Z))
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertTrue(c.r.getFlag(FlagBit.N))
        c.execute([0xC9])
        self.assertFalse(c.r.getFlag(FlagBit.Z))
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertTrue(c.r.getFlag(FlagBit.N))

    def test_cpx(self):
        c = self._cpu(romInit=[0x0F, 0x10, 0x11])

        c.r.x = 0x10
        c.execute([0xE0])
        self.assertFalse(c.r.getFlag(FlagBit.Z))
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertFalse(c.r.getFlag(FlagBit.N))
        c.execute([0xE0])
        self.assertTrue(c.r.getFlag(FlagBit.Z))
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertFalse(c.r.getFlag(FlagBit.N))
        c.execute([0xE0])
        self.assertFalse(c.r.getFlag(FlagBit.Z))
        self.assertFalse(c.r.getFlag(FlagBit.C))
        self.assertTrue(c.r.getFlag(FlagBit.N))

    def test_cpy(self):
        c = self._cpu(romInit=[0x0F, 0x10, 0x11])

        c.r.y = 0x10
        c.execute([0xC0])
        self.assertFalse(c.r.getFlag(FlagBit.Z))
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertFalse(c.r.getFlag(FlagBit.N))
        c.execute([0xC0])
        self.assertTrue(c.r.getFlag(FlagBit.Z))
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertFalse(c.r.getFlag(FlagBit.N))
        c.execute([0xC0])
        self.assertFalse(c.r.getFlag(FlagBit.Z))
        self.assertFalse(c.r.getFlag(FlagBit.C))
        self.assertTrue(c.r.getFlag(FlagBit.N))

    def test_dec(self):
        c = self._cpu(romInit=[0x00])
        c.execute([0xC6])
        self.assertEqual(c.mmu.read(0x00), 0xFF)

    def test_dex(self):
        c = self._cpu()
        c.execute([0xCA])
        self.assertEqual(c.r.x, 0xFF)

    def test_dey(self):
        c = self._cpu()
        c.execute([0x88])
        self.assertEqual(c.r.y, 0xFF)

    def test_eor(self):
        c = self._cpu(romInit=[0x0F, 0xF0, 0xFF])

        c.execute([0x49])
        self.assertEqual(c.r.a, 0x0F)
        c.execute([0x49])
        self.assertEqual(c.r.a, 0xFF)
        c.execute([0x49])
        self.assertEqual(c.r.a, 0x00)

    def test_flag_ops(self):
        c = self._cpu()

        c.execute([0x38])
        self.assertTrue(c.r.getFlag(FlagBit.C))
        c.execute([0x78])
        self.assertTrue(c.r.getFlag(FlagBit.I))
        c.execute([0xF8])
        self.assertTrue(c.r.getFlag(FlagBit.D))

        c.r.setFlag(FlagBit.V)

        c.execute([0x18])
        self.assertFalse(c.r.getFlag(FlagBit.C))
        c.execute([0x58])
        self.assertFalse(c.r.getFlag(FlagBit.I))
        c.execute([0xB8])
        self.assertFalse(c.r.getFlag(FlagBit.V))
        c.execute([0xD8])
        self.assertFalse(c.r.getFlag(FlagBit.D))

    def test_inc(self):
        c = self._cpu(romInit=[0x00])
        c.execute([0xE6])
        self.assertEqual(c.mmu.read(0x00), 0x01)

    def test_inx(self):
        c = self._cpu()
        c.execute([0xE8])
        self.assertEqual(c.r.x, 0x01)

    def test_iny(self):
        c = self._cpu()
        c.execute([0xC8])
        self.assertEqual(c.r.y, 0x01)

    def test_jmp(self):
        c = self._cpu(romInit=[0x00, 0x10])

        c.execute([0x4C])
        self.assertEqual(c.r.pc, 0x1000)

        c.execute([0x6C])
        self.assertEqual(c.r.pc, 0x1000)

    def test_jsr(self):
        c = self._cpu(romInit=[0x00, 0x10])

        c.execute([0x20])
        self.assertEqual(c.r.pc, 0x1000)
        self.assertEqual(c.stackPopWord(), 0x1001)

    def test_lda(self):
        c = self._cpu(romInit=[0x01])
        c.execute([0xA9])
        self.assertEqual(c.r.a, 0x01)

    def test_ldx(self):
        c = self._cpu(romInit=[0x01])
        c.execute([0xA2])
        self.assertEqual(c.r.x, 0x01)

    def test_ldy(self):
        c = self._cpu(romInit=[0x01])
        c.execute([0xA0])
        self.assertEqual(c.r.y, 0x01)

    def test_lsr(self):
        c = self._cpu(romInit=[0x00])

        c.r.a = 0x02

        c.execute([0x4A])
        self.assertEqual(c.r.a, 0x01)
        self.assertFalse(c.r.getFlag(FlagBit.C))

        c.execute([0x4A])
        self.assertEqual(c.r.a, 0x00)
        self.assertTrue(c.r.getFlag(FlagBit.C))

        c.mmu.write(0x00, 0x02)
        c.execute([0x46])
        self.assertEqual(c.mmu.read(0x00), 0x01)

    def test_ora(self):
        c = self._cpu(romInit=[0x0F, 0xF0, 0xFF])
        c.execute([0x09])
        self.assertEqual(c.r.a, 0x0F)
        c.execute([0x09])
        self.assertEqual(c.r.a, 0xFF)
        c.execute([0x09])
        self.assertEqual(c.r.a, 0xFF)

    def test_p(self):
        c = self._cpu()

        c.r.a = 0xCC
        c.execute([0x48])
        self.assertEqual(c.stackPop(), 0xCC)

        c.r.p = 0xFF
        c.execute([0x08])
        self.assertEqual(c.stackPop(), 0xFF)

        c.r.a = 0x00
        c.stackPush(0xDD)
        c.execute([0x68])
        self.assertEqual(c.r.a, 0xDD)

        c.r.p = 0x20
        c.stackPush(0xFD)
        c.execute([0x28])
        self.assertEqual(c.r.p, 0xED)

    def test_rol(self):
        c = self._cpu(romInit=[0x00])

        c.r.a = 0xFF
        c.execute([0x2A])
        self.assertEqual(c.r.a, 0xFE)
        self.assertTrue(c.r.getFlag(FlagBit.C))
        c.execute([0x2A])
        self.assertEqual(c.r.a, 0xFD)
        self.assertTrue(c.r.getFlag(FlagBit.C))

        c.execute([0x26])
        self.assertEqual(c.mmu.read(0x00), 0x01)
        self.assertFalse(c.r.getFlag(FlagBit.C))

    def test_ror(self):
        c = self._cpu(romInit=[0x00])

        c.r.a = 0xFF
        c.execute([0x6A])
        self.assertEqual(c.r.a, 0x7F)
        self.assertTrue(c.r.getFlag(FlagBit.C))
        c.execute([0x6A])
        self.assertEqual(c.r.a, 0xBF)
        self.assertTrue(c.r.getFlag(FlagBit.C))

        c.execute([0x66])
        self.assertEqual(c.mmu.read(0x00), 0x80)
        self.assertFalse(c.r.getFlag(FlagBit.C))

    def test_rti(self):
        c = self._cpu()

        c.stackPushWord(0x1234)
        c.stackPush(0xFD)

        c.execute([0x40])
        self.assertEqual(c.r.pc, 0x1234)
        self.assertTrue(c.r.getFlag(FlagBit.N))
        self.assertTrue(c.r.getFlag(FlagBit.V))
        self.assertTrue(c.r.getFlag(FlagBit.B))
        self.assertTrue(c.r.getFlag(FlagBit.D))
        self.assertTrue(c.r.getFlag(FlagBit.I))
        self.assertFalse(c.r.getFlag(FlagBit.Z))
        self.assertTrue(c.r.getFlag(FlagBit.C))

    def test_rts(self):
        c = self._cpu()
        c.stackPushWord(0x1234)
        c.execute([0x60])
        self.assertEqual(c.r.pc, 0x1235)

    def test_sbc(self):
        c = self._cpu(romInit=[0x10, 0x01, 0x51, 0x80, 0x12, 0x13, 0x02, 0x21])

        c.r.a = 0x15
        c.r.setFlag(FlagBit.C)
        c.execute([0xE9])
        self.assertEqual(c.r.a, 0x05)
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertFalse(c.r.getFlag(FlagBit.V))
        self.assertFalse(c.r.getFlag(FlagBit.N))
        self.assertFalse(c.r.getFlag(FlagBit.Z))

        c.r.a = 0xFF
        c.r.setFlag(FlagBit.C)
        c.execute([0xE9])
        self.assertEqual(c.r.a, 0xFE)
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertFalse(c.r.getFlag(FlagBit.V))
        self.assertTrue(c.r.getFlag(FlagBit.N))
        self.assertFalse(c.r.getFlag(FlagBit.Z))

        c.r.a = 0x50
        c.r.setFlag(FlagBit.C)
        c.execute([0xE9])
        self.assertEqual(c.r.a, 0xFF)
        self.assertFalse(c.r.getFlag(FlagBit.C))
        self.assertFalse(c.r.getFlag(FlagBit.V))
        self.assertTrue(c.r.getFlag(FlagBit.N))
        self.assertFalse(c.r.getFlag(FlagBit.Z))

        c.r.a = 0x01
        c.r.setFlag(FlagBit.C)
        c.execute([0xE9])
        self.assertEqual(c.r.a, 0x81)
        self.assertFalse(c.r.getFlag(FlagBit.C))
        self.assertTrue(c.r.getFlag(FlagBit.V))
        self.assertTrue(c.r.getFlag(FlagBit.N))
        self.assertFalse(c.r.getFlag(FlagBit.Z))

        # decimal mode test
        c.r.setFlag(FlagBit.D)

        c.r.a = 0x46
        c.r.setFlag(FlagBit.C)
        c.execute([0xE9])
        self.assertEqual(c.r.a, 0x34)
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertFalse(c.r.getFlag(FlagBit.V))
        self.assertFalse(c.r.getFlag(FlagBit.N))
        self.assertFalse(c.r.getFlag(FlagBit.Z))

        c.r.a = 0x40
        c.r.setFlag(FlagBit.C)
        c.execute([0xE9])
        self.assertEqual(c.r.a, 0x27)
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertFalse(c.r.getFlag(FlagBit.V))
        self.assertFalse(c.r.getFlag(FlagBit.N))
        self.assertFalse(c.r.getFlag(FlagBit.Z))

        c.r.a = 0x32
        c.r.clearFlag(FlagBit.C)
        c.execute([0xE9])
        self.assertEqual(c.r.a, 0x29)
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertFalse(c.r.getFlag(FlagBit.V))
        self.assertFalse(c.r.getFlag(FlagBit.N))
        self.assertFalse(c.r.getFlag(FlagBit.Z))

        c.r.a = 0x12
        c.r.setFlag(FlagBit.C)
        c.execute([0xE9])
        self.assertEqual(c.r.a, 0x91)
        self.assertFalse(c.r.getFlag(FlagBit.C))
        self.assertFalse(c.r.getFlag(FlagBit.V))
        self.assertTrue(c.r.getFlag(FlagBit.N))
        self.assertFalse(c.r.getFlag(FlagBit.Z))

    def test_sta(self):
        c = self._cpu(romInit=[0x00])
        c.r.a = 0xF0
        c.execute([0x85])
        self.assertEqual(c.mmu.read(0x00), 0xF0)

    def test_stx(self):
        c = self._cpu(romInit=[0x00])
        c.r.x = 0xF0
        c.execute([0x86])
        self.assertEqual(c.mmu.read(0x00), 0xF0)

    def test_sty(self):
        c = self._cpu(romInit=[0x00])
        c.r.y = 0xF0
        c.execute([0x84])
        self.assertEqual(c.mmu.read(0x00), 0xF0)

    def test_t(self):
        c = self._cpu()

        c.r.a = 0xF0
        c.execute([0xAA])
        self.assertEqual(c.r.x, 0xF0)

        c.r.x = 0x0F
        c.execute([0x8A])
        self.assertEqual(c.r.a, 0x0F)

        c.r.a = 0xFF
        c.execute([0xA8])
        self.assertEqual(c.r.y, 0xFF)

        c.r.y = 0x00
        c.execute([0x98])
        self.assertEqual(c.r.a, 0x00)

        c.r.x = 0xFF
        c.execute([0x9A])
        self.assertEqual(c.r.s, 0xFF)

        c.r.s = 0xF0
        c.execute([0xBA])
        self.assertEqual(c.r.x, 0xF0)

    def test_aac(self):
        c = self._cpu(romInit=[0xFF, 0xFF, 0x01, 0x2])

        c.r.a = 0x00
        c.execute([0x0B])
        self.assertEqual(c.r.a, 0)
        self.assertFalse(c.r.getFlag(FlagBit.N))
        self.assertFalse(c.r.getFlag(FlagBit.C))

        c.r.a = 0xFF
        c.execute([0x2B])
        self.assertEqual(c.r.a, 0xFF)
        self.assertTrue(c.r.getFlag(FlagBit.N))
        self.assertTrue(c.r.getFlag(FlagBit.C))

    def test_aax(self):
        c = self._cpu(romInit=[0x00, 0x00])

        c.r.a = 0xF0
        c.r.x = 0xF0
        c.execute([0x87])
        self.assertEqual(c.mmu.read(0x00), 0xF0)

    def test_arr(self):
        c = self._cpu(romInit=[0x80])

        c.r.a = 0xFF
        c.r.setFlag(FlagBit.C)
        c.execute([0x6B])
        self.assertEqual(c.r.a, 0xC0)
        self.assertTrue(c.r.getFlag(FlagBit.C))
        self.assertTrue(c.r.getFlag(FlagBit.V))

    def test_asr(self):
        c = self._cpu(romInit=[0x80])

        c.r.a = 0xFF
        c.r.setFlag(FlagBit.C)
        c.execute([0x4B])
        self.assertEqual(c.r.a, 0x40)

    def test_atx(self):
        c = self._cpu(romInit=[0xF8])

        c.r.a = 0x1F
        c.execute([0xAB])
        # self.assertEqual(c.r.x, 0x18)
        self.assertEqual(c.r.a, 0xf0)
        self.assertEqual(c.r.x, 0xf0)

    def test_axa(self):
        c = self._cpu(ram=(0, 0x400, False), romInit=[0xFF, 0x01])

        c.r.a = c.r.x = 0xFF
        c.r.y = 0x01
        c.execute([0x9F])

        self.assertEqual(c.mmu.read(0x200), 0x02)

    def test_axs(self):
        c = self._cpu(romInit=[0x02])

        c.r.a = 0xF0
        c.r.x = 0x0F
        c.execute([0xCB])
        self.assertEqual(c.r.x, 0xFE)

    def test_dcp(self):
        c = self._cpu(romInit=[0x01])
        c.r.a = 0xFF
        c.execute([0xC7])
        self.assertEqual(c.mmu.read(0x01), 0xFF)
        self.assertTrue(c.r.getFlag(FlagBit.Z))

    def test_isc(self):
        c = self._cpu(romInit=[0x01])
        c.r.a = 0xFF
        c.r.setFlag(FlagBit.C)
        c.execute([0xE7])
        self.assertEqual(c.mmu.read(0x01), 0x01)
        self.assertEqual(c.r.a, 0xFE)

    def test_kil(self):
        c = self._cpu()
        c.execute([0x02])
        self.assertFalse(c.running)

    def test_lar(self):
        c = self._cpu(romInit=[0x01, 0x00])
        c.r.y = 0x01
        c.mmu.write(0x02, 0xF0)

        c.execute([0xBB])
        self.assertEqual(c.r.a, 0xF0)
        self.assertEqual(c.r.x, 0xF0)
        self.assertEqual(c.r.s, 0xF0)

    def test_lax(self):
        c = self._cpu(romInit=[0x01])
        c.mmu.write(0x01, 0xF0)
        c.execute([0xA7])
        self.assertEqual(c.r.a, 0xF0)
        self.assertEqual(c.r.x, 0xF0)

    def test_rla(self):
        c = self._cpu(romInit=[0x01])
        c.mmu.write(0x01, 0x01)
        c.r.a = 0x06
        c.r.setFlag(FlagBit.C)
        c.execute([0x27])
        self.assertEqual(c.mmu.read(0x01), 0x03)
        self.assertEqual(c.r.a, 0x02)

    def test_rra(self):
        c = self._cpu(romInit=[0x01])
        c.mmu.write(0x01, 0x01)
        c.r.a = 0x06
        c.r.setFlag(FlagBit.C)
        c.execute([0x67])
        self.assertEqual(c.mmu.read(0x01), 0x80)
        self.assertEqual(c.r.a, 0x87)

    def test_rra2(self):
        c = self._cpu(romInit=[0x01])
        c.mmu.write(0x01, 0x02)
        c.r.a = 0x06
        c.r.setFlag(FlagBit.C)
        c.execute([0x47])
        self.assertEqual(c.mmu.read(0x01), 0x01)
        self.assertEqual(c.r.a, 0x07)

    def test_slo(self):
        c = self._cpu(romInit=[0x01])
        c.mmu.write(0x01, 0x01)
        c.r.a = 0x06
        c.r.setFlag(FlagBit.C)
        c.execute([0x07])
        self.assertEqual(c.mmu.read(0x01), 0x02)
        self.assertEqual(c.r.a, 0x06)

    def test_sxa(self):
        c = self._cpu(ram=(0, 0x400, False), romInit=[0xFF, 0x01])
        c.r.x = 0xFF
        c.r.y = 0x01
        c.execute([0x9E])

        self.assertEqual(c.mmu.read(0x200), 0x02)

    def test_sya(self):
        c = self._cpu(ram=(0, 0x400, False), romInit=[0xFF, 0x01])

        c.r.y = 0xFF
        c.r.x = 0x01
        c.execute([0x9C])

        self.assertEqual(c.mmu.read(0x200), 0x02)

    def test_xaa(self):
        c = self._cpu(romInit=[0xFF])

        c.r.a = 0b11111110
        c.r.x = 0b11101111

        c.execute([0x8B])

        self.assertEqual(c.r.a, 0b11101110)

    def test_xas(self):
        c = self._cpu(ram=(0, 0x400, False), romInit=[0xFF, 0x01])

        c.r.x = 0xFE
        c.r.a = 0x7F

        c.r.y = 0x01
        c.execute([0x9B])

        self.assertEqual(c.r.s, 0x7E)
        self.assertEqual(c.mmu.read(0x100), 0x02)

    def test_step(self):
        c = self._cpu(romInit=[0xA9, 0x55, 0x69, 0x22])
        c.step()
        self.assertEqual(c.r.a, 0x55)
        c.step()
        self.assertEqual(c.r.a, 0x77)

    def test_run_rom(self):
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "files",
            "test_load_file.bin"
        )

        with open(path, "rb") as f:
            c = self._cpu(romInit=f)

        c.step()
        self.assertEqual(c.r.a, 0x55)

    def test_cycle_counting(self):
        # Adapted from @InvalidCo's test in #7
        c = self._cpu(
            romInit=[
                0xA0,
                0x80,
                0xB9,
                0x00,
                0x00,
                0xB9,
                0x80,
                0x00,
                0x18,
                0xB0,
                0xFF,
                0x90,
                0x01,
                0x90,
                0x8F,
            ]
        )

        expected_cycles = [2, 4, 5, 2, 2, 3, 4]

        for expected_cycle in expected_cycles:
            with self.subTest(expected_cycle=expected_cycle):
                c.step()
                self.assertEqual(c.cc, expected_cycle, f"{c.op.opcode:0>2x}")

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()

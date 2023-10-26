#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_processor
----------------------------------

Processor tests for `py65emu` module.
"""


import unittest

from enum import Enum
from py65emu.cpu import CPU, FlagBit
from py65emu.mmu import MMU


class RegisterMode(Enum):
    """
    An enum helper, used when testing addressing modes for
    Comparison and Store operations
    """

    # CMP Operation
    Accumulator = 0xA9
    # CPX Operation
    XRegister = 0xA2
    # CPY Operation
    YRegister = 0xA0


class Processor(unittest.TestCase):
    c: CPU | None = None

    def setUp(self):
        pass

    def tearDown(self) -> None:
        # Debug.crash_dump(self.c)
        pass

    def _cpu(self, program=None, pc=0x0000) -> CPU:
        self.c = CPU(
            MMU(
                [
                    (0x0, 0x10000, False, program, pc),
                ]
            ),
            pc,
        )
        return self.c

    def get_stack_location(self, offset: int = 0x00) -> int:
        if self.c is None:
            return 0
        return (self.c.stack_page * 0x100) + offset

    def create(self, *args, **kwargs) -> CPU:
        """
        :param array | tuple data: Parameters to program
        :param array | tuple program: Program to run
        :param int | None condition: Instruction to prepend to program if set
        :param int pc: Program counter.
        """
        pc = 0x00
        program = []

        if len(args) > 0:
            if 'program' not in kwargs:
                kwargs['program'] = args[1]
        if 'program' in kwargs and kwargs['program'] is not None:
            program = kwargs['program']

        if 'condition' in kwargs and kwargs['condition'] is not None:
            program = [kwargs['condition']] + (
                program if isinstance(program, tuple) is False else [*program]
            )

        if 'pc' in kwargs and kwargs['pc'] is not None:
            pc = kwargs['pc']

        c = self._cpu(program, pc)
        self.assertEqual(c.r.pc, pc)
        return c


class Initialization(Processor):
    """ Initialization Tests
    """
    def test_Processor_Status_Flags_Initialized_Correctly(self):
        c = self._cpu()
        self.assertEqual(c.r.getFlag(FlagBit.C), False)
        self.assertEqual(c.r.getFlag(FlagBit.Z), False)
        self.assertEqual(c.r.getFlag(FlagBit.I), True)
        self.assertEqual(c.r.getFlag(FlagBit.D), False)
        self.assertEqual(c.r.getFlag(FlagBit.V), False)
        self.assertEqual(c.r.getFlag(FlagBit.N), False)

    def test_Processor_Status_Flags_Sets_And_Unsets_Correctly_Enum(self):
        subtests = [
            (FlagBit.C.name,),
            (FlagBit.Z.name,),
            (FlagBit.I.name,),
            (FlagBit.D.name,),
            (FlagBit.V.name,),
            (FlagBit.N.name,),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self._cpu()
                c.r.p = 0x00
                self.assertFalse(c.r.getFlag(data[0]))

                c.r.setFlag(data[0])
                self.assertTrue(c.r.getFlag(data[0]))

                c.r.clearFlag(data[0])
                self.assertFalse(c.r.getFlag(data[0]))

    def test_Processor_Status_Flags_Sets_And_Unsets_Correctly_String(self):
        subtests = [
            (FlagBit.C.value,),
            (FlagBit.Z.value,),
            (FlagBit.I.value,),
            (FlagBit.D.value,),
            (FlagBit.V.value,),
            (FlagBit.N.value,),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self._cpu()
                c.r.p = 0x00
                self.assertFalse(c.r.getFlag(data[0]))

                c.r.setFlag(data[0])
                self.assertTrue(c.r.getFlag(data[0]))

                c.r.clearFlag(data[0])
                self.assertFalse(c.r.getFlag(data[0]))

    def test_Processor_Status_Flags_Sets_And_Unsets_Correctly_Int(self):
        subtests = [
            (FlagBit.C,),
            (FlagBit.Z,),
            (FlagBit.I,),
            (FlagBit.D,),
            (FlagBit.V,),
            (FlagBit.N,),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self._cpu()
                c.r.p = 0x00
                self.assertFalse(c.r.getFlag(data[0]))

                c.r.setFlag(data[0])
                self.assertTrue(c.r.getFlag(data[0]))

                c.r.clearFlag(data[0])
                self.assertFalse(c.r.getFlag(data[0]))

    def test_Processor_Registers_Initialized_Correctly(self):
        c = self._cpu()
        self.assertEqual(c.r.a, 0)
        self.assertEqual(c.r.x, 0)
        self.assertEqual(c.r.y, 0)
        self.assertEqual(c.mmu.read(c.r.pc), 0)
        self.assertEqual(c.r.pc, 0)

    def test_ProgramCounter_Correct_When_Program_Loaded(self):
        c = self._cpu(pc=0x01)
        self.assertEqual(c.r.pc, 0x01)

    def test_ProgramCounter_Initializes_To_Default_Value_After_Reset(self):
        c = self._cpu()
        c.reset()
        self.assertEqual(c.r.pc, c.interrupts["RESET"])

    def test_Stack_Pointer_Initializes_To_Default_Value_After_Reset(self):
        c = self._cpu()
        c.reset()
        # self.assertEqual(c.r.s, 0xfd)
        self.assertEqual(c.r.s, 0xFF)


class InstructionADC(Processor):
    """ ADC - Add with Carry Tests
    """
    def create(self, *args, **kwargs) -> CPU:
        c = super().create(*args, **kwargs)

        if 'condition' in kwargs and kwargs['condition'] is not None:
            c.step()

        return c

    def test_Accumulator_Correct_When_Not_In_BCD_Mode(self):
        subtests = [
            (0x00, 0x00, False, 0x00),
            (0x00, 0x01, False, 0x01),
            (0x01, 0x02, False, 0x03),
            (0xFF, 0x01, False, 0x00),
            (0xFE, 0x01, False, 0xFF),
            (0xFF, 0x00, False, 0xFF),
            (0x00, 0x00, True, 0x01),
            (0x00, 0x01, True, 0x02),
            (0x01, 0x02, True, 0x04),
            (0xFE, 0x01, True, 0x00),
            (0xFD, 0x01, True, 0xFF),
            (0xFE, 0x00, True, 0xFF),
            (0xFF, 0xFF, True, 0xFF),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA9, data[0], 0x69, data[1]],
                    condition=(0x38 if data[2] is True else None)
                )

                c.step()
                self.assertEqual(c.r.a, data[0])

                c.step()
                self.assertEqual(c.r.a, data[3])

    def test_Accumulator_Correct_When_In_BCD_Mode(self):
        """BCD is a bit special.
        0x99 + 0x99 = 0x98, which equals: 99 + 99 = 198 = 98 (with carry)
        it's hex in dec form
        hex(99) = dec(99)
        """
        subtests = [
            (0x99, 0x99, False, 0x98),
            (0x99, 0x99, True, 0x99),
            (0x90, 0x99, False, 0x89),
            (0x01, 0x69, True, 0x71),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xF8, 0xA9, data[0], 0x69, data[1]],
                    condition=(0x38 if data[2] is True else None)
                )

                c.step()
                c.step()
                self.assertEqual(c.r.a, data[0])

                c.step()
                self.assertEqual(c.r.a, data[3])

    def test_Carry_Correct_When_Not_In_BCD_Mode(self):
        subtests = [
            (0xFE, 0x01, False, False),
            (0xFE, 0x01, True, True),
            (0xFD, 0x01, True, False),
            (0xFF, 0x01, False, True),
            (0xFF, 0x01, True, True),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA9, data[0], 0x69, data[1]],
                    condition=(0x38 if data[2] is True else None)
                )

                c.step()
                self.assertEqual(c.r.a, data[0])

                c.step()
                self.assertEqual(c.r.getFlag(FlagBit.C), data[3])

    def test_Carry_Correct_When_In_BCD_Mode(self):
        subtests = [
            (0x62, 0x01, False, False),
            (0x62, 0x01, True, False),
            (0x63, 0x01, False, False),
            (0x63, 0x01, True, False),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA9, data[0], 0x69, data[1]],
                    condition=(0x38 if data[2] is True else None)
                )

                c.step()
                self.assertEqual(c.r.a, data[0])

                c.step()
                self.assertEqual(c.r.getFlag(FlagBit.C), data[3])

    def test_Zero_Flag_Correct_When_Not_In_BCD_Mode(self):
        subtests = [
            (0x00, 0x00, True),
            (0xFF, 0x01, True),
            (0x00, 0x01, False),
            (0x01, 0x00, False),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x69, data[1]],)

                c.step()
                self.assertEqual(c.r.a, data[0])

                c.step()
                self.assertEqual(c.r.getFlag(FlagBit.Z), data[2])

    def test_Negative_Flag_Correct(self):
        subtests = [
            (0x7E, 0x01, False),
            (0x01, 0x7E, False),
            (0x01, 0x7F, True),
            (0x7F, 0x01, True),
            (0x01, 0xFE, True),
            (0xFE, 0x01, True),
            (0x01, 0xFF, False),
            (0xFF, 0x01, False),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x69, data[1]],)

                c.step()
                self.assertEqual(c.r.a, data[0])

                c.step()
                self.assertEqual(c.r.getFlag(FlagBit.N), data[2])

    def test_Overflow_Flag_Correct(self):
        subtests = [
            (0x00, 0x7F, False, False),
            (0x00, 0x80, False, False),
            (0x01, 0x7F, False, True),
            (0x01, 0x80, False, False),
            (0x7F, 0x01, False, True),
            (0x7F, 0x7F, False, True),
            (0x80, 0x7F, False, False),
            (0x80, 0x80, False, True),
            (0x80, 0x81, False, True),
            (0x80, 0xFF, False, True),
            (0xFF, 0x00, False, False),
            (0xFF, 0x01, False, False),
            (0xFF, 0x7F, False, False),
            (0xFF, 0x80, False, True),
            (0xFF, 0xFF, False, False),
            (0x00, 0x7F, True, True),
            (0x00, 0x80, True, False),
            (0x01, 0x7F, True, True),
            (0x01, 0x80, True, False),
            (0x7F, 0x01, True, True),
            (0x7F, 0x7F, True, True),
            (0x80, 0x7F, True, False),
            (0x80, 0x80, True, True),
            (0x80, 0x81, True, True),
            (0x80, 0xFF, True, False),
            (0xFF, 0x00, True, False),
            (0xFF, 0x01, True, False),
            (0xFF, 0x7F, True, False),
            (0xFF, 0x80, True, False),
            (0xFF, 0xFF, True, False),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA9, data[0], 0x69, data[1]],
                    condition=(0x38 if data[2] is True else None)
                )

                c.step()
                self.assertEqual(c.r.a, data[0])

                c.step()
                self.assertEqual(c.r.getFlag(FlagBit.V), data[3])


class InstructionAND(Processor):
    """ AND - Compare Memory with Accumulator
    """
    def test_Accumulator_Correct(self):
        subtests = [
            (0, 0, 0),
            (255, 255, 255),
            (255, 254, 254),
            (170, 85, 0),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x29, data[1]])

                c.step()
                c.step()
                self.assertEqual(c.r.a, data[2])


class InstructionASL(Processor):
    """ ASL - Arithmetic Shift Left
    """
    def test_Correct_Value_Stored(self):
        subtests = [
            (0x0A, 0x6D, 0xDA, 0x00),  # ASL Accumulator
            (0x0A, 0x6C, 0xD8, 0x00),  # ASL Accumulator
            (0x06, 0x6D, 0xDA, 0x01),  # ASL Zero Page
            (0x16, 0x6D, 0xDA, 0x01),  # ASL Zero Page X
            (0x0E, 0x6D, 0xDA, 0x01),  # ASL Absolute
            (0x1E, 0x6D, 0xDA, 0x01),  # ASL Absolute X
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, data[1], data[0], data[3]]
                c = self.create(program=p)

                c.step()
                self.assertEqual(c.r.a, data[1])

                c.step()

                if data[0] == 0x0A:
                    self.assertEqual(c.r.a, data[2])
                else:
                    self.assertEqual(c.mmu.read(data[3]), data[2])

    def test_Carry_Set_Correctly(self):
        subtests = [
            (0x7F, False),
            (0x80, True),
            (0xFF, True),
            (0x00, False),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x0A])

                c.step()
                self.assertEqual(c.r.a, data[0])

                c.step()
                self.assertEqual(c.r.getFlag(FlagBit.C), data[1])

    def test_Negative_Set_Correctly(self):
        subtests = [
            [0x3f, False],
            [0x40, True],
            [0x7f, True],
            [0x80, False],
            [0x00, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x0A])

                c.step()
                self.assertEqual(c.r.a, data[0])

                c.step()
                self.assertEqual(c.r.getFlag(FlagBit.N), data[1])

    def test_Zero_Set_Correctly(self):
        subtests = [
            [0x7f, False],
            [0x80, True],
            [0x00, True],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x0A])

                c.step()
                self.assertEqual(c.r.a, data[0])

                c.step()
                self.assertEqual(c.r.getFlag(FlagBit.Z), data[1])


class InstructionBCC(Processor):
    """ BCC - Branch On Carry Clear
    """
    def test_Program_Counter_Correct(self):
        subtests = [
            (0x00, 0x01, 0x03),
            (0x80, 0x80, 0x02),
            (0x00, 0x03, 0x05),
            (0x00, 0xFD, 0xFFFF),
            (0x7D, 0x80, 0xFFFF),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0x90, data[1]], pc=data[0])

                c.step()

                self.assertEqual(c.r.pc, data[2])


class InstructionBCS(Processor):
    """ BCS - Branch on Carry Set
    """
    def test_Program_Counter_Correct(self):
        subtests = [
            (0x00, 0x01, 0x04),
            (0x80, 0x80, 0x03),
            (0x00, 0xFC, 0xFFFF),
            (0x7C, 0x80, 0xFFFF),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0x38, 0xB0, data[1]],
                    pc=data[0]
                )

                c.step()
                c.step()

                self.assertEqual(c.r.pc, data[2])


class InstructionBEQ(Processor):
    """ BEQ - Branch on Zero Set
    """
    def test_Program_Counter_Correct(self):
        subtests = [
            (0x00, 0x01, 0x05),
            (0x80, 0x80, 0x04),
            (0x00, 0xFB, 0xFFFF),
            (0x7B, 0x80, 0xFFFF),
            (0x02, 0xFE, 0x04),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA9, 0x00, 0xF0, data[1]],
                    pc=data[0]
                )

                c.step()
                c.step()

                self.assertEqual(c.r.pc, data[2])


class InstructionBIT(Processor):
    """ BIT - Compare Memory with Accumulator
    """
    def test_Negative_Set_When_Comparison_Is_Negative_Number(self):
        subtests = [
            (0x24, 0x7F, 0x7F, False),  # BIT Zero Page
            (0x24, 0x80, 0x7F, False),  # BIT Zero Page
            (0x24, 0x7F, 0x80, True),   # BIT Zero Page
            (0x24, 0x80, 0xFF, True),   # BIT Zero Page
            (0x24, 0xFF, 0x80, True),   # BIT Zero Page
            (0x2C, 0x7F, 0x7F, False),  # BIT Absolute
            (0x2C, 0x80, 0x7F, False),  # BIT Absolute
            (0x2C, 0x7F, 0x80, True),   # BIT Absolute
            (0x2C, 0x80, 0xFF, True),   # BIT Absolute
            (0x2C, 0xFF, 0x80, True),   # BIT Absolute
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, data[1], data[0], 0x06, 0x00, 0x00, data[2]]
                c = self.create(program=p)

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[3])

    def test_Overflow_Set_By_Bit_Six(self):
        subtests = [
            (0x24, 0x3F, 0x3F, False),  # BIT Zero Page
            (0x24, 0x3F, 0x40, True),   # BIT Zero Page
            (0x24, 0x40, 0x3F, False),  # BIT Zero Page
            (0x24, 0x40, 0x7F, True),   # BIT Zero Page
            (0x24, 0x7F, 0x40, True),   # BIT Zero Page
            (0x24, 0x7F, 0x80, False),  # BIT Zero Page
            (0x24, 0x80, 0x7F, True),   # BIT Zero Page
            (0x24, 0xC0, 0xDF, True),   # BIT Zero Page
            (0x24, 0xDF, 0xC0, True),   # BIT Zero Page
            (0x24, 0x3F, 0x3F, False),  # BIT Zero Page
            (0x24, 0xC0, 0xFF, True),   # BIT Zero Page
            (0x24, 0xFF, 0xC0, True),   # BIT Zero Page
            (0x24, 0x40, 0xFF, True),   # BIT Zero Page
            (0x24, 0xFF, 0x40, True),   # BIT Zero Page
            (0x24, 0xC0, 0x7F, True),   # BIT Zero Page
            (0x24, 0x7F, 0xC0, True),   # BIT Zero Page
            (0x2C, 0x3F, 0x3F, False),  # BIT Absolute
            (0x2C, 0x3F, 0x40, True),   # BIT Absolute
            (0x2C, 0x40, 0x3F, False),  # BIT Absolute
            (0x2C, 0x40, 0x7F, True),   # BIT Absolute
            (0x2C, 0x7F, 0x40, True),   # BIT Absolute
            (0x2C, 0x7F, 0x80, False),  # BIT Absolute
            (0x2C, 0x80, 0x7F, True),   # BIT Absolute
            (0x2C, 0xC0, 0xDF, True),   # BIT Absolute
            (0x2C, 0xDF, 0xC0, True),   # BIT Absolute
            (0x2C, 0x3F, 0x3F, False),  # BIT Absolute
            (0x2C, 0xC0, 0xFF, True),   # BIT Absolute
            (0x2C, 0xFF, 0xC0, True),   # BIT Absolute
            (0x2C, 0x40, 0xFF, True),   # BIT Absolute
            (0x2C, 0xFF, 0x40, True),   # BIT Absolute
            (0x2C, 0xC0, 0x7F, True),   # BIT Absolute
            (0x2C, 0x7F, 0xC0, True),   # BIT Absolute
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, data[1], data[0], 0x06, 0x00, 0x00, data[2]]
                c = self.create(program=p)

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.V), data[3])

    def test_Zero_Set_When_Comparison_Is_Zero(self):
        subtests = [
            (0x24, 0x00, 0x00, True),   # BIT Zero Page
            (0x24, 0xFF, 0xFF, False),  # BIT Zero Page
            (0x24, 0xAA, 0x55, True),   # BIT Zero Page
            (0x24, 0x55, 0xAA, True),   # BIT Zero Page
            (0x2C, 0x00, 0x00, True),   # BIT Absolute
            (0x2C, 0xFF, 0xFF, False),  # BIT Absolute
            (0x2C, 0xAA, 0x55, True),   # BIT Absolute
            (0x2C, 0x55, 0xAA, True),   # BIT Absolute
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, data[1], data[0], 0x06, 0x00, 0x00, data[2]]
                c = self.create(program=p)

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[3])


class InstructionBMI(Processor):
    """ BMI - Branch if Negative Set
    """
    def test_Program_Counter_Correct(self):
        subtests = [
            (0x00, 0x01, 0x05),
            (0x80, 0x80, 0x04),
            (0x00, 0xFB, 0xFFFF),
            (0x7B, 0x80, 0xFFFF),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA9, 0x80, 0x30, data[1]],
                    pc=data[0]
                )

                c.step()
                c.step()

                self.assertEqual(c.r.pc, data[2])


class InstructionBNE(Processor):
    """ BNE - Branch On Result Not Zero
    """
    def test_Program_Counter_Correct(self):
        subtests = [
            (0x00, 0x01, 0x05),
            (0x80, 0x80, 0x04),
            (0x00, 0xFB, 0xFFFF),
            (0x7B, 0x80, 0xFFFF),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA9, 0x80, 0xD0, data[1]],
                    pc=data[0]
                )

                c.step()
                c.step()

                self.assertEqual(c.r.pc, data[2])


class InstructionBPL(Processor):
    """ BPL - Branch if Negative Clear
    """
    def test_Program_Counter_Correct(self):
        subtests = [
            (0x00, 0x01, 0x05),
            (0x80, 0x80, 0x04),
            (0x00, 0xFB, 0xFFFF),
            (0x7B, 0x80, 0xFFFF),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA9, 0x79, 0x10, data[1]],
                    pc=data[0]
                )

                c.step()
                c.step()

                self.assertEqual(c.r.pc, data[2])


class InstructionBRK(Processor):
    """ BRK - Simulate Interrupt Request (IRQ)
    """
    def test_Program_Counter_Set_To_Address_At_Break_Vector_Address(self):
        c = self.create(program=[0x00], pc=0x1000)

        c.mmu.write(0xFFFE, 0xBC)
        c.mmu.write(0xFFFF, 0xCD)

        self.assertEqual(c.r.pc, 0x1000)

        c.step()

        self.assertEqual(c.r.pc, 0xCDBC)

    def test_Program_Counter_Stack_Correct(self):
        c = self.create(program=[0x00, 0xFF], pc=0xABCD)

        stackLocation = c.r.s
        c.step()

        self.assertEqual(
            c.mmu.read(self.get_stack_location(stackLocation)),
            0xAB
        )
        self.assertEqual(
            c.mmu.read(self.get_stack_location(stackLocation - 1)),
            0xCF
        )

    def test_Stack_Pointer_Correct(self):
        c = self.create(program=[0x00], pc=0xABCD)

        stackLocation = c.r.s
        c.step()

        self.assertEqual(c.r.s, stackLocation - 3)

    def test_Stack_Set_Flag_Operations_Correctly(self):
        subtests = [
            (0x38, 0x31),  # SEC Carry Flag Test
            (0xF8, 0x38),  # SED Decimal Flag Test
            (0x78, 0x34),  # SEI Interrupt Flag Test
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0x58, data[0], 0x00]
                c = self.create(program=p)

                stackLocation = c.r.s
                c.step()
                c.step()
                c.step()

                # Accounting for the Offset in memory
                self.assertEqual(
                    c.mmu.read(self.get_stack_location(stackLocation - 2)),
                    data[1]
                )

    def test_Stack_Non_Set_Flag_Operations_Correctly(self):
        subtests = [
            (0x01, 0x80, 0xB0),  # Negative
            (0x01, 0x7F, 0xF0),  # Overflow + Negative
            (0x00, 0x00, 0x32),  # Zero
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0x58, 0xA9, data[0], 0x69, data[1], 0x00]
                c = self.create(program=p)

                stackLocation = c.r.s
                c.step()
                c.step()
                c.step()
                c.step()

                # Accounting for the Offset in memory
                self.assertEqual(
                    c.mmu.read(self.get_stack_location(stackLocation - 2)),
                    data[2]
                )


class InstructionBVC(Processor):
    """ BVC - Branch if Overflow Clear
    """
    def test_Program_Counter_Correct(self):
        subtests = [
            (0x00, 0x01, 0x03),
            (0x80, 0x80, 0x02),
            (0x00, 0xFD, 0xFFFF),
            (0x7D, 0x80, 0xFFFF),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0x50, data[1]], pc=data[0])

                c.step()
                self.assertEqual(c.r.pc, data[2])


class InstructionBVS(Processor):
    """ BVS - Branch if Overflow Set
    """
    def test_Program_Counter_Correct(self):
        subtests = [
            [0x00, 0x01, 0x07],
            [0x80, 0x80, 0x06],
            [0x00, 0xF9, 0xFFFF],
            [0x79, 0x80, 0xFFFF],
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, 0x01, 0x69, 0x7F, 0x70, data[1]]
                c = self.create(program=p, pc=data[0])

                c.step()
                c.step()
                c.step()

                self.assertEqual(c.r.pc, data[2])


class InstructionCLC(Processor):
    """ CLC - Clear Carry Flag
    """
    def test_Carry_Flag_Cleared_Correctly(self):
        c = self.create(program=[0x18])

        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.C), False)


class InstructionCLD(Processor):
    """ CLD - Clear Decimal Flag
    """
    def test_Carry_Flag_Set_And_Cleared_Correctly(self):
        c = self.create(program=[0xF8, 0xD8])

        c.step()
        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.D), False)


class InstructionCLI(Processor):
    """ CLI - Clear Interrupt Flag
    """
    def test_Interrup_Flag_Cleared_Correctly(self):
        c = self.create(program=[0x58])

        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.I), False)


class InstructionCLV(Processor):
    """ CLV - Clear Overflow Flag
    """
    def test_Overflow_Flag_Cleared_Correctly(self):
        c = self.create(program=[0xB8])

        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.V), False)


class InstructionCMP(Processor):
    """ CMP - Compare Memory With Accumulator
    """
    def test_Zero_Flag_Set_When_Values_Match(self):
        subtests = [
            (0x00, 0x00, True),
            (0xFF, 0x00, False),
            (0x00, 0xFF, False),
            (0xFF, 0xFF, True),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0xC9, data[1]])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[2])

    def test_Carry_Flag_Set_When_Accumulator_Is_Greater_Than_Or_Equal(
        self
    ):
        subtests = [
            (0x00, 0x00, True),
            (0xFF, 0x00, True),
            (0x00, 0xFF, False),
            (0x00, 0x01, False),
            (0xFF, 0xFF, True),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0xC9, data[1]])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.C), data[2])

    def test_Negative_Flag_Set_When_Result_Is_Negative(self):
        subtests = [
            [0xFE, 0xFF, True],
            [0x81, 0x1, True],
            [0x81, 0x2, False],
            [0x79, 0x1, False],
            [0x00, 0x1, True],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0xC9, data[1]])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[2])


class InstructionCPX(Processor):
    """ CPX - Compare Memory With X Register
    """
    def test_Zero_Flag_Set_When_Values_Match(self):
        subtests = [
            (0x00, 0x00, True),
            (0xFF, 0x00, False),
            (0x00, 0xFF, False),
            (0xFF, 0xFF, True),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA2, data[0], 0xE0, data[1]])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[2])

    def test_Carry_Flag_Set_When_Accumulator_Is_Greater_Than_Or_Equal(
        self
    ):
        subtests = [
            (0x00, 0x00, True),
            (0xFF, 0x00, True),
            (0x00, 0xFF, False),
            (0x00, 0x01, False),
            (0xFF, 0xFF, True),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA2, data[0], 0xE0, data[1]])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.C), data[2])

    def test_Negative_Flag_Set_When_Result_Is_Negative(self):
        subtests = [
            (0xFE, 0xFF, True),
            (0x81, 0x1, True),
            (0x81, 0x2, False),
            (0x79, 0x1, False),
            (0x00, 0x1, True),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA2, data[0], 0xE0, data[1]])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[2])


class InstructionCPY(Processor):
    """ CPY - Compare Memory With Y Register
    """
    def test_Zero_Flag_Set_When_Values_Match(self):
        subtests = [
            (0x00, 0x00, True),
            (0xFF, 0x00, False),
            (0x00, 0xFF, False),
            (0xFF, 0xFF, True),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA0, data[0], 0xC0, data[1]])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[2])

    def test_Carry_Flag_Set_When_Accumulator_Is_Greater_Than_Or_Equal(self):
        subtests = [
            (0x00, 0x00, True),
            (0xFF, 0x00, True),
            (0x00, 0xFF, False),
            (0x00, 0x01, False),
            (0xFF, 0xFF, True),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA0, data[0], 0xC0, data[1]])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.C), data[2])

    def test_Negative_Flag_Set_When_Result_Is_Negative(self):
        subtests = [
            (0xFE, 0xFF, True),
            (0x81, 0x1, True),
            (0x81, 0x2, False),
            (0x79, 0x1, False),
            (0x00, 0x1, True),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA0, data[0], 0xC0, data[1]])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[2])


class InstructionDEC(Processor):
    """ DEC - Decrement Memory by One
    """
    def test_Memory_Has_Correct_Value(self):
        subtests = [
            (0x00, 0xFF),
            (0xFF, 0xFE),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xC6, 0x03, 0x00, data[0]])

                c.step()

                self.assertEqual(c.mmu.read(0x03), data[1])

    def test_Zero_Has_Correct_Value(self):
        subtests = [
            (0x00, False),
            (0x01, True),
            (0x02, False),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xC6, 0x03, 0x00, data[0]])

                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[1])

    def test_Negative_Has_Correct_Value(self):
        subtests = [
            (0x80, False),
            (0x81, True),
            (0x00, True),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xC6, 0x03, 0x00, data[0]])

                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[1])


class InstructionDEX(Processor):
    """ DEX - Decrement X by One
    """
    def test_XRegister_Has_Correct_Value(self):
        subtests = [
            [0x00, 0xFF],
            [0xFF, 0xFE],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA2, data[0], 0xCA])

                c.step()
                c.step()

                self.assertEqual(c.r.x, data[1])

    def test_Zero_Has_Correct_Value(self):
        subtests = [
            [0x00, False],
            [0x01, True],
            [0x02, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA2, data[0], 0xCA])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[1])

    def test_Negative_Has_Correct_Value(self):
        subtests = [
            [0x80, False],
            [0x81, True],
            [0x00, True],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA2, data[0], 0xCA])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[1])


class InstructionDEY(Processor):
    """ DEY - Decrement Y by One
    """
    def test_YRegister_Has_Correct_Value(self):
        subtests = [
            [0x00, 0xFF],
            [0xFF, 0xFE],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA0, data[0], 0x88])

                c.step()
                c.step()

                self.assertEqual(c.r.y, data[1])

    def test_Zero_Has_Correct_Value(self):
        subtests = [
            [0x00, False],
            [0x01, True],
            [0x02, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA0, data[0], 0x88])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[1])

    def test_Negative_Has_Correct_Value(self):
        subtests = [
            [0x80, False],
            [0x81, True],
            [0x00, True],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA0, data[0], 0x88])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[1])


class InstructionEOR(Processor):
    """ EOR - Exclusive OR Compare Accumulator With Memory
    """
    def test_Accumulator_Correct(self):
        subtests = [
            [0x00, 0x00, 0x00],
            [0xFF, 0x00, 0xFF],
            [0x00, 0xFF, 0xFF],
            [0x55, 0xAA, 0xFF],
            [0xFF, 0xFF, 0x00],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x49, data[1]])

                c.step()
                c.step()

                self.assertEqual(c.r.a, data[2])

    def test_Negative_Flag_Correct(self):
        subtests = [
            [0xFF, 0xFF, False],
            [0x80, 0x7F, True],
            [0x40, 0x3F, False],
            [0xFF, 0x7F, True],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x49, data[1]])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[2])

    def test_Zero_Flag_Correct(self):
        subtests = [
            [0xFF, 0xFF, True],
            [0x80, 0x7F, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x49, data[1]])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[2])


class InstructionINC(Processor):
    """ INC - Increment Memory by One
    """
    def test_Memory_Has_Correct_Value(self):
        subtests = [
            [0x00, 0x01],
            [0xFF, 0x00],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xE6, 0x03, 0x00, data[0]])

                c.step()

                self.assertEqual(c.mmu.read(0x03), data[1])

    def test_Zero_Has_Correct_Value(self):
        subtests = [
            [0x00, False],
            [0xFF, True],
            [0xFE, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xE6, 0x03, 0x00, data[0]])

                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[1])

    def test_Negative_Has_Correct_Value(self):
        subtests = [
            [0x78, False],
            [0x80, True],
            [0x00, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xE6, 0x03, 0x00, data[0]])

                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[1])


class InstructionINX(Processor):
    """ INX - Increment X by One
    """
    def test_XRegister_Has_Correct_Value(self):
        subtests = [
            [0x00, 0x01],
            [0xFF, 0x00],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA2, data[0], 0xE8])

                c.step()
                c.step()

                self.assertEqual(c.r.x, data[1])

    def test_Zero_Has_Correct_Value(self):
        subtests = [
            [0x00, False],
            [0xFF, True],
            [0xFE, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA2, data[0], 0xE8])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[1])

    def test_Negative_Has_Correct_Value(self):
        subtests = [
            [0x78, False],
            [0x80, True],
            [0x00, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA2, data[0], 0xE8])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[1])


class InstructionINY(Processor):
    """ INY - Increment Y by One
    """
    def test_YRegisgter_Has_Correct_Value(self):
        subtests = [
            [0x00, 0x01],
            [0xFF, 0x00],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA0, data[0], 0xC8])

                c.step()
                c.step()

                self.assertEqual(c.r.y, data[1])

    def test_Zero_Has_Correct_Value(self):
        subtests = [
            [0x00, False],
            [0xFF, True],
            [0xFE, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA0, data[0], 0xC8])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[1])

    def test_Negative_Has_Correct_Value(self):
        subtests = [
            [0x78, False],
            [0x80, True],
            [0x00, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA0, data[0], 0xC8])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[1])


class InstructionJMP(Processor):
    """ JMP - Jump to New Location
    """
    def test_Program_Counter_Set_Correctly_After_Jump(self):
        c = self.create(program=[0x4C, 0x08, 0x00])

        c.step()

        self.assertEqual(c.r.pc, 0x08)

    def test_Program_Counter_Set_Correctly_After_Indirect_Jump(self):
        c = self.create(program=[0x6C, 0x03, 0x00, 0x08, 0x00])

        c.step()

        self.assertEqual(c.r.pc, 0x08)

    def test_Indirect_Wraps_Correct_If_MSB_IS_FF(self):
        c = self.create(program=[0x6C, 0xFF, 0x01, 0x08, 0x00])
        c.mmu.write(0x01FE, 0x6C)
        c.step()
        c.mmu.write(0x01FF, 0x03)
        c.mmu.write(0x0100, 0x02)
        c.step()

        self.assertEqual(c.r.pc, 0x0203)


class InstructionJSR(Processor):
    """ JSR - Jump to SubRoutine
    """
    def test_Stack_Loads_Correct_Value(self):
        c = self.create(program=[0x20, 0xCC, 0xCC], pc=0xBBAA)

        stackLocation = c.r.s
        c.step()

        self.assertEqual(
            c.mmu.read(self.get_stack_location(stackLocation)),
            0xBB
        )
        self.assertEqual(
            c.mmu.read(self.get_stack_location(stackLocation - 1)),
            0xAC
        )

    def test_Program_Counter_Correct(self):
        c = self.create(program=[0x20, 0xCC, 0xCC])

        c.step()

        self.assertEqual(c.r.pc, 0xCCCC)

    def test_Stack_Pointer_Correct(self):
        c = self.create(program=[0x20, 0xCC, 0xCC])

        stackLocation = c.r.s
        c.step()

        self.assertEqual(c.r.s, stackLocation - 2)


class InstructionLDA(Processor):
    """ LDA - Load Accumulator with Memory
    """
    def test_Accumulator_Has_Correct_Value(self):
        subtests = [
            (0x03,),
            (0x91,),
            (0x01,),
            (0xFF,),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0]])

                c.step()

                self.assertEqual(c.r.a, data[0])

    def test_Zero_Set_Correctly(self):
        subtests = [
            [0x00, True],
            [0x03, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0]])

                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[1])

    def test_Negative_Set_Correctly(self):
        subtests = [
            [0x00, False],
            [0x79, False],
            [0x80, True],
            [0xFF, True],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0]])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[1])


class InstructionLDX(Processor):
    """ LDX - Load X with Memory
    """
    def test_XRegister_Value_Has_Correct_Value(self):
        subtests = [
            (0x00,),
            (0x79,),
            (0x80,),
            (0xFF,),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA2, data[0]])

                c.step()

                self.assertEqual(c.r.x, data[0])

    def test_Negative_Flag_Set_Correctly(self):
        subtests = [
            [0x00, False],
            [0x79, False],
            [0x80, True],
            [0xFF, True],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA2, data[0]])

                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[1])

    def test_Zero_Set_Correctly(self):
        subtests = [
            [0x0, True],
            [0x3, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA2, data[0]])

                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[1])


class InstructionLDY(Processor):
    """ LDY - Load Y with Memory
    """
    def test_YRegister_Value_Has_Correct_Value(self):
        subtests = [
            (0x00,),
            (0x79,),
            (0x80,),
            (0xFF,),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA0, data[0]])

                c.step()

                self.assertEqual(c.r.y, data[0])

    def test_Negative_Flag_Set_Correctly(self):
        subtests = [
            [0x00, False],
            [0x79, False],
            [0x80, True],
            [0xFF, True],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA0, data[0]])

                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[1])

    def test_Zero_Set_Correctly(self):
        subtests = [
            [0x00, True],
            [0x03, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA0, data[0]])

                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[1])


class InstructionLSR(Processor):
    """ LSR - Logical Shift Right
    """
    def test_Negative_Set_Correctly(self):
        subtests = [
            [0xFF, False, False],
            [0xFE, False, False],
            [0xFF, True, False],
            [0x00, True, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA9, data[0], 0x4A],
                    condition=(0x38 if data[1] is True else 0x18)
                )

                c.step()
                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[2])

    def test_Zero_Set_Correctly(self):
        subtests = [
            [0x1, True],
            [0x2, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x4A])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[1])

    def test_Carry_Flag_Set_Correctly(self):
        subtests = [
            [0x1, True],
            [0x2, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x4A])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.C), data[1])

    def test_Correct_Value_Stored(self):
        subtests = [
            [0x4A, 0xFF, 0x7F, 0x00],  # LSR Accumulator
            [0x4A, 0xFD, 0x7E, 0x00],  # LSR Accumulator
            [0x46, 0xFF, 0x7F, 0x01],  # LSR Zero Page
            [0x56, 0xFF, 0x7F, 0x01],  # LSR Zero Page X
            [0x4E, 0xFF, 0x7F, 0x01],  # LSR Absolute
            [0x5E, 0xFF, 0x7F, 0x01],  # LSR Absolute X
        ]
        for data in subtests:
            with self.subTest(data=data):
                program = [0xA9, data[1], data[0], data[3]]
                c = self.create(data, program=program)

                c.step()
                c.step()

                if data[0] == 0x4A:
                    self.assertEqual(c.r.a, data[2])
                else:
                    self.assertEqual(c.mmu.read(data[3]), data[2])


class InstructionORA(Processor):
    """ ORA - Bitwise OR Compare Memory with Accumulator
    """
    def test_Accumulator_Correct(self):
        subtests = [
            [0x00, 0x00, 0x00],
            [0xFF, 0xFF, 0xFF],
            [0x55, 0xAA, 0xFF],
            [0xAA, 0x55, 0xFF],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x09, data[1]])

                c.step()
                c.step()

                self.assertEqual(c.r.a, data[2])

    def test_Zero_Flag_Correct(self):
        subtests = [
            [0x00, 0x00, True],
            [0xFF, 0xFF, False],
            [0x00, 0x01, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x09, data[1]])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[2])

    def test_Negative_Flag_Correct(self):
        subtests = [
            [0x7F, 0x80, True],
            [0x79, 0x00, False],
            [0xFF, 0xFF, True],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x09, data[1]])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[2])


class InstructionPHA(Processor):
    """ PHA - Push Accumulator Onto Stack
    """
    def test_Stack_Has_Correct_Value(self):
        c = self.create(program=[0xA9, 0x03, 0x48])
        stackLocation = c.r.s

        c.step()
        c.step()

        # Accounting for the Offset in memory
        self.assertEqual(
            c.mmu.read(self.get_stack_location(stackLocation)),
            0x03
        )

    def test_Stack_Pointer_Has_Correct_Value(self):
        c = self.create(program=[0xA9, 0x03, 0x48])
        stackLocation = c.r.s

        c.step()
        c.step()

        # A Push will decrement the Pointer by 1
        self.assertEqual(c.r.s, stackLocation - 1)

    def test_Stack_Pointer_Has_Correct_Value_When_Wrapping(self):
        c = self.create(program=[0x9A, 0x48])
        stackLocation = c.r.s

        c.step()
        c.step()

        self.assertEqual(c.r.s, stackLocation)


class InstructionPHP(Processor):
    """ PHP - Push Flags Onto Stack
    """
    def test_Stack_Set_Flag_Operations_Correctly(self):
        subtests = [
            [0x38, 0x31],  # SEC Carry Flag Test
            [0xF8, 0x38],  # SED Decimal Flag Test
            [0x78, 0x34],  # SEI Interrupt Flag Test
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0x58, data[0], 0x08])
                stackLocation = c.r.s

                c.step()
                c.step()
                c.step()

                self.assertEqual(
                    c.mmu.read(self.get_stack_location(stackLocation)),
                    data[1]
                )

    def test_Stack_Non_Set_Flag_Operations_Correctly(self):
        subtests = [
            [0x01, 0x80, 0xB0],  # Negative
            [0x01, 0x7F, 0xF0],  # Overflow + Negative
            [0x00, 0x00, 0x32],  # Zero
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0x58, 0xA9, data[0], 0x69, data[1], 0x08]
                c = self.create(data, program=p)

                stackLocation = c.r.s
                c.step()
                c.step()
                c.step()
                c.step()

                self.assertEqual(
                    c.mmu.read(self.get_stack_location(stackLocation)),
                    data[2]
                )

    def test_Stack_Pointer_Has_Correct_Value(self):
        c = self.create(program=[0x08])
        stackLocation = c.r.s

        c.step()

        # A Push will decrement the Pointer by 1
        self.assertEqual(c.r.s, stackLocation - 1)


class InstructionPLA(Processor):
    """ PLA - Pull From Stack to Accumulator
    """
    def test_Accumulator_Has_Correct_Value(self):
        subtests = [
            [0x03,]
        ]
        for data in subtests:
            with self.subTest(data=data):
                program = [0xA9, 0x03, 0x48, 0xA9, 0x00, 0x68]
                c = self.create(data, program=program)

                c.step()
                c.step()
                c.step()
                c.step()

                self.assertEqual(c.r.a, data[0])

    def test_Zero_Flag_Has_Correct_Value(self):
        subtests = [
            [0x00, True],
            [0x01, False],
            [0xFF, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x48, 0x68])

                c.step()
                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[1])

    def test_Negative_Flag_Has_Correct_Value(self):
        subtests = [
            [0x7F, False],
            [0x80, True],
            [0xFF, True],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x48, 0x68])

                c.step()
                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[1])


class InstructionPLP(Processor):
    """ PLP - Pull From Stack to Flags
    """
    def test_Carry_Flag_Set_Correctly(self):
        c = self._cpu(program=[0xA9, 0x01, 0x48, 0x28])

        c.step()
        c.step()
        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.C), True)

    def test_Zero_Flag_Set_Correctly(self):
        c = self._cpu(program=[0xA9, 0x02, 0x48, 0x28])

        c.step()
        c.step()
        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.Z), True)

    def test_Decimal_Flag_Set_Correctly(self):
        c = self._cpu(program=[0xA9, 0x08, 0x48, 0x28])

        c.step()
        c.step()
        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.D), True)

    def test_Interrupt_Flag_Set_Correctly(self):
        c = self._cpu(program=[0xA9, 0x04, 0x48, 0x28])

        c.step()
        c.step()
        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.I), True)

    def test_Overflow_Flag_Set_Correctly(self):
        c = self._cpu(program=[0xA9, 0x40, 0x48, 0x28])

        c.step()
        c.step()
        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.V), True)

    def test_Negative_Flag_Set_Correctly(self):
        c = self._cpu(program=[0xA9, 0x80, 0x48, 0x28])

        c.step()
        c.step()
        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.N), True)


class InstructionROL(Processor):
    """ ROL - Rotate Left
    """
    def test_Negative_Set_Correctly(self):
        subtests = [
            [0x40, True],
            [0x3F, False],
            [0x80, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[data[0], 0x2A], condition=0xA9)

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[1])

    def test_Zero_Set_Correctly(self):
        subtests = [
            [0x38, False],
            [0x18, True],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[data[0], 0x2A])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[1])

    def test_Carry_Flag_Set_Correctly(self):
        subtests = [
            [0x80, True],
            [0x7F, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[data[0], 0x2A], condition=0xA9)

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.C), data[1])

    def test_Correct_Value_Stored(self):
        subtests = [
            [0x2A, 0x55, 0xAA, 0x00],  # ROL Accumulator
            [0x2A, 0x55, 0xAA, 0x00],  # ROL Accumulator
            [0x26, 0x55, 0xAA, 0x01],  # ROL Zero Page
            [0x36, 0x55, 0xAA, 0x01],  # ROL Zero Page X
            [0x2E, 0x55, 0xAA, 0x01],  # ROL Absolute
            [0x3E, 0x55, 0xAA, 0x01],  # ROL Absolute X
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, data[1], data[0], data[3]]
                c = self.create(program=p)

                c.step()
                c.step()

                if data[0] == 0x2A:
                    self.assertEqual(c.r.a, data[2])
                else:
                    self.assertEqual(
                        c.mmu.read(data[3]),
                        data[2]
                    )


class InstructionROR(Processor):
    """ ROR - Rotate Left
    """
    def test_Negative_Set_Correctly(self):
        subtests = [
            [0xFF, 0x18, False],
            [0xFE, 0x18, False],
            [0xFF, 0x38, True],
            [0x00, 0x38, True],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA9, data[0], 0x6A],
                    condition=data[1]
                )

                c.step()
                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[2])

    def test_Zero_Set_Correctly(self):
        subtests = [
            [0x00, 0x18, True],
            [0x00, 0x38, False],
            [0x01, 0x18, True],
            [0x01, 0x38, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA9, data[0], 0x6A],
                    condition=data[1]
                )

                c.step()
                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[2])

    def test_Carry_Flag_Set_Correctly(self):
        subtests = [
            [0x01, True],
            [0x02, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0x6A])

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.C), data[1])

    def test_Correct_Value_Stored(self):
        subtests = [
            [0x6A, 0xAA, 0x55, 0x00],  # ROR Accumulator
            [0x6A, 0xAA, 0x55, 0x00],  # ROR Accumulator
            [0x66, 0xAA, 0x55, 0x01],  # ROR Zero Page
            [0x76, 0xAA, 0x55, 0x01],  # ROR Zero Page X
            [0x6E, 0xAA, 0x55, 0x01],  # ROR Absolute
            [0x7E, 0xAA, 0x55, 0x01],  # ROR Absolute X
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, data[1], data[0], data[3]]
                c = self.create(data, program=p)

                c.step()
                c.step()

                if data[0] == 0x6A:
                    self.assertEqual(c.r.a, data[2])
                else:
                    self.assertEqual(
                        c.mmu.read(data[3]),
                        data[2]
                    )


class InstructionRTI(Processor):
    """ RTI - Return from Interrupt
    """
    def test_Program_Counter_Correct(self):
        c = self.create(pc=0xABCD)

        # The Reset Vector Points to 0x0000 by default,
        # so load the RTI instruction there.
        c.mmu.write(0x00, 0x40)
        self.assertEqual(c.r.pc, 0xABCD)

        c.step()
        c.step()

        self.assertEqual(c.r.pc, 0xABCF)

    def test_Carry_Flag_Set_Correctly(self):
        p = [0xA9, FlagBit.C.value, 0x48, 0x40]
        c = self.create(program=p)

        c.step()
        c.step()
        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.C), True)

    def test_Zero_Flag_Set_Correctly(self):
        p = [0xA9, FlagBit.Z.value, 0x48, 0x40]
        c = self.create(program=p)

        c.step()
        c.step()
        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.Z), True)

    def test_Decimal_Flag_Set_Correctly(self):
        p = [0xA9, FlagBit.D.value, 0x48, 0x40]
        c = self.create(program=p)

        c.step()
        c.step()
        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.D), True)

    def test_Interrupt_Flag_Set_Correctly(self):
        p = [0xA9, FlagBit.I.value, 0x48, 0x40]
        c = self.create(program=p)

        c.step()
        c.step()
        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.I), True)

    def test_Overflow_Flag_Set_Correctly(self):
        p = [0xA9, FlagBit.V.value, 0x48, 0x40]
        c = self.create(program=p)

        c.step()
        c.step()
        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.V), True)

    def test_Negative_Flag_Set_Correctly(self):
        p = [0xA9, FlagBit.N.value, 0x48, 0x40]
        c = self.create(program=p)

        c.step()
        c.step()
        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.N), True)


class InstructionRTS(Processor):
    """ RTS - Return from SubRoutine
    """
    def test_Program_Counter_Has_Correct_Value(self):
        c = self._cpu(program=[0x20, 0x04, 0x00, 0x00, 0x60])

        c.step()
        c.step()

        self.assertEqual(c.r.pc, 0x03)

    def test_Stack_Pointer_Has_Correct_Value(self):
        c = self._cpu()
        c.mmu.write(0xBBAA, 0x60)
        c.r.pc = 0xBBAA

        stackLocation = c.r.s
        c.step()

        self.assertEqual(c.r.s, (stackLocation + 2) & 0xFF)


class InstructionSBC(Processor):
    """ SBC - Subtraction With Borrow
    """
    def create(self, *args, **kwargs) -> CPU:
        c = super().create(*args, **kwargs)

        if 'condition' in kwargs and kwargs['condition'] is not None:
            c.step()

        return c

    def test_Accumulator_Correct_When_Not_In_BCD_Mode(self):
        subtests = [
            (0x00, 0x00, False, 0xFF),
            (0x00, 0x00, True, 0x00),
            (0x50, 0xF0, False, 0x5F),
            (0x50, 0xB0, True, 0xA0),
            (0xFF, 0xFF, False, 0xFF),
            (0xFF, 0xFF, True, 0x00),
            (0xFF, 0x80, False, 0x7E),
            (0xFF, 0x80, True, 0x7F),
            (0x80, 0xFF, False, 0x80),
            (0x80, 0xFF, True, 0x81),
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA9, data[0], 0xE9, data[1]],
                    condition=(0x38 if data[2] is True else None)
                )

                self.assertEqual(c.r.a, 0x00)

                c.step()
                self.assertEqual(
                    c.r.a,
                    data[0],
                    f"0x{c.r.a:0>2x} != 0x{data[0]:0>2x}"
                )

                c.step()
                self.assertEqual(
                    c.r.a,
                    data[3],
                    f"0x{c.r.a:0>2x} != 0x{data[3]:0>2x}"
                )

    def test_Accumulator_Correct_When_In_BCD_Mode(self):
        """BCD is a bit special.
        0x99 + 0x99 = 0x98, which equals: 99 + 99 = 198 = 98 (with carry)
        it's hex in dec form
        hex(99) = dec(99)
        """
        subtests = [
            [0x00, 0x99, False, 0x00],
            [0x00, 0x99, True, 0x01],
            [0x45, 0x12, False, 0x32],
            [0x99, 0x99, False, 0x99],
            [0x99, 0x99, True, 0x00],
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xF8, 0xA9, data[0], 0xE9, data[1]]
                c = self.create(
                    data,
                    program=p,
                    condition=(0x38 if data[2] is True else None)
                )

                self.assertEqual(
                    c.r.a,
                    0x00,
                    f"Before: 0x{c.r.a:0>2x} != 0x00"
                )

                c.step()
                c.step()
                self.assertEqual(
                    c.r.a,
                    data[0],
                    f"Initial: 0x{c.r.a:0>2x} != 0x{data[0]:0>2x}"
                )

                c.step()
                self.assertEqual(
                    c.r.a,
                    data[3],
                    f"Expected: 0x{c.r.a:0>2x} != 0x{data[3]:0>2x}"
                )

    def test_Overflow_Correct_When_Not_In_BCD_Mode(self):
        subtests = [
            [0xFF, 0x01, False, False],
            [0xFF, 0x00, False, False],
            [0x80, 0x00, False, True],
            [0x80, 0x00, True, False],
            [0x81, 0x01, False, True],
            [0x81, 0x01, True, False],
            [0x00, 0x80, False, False],
            [0x00, 0x80, True, True],
            [0x01, 0x80, True, True],
            [0x01, 0x7F, False, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA9, data[0], 0xE9, data[1]],
                    condition=(0x38 if data[2] is True else None)
                )

                self.assertEqual(c.r.a, 0x00)

                c.step()
                self.assertEqual(
                    c.r.a,
                    data[0],
                    f"0x{c.r.a:0>2x} != 0x{data[0]:0>2x}"
                )

                c.step()
                self.assertEqual(c.r.getFlag(FlagBit.V), data[3])

    def test_Overflow_Correct_When_In_BCD_Mode(self):
        subtests = [
            [0x99, 0x01, False, False],
            [0x99, 0x00, False, False],
            # [0, 1, False, True],
            # [1, 1, True, True],
            # [2, 1, True, False],
            # [1, 1, False, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0xE9, data[1]])

                self.assertEqual(c.r.a, 0x00)

                c.step()
                self.assertEqual(
                    c.r.a,
                    data[0],
                    f"0x{c.r.a:0>2x} != 0x{data[0]:0>2x}"
                )

                c.step()
                self.assertEqual(c.r.getFlag(FlagBit.V), data[2])

    def test_Carry_Correct(self):
        subtests = [
            [0x00, 0x00, False],
            [0x00, 0x01, False],
            [0x01, 0x00, True],
            [0x02, 0x01, True],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0xE9, data[1]])

                self.assertEqual(c.r.a, 0x00)

                c.step()
                self.assertEqual(
                    c.r.a,
                    data[0],
                    f"0x{c.r.a:0>2x} != 0x{data[0]:0>2x}"
                )

                c.step()
                self.assertEqual(c.r.getFlag(FlagBit.C), data[2])

    def test_Zero_Correct(self):
        subtests = [
            [0x00, 0x00, False],
            [0x00, 0x01, False],
            [0x01, 0x00, True],
            [0x01, 0x01, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0xE9, data[1]])

                self.assertEqual(c.r.a, 0x00)

                c.step()
                self.assertEqual(
                    c.r.a,
                    data[0],
                    f"0x{c.r.a:0>2x} != 0x{data[0]:0>2x}"
                )

                c.step()
                self.assertEqual(c.r.getFlag(FlagBit.Z), data[2])

    def test_Negative_Correct(self):
        subtests = [
            [0x80, 0x01, False],
            [0x81, 0x01, False],
            [0x00, 0x01, True],
            [0x01, 0x01, True],
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[0xA9, data[0], 0xE9, data[1]])

                self.assertEqual(c.r.a, 0x00)

                c.step()
                self.assertEqual(
                    c.r.a,
                    data[0],
                    f"0x{c.r.a:0>2x} != 0x{data[0]:0>2x}"
                )

                c.step()
                self.assertEqual(c.r.getFlag(FlagBit.N), data[2])


class InstructionSEC(Processor):
    """ SEC - Set Carry Flag
    """
    def test_Carry_Flag_Set_Correctly(self):
        c = self.create(program=[0x38])

        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.C), True)


class InstructionSED(Processor):
    """ SED - Set Decimal Mode
    """
    def test_Decimal_Mode_Set_Correctly(self):
        c = self.create(program=[0xF8])

        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.D), True)


class InstructionSEI(Processor):
    """ SEI - Set Interrup Flag
    """
    def test_Interrupt_Flag_Set_Correctly(self):
        c = self.create(program=[0x78])

        c.step()

        self.assertEqual(c.r.getFlag(FlagBit.I), True)


class InstructionSTA(Processor):
    """ STA - Store Accumulator In Memory
    """
    def test_Memory_Has_Correct_Value(self):
        c = self.create(program=[0xA9, 0x03, 0x85, 0x05])

        self.assertEqual(c.mmu.read(0x05), 0x00)

        c.step()
        c.step()

        self.assertEqual(c.mmu.read(0x05), 0x03)


class InstructionSTX(Processor):
    """ STX - Set Memory To X
    """
    def test_Memory_Has_Correct_Value(self):
        c = self.create(program=[0xA2, 0x03, 0x86, 0x05])

        self.assertEqual(c.mmu.read(0x05), 0x00)

        c.step()
        c.step()

        self.assertEqual(c.mmu.read(0x05), 0x03)


class InstructionSTY(Processor):
    """ STY - Set Memory To Y
    """
    def test_Memory_Has_Correct_Value(self):
        c = self.create(program=[0xA0, 0x03, 0x84, 0x05])

        self.assertEqual(c.mmu.read(0x05), 0x00)

        c.step()
        c.step()

        self.assertEqual(c.mmu.read(0x05), 0x03)


class InstructionTAX(Processor):
    """ TAX, TAY, TSX, TSY Tests
    """
    def test_Transfer_Correct_Value_Set(self):
        subtests: list[tuple[int, int, RegisterMode, RegisterMode]] = [
            (0xAA, 0x03, RegisterMode.Accumulator, RegisterMode.XRegister),
            (0xA8, 0x03, RegisterMode.Accumulator, RegisterMode.YRegister),
            (0x8A, 0x03, RegisterMode.XRegister, RegisterMode.Accumulator),
            (0x98, 0x03, RegisterMode.YRegister, RegisterMode.Accumulator),
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [data[2].value, data[1], data[0]]
                c = self.create(program=p)

                c.step()
                c.step()

                match data[3]:
                    case RegisterMode.Accumulator:
                        self.assertEqual(c.r.a, data[1])
                    case RegisterMode.XRegister:
                        self.assertEqual(c.r.x, data[1])
                    case RegisterMode.YRegister:
                        self.assertEqual(c.r.y, data[1])

    def test_Transfer_Negative_Value_Set(self):
        subtests: list[tuple[int, int, RegisterMode, bool]] = [
            (0xAA, 0x80, RegisterMode.Accumulator, True),
            (0xA8, 0x80, RegisterMode.Accumulator, True),
            (0x8A, 0x80, RegisterMode.XRegister, True),
            (0x98, 0x80, RegisterMode.YRegister, True),
            (0xAA, 0xFF, RegisterMode.Accumulator, True),
            (0xA8, 0xFF, RegisterMode.Accumulator, True),
            (0x8A, 0xFF, RegisterMode.XRegister, True),
            (0x98, 0xFF, RegisterMode.YRegister, True),
            (0xAA, 0x7F, RegisterMode.Accumulator, False),
            (0xA8, 0x7F, RegisterMode.Accumulator, False),
            (0x8A, 0x7F, RegisterMode.XRegister, False),
            (0x98, 0x7F, RegisterMode.YRegister, False),
            (0xAA, 0x00, RegisterMode.Accumulator, False),
            (0xA8, 0x00, RegisterMode.Accumulator, False),
            (0x8A, 0x00, RegisterMode.XRegister, False),
            (0x98, 0x00, RegisterMode.YRegister, False),
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [data[2].value, data[1], data[0]]
                c = self.create(program=p)

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[3])

    def test_Transfer_Zero_Value_Set(self):
        subtests: list[tuple[int, int, RegisterMode, bool]] = [
            (0xAA, 0xFF, RegisterMode.Accumulator, False),
            (0xA8, 0xFF, RegisterMode.Accumulator, False),
            (0x8A, 0xFF, RegisterMode.XRegister, False),
            (0x98, 0xFF, RegisterMode.YRegister, False),
            (0xAA, 0x00, RegisterMode.Accumulator, True),
            (0xA8, 0x00, RegisterMode.Accumulator, True),
            (0x8A, 0x00, RegisterMode.XRegister, True),
            (0x98, 0x00, RegisterMode.YRegister, True),
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [data[2].value, data[1], data[0]]
                c = self.create(program=p)

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[3])


class InstructionTSX(Processor):
    """ TSX - Transfer Stack Pointer to X Register
    """
    def test_XRegister_Set_Correctly(self):
        c = self.create(program=[0xBA])

        stackPointer = c.r.s
        c.step()

        self.assertEqual(c.r.x, stackPointer)

    def test_Negative_Set_Correctly(self):
        subtests = [
            [0x00, False],
            [0x7F, False],
            [0x80, True],
            [0xFF, True],
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA2, data[0], 0x9A, 0xBA]
                c = self.create(program=p)

                c.step()
                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.N), data[1])

    def test_Zero_Set_Correctly(self):
        subtests = [
            [0x00, True],
            [0x01, False],
            [0xFF, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA2, data[0], 0x9A, 0xBA]
                c = self.create(program=p)

                c.step()
                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), data[1])


class InstructionTXS(Processor):
    """ TXS - Transfer X Register to Stack Pointer
    """
    def test_Stack_Pointer_Set_Correctly(self):
        subtests = [
            [0xAA,],
            [0x00,],
            [0xFF,],
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA2, data[0], 0x9A]
                c = self.create(program=p)

                c.step()
                c.step()

                self.assertEqual(c.r.s, data[0])


class AccumulatorAddress(Processor):
    """ Accumulator Address Tests
    """
    def test_Immediate_Mode_Accumulator_Has_Correct_Result(self):
        subtests = [
            [0x69, 0x01, 0x01, 0x02],  # ADC
            [0x29, 0x03, 0x03, 0x03],  # AND
            [0xA9, 0x04, 0x03, 0x03],  # LDA
            [0x49, 0x55, 0xAA, 0xFF],  # EOR
            [0x09, 0x55, 0xAA, 0xFF],  # ORA
            [0xE9, 0x03, 0x01, 0x01],  # SBC
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, data[1], data[0], data[2]]
                c = self.create(program=p)

                c.step()
                c.step()

                self.assertEqual(c.r.a, data[3])

    def test_ZeroPage_Mode_Accumulator_Has_Correct_Result(self):
        subtests = [
            [0x65, 0x01, 0x01, 0x02],  # ADC
            [0x25, 0x03, 0x03, 0x03],  # AND
            [0xA5, 0x04, 0x03, 0x03],  # LDA
            [0x45, 0x55, 0xAA, 0xFF],  # EOR
            [0x05, 0x55, 0xAA, 0xFF],  # ORA
            [0xE5, 0x03, 0x01, 0x01],  # SBC
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, data[1], data[0], 0x05, 0x00, data[2]]
                c = self.create(program=p)

                c.step()
                c.step()

                self.assertEqual(c.r.a, data[3])

    def test_ZeroPageX_Mode_Accumulator_Has_Correct_Result(self):
        # Just remember that my value's for the STX and ADC were added to the
        # end of the array. In a real program this would be invalid, as an
        # opcode would be next and 0x03 would be somewhere else
        subtests = [
            [0x75, 0x00, 0x03, 0x03],  # ADC
            [0x35, 0x03, 0x03, 0x03],  # AND
            [0xB5, 0x04, 0x03, 0x03],  # LDA
            [0x55, 0x55, 0xAA, 0xFF],  # EOR
            [0x15, 0x55, 0xAA, 0xFF],  # ORA
            [0xF5, 0x03, 0x01, 0x01],  # SBC
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, data[1], 0xA2, 0x01, data[0], 0x06, 0x00, data[2]]
                c = self.create(program=p)

                self.assertEqual(c.r.a, 0x00)

                c.step()
                c.step()
                c.step()

                self.assertEqual(c.r.a, data[3])

    def test_Absolute_Mode_Accumulator_Has_Correct_Result(self):
        subtests = [
            [0x6D, 0x00, 0x03, 0x03],  # ADC
            [0x2D, 0x03, 0x03, 0x03],  # AND
            [0xAD, 0x04, 0x03, 0x03],  # LDA
            [0x4D, 0x55, 0xAA, 0xFF],  # EOR
            [0x0D, 0x55, 0xAA, 0xFF],  # ORA
            [0xED, 0x03, 0x01, 0x01],  # SBC
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, data[1], data[0], 0x06, 0x00, 0x00, data[2]]
                c = self.create(program=p)

                self.assertEqual(c.r.a, 0x00)

                c.step()
                c.step()

                self.assertEqual(c.r.a, data[3])

    def test_AbsoluteX_Mode_Accumulator_Has_Correct_Result(self):
        subtests = [
            [0x7D, 0x01, 0x01, False, 0x02],  # ADC
            [0x3D, 0x03, 0x03, False, 0x03],  # AND
            [0xBD, 0x04, 0x03, False, 0x03],  # LDA
            [0x5D, 0x55, 0xAA, False, 0xFF],  # EOR
            [0x1D, 0x55, 0xAA, False, 0xFF],  # ORA
            [0xFD, 0x03, 0x01, False, 0x01],  # SBC
            [0x7D, 0x01, 0x01, True, 0x02],  # ADC
            [0x3D, 0x03, 0x03, True, 0x03],  # AND
            [0xBD, 0x04, 0x03, True, 0x03],  # LDA
            [0x5D, 0x55, 0xAA, True, 0xFF],  # EOR
            [0x1D, 0x55, 0xAA, True, 0xFF],  # ORA
            [0xFD, 0x03, 0x01, True, 0x01],  # SBC
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, data[1], 0xA2,]

                if data[3]:
                    p += [0x09, data[0], 0xFF, 0xFF, 0x00, data[2]]
                else:
                    p += [0x01, data[0], 0x07, 0x00, 0x00, data[2]]

                c = self.create(program=p)

                self.assertEqual(c.r.a, 0x00)

                c.step()
                c.step()
                c.step()

                self.assertEqual(c.r.a, data[4])

    def test_AbsoluteY_Mode_Accumulator_Has_Correct_Result(self):
        subtests = [
            [0x79, 0x01, 0x01, False, 0x02],  # ADC
            [0x39, 0x03, 0x03, False, 0x03],  # AND
            [0xB9, 0x04, 0x03, False, 0x03],  # LDA
            [0x59, 0x55, 0xAA, False, 0xFF],  # EOR
            [0x19, 0x55, 0xAA, False, 0xFF],  # ORA
            [0xF9, 0x03, 0x01, False, 0x01],  # SBC
            [0x79, 0x01, 0x01, True, 0x02],  # ADC
            [0x39, 0x03, 0x03, True, 0x03],  # AND
            [0xB9, 0x04, 0x03, True, 0x03],  # LDA
            [0x59, 0x55, 0xAA, True, 0xFF],  # EOR
            [0x19, 0x55, 0xAA, True, 0xFF],  # ORA
            [0xF9, 0x03, 0x01, True, 0x01],  # SBC
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, data[1], 0xA0,]

                if data[3]:
                    p += [0x09, data[0], 0xFF, 0xFF, 0x00, data[2]]
                else:
                    p += [0x01, data[0], 0x07, 0x00, 0x00, data[2]]

                c = self.create(program=p)

                self.assertEqual(c.r.a, 0x00)

                c.step()
                c.step()
                c.step()

                self.assertEqual(c.r.a, data[4])

    def test_Indexed_Indirect_Mode_Accumulator_Has_Correct_Result(self):
        subtests = [
            [0x61, 0x01, 0x01, False, 0x02],  # ADC
            [0x21, 0x03, 0x03, False, 0x03],  # AND
            [0xA1, 0x04, 0x03, False, 0x03],  # LDA
            [0x41, 0x55, 0xAA, False, 0xFF],  # EOR
            [0x01, 0x55, 0xAA, False, 0xFF],  # ORA
            [0xE1, 0x03, 0x01, False, 0x01],  # SBC
            [0x61, 0x01, 0x01, True, 0x02],  # ADC
            [0x21, 0x03, 0x03, True, 0x03],  # AND
            [0xA1, 0x04, 0x03, True, 0x03],  # LDA
            [0x41, 0x55, 0xAA, True, 0xFF],  # EOR
            [0x01, 0x55, 0xAA, True, 0xFF],  # ORA
            [0xE1, 0x03, 0x01, True, 0x01],  # SBC
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, data[1], 0xA6, 0x06, data[0]]

                if data[3]:
                    p += [0xFF, 0x08, 0x09, 0x00, data[2]]
                else:
                    p += [0x01, 0x06, 0x09, 0x00, data[2]]

                c = self.create(program=p)

                self.assertEqual(c.r.a, 0x00)

                c.step()
                c.step()
                c.step()

                self.assertEqual(c.r.a, data[4])

    def test_Indirect_Indexed_Mode_Accumulator_Has_Correct_Result(self):
        subtests = [
            [0x71, 0x01, 0x01, False, 0x02],  # ADC
            [0x31, 0x03, 0x03, False, 0x03],  # AND
            [0xB1, 0x04, 0x03, False, 0x03],  # LDA
            [0x51, 0x55, 0xAA, False, 0xFF],  # EOR
            [0x11, 0x55, 0xAA, False, 0xFF],  # ORA
            [0xF1, 0x03, 0x01, False, 0x01],  # SBC
            [0x71, 0x01, 0x01, True, 0x02],  # ADC
            [0x31, 0x03, 0x03, True, 0x03],  # AND
            [0xB1, 0x04, 0x03, True, 0x03],  # LDA
            [0x51, 0x55, 0xAA, True, 0xFF],  # EOR
            [0x11, 0x55, 0xAA, True, 0xFF],  # ORA
            [0xF1, 0x03, 0x01, True, 0x01],  # SBC
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, data[1], 0xA0]
                p = [0xA9, data[1], 0xA0]

                if data[3]:
                    p += [0x0A, data[0], 0x07, 0x00, 0xFF, 0xFF, data[2]]
                else:
                    p += [0x01, data[0], 0x07, 0x00, 0x08, 0x00, data[2]]

                c = self.create(program=p)

                self.assertEqual(c.r.a, 0x00)

                c.step()
                c.step()
                c.step()

                self.assertEqual(c.r.a, data[4])


class IndexAddress(Processor):
    """ Index Address Tests
    """
    def test_ZeroPage_Mode_Index_Has_Correct_Result(self):
        subtests = [
            [0xA6, 0x03, True],  # LDX Zero Page
            [0xB6, 0x03, True],  # LDX Zero Page Y
            [0xA4, 0x03, False],  # LDY Zero Page
            [0xB4, 0x03, False],  # LDY Zero Page X
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [data[0], 0x03, 0x00, data[1]]
                c = self.create(program=p)

                self.assertEqual(c.r.a, 0x00)

                c.step()

                if data[2]:
                    self.assertEqual(c.r.x, data[1])
                else:
                    self.assertEqual(c.r.y, data[1])

    def test_ZeroPage_Mode_Index_Has_Correct_Result_When_Wrapped(self):
        subtests = [
            [0xB6, 0x03, True],  # LDX Zero Page Y
            [0xB4, 0x03, False],  # LDY Zero Page X
        ]
        for data in subtests:
            with self.subTest(data=data):
                if data[2]:
                    p = [0xA0, 0xFF, data[0], 0x06, 0x00, data[1]]
                else:
                    p = [0xA2, 0xFF, data[0], 0x06, 0x00, data[1]]

                c = self.create(program=p)

                self.assertEqual(c.r.a, 0x00)

                c.step()
                c.step()

                if data[2]:
                    self.assertEqual(c.r.x, data[1])
                else:
                    self.assertEqual(c.r.y, data[1])

    def test_Absolute_Mode_Index_Has_Correct_Result(self):
        subtests = [
            [0xAE, 0x03, True],  # LDX Absolute
            [0xAC, 0x03, False],  # LDY Absolute
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [data[0], 0x04, 0x00, 0x00, data[1]]
                c = self.create(program=p)

                self.assertEqual(c.r.a, 0x00)

                c.step()

                if data[2]:
                    self.assertEqual(c.r.x, data[1])
                else:
                    self.assertEqual(c.r.y, data[1])


class CompareAddress(Processor):
    """ Compare Address Tests
    """
    def test_Immediate_Mode_Compare_Operation_Has_Correct_Result(self):
        subtests: list[tuple[int, int, int, RegisterMode]] = [
            (0xC9, 0xFF, 0x00, RegisterMode.Accumulator),  # CMP Immediate
            (0xE0, 0xFF, 0x00, RegisterMode.XRegister),  # CPX Immediate
            (0xC0, 0xFF, 0x00, RegisterMode.YRegister),  # CPY Immediate
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [data[3].value, data[1], data[0], data[2]]
                c = self.create(program=p)

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), False)
                self.assertEqual(c.r.getFlag(FlagBit.N), True)
                self.assertEqual(c.r.getFlag(FlagBit.C), True)

    def test_ZeroPage_Modes_Compare_Operation_Has_Correct_Result(self):
        subtests: list[tuple[int, int, int, RegisterMode]] = [
            (0xC5, 0xFF, 0x00, RegisterMode.Accumulator),  # CMP Zero Page
            (0xD5, 0xFF, 0x00, RegisterMode.Accumulator),  # CMP Zero Page X
            (0xE4, 0xFF, 0x00, RegisterMode.XRegister),  # CPX Zero Page
            (0xC4, 0xFF, 0x00, RegisterMode.YRegister),  # CPY Zero Page
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [data[3].value, data[1], data[0], 0x04, data[2]]
                c = self.create(program=p)

                self.assertEqual(c.r.a, 0x00)

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), False)
                self.assertEqual(c.r.getFlag(FlagBit.N), True)
                self.assertEqual(c.r.getFlag(FlagBit.C), True)

    def test_Absolute_Modes_Compare_Operation_Has_Correct_Result(self):
        subtests: list[tuple[int, int, int, RegisterMode]] = [
            (0xCD, 0xFF, 0x00, RegisterMode.Accumulator),  # CMP Absolute
            (0xDD, 0xFF, 0x00, RegisterMode.Accumulator),  # CMP Absolute X
            (0xEC, 0xFF, 0x00, RegisterMode.XRegister),  # CPX Absolute
            (0xCC, 0xFF, 0x00, RegisterMode.YRegister),  # CPY Absolute
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [data[3].value, data[1], data[0], 0x05, 0x00, data[2]]
                c = self.create(program=p)

                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), False)
                self.assertEqual(c.r.getFlag(FlagBit.N), True)
                self.assertEqual(c.r.getFlag(FlagBit.C), True)

    def test_Indexed_Indirect_Mode_CMP_Operation_Has_Correct_Result(self):
        subtests = [
            [0xC1, 0xFF, 0x00, True],
            [0xC1, 0xFF, 0x00, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, data[1], 0xA6, 0x06, data[0]]
                if data[3]:
                    p += [0xFF, 0x08, 0x09, 0x00, data[2]]
                else:
                    p += [0x01, 0x06, 0x09, 0x00, data[2]]

                c = self.create(program=p)
                self.assertEqual(c.r.a, 0x00)

                c.step()
                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), False)
                self.assertEqual(c.r.getFlag(FlagBit.N), True)
                self.assertEqual(c.r.getFlag(FlagBit.C), True)

    def test_Indirect_Indexed_Mode_CMP_Operation_Has_Correct_Result(self):
        subtests = [
            [0xD1, 0xFF, 0x00, True],
            [0xD1, 0xFF, 0x00, False],
        ]
        for data in subtests:
            with self.subTest(data=data):
                p = [0xA9, data[1], 0x84, 0x06, data[0], 0x07,]
                if data[3]:
                    p += [0x0A, 0xFF, 0xFF, data[2]]
                else:
                    p += [0x01, 0x08, 0x00, data[2]]

                c = self.create(program=p)

                c.step()
                c.step()
                c.step()

                self.assertEqual(c.r.getFlag(FlagBit.Z), False)
                self.assertEqual(c.r.getFlag(FlagBit.N), True)
                self.assertEqual(c.r.getFlag(FlagBit.C), True)


class DecrementIncrementAddress(Processor):
    """ Decrement/Increment Address Tests
    """
    def test_Zero_Page_DEC_INC_Has_Correct_Result(self):
        subtests = [
            [0xC6, 0xFF, 0xFE],  # DEC Zero Page
            [0xD6, 0xFF, 0xFE],  # DEC Zero Page X
            [0xE6, 0xFF, 0x00],  # INC Zero Page
            [0xF6, 0xFF, 0x00],  # INC Zero Page X
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[data[0], 0x02, data[1]])

                c.step()

                self.assertEqual(c.mmu.read(0x02), data[2])

    def test_Absolute_DEC_INC_Has_Correct_Result(self):
        subtests = [
            [0xCE, 0xFF, 0xFE],  # DEC Zero Page
            [0xDE, 0xFF, 0xFE],  # DEC Zero Page X
            [0xEE, 0xFF, 0x00],  # INC Zero Page
            [0xFE, 0xFF, 0x00],  # INC Zero Page X
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[data[0], 0x03, 0x00, data[1]])

                c.step()

                self.assertEqual(c.mmu.read(0x03), data[2])


class StoreInMemoryAddress(Processor):
    """ Store In Memory Address Tests
    """
    def test_ZeroPage_Mode_Memory_Has_Correct_Result(self):
        subtests: list[tuple[int, RegisterMode]] = [
            (0x85, RegisterMode.Accumulator),  # STA Zero Page
            (0x95, RegisterMode.Accumulator),  # STA Zero Page X
            (0x86, RegisterMode.XRegister),  # STX Zero Page
            (0x96, RegisterMode.XRegister),  # STX Zero Page Y
            (0x84, RegisterMode.YRegister),  # STY Zero Page
            (0x94, RegisterMode.YRegister),  # STY Zero Page X
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[
                    data[1].value, 0x04, data[0], 0x00, 0x05
                ])

                c.step()
                c.step()

                self.assertEqual(c.mmu.read(0x04), 0x05)

    def test_Absolute_Mode_Memory_Has_Correct_Result(self):
        subtests: list[tuple[int, int, RegisterMode]] = [
            (0x8D, 0x03, RegisterMode.Accumulator),  # STA Absolute
            (0x9D, 0x03, RegisterMode.Accumulator),  # STA Absolute X
            (0x99, 0x03, RegisterMode.Accumulator),  # STA Absolute X
            (0x8E, 0x03, RegisterMode.XRegister),  # STX Zero Page
            (0x8C, 0x03, RegisterMode.YRegister),  # STY Zero Page
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[
                    data[2].value, data[1], data[0], 0x04
                ])

                c.step()
                c.step()

                self.assertEqual(c.mmu.read(0x04), data[1])


class Cycle(Processor):
    """ Cycle Tests
    """
    def test_Cycles_Correct_After_Operations_That_Do_Not_Wrap(self):
        """
        Test number of cycles remaining is correct after operations and the
        result do not wrap
        """
        subtests = [
            [0x69, 2],  # ADC Immediate
            [0x65, 3],  # ADC Zero Page
            [0x75, 4],  # ADC Zero Page X
            [0x6D, 4],  # ADC Absolute
            [0x7D, 4],  # ADC Absolute X
            [0x79, 4],  # ADC Absolute Y
            [0x61, 6],  # ADC Indrect X
            [0x71, 5],  # ADC Indirect Y
            [0x29, 2],  # AND Immediate
            [0x25, 3],  # AND Zero Page
            [0x35, 4],  # AND Zero Page X
            [0x2D, 4],  # AND Absolute
            [0x3D, 4],  # AND Absolute X
            [0x39, 4],  # AND Absolute Y
            [0x21, 6],  # AND Indirect X
            [0x31, 5],  # AND Indirect Y
            [0x0A, 2],  # ASL Accumulator
            [0x06, 5],  # ASL Zero Page
            [0x16, 6],  # ASL Zero Page X
            [0x0E, 6],  # ASL Absolute
            [0x1E, 7],  # ASL Absolute X
            [0x24, 3],  # BIT Zero Page
            [0x2C, 4],  # BIT Absolute
            [0x00, 7],  # BRK Implied
            [0x18, 2],  # CLC Implied
            [0xD8, 2],  # CLD Implied
            [0x58, 2],  # CLI Implied
            [0xB8, 2],  # CLV Implied
            [0xC9, 2],  # CMP Immediate
            [0xC5, 3],  # CMP ZeroPage
            [0xD5, 4],  # CMP Zero Page X
            [0xCD, 4],  # CMP Absolute
            [0xDD, 4],  # CMP Absolute X
            [0xD9, 4],  # CMP Absolute Y
            [0xC1, 6],  # CMP Indirect X
            [0xD1, 5],  # CMP Indirect Y
            [0xE0, 2],  # CPX Immediate
            [0xE4, 3],  # CPX ZeroPage
            [0xEC, 4],  # CPX Absolute
            [0xC0, 2],  # CPY Immediate
            [0xC4, 3],  # CPY ZeroPage
            [0xCC, 4],  # CPY Absolute
            [0xC6, 5],  # DEC Zero Page
            [0xD6, 6],  # DEC Zero Page X
            [0xCE, 6],  # DEC Absolute
            [0xDE, 7],  # DEC Absolute X
            [0xCA, 2],  # DEX Implied
            [0x88, 2],  # DEY Implied
            [0x49, 2],  # EOR Immediate
            [0x45, 3],  # EOR Zero Page
            [0x55, 4],  # EOR Zero Page X
            [0x4D, 4],  # EOR Absolute
            [0x5D, 4],  # EOR Absolute X
            [0x59, 4],  # EOR Absolute Y
            [0x41, 6],  # EOR Indrect X
            [0x51, 5],  # EOR Indirect Y
            [0xE6, 5],  # INC Zero Page
            [0xF6, 6],  # INC Zero Page X
            [0xEE, 6],  # INC Absolute
            [0xFE, 7],  # INC Absolute X
            [0xE8, 2],  # INX Implied
            [0xC8, 2],  # INY Implied
            [0x4C, 3],  # JMP Absolute
            [0x6C, 5],  # JMP Indirect
            [0x20, 6],  # JSR Absolute
            [0xA9, 2],  # LDA Immediate
            [0xA5, 3],  # LDA Zero Page
            [0xB5, 4],  # LDA Zero Page X
            [0xAD, 4],  # LDA Absolute
            [0xBD, 4],  # LDA Absolute X
            [0xB9, 4],  # LDA Absolute Y
            [0xA1, 6],  # LDA Indirect X
            [0xB1, 5],  # LDA Indirect Y
            [0xA2, 2],  # LDX Immediate
            [0xA6, 3],  # LDX Zero Page
            [0xB6, 4],  # LDX Zero Page Y
            [0xAE, 4],  # LDX Absolute
            [0xBE, 4],  # LDX Absolute Y
            [0xA0, 2],  # LDY Immediate
            [0xA4, 3],  # LDY Zero Page
            [0xB4, 4],  # LDY Zero Page Y
            [0xAC, 4],  # LDY Absolute
            [0xBC, 4],  # LDY Absolute Y
            [0x4A, 2],  # LSR Accumulator
            [0x46, 5],  # LSR Zero Page
            [0x56, 6],  # LSR Zero Page X
            [0x4E, 6],  # LSR Absolute
            [0x5E, 7],  # LSR Absolute X
            [0xEA, 2],  # NOP Implied
            [0x09, 2],  # ORA Immediate
            [0x05, 3],  # ORA Zero Page
            [0x15, 4],  # ORA Zero Page X
            [0x0D, 4],  # ORA Absolute
            [0x1D, 4],  # ORA Absolute X
            [0x19, 4],  # ORA Absolute Y
            [0x01, 6],  # ORA Indirect X
            [0x11, 5],  # ORA Indirect Y
            [0x48, 3],  # PHA Implied
            [0x08, 3],  # PHP Implied
            [0x68, 4],  # PLA Implied
            [0x28, 4],  # PLP Implied
            [0x2A, 2],  # ROL Accumulator
            [0x26, 5],  # ROL Zero Page
            [0x36, 6],  # ROL Zero Page X
            [0x2E, 6],  # ROL Absolute
            [0x3E, 7],  # ROL Absolute X
            [0x6A, 2],  # ROR Accumulator
            [0x66, 5],  # ROR Zero Page
            [0x76, 6],  # ROR Zero Page X
            [0x6E, 6],  # ROR Absolute
            [0x7E, 7],  # ROR Absolute X
            [0x40, 6],  # RTI Implied
            [0x60, 6],  # RTS Implied
            [0xE9, 2],  # SBC Immediate
            [0xE5, 3],  # SBC Zero Page
            [0xF5, 4],  # SBC Zero Page X
            [0xED, 4],  # SBC Absolute
            [0xFD, 4],  # SBC Absolute X
            [0xF9, 4],  # SBC Absolute Y
            [0xE1, 6],  # SBC Indrect X
            [0xF1, 5],  # SBC Indirect Y
            [0x38, 2],  # SEC Implied
            [0xF8, 2],  # SED Implied
            [0x78, 2],  # SEI Implied
            [0x85, 3],  # STA ZeroPage
            [0x95, 4],  # STA Zero Page X
            [0x8D, 4],  # STA Absolute
            [0x9D, 5],  # STA Absolute X
            [0x99, 5],  # STA Absolute Y
            [0x81, 6],  # STA Indirect X
            [0x91, 6],  # STA Indirect Y
            [0x86, 3],  # STX Zero Page
            [0x96, 4],  # STX Zero Page Y
            [0x8E, 4],  # STX Absolute
            [0x84, 3],  # STY Zero Page
            [0x94, 4],  # STY Zero Page X
            [0x8C, 4],  # STY Absolute
            [0xAA, 2],  # TAX Implied
            [0xA8, 2],  # TAY Implied
            [0xBA, 2],  # TSX Implied
            [0x8A, 2],  # TXA Implied
            [0x9A, 2],  # TXS Implied
            [0x98, 2],  # TYA Implied
            # Illegal Instructions
            [0x0B, 2],  # AAC Immediate
            [0x87, 3],  # AAX Zero Page
            [0x97, 4],  # AAX Zero Page Y
            [0x8F, 4],  # AAX Absolute
            [0x83, 6],  # AAX Indirect X
            [0x6B, 2],  # ARR Immediate
            [0x4B, 2],  # ASR Immediate
            [0xAB, 2],  # ATX Immediate
            [0x9F, 5],  # AXA Absolute Y
            [0x93, 6],  # AXA Indirect Y
            [0xCB, 2],  # AXS Immediate
            [0xC7, 5],  # DCP Zero Page
            [0xD7, 6],  # DCP Zero Page X
            [0xCF, 6],  # DCP Absolute
            [0xDF, 7],  # DCP Absolute X
            [0xDB, 7],  # DCP Absolute Y
            [0xC3, 8],  # DCP Indirect X
            [0xD3, 8],  # DCP Indirect Y
            [0xE7, 5],  # ISC Zero Page
            [0xF7, 6],  # ISC Zero Page X
            [0xEF, 6],  # ISC Absolute
            [0xFF, 7],  # ISC Absolute X
            [0xFB, 7],  # ISC Absolute Y
            [0xE3, 8],  # ISC Indirect X
            [0xF3, 8],  # ISC Indirect Y
            # [0x, ], # KIL Immediate (Untestable)
            [0xBB, 4],  # LAR Absolute Y
            [0xA7, 3],  # LAX Zero Page
            [0xB7, 4],  # LAX Zero Page Y
            [0xAF, 4],  # LAX Absolute
            [0xBF, 4],  # LAX Absolute Y
            [0xA3, 6],  # LAX Indirect X
            [0xB3, 5],  # LAX Indirect Y
            [0x27, 5],  # RLA Zero Page
            [0x37, 6],  # RLA Zero Page X
            [0x2F, 6],  # RLA Absolute
            [0x3F, 7],  # RLA Absolute X
            [0x3B, 7],  # RLA Absolute Y
            [0x23, 8],  # RLA Indirect X
            [0x33, 8],  # RLA Indirect Y
            [0x67, 5],  # RRA Zero Page
            [0x77, 6],  # RRA Zero Page X
            [0x6F, 6],  # RRA Absolute
            [0x7F, 7],  # RRA Absolute X
            [0x7B, 7],  # RRA Absolute Y
            [0x63, 8],  # RRA Indirect X
            [0x73, 8],  # RRA Indirect Y
            [0xEB, 2],  # USBC Immediate (SBC)
            [0x07, 5],  # SLO Zero Page
            [0x17, 6],  # SLO Zero Page X
            [0x0F, 6],  # SLO Absolute
            [0x1F, 7],  # SLO Absolute X
            [0x1B, 7],  # SLO Absolute Y
            [0x03, 8],  # SLO Indirect X
            [0x13, 8],  # SLO Indirect Y
            [0x47, 5],  # SRE Zero Page
            [0x57, 6],  # SRE Zero Page X
            [0x4F, 6],  # SRE Absolute
            [0x5F, 7],  # SRE Absolute X
            [0x5B, 7],  # SRE Absolute Y
            [0x43, 8],  # SRE Indirect X
            [0x53, 8],  # SRE Indirect Y
            [0x9E, 5],  # SXA Absolute Y
            [0x9C, 5],  # SYA Absolute X
            [0x8B, 2],  # XAA Immediate
            [0x9B, 5],  # XAS Absolute Y
            [0x1A, 2],  # NOP Implied
            [0x3A, 2],  # NOP Implied
            [0x5A, 2],  # NOP Implied
            [0x7A, 2],  # NOP Implied
            [0xDA, 2],  # NOP Implied
            [0xFA, 2],  # NOP Implied
            [0x80, 2],  # NOP Immediate   (SKB)
            [0x82, 2],  # NOP Immediate   (SKB)
            [0x89, 2],  # NOP Immediate   (SKB)
            [0xC2, 2],  # NOP Immediate   (SKB)
            [0xE2, 2],  # NOP Immediate   (SKB)
            [0x04, 3],  # NOP Zero Page   (IGN)
            [0x44, 3],  # NOP Zero Page   (IGN)
            [0x64, 3],  # NOP Zero Page   (IGN)
            [0x14, 4],  # NOP Zero Page X (IGN)
            [0x34, 4],  # NOP Zero Page X (IGN)
            [0x54, 4],  # NOP Zero Page X (IGN)
            [0x74, 4],  # NOP Zero Page X (IGN)
            [0xD4, 4],  # NOP Zero Page X (IGN)
            [0xF4, 4],  # NOP Zero Page X (IGN)
            [0x0C, 4],  # NOP Absolute    (IGN)
            [0x1C, 4],  # NOP Absolute X  (IGN)
            [0x3C, 4],  # NOP Absolute X  (IGN)
            [0x5C, 4],  # NOP Absolute X  (IGN)
            [0x7C, 4],  # NOP Absolute X  (IGN)
            [0xDC, 4],  # NOP Absolute X  (IGN)
            [0xFC, 4],  # NOP Absolute X  (IGN)
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[data[0], 0x00])

                self.assertEqual(c.cc, 0)

                c.step()

                self.assertEqual(c.cc, data[1], f"OP: {data[0]:0>2x}")

    def test_Cycles_Correct_When_In_AbsoluteX_Or_AbsoluteY_And_Wrap(self):
        """
        Test number of cycles remaining is correct when in AbsoluteX or
        AbsoluteY and the result wraps
        """
        subtests = [
            [0x7D, True, 5],  # ADC Absolute X
            [0x79, False, 5],  # ADC Absolute Y
            [0x3D, True, 5],  # AND Absolute X
            [0x39, False, 5],  # AND Absolute Y
            [0x1E, True, 7],  # ASL Absolute X
            [0xDD, True, 5],  # CMP Absolute X
            [0xD9, False, 5],  # CMP Absolute Y
            [0xDE, True, 7],  # DEC Absolute X
            [0x5D, True, 5],  # EOR Absolute X
            [0x59, False, 5],  # EOR Absolute Y
            [0xFE, True, 7],  # INC Absolute X
            [0xBD, True, 5],  # LDA Absolute X
            [0xB9, False, 5],  # LDA Absolute Y
            [0xBE, False, 5],  # LDX Absolute Y
            [0xBC, True, 5],  # LDY Absolute X
            [0x5E, True, 7],  # LSR Absolute X
            [0x1D, True, 5],  # ORA Absolute X
            [0x19, False, 5],  # ORA Absolute Y
            [0x3E, True, 7],  # ROL Absolute X
            [0x7E, True, 7],  # ROR Absolute X
            [0xFD, True, 5],  # SBC Absolute X
            [0xF9, False, 5],  # SBC Absolute Y
            [0x9D, True, 5],  # STA Absolute X
            [0x99, True, 5],  # STA Absolute Y
            # Illegal Instructions
            [0xBB, False, 5],  # LAR Absolute Y
            [0xBF, False, 5],  # LAX Absolute Y
            [0x1C, True, 5],  # TOP Absolute X
            [0x3C, True, 5],  # TOP Absolute X
            [0x5C, True, 5],  # TOP Absolute X
            [0x7C, True, 5],  # TOP Absolute X
            [0xDC, True, 5],  # TOP Absolute X
            [0xFC, True, 5],  # TOP Absolute X
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0x06, data[0], 0xFF, 0xFF, 0x00, 0x03],
                    condition=(0xA6 if data[1] else 0xA4)
                )

                self.assertEqual(c.cc, 0)

                c.step()
                c.step()

                self.assertEqual(c.cc, data[2], f"OP: {data[0]:0>2x}")

    def test_Cycles_Correct_When_In_IndirectIndexed_And_Wrap(self):
        """
        Test number of cycles remaining is correct when in Indirect Indexed
        and the result wraps
        """
        subtests = [
            [0x71, 6],  # ADC Indirect Y
            [0x31, 6],  # AND Indirect Y
            [0xB1, 6],  # LDA Indirect Y
            [0xD1, 6],  # CMP Indirect Y
            [0x51, 6],  # EOR Indirect Y
            [0x11, 6],  # ORA Indirect Y
            [0xF1, 6],  # SBC Indirect Y
            [0x91, 6],  # STA Indirect Y
            # Illegal Instructions
            [0xB3, 6],  # LAX Indirect Y
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA0, 0x04, data[0], 0x05, 0x08, 0xFF, 0xFF, 0x03]
                )

                self.assertEqual(c.cc, 0)

                c.step()
                c.step()

                self.assertEqual(c.cc, data[1], f"OP: {data[0]:0>2x}")

    def test_Cycles_Correct_When_Relative_And_Branch_On_Carry(self):
        """
        Test number of cycles remaining is correct when Relative and
        Branch On Carry, and result do not wrap
        """
        subtests = [
            [0x90, 2, 0x38],  # BCC
            [0x90, 3, 0x18],  # BCC
            [0xB0, 2, 0x18],  # BCS
            [0xB0, 3, 0x38],  # BCS
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[data[2], data[0], 0x00]
                )
                self.assertEqual(c.cc, 0)

                c.step()
                c.step()

                self.assertEqual(c.cc, data[1])

    def test_Cycles_Correct_When_Relative_And_Branch_On_Carry_And_Wrap(self):
        """
        Test number of cycles remaining is correct when Relative and
        Branch On Carry, and result wrap
        """
        subtests = [
            [0x90, 4, 0x18, True],  # BCC
            [0x90, 4, 0x18, False],  # BCC
            [0xB0, 4, 0x38, True],  # BCC
            [0xB0, 4, 0x38, False],  # BCC
        ]
        for data in subtests:
            with self.subTest(data=data):
                if data[3]:
                    initialAddress = 0xFFF0
                    amountToMove = 0x0F
                else:
                    initialAddress = 0x00
                    amountToMove = 0x84

                c = self.create(
                    program=[data[2], data[0], amountToMove, 0x00],
                    pc=initialAddress
                )

                self.assertEqual(c.cc, 0)

                c.step()
                c.step()

                self.assertEqual(c.cc, data[1])

    def test_Cycles_Correct_When_Relative_And_Branch_On_Zero(self):
        """
        Test number of cycles remaining is correct when Relative and
        Branch On Zerp, and result do not wrap
        """
        subtests = [
            [0xF0, 3, True],  # BEQ
            [0xF0, 2, False],  # BEQ
            [0xD0, 3, False],  # BNE
            [0xD0, 2, True],  # BNE
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA9, (0x00 if data[2] else 0x01), data[0], 0x00]
                )

                self.assertEqual(c.cc, 0)

                c.step()
                c.step()

                self.assertEqual(c.cc, data[1])

    def test_Cycles_Correct_When_Relative_And_Branch_On_Zero_And_Wrap(self):
        """
        Test number of cycles remaining is correct when Relative and
        Branch On Zero, and result wrap
        """
        subtests = [
            [0xF0, 4, True, True],  # BEQ
            [0xF0, 4, True, False],  # BEQ
            [0xD0, 4, False, True],  # BNE
            [0xD0, 4, False, False],  # BNE
        ]
        for data in subtests:
            with self.subTest(data=data):
                if data[3]:
                    initialAddress = 0xFFF0
                    amountToMove = 0x0D
                else:
                    initialAddress = 0x00
                    amountToMove = 0x84

                c = self.create(
                    program=[
                        0xA9, (0x00 if data[2] else 0x01),
                        data[0], amountToMove, 0x00
                    ],
                    pc=initialAddress
                )

                self.assertEqual(c.cc, 0)

                c.step()
                c.step()

                self.assertEqual(c.cc, data[1])

    def test_Cycles_Correct_When_Relative_And_Branch_On_Negative(self):
        """
        Test number of cycles remaining is correct when Relative and
        Branch On Negative, and result do not wrap
        """
        subtests = [
            [0x30, 3, True],  # BEQ
            [0x30, 2, False],  # BEQ
            [0x10, 3, False],  # BNE
            [0x10, 2, True],  # BNE
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[
                    0xA9, (0x80 if data[2] else 0x79), data[0], 0x00
                ])

                self.assertEqual(c.cc, 0)

                c.step()
                c.step()

                self.assertEqual(c.cc, data[1])

    def test_Cycles_Correct_When_Relative_And_Branch_On_Negative_And_Wrap(
        self
    ):
        """
        Test number of cycles remaining is correct when Relative and
        Branch On Negative, and result wrap
        """
        subtests = [
            [0x30, 4, True, True],  # BEQ
            [0x30, 4, True, False],  # BEQ
            [0x10, 4, False, True],  # BNE
            [0x10, 4, False, False],  # BNE
        ]
        for data in subtests:
            with self.subTest(data=data):
                if data[3]:
                    initialAddress = 0xFFF0
                    amountToMove = 0x0D
                else:
                    initialAddress = 0x00
                    amountToMove = 0x84

                c = self.create(
                    program=[
                        0xA9, (0x80 if data[2] else 0x79), data[0],
                        amountToMove, 0x00
                    ],
                    pc=initialAddress
                )

                c.step()
                c.step()

                self.assertEqual(c.cc, data[1])

    def test_Cycles_Correct_When_Relative_And_Branch_On_Overflow(self):
        """
        Test number of cycles remaining is correct when Relative and
        Branch On Overflow, and result do not wrap
        """
        subtests = [
            [0x50, 3, False],  # BVC
            [0x50, 2, True],  # BVC
            [0x70, 3, True],  # BVS
            [0x70, 2, False],  # BVS
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[
                    0xA9, 0x01, 0x69, (0x7F if data[2] else 0x01),
                    data[0], 0x00
                ])

                self.assertEqual(c.cc, 0)

                c.step()
                c.step()
                c.step()

                self.assertEqual(c.cc, data[1])

    def test_Cycles_Correct_When_Relative_And_Branch_On_Overflow_And_Wrap(
        self
    ):
        """
        Test number of cycles remaining is correct when Relative and
        Branch On Overflow, and result wrap
        """
        subtests = [
            [0x50, 4, False, True],  # BVC
            [0x50, 4, False, False],  # BVC
            [0x70, 4, True, True],  # BVS
            [0x70, 4, True, False],  # BVS
        ]
        for data in subtests:
            with self.subTest(data=data):
                if data[3]:
                    initialAddress = 0xFFF0
                    amountToMove = 0x0B
                else:
                    initialAddress = 0x00
                    amountToMove = 0x86

                p = [
                    0xA9, (0x7F if data[2] else 0x00), 0x69,
                    0x01, data[0], amountToMove, 0x00
                ]
                c = self.create(program=p, pc=initialAddress)
                self.assertEqual(c.cc, 0)

                c.step()
                c.step()
                c.step()
                """
                print(
                    f'${initialAddress:0>4x}',
                    f'${c.r.pc:0>4x}',
                    [f'{i:0>2x}' for i in p]
                )
                """

                self.assertEqual(c.cc, data[1])


class ProgramCounter(Processor):
    """ Program Counter Tests
    """
    def test_Program_Counter_Correct(self):
        subtests = [
            [0x69, 2],  # ADC Immediate
            [0x65, 2],  # ADC Zero Page
            [0x75, 2],  # ADC Zero Page X
            [0x6D, 3],  # ADC Absolute
            [0x7D, 3],  # ADC Absolute X
            [0x79, 3],  # ADC Absolute Y
            [0x61, 2],  # ADC Indirect X
            [0x71, 2],  # ADC Indirect Y
            [0x29, 2],  # AND Immediate
            [0x25, 2],  # AND Zero Page
            [0x35, 2],  # AND Zero Page X
            [0x2D, 3],  # AND Absolute
            [0x3D, 3],  # AND Absolute X
            [0x39, 3],  # AND Absolute Y
            [0x21, 2],  # AND Indirect X
            [0x31, 2],  # AND Indirect Y
            [0x0A, 1],  # ASL Accumulator
            [0x06, 2],  # ASL Zero Page
            [0x16, 2],  # ASL Zero Page X
            [0x0E, 3],  # ASL Absolute
            [0x1E, 3],  # ASL Absolute X
            [0x24, 2],  # BIT Zero Page
            [0x2C, 3],  # BIT Absolute
            [0x18, 1],  # CLC Implied
            [0xD8, 1],  # CLD Implied
            [0x58, 1],  # CLI Implied
            [0xB8, 1],  # CLV Implied
            [0xC9, 2],  # CMP Immediate
            [0xC5, 2],  # CMP ZeroPage
            [0xD5, 2],  # CMP Zero Page X
            [0xCD, 3],  # CMP Absolute
            [0xDD, 3],  # CMP Absolute X
            [0xD9, 3],  # CMP Absolute Y
            [0xC1, 2],  # CMP Indirect X
            [0xD1, 2],  # CMP Indirect Y
            [0xE0, 2],  # CPX Immediate
            [0xE4, 2],  # CPX ZeroPage
            [0xEC, 3],  # CPX Absolute
            [0xC0, 2],  # CPY Immediate
            [0xC4, 2],  # CPY ZeroPage
            [0xCC, 3],  # CPY Absolute
            [0xC6, 2],  # DEC Zero Page
            [0xD6, 2],  # DEC Zero Page X
            [0xCE, 3],  # DEC Absolute
            [0xDE, 3],  # DEC Absolute X
            [0xCA, 1],  # DEX Implied
            [0x88, 1],  # DEY Implied
            [0x49, 2],  # EOR Immediate
            [0x45, 2],  # EOR ZeroPage
            [0x55, 2],  # EOR Zero Page X
            [0x4D, 3],  # EOR Absolute
            [0x5D, 3],  # EOR Absolute X
            [0x59, 3],  # EOR Absolute Y
            [0x41, 2],  # EOR Indirect X
            [0x51, 2],  # EOR Indirect Y
            [0xE6, 2],  # INC Zero Page
            [0xF6, 2],  # INC Zero Page X
            [0xEE, 3],  # INC Absolute
            [0xFE, 3],  # INC Absolute X
            [0xE8, 1],  # INX Implied
            [0xC8, 1],  # INY Implied
            [0xA9, 2],  # LDA Immediate
            [0xA5, 2],  # LDA Zero Page
            [0xB5, 2],  # LDA Zero Page X
            [0xAD, 3],  # LDA Absolute
            [0xBD, 3],  # LDA Absolute X
            [0xB9, 3],  # LDA Absolute Y
            [0xA1, 2],  # LDA Indirect X
            [0xB1, 2],  # LDA Indirect Y
            [0xA2, 2],  # LDX Immediate
            [0xA6, 2],  # LDX Zero Page
            [0xB6, 2],  # LDX Zero Page Y
            [0xAE, 3],  # LDX Absolute
            [0xBE, 3],  # LDX Absolute Y
            [0xA0, 2],  # LDY Immediate
            [0xA4, 2],  # LDY Zero Page
            [0xB4, 2],  # LDY Zero Page Y
            [0xAC, 3],  # LDY Absolute
            [0xBC, 3],  # LDY Absolute Y
            [0x4A, 1],  # LSR Accumulator
            [0x46, 2],  # LSR Zero Page
            [0x56, 2],  # LSR Zero Page X
            [0x4E, 3],  # LSR Absolute
            [0x5E, 3],  # LSR Absolute X
            [0xEA, 1],  # NOP Implied
            [0x09, 2],  # ORA Immediate
            [0x05, 2],  # ORA Zero Page
            [0x15, 2],  # ORA Zero Page X
            [0x0D, 3],  # ORA Absolute
            [0x1D, 3],  # ORA Absolute X
            [0x19, 3],  # ORA Absolute Y
            [0x01, 2],  # ORA Indirect X
            [0x11, 2],  # ORA Indirect Y
            [0x48, 1],  # PHA Implied
            [0x08, 1],  # PHP Implied
            [0x68, 1],  # PLA Implied
            [0x28, 1],  # PLP Implied
            [0x2A, 1],  # ROL Accumulator
            [0x26, 2],  # ROL Zero Page
            [0x36, 2],  # ROL Zero Page X
            [0x2E, 3],  # ROL Absolute
            [0x3E, 3],  # ROL Absolute X
            [0x6A, 1],  # ROR Accumulator
            [0x66, 2],  # ROR Zero Page
            [0x76, 2],  # ROR Zero Page X
            [0x6E, 3],  # ROR Absolute
            [0x7E, 3],  # ROR Absolute X
            [0xE9, 2],  # SBC Immediate
            [0xE5, 2],  # SBC Zero Page
            [0xF5, 2],  # SBC Zero Page X
            [0xED, 3],  # SBC Absolute
            [0xFD, 3],  # SBC Absolute X
            [0xF9, 3],  # SBC Absolute Y
            [0xE1, 2],  # SBC Indrect X
            [0xF1, 2],  # SBC Indirect Y
            [0x38, 1],  # SEC Implied
            [0xF8, 1],  # SED Implied
            [0x78, 1],  # SEI Implied
            [0x85, 2],  # STA ZeroPage
            [0x95, 2],  # STA Zero Page X
            [0x8D, 3],  # STA Absolute
            [0x9D, 3],  # STA Absolute X
            [0x99, 3],  # STA Absolute Y
            [0x81, 2],  # STA Indirect X
            [0x91, 2],  # STA Indirect Y
            [0x86, 2],  # STX Zero Page
            [0x96, 2],  # STX Zero Page Y
            [0x8E, 3],  # STX Absolute
            [0x84, 2],  # STY Zero Page
            [0x94, 2],  # STY Zero Page X
            [0x8C, 3],  # STY Absolute
            [0xAA, 1],  # TAX Implied
            [0xA8, 1],  # TAY Implied
            [0xBA, 1],  # TSX Implied
            [0x8A, 1],  # TXA Implied
            [0x9A, 1],  # TXS Implied
            [0x98, 1],  # TYA Implied
            # Illegal Instructions
            [0x0B, 2],  # AAC Immediate
            [0x87, 2],  # AAX Zero Page
            [0x97, 2],  # AAX Zero Page Y
            [0x8F, 3],  # AAX Absolute
            [0x83, 2],  # AAX Indirect X
            [0x6B, 2],  # ARR Immediate
            [0x4B, 2],  # ASR Immediate
            [0xAB, 2],  # ATX Immediate
            [0x9F, 3],  # AXA Absolute Y
            [0x93, 2],  # AXA Indirect Y
            [0xCB, 2],  # AXS Immediate
            [0xC7, 2],  # DCP Zero Page
            [0xD7, 2],  # DCP Zero Page X
            [0xCF, 3],  # DCP Absolute
            [0xDF, 3],  # DCP Absolute X
            [0xDB, 3],  # DCP Absolute Y
            [0xC3, 2],  # DCP Indirect X
            [0xD3, 2],  # DCP Indirect Y
            [0xE7, 2],  # ISC Zero Page
            [0xF7, 2],  # ISC Zero Page X
            [0xEF, 3],  # ISC Absolute
            [0xFF, 3],  # ISC Absolute X
            [0xFB, 3],  # ISC Absolute Y
            [0xE3, 2],  # ISC Indirect X
            [0xF3, 2],  # ISC Indirect Y
            # [0x, ], # KIL Immediate (Untestable)
            [0xBB, 3],  # LAR Absolute Y
            [0xA7, 2],  # LAX Zero Page
            [0xB7, 2],  # LAX Zero Page Y
            [0xAF, 3],  # LAX Absolute
            [0xBF, 3],  # LAX Absolute Y
            [0xA3, 2],  # LAX Indirect X
            [0xB3, 2],  # LAX Indirect Y
            [0x27, 2],  # RLA Zero Page
            [0x37, 2],  # RLA Zero Page X
            [0x2F, 3],  # RLA Absolute
            [0x3F, 3],  # RLA Absolute X
            [0x3B, 3],  # RLA Absolute Y
            [0x23, 2],  # RLA Indirect X
            [0x33, 2],  # RLA Indirect Y
            [0x67, 2],  # RRA Zero Page
            [0x77, 2],  # RRA Zero Page X
            [0x6F, 3],  # RRA Absolute
            [0x7F, 3],  # RRA Absolute X
            [0x7B, 3],  # RRA Absolute Y
            [0x63, 2],  # RRA Indirect X
            [0x73, 2],  # RRA Indirect Y
            [0xEB, 2],  # USBC Immediate (SBC)
            [0x07, 2],  # SLO Zero Page
            [0x17, 2],  # SLO Zero Page X
            [0x0F, 3],  # SLO Absolute
            [0x1F, 3],  # SLO Absolute X
            [0x1B, 3],  # SLO Absolute Y
            [0x03, 2],  # SLO Indirect X
            [0x13, 2],  # SLO Indirect Y
            [0x47, 2],  # SRE Zero Page
            [0x57, 2],  # SRE Zero Page X
            [0x4F, 3],  # SRE Absolute
            [0x5F, 3],  # SRE Absolute X
            [0x5B, 3],  # SRE Absolute Y
            [0x43, 2],  # SRE Indirect X
            [0x53, 2],  # SRE Indirect Y
            [0x9E, 3],  # SXA Absolute Y
            [0x9C, 3],  # SYA Absolute X
            [0x8B, 2],  # XAA Immediate
            [0x9B, 3],  # XAS Absolute Y
            [0x1A, 1],  # NOP Implied
            [0x3A, 1],  # NOP Implied
            [0x5A, 1],  # NOP Implied
            [0x7A, 1],  # NOP Implied
            [0xDA, 1],  # NOP Implied
            [0xFA, 1],  # NOP Implied
            [0x80, 2],  # NOP Immediate   (SKB)
            [0x82, 2],  # NOP Immediate   (SKB)
            [0x89, 2],  # NOP Immediate   (SKB)
            [0xC2, 2],  # NOP Immediate   (SKB)
            [0xE2, 2],  # NOP Immediate   (SKB)
            [0x04, 2],  # NOP Zero Page   (IGN)
            [0x44, 2],  # NOP Zero Page   (IGN)
            [0x64, 2],  # NOP Zero Page   (IGN)
            [0x14, 2],  # NOP Zero Page X (IGN)
            [0x34, 2],  # NOP Zero Page X (IGN)
            [0x54, 2],  # NOP Zero Page X (IGN)
            [0x74, 2],  # NOP Zero Page X (IGN)
            [0xD4, 2],  # NOP Zero Page X (IGN)
            [0xF4, 2],  # NOP Zero Page X (IGN)
            [0x0C, 3],  # NOP Absolute    (IGN)
            [0x1C, 3],  # NOP Absolute X  (IGN)
            [0x3C, 3],  # NOP Absolute X  (IGN)
            [0x5C, 3],  # NOP Absolute X  (IGN)
            [0x7C, 3],  # NOP Absolute X  (IGN)
            [0xDC, 3],  # NOP Absolute X  (IGN)
            [0xFC, 3],  # NOP Absolute X  (IGN)
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(program=[data[0], 0x00])

                self.assertEqual(c.r.pc, 0)

                c.step()

                self.assertEqual(c.r.pc, data[1], f"OP: {data[0]:0>2x}")

    def test_Branch_On_Carry_Program_Counter_Correct_When_NoBranch_Occurs(
        self
    ):
        subtests = [
            [0x90, True, 2],  # BCC
            [0xB0, False, 2],  # BCS
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[(0x38 if data[1] else 0x18), data[0], 0x48]
                )

                c.step()
                currentProgramCounter = c.r.pc

                c.step()
                self.assertEqual(c.r.pc, currentProgramCounter + data[2])

    def test_Branch_On_Zero_Program_Counter_Correct_When_NoBranch_Occurs(
        self
    ):
        subtests = [
            [0xF0, False, 2],  # BEQ
            [0xD0, True, 2],  # BNE
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA9, (0x00 if data[1] else 0x01), data[0]]
                )

                self.assertEqual(c.r.pc, 0x0000)

                c.step()
                currentProgramCounter = c.r.pc

                c.step()
                self.assertEqual(c.r.pc, currentProgramCounter + data[2])

    def test_Branch_On_Negative_Program_Counter_Correct_When_NoBranch_Occurs(
        self
    ):
        subtests = [
            [0x30, False, 2],  # BMI
            [0x10, True, 2],  # BPL
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[0xA9, (0x80 if data[1] else 0x79), data[0]]
                )

                c.step()
                currentProgramCounter = c.r.pc

                c.step()
                self.assertEqual(c.r.pc, currentProgramCounter + data[2])

    def test_Branch_On_Overflow_Program_Counter_Correct_When_NoBranch_Occurs(
        self
    ):
        subtests = [
            [0x50, True, 2],  # BVC
            [0x70, False, 2],  # BVS
        ]
        for data in subtests:
            with self.subTest(data=data):
                c = self.create(
                    program=[
                        0xA9, 0x01, 0x69, (0x7F if data[1] else 0x01),
                        data[0], 0x00
                    ]
                )

                c.step()
                c.step()
                currentProgramCounter = c.r.pc

                c.step()
                self.assertEqual(c.r.pc, currentProgramCounter + data[2])

    def test_Program_Counter_Wraps_Correctly(self):
        c = self.create(program=[0x38], pc=0xFFFF)
        self.assertEqual(c.r.pc, 0xFFFF)
        c.step()
        self.assertEqual(c.r.pc, 0)


if __name__ == "__main__":
    unittest.main()

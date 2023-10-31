#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_suites
----------------------------------

Tests for `py65emu` module.
"""

import unittest
import io
import unittest.mock

from typing import Sequence
from py65emu.cpu import CPU
from py65emu.debug import Disassembly, Debug
from py65emu.mmu import MMU


class BaseDebug(unittest.TestCase):
    subtests = [
        # OPC   LO    HI
        (0xEA, 0x00, 0x00, 'NOP'),
        (0x0A, 0x00, 0x00, 'ASL A'),
        (0xA9, 0x38, 0x00, 'LDA #$38'),
        (0xAD, 0x00, 0x00, 'LDA $0000'),
        (0xAD, 0x01, 0x02, 'LDA $0201'),
        (0xA5, 0x34, 0x12, 'LDA $34'),
        (0xB5, 0x34, 0x12, 'LDA $34, X [$0035]'),
        (0xB6, 0x34, 0x12, 'LDX $34, Y [$0036]'),
        (0xAD, 0x34, 0x12, 'LDA $1234'),
        (0xBD, 0x34, 0x12, 'LDA $1234, X [$1235]'),
        (0xB9, 0x00, 0x12, 'LDA $1200, Y [$1202]'),
        (0xA1, 0x01, 0x00, 'LDA ($01, X) [$0000]'),
        (0xB1, 0x00, 0x00, 'LDA ($00), Y [$00b3]'),
        (0x6C, 0x01, 0x00, 'JMP ($0001)'),
        (0x6C, 0x01, 0x02, 'JMP ($0201)'),
        (0x6C, 0x00, 0x00, 'JMP ($0000)'),
        (0x90, 0x0C, 0x00, 'BCC $0c [$000d]'),
        (0xB0, 0x90, 0x00, 'BCS $90 [$ff91]'),
    ]

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _cpu(self, program=None, pc=0x0000, **kwargs) -> CPU:

        self.c = CPU(
            mmu=MMU([(0x0, 0x10000, False, program, pc)]),
            pc=pc,
            **kwargs
        )
        self.c.r.a = 0xAA
        self.c.r.x = 0x01
        self.c.r.y = 0x02

        return self.c

    def data_to_str(self, data: Sequence, pc: int | None = None) -> str:
        o = self.c.opcodes[data[0]]
        op = {
            "pc":     pc if pc is not None else self.c.r.pc,
            "opc":    data[0],
            "memory": data[3],
            "lo":     0x00,
            "hi":     0x00,
        }
        if o.bytes == 2:
            op['lo'] = data[1]
        elif o.bytes == 3:
            op['lo'] = data[1]
            op['hi'] = data[2]

        val = f'${op["pc"]:0>4x} {op["opc"]:0>2x} {op["lo"]:0>2x} '\
              f'{op["hi"]:0>2x} {o.opname: >3s}: {op["memory"]}'
        return val.strip()

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def assert_stdout(self, n, expected_output, mock_stdout):
        call = getattr(n[0], n[1])
        call()
        self.assertIn(mock_stdout.getvalue(), expected_output)

    def test_debug_active(self):
        # OPC   LO    HI
        data = (0xEA, 'NOP')
        c = self._cpu(program=[data[0],], debug=True)
        expected_pc = (c.r.pc + 1)
        expected = (
            '${r.pc:0>4x} {data[0]:0>2x} 00 00 {data[1]:s}: '
            '{data[1]: <24s} A: {r.a:0>2X} X: {r.x:0>2X} Y: {r.y:0>2X} '
            'S: {r.s:0>2X} PC: {pc1:0>4X} P: {r.p:0>2X}'
        )
        v = expected.format(
                data=data,
                pc1=expected_pc,
                r=c.r,
            )

        with unittest.mock.patch(
            'sys.stdout', new_callable=io.StringIO
        ) as mock_stdout:
            c.step()
            self.assertIn(
                v,
                mock_stdout.getvalue().strip(),
            )


class TestDisassembly(BaseDebug):
    def _dasm(self, **kwargs) -> Disassembly:
        self._cpu(**kwargs)
        opc = self.c.nextByte()
        op = self.c.opcodes[opc]

        dasm = Disassembly(op, *op.get_operands())
        return dasm

    def test_initialized_correct(self):
        for data in self.subtests:
            with self.subTest(data=data):
                dasm = self._dasm(program=[data[0], data[1], data[2]])

                self.assertIsInstance(dasm, Disassembly)

                self.assertEqual(dasm.op.opcode, data[0], f'0x{data[0]:0>2X}')
                if dasm.op.bytes == 2:
                    self.assertEqual(dasm.lo, data[1], f'0x{data[0]:0>2X}')
                elif dasm.op.bytes == 3:
                    self.assertEqual(dasm.lo, data[1], f'0x{data[0]:0>2X}')
                    self.assertEqual(dasm.hi, data[2], f'0x{data[0]:0>2X}')

    def test_as_word(self):
        for data in self.subtests:
            with self.subTest(data=data):
                dasm = self._dasm(program=[data[0], data[1], data[2]])
                dasm.lo = data[1]
                dasm.hi = data[2]

                word = (data[2] << 8) + data[1]

                self.assertEqual(dasm.op.opcode, data[0], f'0x{data[0]:0>2X}')
                self.assertEqual(dasm.as_word(), word, f'0x{data[0]:0>2X}')

    def test_memory_string(self):
        for data in self.subtests:
            with self.subTest(data=data):
                dasm = self._dasm(program=[data[0], data[1], data[2]])

                self.assertEqual(dasm.op.opcode, data[0], f'0x{data[0]:0>2X}')
                self.assertEqual(
                    dasm.memory.strip(),
                    data[3],
                    f'0x{data[0]:0>2X}'
                )

    def test_repr_string(self):
        for data in self.subtests:
            with self.subTest(data=data):
                dasm = self._dasm(program=[data[0], data[1], data[2]])

                self.assertEqual(dasm.op.opcode, data[0], f'0x{data[0]:0>2X}')
                self.assertEqual(
                    repr(dasm).strip(),
                    self.data_to_str(data, 0x0000),
                    f'0x{data[0]:0>2X}'
                )


class TestDebug(BaseDebug):
    def _debug(self, **kwargs) -> Debug:
        self._cpu(**kwargs)
        debug = Debug(self.c)

        return debug

    def memdump_str(
        self,
        addr: int,
        data: tuple[int, int, int, str] | None = None,
        suffix: bool = False
    ) -> str:
        memory_prefix = (
            "MEMORY DUMP FOR: ${start:0>4x} - ${stop:0>4x}\n"
            "ADDR 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F\n"
        )
        memory_suffix = " {offset: >{width}} ^^"

        res = [0x0] * 17
        res[0] = addr
        if data is not None and len(data) > 0:
            for k in range(3):
                val = data[k]
                if type(val) is int:
                    res[(k + 1)] = val

        row = "{:0>4x}"
        row += " {:0>2x}" * 16

        t = memory_prefix + row.format(*res)
        if suffix is True:
            t += "\n" + memory_suffix
        return t

    def test_initialized_correct(self):
        for data in self.subtests:
            with self.subTest(data=data):
                debug = self._debug(program=[data[0], data[1], data[2]])

                self.assertIsInstance(debug, Debug)
                self.assertIsInstance(debug.cpu, CPU)

    def test_get_assembly(self):
        for data in self.subtests:
            with self.subTest(data=data):
                dbg = self._debug(program=[data[0], data[1], data[2]])
                o = self.c.opcodes[data[0]]

                bytes, dasm = dbg._get_assembly(self.c.r.pc)

                self.assertEqual(
                    o.opcode,
                    data[0],
                    f'0x{data[0]:0>2X} - OPC'
                )
                self.assertEqual(
                    o.bytes,
                    bytes,
                    f'0x{data[0]:0>2X} - Bytes'
                )
                self.assertEqual(
                    repr(dasm).strip(),
                    self.data_to_str(data),
                    f'0x{data[0]:0>2X} - Repr.'
                )

    def test_get_memory(self):
        for data in self.subtests:
            with self.subTest(data=data):
                dbg = self._debug(program=[data[0], data[1], data[2]])

                memory = dbg._get_memory(0x0000, 0x0003)
                mem = memory.pop()

                self.assertEqual(mem[0], 0x00)  # Memory Offset
                self.assertEqual(mem[1], data[0])
                self.assertEqual(mem[2], data[1])
                self.assertEqual(mem[3], data[2])

    def test_shorthand_d(self):
        for data in self.subtests:
            with self.subTest(data=data):
                dbg = self._debug(program=[data[0], data[1], data[2]])
                self.c.r.pc = 0x0000

                with unittest.mock.patch(
                    'sys.stdout', new_callable=io.StringIO
                ) as mock_stdout:
                    dbg.d(0x0000)
                    self.assertIsNotNone(mock_stdout.getvalue())

    def test_crash_dump(self):
        for data in self.subtests:
            with self.subTest(data=data):
                cpu = self._cpu(program=[data[0], data[1], data[2]])
                self.c.r.pc = 0x0000

                v = 'DISASSEMBLE: $0000 - $0000\n'\
                    'OP LO HI OPS DISASSEMBLY\n'\
                    + self.data_to_str(data)

                with unittest.mock.patch(
                    'sys.stdout', new_callable=io.StringIO
                ) as mock_stdout:
                    Debug.crash_dump(cpu)
                    self.assertIn(mock_stdout.getvalue().strip(), v)

    def test_disassemble(self):
        for data in self.subtests:
            with self.subTest(data=data):
                dbg = self._debug(program=[data[0], data[1], data[2]])

                start = 0
                stop = 0
                v = f'DISASSEMBLE: ${start:0>4x} - ${stop:0>4x}\n'\
                    'OP LO HI OPS DISASSEMBLY\n' + self.data_to_str(data)

                with unittest.mock.patch(
                    'sys.stdout', new_callable=io.StringIO
                ) as mock_stdout:
                    dbg.disassemble()
                    self.assertIn(mock_stdout.getvalue().strip(), v)

    def test_disassemble_with_negative_stop(self):
        for data in self.subtests:
            with self.subTest(data=data):
                dbg = self._debug(program=[data[0], data[1], data[2]])

                start = 0
                stop = 0
                v = f'DISASSEMBLE: ${start:0>4x} - ${stop:0>4x}\n'\
                    'OP LO HI OPS DISASSEMBLY\n' + self.data_to_str(data)
                stop = -1

                with unittest.mock.patch(
                    'sys.stdout', new_callable=io.StringIO
                ) as mock_stdout:
                    dbg.disassemble(start, stop)
                    self.assertIn(mock_stdout.getvalue().strip(), v)

    def test_disassemble_list(self):
        for data in self.subtests:
            with self.subTest(data=data):
                dbg = self._debug(program=[data[0], data[1], data[2]])

                res = dbg.disassemble_list(0x0000, 0x0000)
                self.assertIsInstance(res, list)
                self.assertEqual(len(res), 1)

    def test_memdump(self):
        for data in self.subtests:
            with self.subTest(data=data):
                dbg = self._debug(program=[data[0], data[1], data[2]])

                offset = 0x000F
                offset_width = ((offset | 0x000F) - ((offset & 0xFFF0)) + 1)

                t = self.memdump_str(
                    addr=(offset & 0xFFF0),
                    data=data,
                    suffix=True
                )

                v = t.format(
                    start=(offset & 0xFFF0),
                    stop=(offset | 0x000F),
                    offset="",
                    width=(offset_width * 3)
                )
                with unittest.mock.patch(
                    'sys.stdout', new_callable=io.StringIO
                ) as mock_stdout:
                    dbg.memdump(offset)
                    self.assertIn(mock_stdout.getvalue().strip(), v)

    def test_stackdump(self):
        for data in self.subtests:
            with self.subTest(data=data):
                dbg = self._debug(program=[data[0], data[1], data[2]])

                offset = 0x01FD
                offset_width = ((offset | 0x000F) - ((offset & 0xFFF0)) + 1)

                t = self.memdump_str(
                    addr=(offset & 0xFFF0),
                    suffix=True
                )

                v = t.format(
                    start=(offset & 0xFFF0),
                    stop=(offset | 0x000F),
                    offset="",
                    width=(offset_width * 3)
                )
                with unittest.mock.patch(
                    'sys.stdout', new_callable=io.StringIO
                ) as mock_stdout:
                    dbg.stackdump()
                    self.assertIn(mock_stdout.getvalue().strip(), v)


if __name__ == "__main__":
    unittest.main()

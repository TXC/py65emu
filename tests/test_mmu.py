#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mmu
----------------------------------
"""

import os
import unittest

from py65emu.mmu import MMU, MemoryRangeError, ReadOnlyError


class TestMMU(unittest.TestCase):
    def setUp(self):
        pass

    def test_create_empty(self):
        MMU([])

    def test_create(self):
        MMU([(0, 128, False, None)])

        MMU([(0, 128, False, None), (128, 128, True, None)])

    def test_create_with_list(self):
        m = MMU([(0, 128, False, [1, 2, 3])])

        self.assertEqual(m.blocks[0].get(0), 1)
        self.assertEqual(m.blocks[0].get(2), 3)
        self.assertEqual(m.blocks[0].get(3), 0)

    def test_create_with_file(self):
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "files",
            "test_load_file.bin"
        )

        with open(path, "rb") as f:
            m = MMU([(0, 128, True, f)])

        self.assertEqual(m.blocks[0].get(0), 0xA9)

    def test_create_overlapping(self):
        with self.assertRaises(MemoryRangeError):
            MMU([(0, 129), (128, 128)])

    def test_addBlock(self):
        m = MMU([])
        m.addBlock(0, 128, False, None)
        m.addBlock(128, 128, True, None)

    def test_addBlock_overlapping(self):
        m = MMU([])
        m.addBlock(128, 128)
        with self.assertRaises(MemoryRangeError):
            m.addBlock(0, 129)
        with self.assertRaises(MemoryRangeError):
            m.addBlock(255, 128)

    def test_write(self):
        m = MMU([(0, 128)])
        m.cpu_write(16, 25)
        self.assertEqual(m.blocks[0].get(16), 25)

    def test_write_multiple_blocks(self):
        m = MMU([(0, 128), (1024, 128)])
        m.cpu_write(16, 25)
        self.assertEqual(m.blocks[0][16], 25)
        m.cpu_write(1056, 55)
        self.assertEqual(m.blocks[1][32], 55, m.blocks[1])

    def test_write_readonly(self):
        m = MMU([(0, 16, True), (16, 16), (32, 16, True)])
        with self.assertRaises(ReadOnlyError):
            m.cpu_write(8, 1)
        m.cpu_write(20, 1)
        with self.assertRaises(ReadOnlyError):
            m.cpu_write(40, 1)

    def test_read(self):
        m = MMU([(0, 128)])
        m.cpu_write(16, 5)
        m.cpu_write(64, 111)
        self.assertEqual(5, m.cpu_read(16))
        self.assertEqual(111, m.cpu_read(64))

    def test_index_error(self):
        m = MMU([(0, 128)])
        with self.assertRaises(IndexError):
            m.cpu_write(-1, 0)
        with self.assertRaises(IndexError):
            m.cpu_write(128, 0)
        with self.assertRaises(IndexError):
            m.cpu_read(-1)
        with self.assertRaises(IndexError):
            m.cpu_read(128)
        with self.assertRaises(IndexError):
            b = m.getBlock(127)
            _ = b[256]
        with self.assertRaises(IndexError):
            b = m.getBlock(127)
            b[256] = 0
        with self.assertRaises(IndexError):
            b = m.getBlock(127)
            b.get(256)
        with self.assertRaises(IndexError):
            b = m.getBlock(127)
            b.set(256, 0)

    def test_that_repr_is_strin(self):
        m = MMU([(0, 128)])
        b = m.getBlock(120)
        s = repr(b)
        self.assertIsInstance(s, str)

    def test_reset(self):
        m = MMU([(0, 16, True), (16, 16, False)])
        m.blocks[0][0] = 5
        m.cpu_write(16, 10)
        m.reset()
        self.assertEqual(m.cpu_read(0), 5)
        self.assertEqual(m.cpu_read(16), 0)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()

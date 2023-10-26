#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_timing
----------------------------------

Timing tests for `py65emu` module.
"""


import os
import unittest
import csv

from py65emu.cpu import CPU
from py65emu.mmu import MMU


class Timing(unittest.TestCase):
    def setUp(self):
        pass

    def load_csv_data(self) -> dict[int, dict[str, object]]:
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "files",
            "6502_cycle_test.csv"
        )

        cycle_test_data_results = {}
        with open(path) as fp:
            test_data = csv.reader(fp, delimiter=",", quotechar='"')
            line_number = 0
            for row in test_data:
                if int(row[0]) % 2 != 0:
                    line_number += 1
                    continue

                if row[8].strip() == "":
                    continue

                cycle_test_data_results[line_number] = {
                    "PC": int(row[1], 16),
                    "A": int(row[2], 16),
                    "X": int(row[3], 16),
                    "Y": int(row[4], 16),
                    "P": int(row[5], 16),
                    "SP": int(row[6], 16),
                    "CC": int(row[7]),
                    "CMD": row[8],
                }

        return cycle_test_data_results

    def test_cycle_test(self):
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "files",
            "6502_cycle_test.bin"
        )

        mmu = MMU()
        with open(path, "rb") as f:
            mmu.addBlock(0x0000, 0x10000, False, f)

        c = CPU(mmu, 0x00)
        c.r.s = 0xFD

        total_no_of_cycles = 0
        loop = 1

        cycle_test_data_results = self.load_csv_data()

        while 1:
            # c.step()
            # total_no_of_cycles = total_no_of_cycles + c.cc

            if loop not in cycle_test_data_results:
                loop += 1
                continue

            self.assertEqual(
                c.r.pc,
                cycle_test_data_results[loop]["PC"],
                "Step: {:d} PC: {:0>4x} - {} - {}".format(
                    loop,
                    c.r.pc,
                    cycle_test_data_results[loop]["CMD"],
                    c.r
                )
            )
            self.assertEqual(
                total_no_of_cycles,
                cycle_test_data_results[loop]["CC"],
                "Step: {:d} Cycle: {:d} - {} - {}".format(
                    loop,
                    total_no_of_cycles,
                    cycle_test_data_results[loop]["CMD"],
                    c.r,
                )
            )
            c.step()
            total_no_of_cycles += c.cc

            loop += 1

            if c.r.pc == 0x1266:
                break

            self.assertLess(
                total_no_of_cycles, 1147, "Maximum number of cycles exceeded"
            )
            self.assertLess(loop, 2295, "Maximum number of loops exceeded")

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()

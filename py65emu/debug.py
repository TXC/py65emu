from typing import TYPE_CHECKING
import math

if TYPE_CHECKING:
    from py65emu.operator import Operation
    from py65emu.cpu import CPU

MemoryType = tuple[int,
                   int, int, int, int,
                   int, int, int, int,
                   int, int, int, int,
                   int, int, int, int,
                   ]


class Disassembly:
    def __init__(
            self,
            op: "Operation",
            pc: int,
            hi: int = 0x00,
            lo: int = 0x00
    ):
        self.op = op
        self.pc = pc
        self.hi = hi
        self.lo = lo

    @property
    def memory(self) -> int:
        prefix = f"{self.op.name} "
        match self.op.mode:
            case "acc":
                return prefix + "A "
            case "imp":
                return prefix
            case "im":
                return prefix + f"#${self.lo:0>2x} "
            case "z":
                return prefix + f"${self.lo:0>2x} "
            case "zx":
                return prefix + f"${self.lo:0>2x}, X "
            case "zy":
                return prefix + f"${self.lo:0>2x}, Y "
            case "a":
                return prefix + f"${self.lo:0>2x}{self.hi:0>2x} "
            case "ax":
                return prefix + f"${self.lo:0>2x}{self.hi:0>2x}, X "
            case "ay":
                return prefix + f"${self.lo:0>2x}{self.hi:0>2x}, Y "
            case "i":
                return prefix + f"(${self.lo:0>2x}{self.hi:0>2x}) "
            case "ix":
                return prefix + f"(${self.lo:0>2x}, X) "
            case "iy":
                return prefix + f"(${self.lo:0>2x}), Y "
            case "rel":
                addr = self.pc + 1
                if self.lo & 0x80:
                    addr -= (self.lo ^ 0xff) + 1
                else:
                    addr += self.lo
                return prefix + f"${self.lo:0>2x} [{addr:0>4x}] "

    def __repr__(self) -> str:
        return f"${self.pc:0>4x} {self.op.opcode:0>2x} {self.lo:0>2x} "\
               f"{self.hi:0>2x} {self.op.opname: >3s}: {self.memory}"


class Debug:
    def __init__(self, cpu: "CPU"):
        self.cpu = cpu

    def _get_assembly(self, addr: int) -> tuple[int, "Disassembly"]:
        """
        This is the disassembly function.
        Its workings are not required for emulation.
        It is merely a convenience function to turn the binary instruction
        code into human readable form. Its included as part of the emulator
        because it can take advantage of many of the CPUs internal operations
        to do

        :param int addr: address to disassemble
        """
        addr_org = addr
        opcode = self.cpu.mmu.read(addr)
        addr += 1
        hi = 0x00
        lo = 0x00

        op = self.cpu.operators[opcode]
        if op.bytes == 2:
            lo = self.cpu.mmu.read(addr)
            addr += 1
        elif op.bytes == 3:
            lo = self.cpu.mmu.read(addr)
            addr += 1
            hi = self.cpu.mmu.read(addr)
            addr += 1

        assembly = Disassembly(
            op=op,
            pc=addr_org,
            hi=hi,
            lo=lo
        )

        return addr, assembly

    def _get_memory(self, start: int, stop: int) -> list[MemoryType]:
        """
        Dumps a portion of the memory.
        This reads over blocks.

        :param int start: Start offset
        :param int stop: Stop offset
        """
        start_offset = start & 0xFFF0
        stop_offset = stop | 0x000F

        offset_length = math.ceil((stop_offset - start_offset) / 16)

        memory = []

        for multiplier in range(offset_length):
            offset = start_offset + (multiplier * 0x0010)

            value = (offset,)
            for addr in range(0x10):
                res = self.cpu.mmu.read(offset + addr),
                value += res

            memory.append(value)
        return memory

    def d(self, addr: int) -> None:
        self.memdump(addr)
        self.disassemble(addr)

    """Disassembly methods."""
    def disassemble(
        self, start: int | None = None, stop: int | None = None
    ) -> None:
        """
        :param int|None start: The starting address. (Default: PC - 20)
        :param int|None stop: The stopping address. If stop is lower than start
            then parameter will be used as "length" (ie. incr. by start)
            instead. (Default: PC)
        """
        addr = start & 0xFFFF

        if start is None:
            start = self.cpu.r.pc - 10

        if stop is None:
            stop = self.cpu.r.pc
        if stop < start:
            stop += start

        print(
            "DISASSEMBLE: ${:0>4x} - ${:0>4x}\n"
            "OP LO HI OPS DISASSEMBLY"
            .format(start, stop)
        )

        while addr <= (stop & 0xFFFF):
            addr, obj = self._get_assembly(addr)
            print(f"{obj!r}")

    def disassemble_list(self, start: int, stop: int) -> list[Disassembly]:
        """
        This disassembly function will turn a chunk of binary code into human
        readable form.
        See the above function for a more descriptive text
        """
        addr = start & 0xFFFF

        if stop is None:
            stop = start
        if stop < start:
            stop += start

        lines = []
        while addr <= (stop & 0xFFFF):
            cur = addr
            addr, assembly = self._get_assembly(addr)
            lines[cur] = assembly
        return lines

    """Memory methods."""
    def memdump(self, start: int, stop: int | None = None) -> None:
        """
        Prints a portion of the memory.
        This reads over blocks.

        :param int start: Start offset
        :param int | None stop: Stop offset. (Default: None)
        """
        offset = None
        if stop is None:
            offset = start - ((start & 0xFFF0) + 1)
            start = start & 0xFFF0
            stop = (start | 0x000F)

        print(
            f"MEMORY DUMP FOR: ${start:0>4x} - ${stop:0>4x}\n"
            "ADDR 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F"
        )

        memory = self._get_memory(start, stop)
        output = "{:0>4x}"
        output += " {:0>2x}" * 16

        for row in memory:
            print(output.format(*row))

        if offset is not None:
            offset += 2
            print("".ljust((offset * 3) + 1, " ") + " ^^")

    def stackdump(self, pointer: int = 0) -> None:
        self.memdump((self.cpu.stack_page * 0x100) + pointer)

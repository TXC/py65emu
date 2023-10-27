from typing import TYPE_CHECKING
import math

if TYPE_CHECKING:
    from py65emu.operation import Operation
    from py65emu.cpu import CPU


class Disassembly:
    """
    Disassembly class, to help disassemble a program in memory
    """

    def __init__(
        self,
        op: "Operation",
        pc: int,
        hi: int = 0x00,
        lo: int = 0x00
    ):
        """
        Init Disassembly Method

        :param Operation op: Operation to disassemble
        :param int pc: Current programcounter
        :param int hi: Hi-byte of memory. (Default: `0x00`)
        :param int lo: Lo-byte of memory. (Default: `0x00`)
        """
        self.op = op
        self.pc = pc
        self.hi = hi
        self.lo = lo

    def as_word(self) -> int:
        """Returns hi & lo byte as word (16 bit)"""
        return (self.hi << 8) + self.lo

    @property
    def memory(self) -> str:
        """
        Returns mnemonic string with current memory addresses
        """
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
                addr = (self.lo + self.op.cpu.r.x) & 0xFFFF
                return prefix + f"${self.lo:0>2x}, X [${addr:0>4x}]"
            case "zy":
                addr = (self.lo + self.op.cpu.r.y) & 0xFFFF
                return prefix + f"${self.lo:0>2x}, Y [${addr:0>4x}]"
            case "a":
                return prefix + f"${self.hi:0>2x}{self.lo:0>2x} "
            case "ax":
                addr = (self.as_word() + self.op.cpu.r.x) & 0xFFFF
                return prefix + (
                    f"${self.hi:0>2x}{self.lo:0>2x}, "
                    f"X [${addr:0>4x}]"
                )
            case "ay":
                addr = (self.as_word() + self.op.cpu.r.y) & 0xFFFF
                return prefix + (
                    f"${self.hi:0>2x}{self.lo:0>2x}, "
                    f"Y [${addr:0>4x}]"
                )
            case "i":
                return prefix + f"(${self.hi:0>2x}{self.lo:0>2x}) "
            case "ix":
                loc_addr = (self.lo + self.op.cpu.r.x) & 0xFFFF
                addr = self.op.cpu.mmu.readWord(loc_addr) & 0xFFFF
                return prefix + f"(${self.lo:0>2x}, X) [${addr:0>4x}]"
            case "iy":
                loc_addr = self.op.cpu.mmu.readWord(self.lo) & 0xFFFF
                addr = loc_addr + self.op.cpu.r.y
                return prefix + f"(${self.lo:0>2x}), Y [${addr:0>4x}]"
            case _:  # Actually: self.op.mode == 'rel'
                addr = (
                    (self.pc + 1) + self.op.cpu.fromTwosCom(self.lo)
                ) & 0xFFFF
                return prefix + f"${self.lo:0>2x} [${addr:0>4x}] "

    def __repr__(self) -> str:
        """Returns everything nicely formated"""
        return f"${self.pc:0>4x} {self.op.opcode:0>2x} {self.lo:0>2x} "\
               f"{self.hi:0>2x} {self.op.opname: >3s}: {self.memory}"


class Debug:
    """
    Debug class, do help debug a program or the module itself.
    """

    def __init__(self, cpu: "CPU"):
        """
        :param CPU cpu: The CPU object to debug
        """
        self.cpu = cpu

    def _get_assembly(self, addr: int) -> tuple[int, "Disassembly"]:
        """
        This is the disassembly function.
        Its workings are not required for emulation.

        It is merely a convenience function to turn the binary instruction
        code into human readable form. Its included as part of the emulator
        because it can take advantage of many of the CPUs internal operations
        to do

        :mode private:
        :param int addr: address to disassemble
        :rtype: tuple[int, Disassembly]
        :return: Tuple with the current address and corresponding
                 Disassembly object
        """
        addr_org = addr
        opcode = self.cpu.mmu.read(addr)
        addr += 1
        hi = 0x00
        lo = 0x00

        op = self.cpu.opcodes[opcode]
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

    def _get_memory(self, start: int, stop: int) -> list[tuple[int, ...]]:
        """
        Dumps a portion of the memory.
        This reads over blocks.

        :mode private:
        :param int start: Start offset
        :param int stop: Stop offset
        :rtype: list[tuple[int, ...]]
        :return: A list with a tuple of the memory data from the selected
                 memory span
        """
        start_offset = start & 0xFFF0
        stop_offset = stop | 0x000F

        offset_length = math.ceil((stop_offset - start_offset) / 16)

        memory = []

        for multiplier in range(offset_length):
            offset = start_offset + (multiplier * 0x0010)

            value = [offset,]
            for addr in range(0x10):
                value += self.cpu.mmu.read(offset + addr),

            memory.append(tuple(value))
        return memory

    def _disassemble_params(
        self, start: int | None = None, stop: int | None = None
    ) -> tuple[int, int, int]:
        if start is None:
            start = self.cpu.r.pc - 10
        if start < 0:
            start = 0

        addr = start & 0xFFFF

        if stop is None:
            stop = self.cpu.r.pc
        if stop < start:
            stop = start

        return start, stop, addr

    def d(self, addr: int) -> None:
        """
        Shorthand method for print out dump of memory and disassembly for
        selected address

        :param addr: Address to dump
        :type addr: int
        """
        self.memdump(addr)
        self.disassemble(addr)

    @staticmethod
    def crash_dump(cpu: "CPU") -> None:
        """
        Static method to dump out backtrace of the program that was running
        """
        entry = cpu.mmu.read(0xFFFD) + (cpu.mmu.read(0xFFFC) << 8)

        pc = cpu.r.pc - 20
        if pc < entry:
            pc = entry

        d = Debug(cpu)
        d.disassemble(pc, cpu.r.pc)

    """Disassembly methods."""
    def disassemble(
        self, start: int | None = None, stop: int | None = None
    ) -> None:
        """
        :param start: The starting address. (Default: PC - 20)
        :param stop: The stopping address. If stop is lower than start then
                     parameter will be used as "length" (ie. incr. by start)
                     instead. (Default: PC)
        :type start: int | None
        :type stop: int | None
        """
        start, stop, addr = self._disassemble_params(start, stop)

        print(
            "DISASSEMBLE: ${:0>4x} - ${:0>4x}\n"
            "OP LO HI OPS DISASSEMBLY"
            .format(start, stop)
        )

        while addr <= (stop & 0xFFFF):
            addr, obj = self._get_assembly(addr)
            print(f"{obj!r}")

    def disassemble_list(
        self, start: int | None = None, stop: int | None = None
    ) -> list[Disassembly]:
        """
        See :py:meth:`py65emu.debug.Debug.disassemble` for a more descriptive
        text

        .. seealso::
           :py:meth:`py65emu.debug.Debug.disassemble`

        :param start: The starting address. (Default: PC - 20)
        :param stop: The stopping address. If stop is lower than start then
                     parameter will be used as "length" (ie. incr. by start)
                     instead. (Default: PC)
        :type start: int | None
        :type stop: int | None
        :rtype: list[Disassembly]
        :return: A list of Disassembly objects
        """
        start, stop, addr = self._disassemble_params(start, stop)

        lines: list[Disassembly] = []
        while addr <= (stop & 0xFFFF):
            cur = addr
            addr, assembly = self._get_assembly(addr)
            lines.insert(cur, assembly)
        return lines

    """Memory methods."""
    def memdump(self, start: int, stop: int | None = None) -> None:
        """
        Prints the memory data between `start` and `stop` parameters.
        If `stop` parameter is left out, it will dump the 16 byte segment where
        the `start` address exists.
        (`start = start & 0xFFF0` and `stop = start | 0x000F`)

        This method reads over blocks.

        :param int start: Start offset
        :param int | None stop: Stop offset. (Default: None)
        """
        offset = None
        if stop is None:
            offset = ((start | 0x000F) - ((start & 0xFFF0)) + 1)
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
            print("".ljust((offset * 3) + 1, " ") + " ^^")

    def stackdump(self, pointer: int | None = None) -> None:
        """
        Convience method to dump the stack around the current stackpointer
        (Register S)

        :param pointer: Stack Pointer Location
        :type pointer: int | None
        """
        if not pointer:
            pointer = self.cpu.r.s

        self.memdump((self.cpu.stack_page * 0x100) + pointer)

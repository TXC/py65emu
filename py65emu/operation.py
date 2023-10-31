import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py65emu.cpu import CPU

InstructionConfigType = int | str | tuple[str, str | bool]

InstructionType = tuple[str, str, int, str, InstructionConfigType | None]


class UndefinedOperation(LookupError):
    pass


class Operation:
    cpu: "CPU"
    """CPU Object"""

    opcode: int
    """Opcode"""

    name: str
    """Operation mnemonic"""

    mode: str
    """Address mode for operation"""
    cycles: int
    """Number of cycles operation should take"""

    _type: str
    """Address mode type"""

    config: InstructionConfigType | None = None
    """Operation configuration"""

    def __init__(
        self,
        cpu: "CPU",
        opcode: int,
        name: str,
        mode: str,
        cycles: int,
        type: str,
        config: InstructionConfigType | None = None
    ):
        """
        Operation Object

        :param CPU cpu: CPU Object
        :param int opcode: Opcode
        :param str name: Operation mnemonic
        :param str mode: Address mode for operation
        :param int cycles: No. of cycles operation should take
        :param str type: Address mode type
        :param InstructionConfigType config: Operation configuration
        """
        self.cpu: "CPU" = cpu
        self.opcode = opcode
        self.name = name
        self.mode = mode
        self.cycles = cycles
        self.config = config
        self._type = type

    def target(self, config: InstructionConfigType) -> InstructionConfigType:
        """Wrapper method to return static configuration"""
        return config

    def execute(self) -> None:
        """Execute operation"""

        op_f = getattr(self.cpu, self.opname)
        if self.config:
            a_f = functools.partial(self.target, self.config)
        else:
            a_f = getattr(self.cpu, self.amode)

        op_f(a_f())

    def get_operands(self, addr: int | None = None) -> tuple[int, int, int]:
        """
        Read operands from memory

        :param addr: Address to read from, if None, read from CPU Register PC
        :type addr: int | None
        :rtype: tuple[int, int, int]
        :return: Tuple with address, hi-byte, lo-byte
        """

        if addr is None:
            addr = self.cpu.r.pc - 1

        hi = 0x00
        lo = 0x00
        if self.bytes == 2:
            lo = self.cpu.mmu.read(addr + 1)
        elif self.bytes == 3:
            lo = self.cpu.mmu.read(addr + 1)
            hi = self.cpu.mmu.read(addr + 2)

        return (addr, hi, lo)

    @property
    def bytes(self) -> int:
        """
        Number of bytes operation reads

        :meta private:
        """
        if self.mode in ['acc', 'imp']:
            return 1
        elif self.mode in ['im', 'z', 'zx', 'zy', 'ix', 'iy', 'rel']:
            return 2
        else:  # ['a', 'ax', 'ay', 'i']:
            return 3

    @property
    def amode(self) -> str:
        """
        Method name for Address Mode

        :meta private:
        """
        if self._type == "v":
            return self.mode

        return "{}_a".format(self.mode)

    @property
    def opname(self) -> str:
        """
        Method name for mnemonic

        :meta private:
        """

        if self.name in [
            "BPL", "BMI", "BVC", "BVS", "BCC", "BCS", "BNE", "BEQ"
        ]:
            return "B"
        elif self.name in ["CLC", "CLI", "CLV", "CLD"]:
            return "CL"
        elif self.name in ["SEC", "SEI", "SED"]:
            return "SE"
        elif self.name in ["PHA", "PLA", "PHP", "PLP"]:
            return "P"
        elif self.name in ["TAX", "TXA", "TAY", "TYA", "TXS", "TSX"]:
            return "T"
        elif self.name in ["SKB", "IGN"]:
            return "NOP"
        return self.name

    def __repr__(self) -> str:
        prefix = f"{self.opcode:0>2X}: {self.name} "

        match self.mode:
            case "acc":
                return prefix + "A "
            case "imp":
                return prefix
            case "im":
                return prefix + "#$BB "
            case "z":
                return prefix + "$LL "
            case "zx":
                return prefix + "$LL, X "
            case "zy":
                return prefix + "$LL, Y "
            case "a":
                return prefix + "$LLHH "
            case "ax":
                return prefix + "$LLHH, X "
            case "ay":
                return prefix + "$LLHH, Y "
            case "i":
                return prefix + "($LLHH) "
            case "ix":
                return prefix + "($LL, X) "
            case "iy":
                return prefix + "($LL), Y "
            case _:  # Actually: self.mode == 'rel'
                return prefix + "$BB [PC + $BB] "


class OpCodes:
    """
    OpCodes
    For each operation have the opcode as key, and 5-tuples containing the name
    of the operation, addressing mode, the base number of cycles it should
    take, whether it acts on values, "v" or on addresses, "a" and a target
    register, if valid.

    Addressing modes:

    =====  ====
    Func.  Description
    =====  ====
    acc    Accumulator (i.e CPU Register A)
    im     Immediate
    imp    Implied
    z      Zeropage
    zx     Zeropage, X-indexed
    zy     Zeropage, Y-indexed
    a      Absolute
    ax     Absolute, X-indexed
    ay     Absolute, Y-indexed
    i      Indirect
    ix     X-indexed, Indirect
    iy     Indirect, Y-indexed
    rel    Relative
    =====  ====
    """

    ops: list[Operation | None]

    def __init__(self, cpu: "CPU"):
        """
        Object to hold the instruction set

        :param CPU cpu: CPU Object
        """
        self.cpu = cpu
        self.ops = [None] * 0x100

        for opcode, config in self.instructions().items():
            if opcode not in self.ops:
                self.ops[opcode] = Operation(cpu, opcode, *config)

    def __getitem__(self, key: int) -> Operation:
        """
        Get a single operation

        :param int key: Opcode to get
        :rtype: Operation
        :return: Operation object for Opcode
        :raises: UndefinedOperation
        """
        op = next(
            (
                x for x in self.ops
                if isinstance(x, Operation) and x.opcode == key
            ),
            None
        )

        if isinstance(op, Operation):
            return op
        raise UndefinedOperation(
            'Operation {0:d} ({0:0>2X}) not instantiated'.format(key)
        )

    def instructions(self) -> dict[int, InstructionType]:
        """
        Returns 6502 instruction set (incl. illegal instructions)

        :rtype: dict[int, InstructionType]
        :returns: Full instruction set
        """
        return {
            # region Instructions 0x0"
            0x00: ("BRK", "imp", 7, "v", 1),
            0x01: ("ORA", "ix", 6, "v", None),
            0x02: ("KIL", "imp", 1, "v", 1),
            0x03: ("SLO", "ix", 8, "a", None),
            0x04: ("IGN", "z", 3, "a", None),
            0x05: ("ORA", "z", 3, "v", None),
            0x06: ("ASL", "z", 5, "a", None),
            0x07: ("SLO", "z", 5, "a", None),
            0x08: ("PHP", "imp", 3, "a", ("PH", "p")),
            0x09: ("ORA", "im", 2, "v", None),
            0x0A: ("ASL", "acc", 2, "a", "a"),
            0x0B: ("AAC", "im", 2, "v", None),  # ANC
            0x0C: ("IGN", "a", 4, "a", None),
            0x0D: ("ORA", "a", 4, "v", None),
            0x0E: ("ASL", "a", 6, "a", None),
            0x0F: ("SLO", "a", 6, "a", None),
            # endregion Instructions 0x0

            # region Instructions 0x1
            0x10: ("BPL", "rel", 2, "v", ("N", False)),
            0x11: ("ORA", "iy", 5, "v", None),
            0x12: ("KIL", "imp", 1, "v", 1),
            0x13: ("SLO", "iy", 8, "a", None),
            0x14: ("IGN", "zx", 4, "a", None),
            0x15: ("ORA", "zx", 4, "v", None),
            0x16: ("ASL", "zx", 6, "a", None),
            0x17: ("SLO", "zx", 6, "a", None),
            0x18: ("CLC", "im", 2, "v", "C"),
            0x19: ("ORA", "ay", 4, "v", None),
            0x1A: ("NOP", "imp", 2, "v", 1),
            0x1B: ("SLO", "ay", 7, "a", None),
            0x1C: ("IGN", "ax", 4, "a", None),
            0x1D: ("ORA", "ax", 4, "v", None),
            0x1E: ("ASL", "ax", 7, "a", None),
            0x1F: ("SLO", "ax", 7, "a", None),
            # endregion Instructions 0x1

            # region Instructions 0x2
            0x20: ("JSR", "a", 6, "a", None),
            0x21: ("AND", "ix", 6, "v", None),
            0x22: ("KIL", "imp", 1, "v", 1),
            0x23: ("RLA", "ix", 8, "a", None),
            0x24: ("BIT", "z", 3, "v", None),
            0x25: ("AND", "z", 3, "v", None),
            0x26: ("ROL", "z", 5, "a", None),
            0x27: ("RLA", "z", 5, "a", None),
            0x28: ("PLP", "imp", 4, "a", ("PL", "p")),
            0x29: ("AND", "im", 2, "v", None),
            0x2A: ("ROL", "acc", 2, "a", "a"),
            0x2B: ("AAC", "im", 2, "v", None),  # ANC
            0x2C: ("BIT", "a", 4, "v", None),
            0x2D: ("AND", "a", 4, "v", None),
            0x2E: ("ROL", "a", 6, "a", None),
            0x2F: ("RLA", "a", 6, "a", None),
            # endregion Instructions 0x2

            # region Instructions 0x3
            0x30: ("BMI", "rel", 2, "v", ("N", True)),
            0x31: ("AND", "iy", 5, "v", None),
            0x32: ("KIL", "imp", 1, "v", 1),
            0x33: ("RLA", "iy", 8, "a", None),
            0x34: ("IGN", "zx", 4, "a", None),
            0x35: ("AND", "zx", 4, "v", None),
            0x36: ("ROL", "zx", 6, "a", None),
            0x37: ("RLA", "zx", 6, "a", None),
            0x38: ("SEC", "imp", 2, "v", "C"),
            0x39: ("AND", "ay", 4, "v", None),
            0x3A: ("NOP", "imp", 2, "v", 1),
            0x3B: ("RLA", "ay", 7, "a", None),
            0x3C: ("IGN", "ax", 4, "a", None),
            0x3D: ("AND", "ax", 4, "v", None),
            0x3E: ("ROL", "ax", 7, "a", None),
            0x3F: ("RLA", "ax", 7, "a", None),
            # endregion Instructions 0x3

            # region Instructions 0x4
            0x40: ("RTI", "imp", 6, "a", 1),
            0x41: ("EOR", "ix", 6, "v", None),
            0x42: ("KIL", "imp", 1, "v", 1),
            0x43: ("SRE", "ix", 8, "a", None),
            0x44: ("IGN", "z", 3, "a", None),
            0x45: ("EOR", "z", 3, "v", None),
            0x46: ("LSR", "z", 5, "a", None),
            0x47: ("SRE", "z", 5, "a", None),
            0x48: ("PHA", "imp", 3, "a", ("PH", "a")),
            0x49: ("EOR", "im", 2, "v", None),
            0x4A: ("LSR", "acc", 2, "a", "a"),
            0x4B: ("ASR", "im", 2, "v", None),  # ALR
            0x4C: ("JMP", "a", 3, "a", None),
            0x4D: ("EOR", "a", 4, "v", None),
            0x4E: ("LSR", "a", 6, "a", None),
            0x4F: ("SRE", "a", 6, "a", None),
            # endregion Instructions 0x4

            # region Instructions 0x5
            0x50: ("BVC", "rel", 2, "v", ("V", False)),
            0x51: ("EOR", "iy", 5, "v", None),
            0x52: ("KIL", "imp", 1, "v", 1),
            0x53: ("SRE", "iy", 8, "a", None),
            0x54: ("IGN", "zx", 4, "a", None),
            0x55: ("EOR", "zx", 4, "v", None),
            0x56: ("LSR", "zx", 6, "a", None),
            0x57: ("SRE", "zx", 6, "a", None),
            0x58: ("CLI", "im", 2, "v", "I"),
            0x59: ("EOR", "ay", 4, "v", None),
            0x5A: ("NOP", "imp", 2, "v", 1),
            0x5B: ("SRE", "ay", 7, "a", None),
            0x5C: ("IGN", "ax", 4, "a", None),
            0x5D: ("EOR", "ax", 4, "v", None),
            0x5E: ("LSR", "ax", 7, "a", None),
            0x5F: ("SRE", "ax", 7, "a", None),
            # endregion Instructions 0x5

            # region Instructions 0x6
            0x60: ("RTS", "imp", 6, "a", 1),
            0x61: ("ADC", "ix", 6, "v", None),
            0x62: ("KIL", "imp", 1, "v", 1),
            0x63: ("RRA", "ix", 8, "a", None),
            0x64: ("IGN", "z", 3, "a", None),
            0x65: ("ADC", "z", 3, "v", None),
            0x66: ("ROR", "z", 5, "a", None),
            0x67: ("RRA", "z", 5, "a", None),
            0x68: ("PLA", "imp", 4, "a", ("PL", "a")),
            0x69: ("ADC", "im", 2, "v", None),
            0x6A: ("ROR", "acc", 2, "a", "a"),
            0x6B: ("ARR", "im", 2, "v", None),
            0x6C: ("JMP", "i", 5, "a", None),
            0x6D: ("ADC", "a", 4, "v", None),
            0x6E: ("ROR", "a", 6, "a", None),
            0x6F: ("RRA", "a", 6, "a", None),
            # endregion Instructions 0x6

            # region Instructions 0x7
            0x70: ("BVS", "rel", 2, "v", ("V", True)),
            0x71: ("ADC", "iy", 5, "v", None),
            0x72: ("KIL", "imp", 1, "v", 1),
            0x73: ("RRA", "iy", 8, "a", None),
            0x74: ("IGN", "zx", 4, "a", None),
            0x75: ("ADC", "zx", 4, "v", None),
            0x76: ("ROR", "zx", 6, "a", None),
            0x77: ("RRA", "zx", 6, "a", None),
            0x78: ("SEI", "imp", 2, "v", "I"),
            0x79: ("ADC", "ay", 4, "v", None),
            0x7A: ("NOP", "imp", 2, "v", 1),
            0x7B: ("RRA", "ay", 7, "a", None),
            0x7C: ("IGN", "ax", 4, "a", None),
            0x7D: ("ADC", "ax", 4, "v", None),
            0x7E: ("ROR", "ax", 7, "a", None),
            0x7F: ("RRA", "ax", 7, "a", None),
            # endregion Instructions 0x7

            # region Instructions 0x8
            0x80: ("SKB", "im", 2, "v", None),
            0x81: ("STA", "ix", 6, "a", None),
            0x82: ("SKB", "im", 2, "v", None),
            0x83: ("AAX", "ix", 6, "a", None),  # SAX
            0x84: ("STY", "z", 3, "a", None),
            0x85: ("STA", "z", 3, "a", None),
            0x86: ("STX", "z", 3, "a", None),
            0x87: ("AAX", "z", 3, "a", None),  # SAX
            0x88: ("DEY", "imp", 2, "a", 1),
            0x89: ("SKB", "im", 2, "v", None),
            0x8A: ("TXA", "imp", 2, "a", ('x', 'a')),
            0x8B: ("XAA", "im", 2, "v", None),
            0x8C: ("STY", "a", 4, "a", None),
            0x8D: ("STA", "a", 4, "a", None),
            0x8E: ("STX", "a", 4, "a", None),
            0x8F: ("AAX", "a", 4, "a", None),  # SAX
            # endregion Instructions 0x8

            # region Instructions 0x9
            0x90: ("BCC", "rel", 2, "v", ("C", False)),
            0x91: ("STA", "iy", 6, "a", None),
            0x92: ("KIL", "imp", 1, "v", 1),
            0x93: ("AXA", "iy", 6, "a", None),  # SHA
            0x94: ("STY", "zx", 4, "a", None),
            0x95: ("STA", "zx", 4, "a", None),
            0x96: ("STX", "zy", 4, "a", None),
            0x97: ("AAX", "zy", 4, "a", None),  # SAX
            0x98: ("TYA", "imp", 2, "a", ('y', 'a')),
            0x99: ("STA", "ay", 5, "a", None),
            0x9A: ("TXS", "imp", 2, "a", ('x', 's')),
            0x9B: ("XAS", "ay", 5, "a", None),  # SHS, TAS
            0x9C: ("SYA", "ax", 5, "a", None),
            0x9D: ("STA", "ax", 5, "a", None),
            0x9E: ("SXA", "ay", 5, "a", None),
            0x9F: ("AXA", "ay", 5, "a", None),  # SHA
            # endregion Instructions 0x9

            # region Instructions 0xA
            0xA0: ("LDY", "im", 2, "v", None),
            0xA1: ("LDA", "ix", 6, "v", None),
            0xA2: ("LDX", "im", 2, "v", None),
            0xA3: ("LAX", "ix", 6, "v", None),
            0xA4: ("LDY", "z", 3, "v", None),
            0xA5: ("LDA", "z", 3, "v", None),
            0xA6: ("LDX", "z", 3, "v", None),
            0xA7: ("LAX", "z", 3, "v", None),
            0xA8: ("TAY", "imp", 2, "a", ('a', 'y')),
            0xA9: ("LDA", "im", 2, "v", None),
            0xAA: ("TAX", "imp", 2, "a", ('a', 'x')),
            0xAB: ("ATX", "im", 2, "v", None),  #
            0xAC: ("LDY", "a", 4, "v", None),
            0xAD: ("LDA", "a", 4, "v", None),
            0xAE: ("LDX", "a", 4, "v", None),
            0xAF: ("LAX", "a", 4, "v", None),
            # endregion Instructions 0xA

            # region Instructions 0xB
            0xB0: ("BCS", "rel", 2, "v", ("C", True)),
            0xB1: ("LDA", "iy", 5, "v", None),
            0xB2: ("KIL", "imp", 1, "v", 1),
            0xB3: ("LAX", "iy", 5, "v", None),
            0xB4: ("LDY", "zx", 4, "v", None),
            0xB5: ("LDA", "zx", 4, "v", None),
            0xB6: ("LDX", "zy", 4, "v", None),
            0xB7: ("LAX", "zy", 4, "v", None),
            0xB8: ("CLV", "imp", 2, "v", "V"),
            0xB9: ("LDA", "ay", 4, "v", None),
            0xBA: ("TSX", "imp", 2, "a", ('s', 'x')),
            0xBB: ("LAR", "ay", 4, "v", None),
            0xBC: ("LDY", "ax", 4, "v", None),
            0xBD: ("LDA", "ax", 4, "v", None),
            0xBE: ("LDX", "ay", 4, "v", None),
            0xBF: ("LAX", "ay", 4, "v", None),
            # endregion Instructions 0xB

            # region Instructions 0xC
            0xC0: ("CPY", "im", 2, "v", None),
            0xC1: ("CMP", "ix", 6, "v", None),
            0xC2: ("SKB", "im", 2, "v", None),
            0xC3: ("DCP", "ix", 8, "a", None),
            0xC4: ("CPY", "z", 3, "v", None),
            0xC5: ("CMP", "z", 3, "v", None),
            0xC6: ("DEC", "z", 5, "a", None),
            0xC7: ("DCP", "z", 5, "a", None),
            0xC8: ("INY", "imp", 2, "a", 1),
            0xC9: ("CMP", "im", 2, "v", None),
            0xCA: ("DEX", "imp", 2, "a", 1),
            0xCB: ("AXS", "im", 2, "v", None),  # SBX
            0xCC: ("CPY", "a", 4, "v", None),
            0xCD: ("CMP", "a", 4, "v", None),
            0xCE: ("DEC", "a", 6, "a", None),
            0xCF: ("DCP", "a", 6, "a", None),
            # endregion Instructions 0xC

            # region Instructions 0xD
            0xD0: ("BNE", "rel", 2, "v", ("Z", False)),
            0xD1: ("CMP", "iy", 5, "v", None),
            0xD2: ("KIL", "imp", 1, "v", 1),
            0xD3: ("DCP", "iy", 8, "a", None),
            0xD4: ("IGN", "zx", 4, "a", None),
            0xD5: ("CMP", "zx", 4, "v", None),
            0xD6: ("DEC", "zx", 6, "a", None),
            0xD7: ("DCP", "zx", 6, "a", None),
            0xD8: ("CLD", "imp", 2, "v", "D"),
            0xD9: ("CMP", "ay", 4, "v", None),
            0xDA: ("NOP", "imp", 2, "v", 1),
            0xDB: ("DCP", "ay", 7, "a", None),
            0xDC: ("IGN", "ax", 4, "a", None),
            0xDD: ("CMP", "ax", 4, "v", None),
            0xDE: ("DEC", "ax", 7, "a", None),
            0xDF: ("DCP", "ax", 7, "a", None),
            # endregion Instructions 0xD

            # region Instructions 0xE
            0xE0: ("CPX", "im", 2, "v", None),
            0xE1: ("SBC", "ix", 6, "v", None),
            0xE2: ("SKB", "im", 2, "v", None),
            0xE3: ("ISC", "ix", 8, "a", None),
            0xE4: ("CPX", "z", 3, "v", None),
            0xE5: ("SBC", "z", 3, "v", None),
            0xE6: ("INC", "z", 5, "a", None),
            0xE7: ("ISC", "z", 5, "a", None),
            0xE8: ("INX", "imp", 2, "a", 1),
            0xE9: ("SBC", "im", 2, "v", None),
            0xEA: ("NOP", "imp", 2, "v", 1),
            0xEB: ("SBC", "im", 2, "v", None),  # USBC
            0xEC: ("CPX", "a", 4, "v", None),
            0xED: ("SBC", "a", 4, "v", None),
            0xEE: ("INC", "a", 6, "a", None),
            0xEF: ("ISC", "a", 6, "a", None),
            # endregion Instructions 0xE

            # region Instructions 0xF
            0xF0: ("BEQ", "rel", 2, "v", ("Z", True)),
            0xF1: ("SBC", "iy", 5, "v", None),
            0xF2: ("KIL", "imp", 1, "v", 1),
            0xF3: ("ISC", "iy", 8, "a", None),
            0xF4: ("IGN", "zx", 4, "a", None),
            0xF5: ("SBC", "zx", 4, "v", None),
            0xF6: ("INC", "zx", 6, "a", None),
            0xF7: ("ISC", "zx", 6, "a", None),
            0xF8: ("SED", "imp", 2, "v", "D"),
            0xF9: ("SBC", "ay", 4, "v", None),
            0xFA: ("NOP", "imp", 2, "v", 1),
            0xFB: ("ISC", "ay", 7, "a", None),
            0xFC: ("IGN", "ax", 4, "a", None),
            0xFD: ("SBC", "ax", 4, "v", None),
            0xFE: ("INC", "ax", 7, "a", None),
            0xFF: ("ISC", "ax", 7, "a", None),
            # endregion Instructions 0xF
        }

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from typing import Union
from enum import Enum
from py65emu.mmu import MMU
from py65emu.operator import Operator
from py65emu.debug import Debug


class FlagBit(Enum):
    """
    The status register stores 8 flags. Ive enumerated these here for ease
    of access. You can access the status register directly since its public.
    The bits have different interpretations depending upon the context and
    instruction being executed.
    """
    # Negative (128)
    N = 1 << 7

    # Overflow  (64)
    V = 1 << 6

    # Unused (32)
    U = 1 << 5

    # Break Command (16)
    B = 1 << 4

    # Decimal Mode (8)
    D = 1 << 3

    # IRQ Disable (4)
    I = 1 << 2

    # Zero (2)
    Z = 1 << 1

    # Carry (1)
    C = 1 << 0


class Registers:
    """An object to hold the CPU registers."""

    a: int   # Accumulator
    x: int   # General Purpose X
    y: int   # General Purpose Y
    s: int   # Stack Pointer
    pc: int  # Program Counter
    p: int   # Flag Pointer - N|V|1|B|D|I|Z|C

    def __init__(self, pc: int = 0x0000):
        """
        Init Registers
        :param int | None pc: Positon of Program Counter
        """

        self.reset(pc)

    def reset(self, pc: int = 0x0000):
        """
        Reset Registers
        :param int | None pc: Positon of Program Counter
        """
        self.a: int = 0      # Accumulator
        self.x: int = 0      # General Purpose X
        self.y: int = 0      # General Purpose Y
        self.s: int = 0xFF   # Stack Pointer
        self.pc: int = pc    # Program Counter
        self.p = 0b00100100  # Flag Pointer - N|V|1|B|D|I|Z|C

    def getFlag(self, flag: Union["FlagBit", int, str]) -> bool:
        """
        Get flag value
        :param FlagBit flag: One of 'N', 'V', 'B', 'D', 'I', 'Z', 'C'
        :return Wether the flag is set or not
        :rtype bool
        """
        if isinstance(flag, FlagBit):
            flag = flag.value
        elif isinstance(flag, str):
            flag = FlagBit[flag].value

        return bool(self.p & flag)

    def setFlag(
        self,
        flag: Union["FlagBit", int, str],
        v: bool = True
    ) -> None:
        """
        Set flag value
        :param FlagBit flag: One of 'N', 'V', 'B', 'D', 'I', 'Z', 'C'
        :param bool v: Flag is set or not
        """
        if isinstance(flag, FlagBit):
            flag = flag.value
        elif isinstance(flag, str):
            flag = FlagBit[flag].value

        if v:
            self.p = self.p | flag
        else:
            self.clearFlag(flag)

    def clearFlag(self, flag: Union["FlagBit", int, str]) -> None:
        """
        Set flag value
        :param str flag: One of 'N', 'V', 'B', 'D', 'I', 'Z', 'C'
        """
        if isinstance(flag, FlagBit):
            flag = flag.value
        elif isinstance(flag, str):
            flag = FlagBit[flag].value

        self.p = self.p & (255 - flag)

    def clearFlags(self) -> None:
        self.p = 0

    def ZN(self, v) -> None:
        """
        The criteria for Z and N flags are standard.  Z gets set if the
        value is zero and N gets set to the same value as bit 7 of the value.
        :param int v: Value
        """
        self.setFlag(FlagBit.Z, v == 0)
        self.setFlag(FlagBit.N, v & 0x80)

    @property
    def flags(self) -> str:
        flags = ""
        flags += "N" if self.getFlag(FlagBit.N) else "."
        flags += "V" if self.getFlag(FlagBit.V) else "."
        flags += "U" if self.getFlag(FlagBit.U) else "."
        flags += "B" if self.getFlag(FlagBit.B) else "."
        flags += "D" if self.getFlag(FlagBit.D) else "."
        flags += "I" if self.getFlag(FlagBit.I) else "."
        flags += "Z" if self.getFlag(FlagBit.Z) else "."
        flags += "C" if self.getFlag(FlagBit.C) else "."
        return flags

    def __repr__(self) -> str:
        return f"A: {self.a:0>2X} X: {self.x:0>2X} "\
               f"Y: {self.y:0>2X} S: {self.s:0>2X} "\
               f"PC: {self.pc:0>4X} P: {self.flags}"


class CPU:
    running: bool

    def __init__(
        self,
        mmu: "MMU" = None,
        pc: int | None = None,
        stack_page: int = 0x1,
        magic: int = 0xEE
    ):
        """
        :param "MMU"|None mmu: An instance of MMU
        :param int pc: The starting address of the pc (program counter)
        :param int stack_page: The index of the page which contains the stack.
            The default for a 6502 is page 1 (the stack from 0x0100-0x1ff) but
            in some varients the stack page may be elsewhere.
        :param int magic: A value needed for the illegal opcodes, XAA.
            This value differs between different versions, even of the same
            CPU. The default is 0xee.
        """
        self.mmu: "MMU" = mmu
        self.r: "Registers" = Registers()

        # Hold the number of CPU cycles used during the last call to
        # `self.step()`.
        # Since the CPU is in a "resetted" state, and a reset takes 7 cycles
        # we set the current cycles to 7
        self.cc: int = 0

        # Which page the stack is in.  0x1 means that the stack is from
        # 0x100-0x1ff. In the 6502 this is always true but it's different
        # for other 65* varients.
        self.stack_page: int = stack_page

        self.magic: int = magic
        self.opcode: int | None = None

        self.trigger_nmi: bool = False
        self.trigger_irq: bool = False
        self._previous_interrupt: bool = False

        if pc:
            self.r.pc = pc
        else:
            # if pc is none get the address from $FFFD,$FFFC
            pass

        # self._ops = Operator()
        # self._create_ops()
        # self.ops = self.operators.create()
        self.operators = Operator(self)
        self.debug = Debug(self)

    def reset(self) -> None:
        self.r.reset(self.interrupts["RESET"])
        self.mmu.reset()
        self.trigger_nmi = False
        self.trigger_irq = False
        self._previous_interrupt = False
        self.opcode = None
        self.cc = 7  # Reset takes 7 cycles

        self.running = True

    def step(self) -> None:
        self.cc = 0

        self.opcode = self.nextByte()
        self.operators[self.opcode].execute(),
        self.handle_interrupt()

    def execute(self, instruction: list[int]) -> None:
        """
        Execute a single instruction independent of the program in memory.
        instruction is an array of bytes.
        """
        self.cc = 0
        for i in instruction:
            self.operators[i].execute(),
            self.handle_interrupt()

    def handle_interrupt(self) -> None:
        if self._previous_interrupt:
            if self.trigger_nmi:
                self.process_nmi()
                self.trigger_nmi = False
            elif self.trigger_irq:
                self.process_irq()
                self.trigger_irq = False

    def nextByte(self) -> int:
        v = self.mmu.read(self.r.pc)
        self.r.pc += 1
        return v

    def nextWord(self) -> int:
        low = self.nextByte()
        high = self.nextByte()
        return (high << 8) + low

    def stackPush(self, v: int) -> None:
        self.mmu.write((self.stack_page * 0x100) + self.r.s, v)
        self.r.s = (self.r.s - 1) & 0xFF

    def stackPushWord(self, v: int) -> None:
        self.stackPush(v >> 8)
        self.stackPush(v & 0xFF)

    def stackPop(self) -> int:
        v = self.mmu.read(self.stack_page * 0x100 + ((self.r.s + 1) & 0xFF))
        self.r.s = (self.r.s + 1) & 0xFF
        return v

    def stackPopWord(self) -> int:
        return self.stackPop() + (self.stackPop() << 8)

    def fromBCD(self, v) -> int:
        return (((v & 0xF0) // 0x10) * 10) + (v & 0xF)

    def toBCD(self, v) -> int:
        return int(math.floor(v / 10)) * 16 + (v % 10)

    def fromTwosCom(self, v) -> int:
        return (v & 0x7F) - (v & 0x80)

    interrupts = {
        "ABORT": 0xfff8,
        "COP": 0xfff4,
        "IRQ": 0xfffe,
        "BRK": 0xfffe,
        "NMI": 0xfffa,
        "RESET": 0xfffc
    }

    def interruptAddress(self, irq_type: str) -> int:
        return self.mmu.readWord(self.interrupts[irq_type])

    def interruptRequest(self) -> None:
        """Trigger """
        self.trigger_irq = True

    def breakOperation(self, irq_type: str) -> None:
        self.stackPushWord(self.r.pc + 1)

        if irq_type == "BRK":
            self.stackPush(self.r.p | FlagBit.B.value)
        else:
            self.stackPush(self.r.p)

        self.r.setFlag("I")
        self.r.pc = self.interruptAddress(type)

    def process_nmi(self) -> None:
        self.r.pc -= 1
        self.breakOperation("NMI")

    def process_irq(self) -> None:
        if self.r.getFlag(FlagBit.I):
            return None
        self.r.pc -= 1
        self.breakOperation("IRQ")

    # Addressing modes
    def z_a(self) -> int:
        return self.nextByte()

    def zx_a(self) -> int:
        return (self.nextByte() + self.r.x) & 0xFF

    def zy_a(self) -> int:
        return (self.nextByte() + self.r.y) & 0xFF

    def a_a(self) -> int:
        return self.nextWord()

    def ax_a(self) -> int:
        o = self.nextWord()
        a = o + self.r.x
        ignore_op = [0x1E, 0xDE, 0xFE, 0x5E, 0x3E, 0x7E, 0x9D]
        if o & 0xFF00 != a & 0xFF00 and self.opcode not in ignore_op:
            self.cc += 1

        return a & 0xFFFF

    def ay_a(self) -> int:
        o = self.nextWord()
        a = o + self.r.y
        ignore_op = [0x99]
        if o & 0xFF00 != a & 0xFF00 and self.opcode not in ignore_op:
            self.cc += 1

        return a & 0xFFFF

    def i_a(self) -> int:
        """Only used by indirect JMP"""
        i = self.nextWord()
        # Doesn't carry, so if the low byte is in the XXFF position
        # Then the high byte will be XX00 rather than XY00
        if i & 0xFF == 0xFF:
            j = i - 0xFF
        else:
            j = i + 1

        return ((self.mmu.read(j) << 8) + self.mmu.read(i)) & 0xFFFF

    def ix_a(self) -> int:
        i = (self.nextByte() + self.r.x) & 0xFF
        return (
            ((self.mmu.read((i + 1) & 0xFF) << 8) + self.mmu.read(i)) & 0xFFFF
        )

    def iy_a(self) -> int:
        i = self.nextByte()
        o = (self.mmu.read((i + 1) & 0xFF) << 8) + self.mmu.read(i)
        a = o + self.r.y

        ignore_op = [0x91]
        if o & 0xFF00 != a & 0xFF00 and self.opcode not in ignore_op:
            self.cc += 1

        return a & 0xFFFF

    # Return values based on the addressing mode
    def im(self) -> int:
        return self.nextByte()

    def z(self) -> int:
        return self.mmu.read(self.z_a())

    def zx(self) -> int:
        return self.mmu.read(self.zx_a())

    def zy(self) -> int:
        return self.mmu.read(self.zy_a())

    def a(self) -> int:
        return self.mmu.read(self.a_a())

    def ax(self) -> int:
        return self.mmu.read(self.ax_a())

    def ay(self) -> int:
        return self.mmu.read(self.ay_a())

    def i(self) -> int:
        return self.mmu.read(self.i_a())

    def ix(self) -> int:
        return self.mmu.read(self.ix_a())

    def iy(self) -> int:
        return self.mmu.read(self.iy_a())

    """Instructions."""

    def ADC(self, v2) -> None:
        v1 = self.r.a

        if self.r.getFlag("D"):  # decimal mode
            """
            d1 = self.fromBCD(v1)
            d2 = self.fromBCD(v2)
            r = d1 + d2 + self.r.getFlag("C")
            self.r.a = self.toBCD(r % 100)
            """
            d1 = (v1 & 0x0F) + (v2 & 0x0F) + self.r.getFlag("C")
            d2 = (v1 >> 4) + (v2 >> 4) + (1 if d1 > 9 else 0)

            r = d1 % 10 | (d2 % 10 << 4)
            self.r.a = r & 0xFF

            # self.r.setFlag("C", r > 99)
            self.r.setFlag("C", d1 > 9)
        else:
            r = v1 + v2 + self.r.getFlag("C")
            self.r.a = r & 0xFF

            self.r.setFlag("C", r > 0xFF)

        self.r.ZN(self.r.a)
        # self.r.ZN(r)
        self.r.setFlag("V", ((~(v1 ^ v2)) & (v1 ^ r) & 0x80))

    def AND(self, v) -> None:
        self.r.a = (self.r.a & v) & 0xFF
        self.r.ZN(self.r.a)

    def ASL(self, a) -> None:
        if a == "a":
            v = self.r.a << 1
            self.r.a = v & 0xFF
        else:
            v = self.mmu.read(a) << 1
            self.mmu.write(a, v)

        self.r.setFlag("C", v > 0xFF)
        self.r.ZN(v & 0xFF)

    def BIT(self, v) -> None:
        self.r.setFlag("Z", self.r.a & v == 0)
        self.r.setFlag("N", v & 0x80)
        self.r.setFlag("V", v & 0x40)

    def B(self, v) -> None:
        """
        v is a tuple of (flag, boolean).
        For instance, BCC (Branch Carry Clear) will call B(('C', False)).
        """
        d = self.im()
        if self.r.getFlag(v[0]) is v[1]:
            o = self.r.pc
            self.r.pc = (self.r.pc + self.fromTwosCom(d)) & 0xFFFF
            if o & 0xFF00 == self.r.pc & 0xFF00:
                self.cc += 1
            else:
                self.cc += 2

    def BRK(self, _: int) -> None:
        self.breakOperation("BRK")

    def CP(self, r, v) -> None:
        o = (r - v) & 0xFF
        self.r.setFlag("Z", o == 0)
        self.r.setFlag("C", v <= r)
        self.r.setFlag("N", o & 0x80)

    def CMP(self, v) -> None:
        self.CP(self.r.a, v)

    def CPX(self, v) -> None:
        self.CP(self.r.x, v)

    def CPY(self, v) -> None:
        self.CP(self.r.y, v)

    def DEC(self, a) -> None:
        v = (self.mmu.read(a) - 1) & 0xFF
        self.mmu.write(a, v)
        self.r.ZN(v)

    def DEX(self, _) -> None:
        self.r.x = (self.r.x - 1) & 0xFF
        self.r.ZN(self.r.x)

    def DEY(self, _) -> None:
        self.r.y = (self.r.y - 1) & 0xFF
        self.r.ZN(self.r.y)

    def EOR(self, v) -> None:
        self.r.a = self.r.a ^ v
        self.r.ZN(self.r.a)

    """Flag Instructions."""

    def SE(self, v) -> None:
        """Set the flag to True."""
        self.r.setFlag(v)

    def CL(self, v) -> None:
        """Clear the flag to False."""
        self.r.clearFlag(v)

    def INC(self, a) -> None:
        v = (self.mmu.read(a) + 1) & 0xFF
        self.mmu.write(a, v)
        self.r.ZN(v)

    def INX(self, _) -> None:
        self.r.x = (self.r.x + 1) & 0xFF
        self.r.ZN(self.r.x)

    def INY(self, _) -> None:
        self.r.y = (self.r.y + 1) & 0xFF
        self.r.ZN(self.r.y)

    def JMP(self, a) -> None:
        self.r.pc = a

    def JSR(self, a) -> None:
        self.stackPushWord(self.r.pc - 1)
        self.r.pc = a

    def LDA(self, v) -> None:
        self.r.a = v
        self.r.ZN(self.r.a)

    def LDX(self, v) -> None:
        self.r.x = v
        self.r.ZN(self.r.x)

    def LDY(self, v) -> None:
        self.r.y = v
        self.r.ZN(self.r.y)

    def LSR(self, a) -> None:
        if a == "a":
            self.r.setFlag("C", self.r.a & 0x01)
            self.r.a = v = self.r.a >> 1
        else:
            v = self.mmu.read(a)
            self.r.setFlag("C", v & 0x01)
            v = v >> 1
            self.mmu.write(a, v)

        self.r.ZN(v)

    def NOP(self, _) -> None:
        pass

    def ORA(self, v) -> None:
        self.r.a = self.r.a | v
        self.r.ZN(self.r.a)

    def P(self, v: tuple[str, str]) -> None:
        """
        Stack operations, PusH and PulL. v is a tuple where the
        first value is either PH or PL, specifying the action and
        the second is the source or target register, either A or P,
        meaning the Accumulator or the Processor status flag.
        """
        a, r = v

        if a == "PH":
            register = getattr(self.r, r)
            if r == "p":
                register |= 0b00110000

            self.stackPush(register)
        else:
            setattr(self.r, r, self.stackPop())

            if r == "a":
                self.r.ZN(self.r.a)
            elif r == "p":
                self.r.p = self.r.p | 0b00100000

    def ROL(self, a) -> None:
        if a == "a":
            v_old = self.r.a
            self.r.a = v_new = ((v_old << 1) + self.r.getFlag("C")) & 0xFF
        else:
            v_old = self.mmu.read(a)
            v_new = ((v_old << 1) + self.r.getFlag("C")) & 0xFF
            self.mmu.write(a, v_new)

        self.r.setFlag("C", v_old & 0x80)
        self.r.ZN(v_new)

    def ROR(self, a) -> None:
        if a == "a":
            v_old = self.r.a
            self.r.a = v_new = (
                ((v_old >> 1) + self.r.getFlag("C") * 0x80) & 0xFF
            )
        else:
            v_old = self.mmu.read(a)
            v_new = ((v_old >> 1) + self.r.getFlag("C") * 0x80) & 0xFF
            self.mmu.write(a, v_new)

        self.r.setFlag("C", v_old & 0x01)
        self.r.ZN(v_new)

    def RTI(self, _) -> None:
        self.r.p = self.stackPop()
        self.r.pc = self.stackPopWord()

    def RTS(self, _) -> None:
        self.r.pc = (self.stackPopWord() + 1) & 0xFFFF

    def SBC(self, v2) -> None:
        v1 = self.r.a
        if self.r.getFlag("D"):
            d1 = self.fromBCD(v1)
            d2 = self.fromBCD(v2)
            r = d1 - d2 - (not self.r.getFlag("C"))
            self.r.a = self.toBCD(r % 100)

            self.r.setFlag("C", r > 99)
        else:
            r = v1 + (v2 ^ 0xFF) + self.r.getFlag("C")

            self.r.a = r & 0xFF

            self.r.setFlag("C", r & 0xFF00)

        self.r.ZN(self.r.a)
        # self.r.ZN(r)
        self.r.setFlag("V", ((v1 ^ v2) & (v1 ^ r) & 0x80))

    def STA(self, a) -> None:
        self.mmu.write(a, self.r.a)

    def STX(self, a) -> None:
        self.mmu.write(a, self.r.x)

    def STY(self, a) -> None:
        self.mmu.write(a, self.r.y)

    def T(self, a) -> None:
        """
        Transfer registers
        a is a tuple with (source, destination) so TAX
        would be T(('a', 'x'))self.
        """
        s, d = a
        setattr(self.r, d, getattr(self.r, s))
        if d != "s":
            self.r.ZN(getattr(self.r, d))

    """
    Illegal Opcodes
    ---------------

    Opcodes which were not officially documented but still have
    and effect.  The behavior for each of these is based on the following:

    -http://www.ataripreservation.org/websites/freddy.offenga/illopc31.txt
    -http://wiki.nesdev.com/w/index.php/Programming_with_unofficial_opcodes
    -www.ffd2.com/fridge/docs/6502-NMOS.extra.opcodes

    The behavior is not consistent across the various resources so I don't
    promise 100% hardware accuracy here.

    Other names for the opcode are in comments on the function defintion
    line.
    """

    def AAC(self, v) -> None:  # ANC
        self.AND(v)
        self.r.setFlag("C", self.r.getFlag("N"))

    def AAX(self, a) -> None:  # SAX, AXS
        r = self.r.a & self.r.x
        self.mmu.write(a, r)
        # There is conflicting information whether this effects P.
        # self.r.ZN(r)

    def ARR(self, v) -> None:
        self.AND(v)
        self.ROR("a")
        self.r.setFlag("C", self.r.a & 0x40)
        self.r.setFlag("V", bool(self.r.a & 0x40) ^ bool(self.r.a & 0x20))

    def ASR(self, v) -> None:  # ALR
        self.AND(v)
        self.LSR("a")

    def ATX(self, v) -> None:  # LXA, OAL
        self.AND(v)
        self.T(("a", "x"))

    def AXA(self, a) -> None:  # SHA
        """
        There are a few illegal opcodes which and the high
        bit of the address with registers and write the values
        back into that address.  These operations are
        particularly screwy.  These posts are used as reference
        but I am unsure whether they are correct.
        - forums.nesdev.com/viewtopic.php?f=3&t=3831&start=30#p113343
        - forums.nesdev.com/viewtopic.php?f=3&t=10698
        """
        o = (a - self.r.y) & 0xFFFF
        low = o & 0xFF
        high = o >> 8
        if low + self.r.y > 0xFF:  # crossed page
            a = ((high & self.r.x) << 8) + low + self.r.y
        else:
            a = (high << 8) + low + self.r.y

        v = self.r.a & self.r.x & (high + 1)
        self.mmu.write(a, v)

    def AXS(self, v) -> None:  # SBX, SAX
        o = self.r.a & self.r.x
        self.r.x = (o - v) & 0xFF

        self.r.setFlag("C", v <= o)
        self.r.ZN(self.r.x)

    def DCP(self, a) -> None:  # DCM
        self.DEC(a)
        self.CMP(self.mmu.read(a))

    def ISC(self, a) -> None:  # ISB, INS
        self.INC(a)
        self.SBC(self.mmu.read(a))

    def KIL(self, _) -> None:  # JAM, HLT
        self.running = False

    def LAR(self, v) -> None:  # LAE, LAS
        self.r.a = self.r.x = self.r.s = self.r.s & v
        self.r.ZN(self.r.a)

    def LAX(self, v) -> None:
        self.r.a = self.r.x = v
        self.r.ZN(self.r.a)

    def RLA(self, a) -> None:
        self.ROL(a)
        self.AND(self.mmu.read(a))

    def RRA(self, a) -> None:
        self.ROR(a)
        self.ADC(self.mmu.read(a))

    def SLO(self, a) -> None:  # ASO
        self.ASL(a)
        self.ORA(self.mmu.read(a))

    def SRE(self, a) -> None:  # LSE
        self.LSR(a)
        self.EOR(self.mmu.read(a))

    def SXA(self, a) -> None:  # SHX, XAS
        # See AXA
        o = (a - self.r.y) & 0xFFFF
        low = o & 0xFF
        high = o >> 8
        if low + self.r.y > 0xFF:  # crossed page
            a = ((high & self.r.x) << 8) + low + self.r.y
        else:
            a = (high << 8) + low + self.r.y

        v = self.r.x & (high + 1)
        self.mmu.write(a, v)

    def SYA(self, a) -> None:  # SHY, SAY
        # See AXA
        o = (a - self.r.x) & 0xFFFF
        low = o & 0xFF
        high = o >> 8
        if low + self.r.x > 0xFF:  # crossed page
            a = ((high & self.r.y) << 8) + low + self.r.x
        else:
            a = (high << 8) + low + self.r.x

        v = self.r.y & (high + 1)
        self.mmu.write(a, v)

    def XAA(self, v) -> None:  # ANE
        """
        Another very wonky operation.  It's fully described here:
        http://visual6502.org/wiki/index.php?title=6502_Opcode_8B_%28XAA,_ANE%29
        "magic" varies by version of the processor.  0xee seems to be common.
        The formula is: A = (A | magic) & X & imm
        """
        self.r.a = (self.r.a | self.magic) & self.r.x & v
        self.r.ZN(self.r.a)

    def XAS(self, a) -> None:  # SHS, TAS
        # First set the stack pointer's value
        self.r.s = self.r.a & self.r.x

        # Then write to memory using the new value of the stack pointer
        o = (a - self.r.y) & 0xFFFF
        low = o & 0xFF
        high = o >> 8
        if low + self.r.y > 0xFF:  # crossed page
            a = ((high & self.r.s) << 8) + low + self.r.y
        else:
            a = (high << 8) + low + self.r.y

        v = self.r.s & (high + 1)
        self.mmu.write(a, v)

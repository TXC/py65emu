#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from enum import Enum
from py65emu.mmu import MMU
from py65emu.operation import Operation, OpCodes
from py65emu.debug import Disassembly


class FlagBit(Enum):
    """
    The status register stores 8 flags. Ive enumerated these here for ease
    of access. You can access the status register directly since its public.
    The bits have different interpretations depending upon the context and
    instruction being executed.
    """

    N = 1 << 7
    """
    Negative

    The negative flag is set if the result of the last operation had bit 7
    set to a one.
    """

    V = 1 << 6
    """
    Overflow

    The overflow flag is set during arithmetic operations if the result has
    yielded an invalid 2's complement result (e.g. adding to positive numbers
    and ending up with a negative result: 64 + 64 => -128). It is determined
    by looking at the carry between bits 6 and 7 and between bit 7 and the
    carry flag.
    """

    U = 1 << 5
    """
    Unused

    Is always set
    """

    B = 1 << 4
    """
    Break Command

    The break command bit is set when a BRK instruction has been executed and
    an interrupt has been generated to process it.
    """

    D = 1 << 3
    """
    Decimal Mode

    While the decimal mode flag is set the processor will obey the rules of
    Binary Coded Decimal (BCD) arithmetic during addition and subtraction. The
    flag can be explicitly set using 'Set Decimal Flag' (SED) and cleared with
    'Clear Decimal Flag' (CLD).
    """

    I = 1 << 2
    """
    IRQ Disable

    The interrupt disable flag is set if the program has executed a
    'Set Interrupt Disable' (SEI) instruction. While this flag is set the
    processor will not respond to interrupts from devices until it is cleared
    by a 'Clear Interrupt Disable' (CLI) instruction.
    """

    Z = 1 << 1
    """
    Zero

    The zero flag is set if the result of the last operation as was zero.
    """

    C = 1 << 0
    """
    Carry

    The carry flag is set if the last operation caused an overflow from bit 7
    of the result or an underflow from bit 0. This condition is set during
    arithmetic, comparison and during logical shifts. It can be explicitly set
    using the 'Set Carry Flag' (SEC) instruction and cleared with
    'Clear Carry Flag' (CLC).
    """


class Registers:
    """
    CPU registers

    The 6502 has only a small number of registers compared to other processor
    of the same era. This makes it especially challenging to program as
    algorithms must make efficient use of both registers and memory.
    """

    a: int
    """
    Accumulator - 8 bit

    The 8 bit accumulator is used all arithmetic and logical operations (with
    the exception of increments and decrements). The contents of the
    accumulator can be stored and retrieved either from memory or the stack.

    Most complex operations will need to use the accumulator for arithmetic
    and efficient optimisation of its use is a key feature of time
    critical routines.
    """

    x: int
    """
    General Purpose X - 8 bit

    The 8 bit index register is most commonly used to hold counters or offsets
    for accessing memory. The value of the X register can be loaded and saved
    in memory, compared with values held in memory or incremented and
    decremented.

    The X register has one special function. It can be used to get a copy of
    the stack pointer or change its value.
    """

    y: int
    """
    General Purpose Y

    The Y register is similar to the X register in that it is available for
    holding counter or offsets memory access and supports the same set of
    memory load, save and compare operations as wells as increments and
    decrements. It has no special functions.
    """

    s: int
    """
    Stack Pointer - 8 bit

    The processor supports a 256 byte stack located between $0100 and $01FF.
    The stack pointer is an 8 bit register and holds the low 8 bits of the next
    free location on the stack. The location of the stack is fixed and
    cannot be moved.

    Pushing bytes to the stack causes the stack pointer to be decremented.
    Conversely pulling bytes causes it to be incremented.

    The CPU does not detect if the stack is overflowed by excessive pushing or
    pulling operations and will most likely result in the program crashing.
    """

    pc: int
    """
    Program Counter - 16 bit

    The program counter is a 16 bit register which points to the next
    instruction to be executed. The value of program counter is modified
    automatically as instructions are executed.

    The value of the program counter can be modified by executing a jump, a
    relative branch or a subroutine call to another memory address or by
    returning from a subroutine or interrupt.
    """

    p: int
    """
    Flag Pointer - 8 bit - N|V|1|B|D|I|Z|C

    As instructions are executed a set of processor flags are set or clear to
    record the results of the operation. This flags and some additional control
    flags are held in a special status register. Each flag has a single bit
    within the register.

    Instructions exist to test the values of the various bits, to set or clear
    some of them and to push or pull the entire set to or from the stack.
    """

    def __init__(self, pc: int = 0x0000):
        """Init Registers

        :param int | None pc: Position of Program Counter
        """
        self.a = 0           # Accumulator
        self.x = 0           # General Purpose X
        self.y = 0           # General Purpose Y
        self.s = 0xFF        # Stack Pointer
        self.pc = pc         # Program Counter
        self.p = 0b00100100  # Flag Pointer
        #          NV1BDIZC

    def reset(self, pc: int = 0x0000):
        """Reset Registers

        :param int | None pc: Position of Program Counter
        """
        self.a = 0           # Accumulator
        self.x = 0           # General Purpose X
        self.y = 0           # General Purpose Y
        self.s = 0xFF        # Stack Pointer
        self.pc = pc         # Program Counter
        self.p = 0b00100100  # Flag Pointer - N|V|1|B|D|I|Z|C

    def getFlag(self, flag: FlagBit | int | str) -> bool:
        """Get flag value

        :param flag: One of `N`, `V`, `B`, `D`, `I`, `Z`, `C`
        :type flag: FlagBit or int or str
        :rtype: bool
        :return: Whether the flag is set or not
        """
        if isinstance(flag, FlagBit):
            pass
        elif isinstance(flag, str):
            flag = FlagBit[flag]
        elif isinstance(flag, int):
            flag = FlagBit(flag)

        return bool(self.p & flag.value)

    def setFlag(
        self,
        flag: FlagBit | int | str,
        v: bool | int = True
    ) -> None:
        """
        Set flag value

        :param flag: One of `N`, `V`, `B`, `D`, `I`, `Z`, `C`
        :type flag: FlagBit or int or str
        :param bool | int v: Flag is set or not
        """
        if isinstance(flag, FlagBit):
            pass
        elif isinstance(flag, str):
            flag = FlagBit[flag]
        elif isinstance(flag, int):
            flag = FlagBit(flag)

        if v > 0:
            self.p = self.p | flag.value
        else:
            self.clearFlag(flag)

    def clearFlag(self, flag: FlagBit | int | str) -> None:
        """Clear flag value

        :param flag: One of `N`, `V`, `B`, `D`, `I`, `Z`, `C`
        :type flag: FlagBit or int or str
        """
        if isinstance(flag, FlagBit):
            pass
        elif isinstance(flag, str):
            flag = FlagBit[flag]
        elif isinstance(flag, int):
            flag = FlagBit(flag)

        self.p = self.p & ~flag.value
        # self.p = self.p & (255 - flag)

    def clearFlags(self) -> None:
        """Clear all flag values
        """
        self.p = 0

    def ZN(self, v) -> None:
        """
        | The criteria for Z and N flags are standard.
        | Z gets set if the value is zero and
        | N gets set to the same value as bit 7 of the value.

        :param int v: Value
        """
        self.setFlag(FlagBit.Z, v == 0)
        self.setFlag(FlagBit.N, v & 0x80)

    @property
    def flags(self) -> str:
        return "{0:s}{1:s}{2:s}{3:s}{4:s}{5:s}{6:s}{7:s}".format(
            "N" if self.getFlag(FlagBit.N) else ".",
            "V" if self.getFlag(FlagBit.V) else ".",
            "U" if self.getFlag(FlagBit.U) else ".",
            "B" if self.getFlag(FlagBit.B) else ".",
            "D" if self.getFlag(FlagBit.D) else ".",
            "I" if self.getFlag(FlagBit.I) else ".",
            "Z" if self.getFlag(FlagBit.Z) else ".",
            "C" if self.getFlag(FlagBit.C) else ".",
        )

    def __repr__(self) -> str:
        return f"A: {self.a:0>2X} X: {self.x:0>2X} "\
               f"Y: {self.y:0>2X} S: {self.s:0>2X} "\
               f"PC: {self.pc:0>4X} P: {self.p:0>2X} [{self.flags}]"


class CPU:
    """The CPU object"""

    running: bool
    """CPU is running"""

    mmu: MMU
    """MMU"""

    op: Operation | None = None
    """Current Operation"""

    cc: int = 0
    """Holds the number of CPU cycles used during the last call to
    :py:meth:`py65emu.CPU.step`.
    Includes :py:attr:`py65emu.CPU.cc_extra`
    """

    cc_total: int = 0
    """
    Holds the total number of cycles during until
    :py:meth:`py65emu.CPU.reset`
    """

    cc_extra = 0
    """
    Holds the number of extra CPU cycles used during the last call to
    :py:meth:`py65emu.CPU.step`
    """

    def __init__(
        self,
        mmu: MMU,
        pc: int | None = None,
        stack_page: int = 0x1,
        magic: int = 0xEE,
        disable_bcd: bool = False,
        debug: bool = False,
    ):
        """
        Initialize CPU

        :param mmu: An instance of MMU
        :param pc: The starting address of the pc (program counter)
        :param stack_page: The index of the page which contains the stack. The
                           default for a 6502 is page 1 (the stack from
                           0x0100-0x1ff) but in some variants the stack page
                           may be elsewhere
        :param magic: A value needed for the illegal opcodes, XAA. This value
                      differs between different versions, even of the same
                      CPU. The default is 0xee
        :type mmu: MMU | None
        :type pc: int | None
        :type stack_page: int
        :type magic: int
        """
        self.mmu: MMU = mmu

        self.r: Registers = Registers()
        self.bcd_disabled = disable_bcd

        # Hold the number of CPU cycles used during the last call to
        # `self.step()`.
        # Since the CPU is in a "resetted" state, and a reset takes 7 cycles
        # we set the total cycles to 7
        self.cc: int = 0
        self.cc_extra = 0
        self.cc_total: int = 7

        # Which page the stack is in.  0x1 means that the stack is from
        # 0x100-0x1ff. In the 6502 this is always true but it's different
        # for other 65* varients.
        self.stack_page: int = stack_page

        self.magic: int = magic

        self.trigger_nmi: bool = False
        self.trigger_irq: bool = False
        self._previous_interrupt: bool = False
        self._interrupt: bool = False

        if pc:
            self.r.pc = pc
        else:
            # if pc is none get the address from $FFFD,$FFFC
            pass

        self.debug = debug
        self.opcodes = OpCodes(self)
        self.op = None

    def reset(self) -> None:
        """Reset everything (CPU, MMU, ...)"""
        self.r.reset(self.interrupts["RESET"])
        self.mmu.reset()
        self.trigger_nmi = False
        self.trigger_irq = False
        self._previous_interrupt = False
        self._interrupt = False
        self.op = None
        self.cc = 0
        self.cc_extra = 0
        self.cc_total = 7  # Reset takes 7 cycles

        self.running = True

    def step(self) -> None:
        """Execute the operation"""
        self.cc = 0
        self.cc_extra = 0

        opcode = self.nextByte()
        self._run_operation(opcode)

    def execute(self, instruction: list[int]) -> None:
        """
        Execute a single instruction independent of the program in memory.
        instruction is an array of bytes.

        :param list[int] instruction: List of unsigned integers to execute
        :rtype: None
        :return: None
        """
        for i in instruction:
            self.cc = 0
            self.cc_extra = 0

            self._run_operation(i)

    def _run_operation(self, opcode: int) -> None:
        """
        Execute a single instruction.

        :param list[int] instruction: List of unsigned integers to execute
        :rtype: None
        :return: None
        """
        self.op = self.opcodes[opcode]

        if self.debug is True:
            dasm = Disassembly(self.op, *self.op.get_operands())

        self.op.execute()

        self.cc += self.cc_extra

        if self.debug is True:
            print(
                f"{dasm!r: <44} {self.r!r: >44} "
                f"C: {self.cc:d} TC: {self.cc_total:d}"
            )
        self.cc_total += self.cc

        self.handle_interrupt()

    def handle_interrupt(self) -> None:
        """Handle interrupts (IRQ/NMI)"""
        if self._previous_interrupt:
            if self.trigger_nmi:
                self.process_nmi()
                self.trigger_nmi = False
            elif self.trigger_irq:
                self.process_irq()
                self.trigger_irq = False

    def readByte(self, addr: int) -> int:
        """
        Read byte from MMU

        :param int addr: 16 bit memory address
        :rtype: int
        :return: 8 bit (1 byte)
        """
        v = self.mmu.read(addr)
        self.increment_cycle_count()
        return v

    def readWord(self, addr: int) -> int:
        """
        Read 2 bytes (word) from MMU

        :param int addr: 16 bit memory address
        :rtype: int
        :return: 16 bit (2 byte)
        """
        low = self.readByte(addr)
        high = self.readByte(addr + 1)
        return (high << 8) + low

    def writeByte(self, addr: int, value: int) -> None:
        """
        Write byte to MMU

        :param int addr: 16 bit memory address
        :param int value: 8 bit value
        :rtype: None
        :return: None
        """
        self.mmu.write(addr, value)
        self.increment_cycle_count()

    def writeWord(self, addr: int, value: int) -> None:
        """
        Write 2 bytes (word) to MMU

        :param int addr: 16 bit memory address
        :param int value: 16 bit value
        :rtype: None
        :return: None
        """
        self.writeByte(addr, value & 0x00FF)
        self.writeByte(addr, (value >> 8) & 0x00FF)

    def nextByte(self) -> int:
        """
        Read next value (8 bit) from program counter

        :rtype: int
        :return: 8 bit
        """
        v = self.readByte(self.r.pc)
        self.r.pc = (self.r.pc + 1) & 0xFFFF
        return v

    def nextWord(self) -> int:
        """Read next value (16 bit) from program counter

        :rtype: int
        :return: 16 bit
        """
        low = self.nextByte()
        high = self.nextByte()
        return (high << 8) + low

    def stackPush(self, v: int) -> None:
        """Pushes value (8 bit) to stack

        :param int v: 8 bit
        """
        self.writeByte((self.stack_page * 0x100) + self.r.s, v)
        self.r.s = (self.r.s - 1) & 0xFF

    def stackPushWord(self, v: int) -> None:
        """Pushes value (16 bit) to stack

        :param int v: 16 bit
        """
        self.stackPush(v >> 8)
        self.stackPush(v & 0xFF)

    def stackPop(self) -> int:
        """Pops value (8 bit) from stack

        :rtype: int
        :return: 8 bit
        """
        v = self.readByte(self.stack_page * 0x100 + ((self.r.s + 1) & 0xFF))
        self.r.s = (self.r.s + 1) & 0xFF
        return v

    def stackPopWord(self) -> int:
        """Pops value (16 bit) from stack

        :rtype: int
        :return: 16 bit
        """
        return self.stackPop() + (self.stackPop() << 8)

    def fromBCD(self, v) -> int:
        """Converts BCD (Binary Coded Decimal) to int

        :param int v: BCD coded value
        :rtype: int
        :return: 8 bit
        """
        return (((v & 0xF0) // 0x10) * 10) + (v & 0xF)

    def toBCD(self, v) -> int:
        """
        Converts int to BCD (Binary Coded Decimal)

        :param int v: Integer
        :rtype: int
        :return: BCD coded value

        """
        return int(math.floor(v / 10)) * 16 + (v % 10)

    def signedHex(self, v: int, bits: int = 8) -> int:
        """
        Converts unsigned int value to signed int value

        :param int v: The unsigned int value
        :param int bits: | Number of bits the value has. (Default 8)
                         | 8 bits   = byte
                         | 16 bits  = word
                         | 32 bits  = long
                         | 64 bits  = long long
                         | 128 bits = double
        :rtype: int
        :return: Signed integer

        """
        if v >= (1 << (bits - 1)):
            v -= (1 << bits)
        return v

    def fromTwosCom(self, v: int) -> int:
        """Converts unsigned int value to signed int value

        :param int v: The unsigned int value
        :rtype: int
        :return: Signed integer
        """
        return (v & 0x7F) - (v & 0x80)

    def increment_cycle_count(self, cycles: int = 1) -> None:
        """Increment cycle on current operation

        :param int cycles: Number of cycles to increment with. (Default: 1)
        """
        self.cc = (self.cc + cycles) & 0xFF
        self._previousInterrupt = self._interrupt
        self._interrupt = (
            self.trigger_nmi or
            (self.trigger_irq and self.r.getFlag(FlagBit.I) is False)
        )

    def increment_extra_cycle(self, cycles: int = 1) -> None:
        """Add extra cycle to current operation"""
        self.cc_extra = (self.cc_extra + cycles) & 0xFF

    interrupts = {
        "ABORT": 0xfff8,
        "COP": 0xfff4,
        "IRQ": 0xfffe,
        "BRK": 0xfffe,
        "NMI": 0xfffa,
        "RESET": 0xfffc
    }
    """Dictionary with interrupt types and their corresponding addresses"""

    def interruptAddress(self, irq_type: str) -> int:
        """
        Get values from interrupt vector

        .. seealso::
           :py:attr:`.interrupts`


        :param str irq_type: Interrupt type, key
                             from :py:attr:`.interrupts`
        """
        return self.mmu.readWord(self.interrupts[irq_type])

    def interruptRequest(self) -> None:
        """Trigger interrupt on next operation"""
        self.trigger_irq = True

    def breakOperation(self, irq_type: str) -> None:
        """
        The BRK routine. Called when a BRK occurs.
        Also called from NMI/IRQ operations

        :param str irq_type: Interrupt type, key
                             from :py:attr:`py65emu.CPU.interrupts`
        """
        self.increment_cycle_count()
        self.stackPushWord(self.r.pc + 1)
        self.increment_cycle_count()

        if irq_type == "BRK":
            self.stackPush(self.r.p | FlagBit.B.value)
        else:
            self.stackPush(self.r.p)

        self.increment_cycle_count()

        self.r.setFlag("I")
        self.r.pc = self.interruptAddress(irq_type)

    def process_nmi(self) -> None:
        """
        Proccesses NMI Interrupt.

        .. seealso::
           :py:meth:`.breakOperation`
        """
        self.r.pc -= 1
        self.breakOperation("NMI")

    def process_irq(self) -> None:
        """
        Proccesses IRQ Interrupt.

        .. seealso::
           :py:meth:`.breakOperation`
        """
        if self.r.getFlag(FlagBit.I):
            return None
        self.r.pc -= 1
        self.breakOperation("IRQ")

    # region Addressing modes
    def im_a(self) -> int:
        """
        Immediate - OPC #$BB (Also: Accumulator - OPC A, Implied - OPC)

        +--------------+--------------------------------------------------+
        | Address Mode | Description                                      |
        +==============+==================================================+
        | Immediate    | The instruction expects the next byte to be used |
        |              | as a value, so we'll prep the read address to    |
        |              | point to the next byte                           |
        +--------------+--------------------------------------------------+
        | Method is also used for following address modes                 |
        +--------------+--------------------------------------------------+
        | Accumulator  | Operand is always AC                             |
        +--------------+--------------------------------------------------+
        | Implied      | There is no additional data required for this    |
        |              | instruction. The instruction does something very |
        |              | simple like like sets a status bit. However, we  |
        |              | will target the accumulator, for instructions    |
        |              | like PHA                                         |
        +--------------+--------------------------------------------------+

        :rtype: int
        :return: 8 bit value
        """
        return self.nextByte()

    def z_a(self) -> int:
        """
        Zero Page - OPC $LL

        +--------------+--------------------------------------------------+
        | Address Mode | Description                                      |
        +--------------+--------------------------------------------------+
        | Zero Page    | To save program bytes, zero page addressing      |
        |              | allows you to absolutely address a location in   |
        |              | first 0xFF bytes of address range. Clearly this  |
        |              | only requires one byte instead of the usual two. |
        +--------------+--------------------------------------------------+

        :rtype: int
        :return: 8 bit address
        """
        return self.nextByte() & 0xFF

    def zx_a(self) -> int:
        """
        Zero Page with X Offset - OPC $LL, X

        +-------------------------+---------------------------------------+
        | Address Mode            | Description                           |
        +-------------------------+---------------------------------------+
        | Zero Page with X Offset | Fundamentally the same as Zero Page   |
        |                         | addressing, but the contents of the   |
        |                         | X Register is added to the supplied   |
        |                         | single byte address. This is useful   |
        |                         | for iterating through ranges within   |
        |                         | the first page.                       |
        +-------------------------+---------------------------------------+

        :rtype: int
        :return: 8 bit address
        """
        self.increment_cycle_count()
        return (self.nextByte() + self.r.x) & 0xFF

    def zy_a(self) -> int:
        """
        Zero Page with Y Offset - OPC $LL, Y

        +-------------------------+---------------------------------------+
        | Address Mode            | Description                           |
        +-------------------------+---------------------------------------+
        | Zero Page with Y Offset | Same as above but uses Y Register for |
        |                         | offset                                |
        +-------------------------+---------------------------------------+

        :rtype: int
        :return: 8 bit address
        """
        self.increment_cycle_count()
        return (self.nextByte() + self.r.y) & 0xFF

    def rel_a(self) -> int:
        """
        Relative - OPC $BB (This address mode is exclusive to branch instr.)

        +--------------+--------------------------------------------------+
        | Address Mode | Description                                      |
        +--------------+--------------------------------------------------+
        | Relative     | This address mode is exclusive to branch         |
        |              | instructions. The address must reside within     |
        |              | -128 to +127 of the branch instruction, i.e. you |
        |              | cant directly branch to any address in the       |
        |              | addressable range.                               |
        +--------------+--------------------------------------------------+

        :rtype: int
        :return: 16 bit address
        """
        d = self.nextByte()
        return (self.r.pc + self.fromTwosCom(d)) & 0xFFFF

    def a_a(self) -> int:
        """
        Absolute - OPC $LLHH

        +--------------+--------------------------------------------------+
        | Address Mode | Description                                      |
        +--------------+--------------------------------------------------+
        | Absolute     | A full 16-bit address is loaded and used         |
        +--------------+--------------------------------------------------+

        :rtype: int
        :return: 16 bit address
        """
        return self.nextWord()

    def ax_a(self) -> int:
        """
        Absolute with X Offset - OPC $LLHH, X

        +------------------------+----------------------------------------+
        | Address Mode           | Description                            |
        +------------------------+----------------------------------------+
        | Absolute with X Offset | Fundamentally the same as absolute     |
        |                        | addressing, but the contents of the    |
        |                        | X Register is added to the supplied    |
        |                        | two byte address. If the resulting     |
        |                        | address changes the page, an           |
        |                        | additional clock cycle is required     |
        +------------------------+----------------------------------------+

        :rtype: int
        :return: 16 bit address
        """
        o = self.nextWord()
        a = o + self.r.x

        special_op = [
            0x1E, 0x1F, 0x3E, 0x3F, 0x5E, 0x5F, 0x7E,
            0x7F, 0x9D, 0x9F, 0xDE, 0xDF, 0xFE, 0xFF
        ]
        if o & 0xFF00 != a & 0xFF00 and (
            self.op and self.op.opcode not in special_op
        ):
            # self.cc += 1
            self.increment_extra_cycle()
        elif self.op and self.op.opcode in special_op:
            self.increment_cycle_count()

        return a & 0xFFFF

    def ay_a(self) -> int:
        """
        Absolute with Y Offset - OPC $LLHH, Y

        +------------------------+----------------------------------------+
        | Address Mode           | Description                            |
        +------------------------+----------------------------------------+
        | Absolute with Y Offset | Fundamentally the same as absolute     |
        |                        | addressing, but the contents of the    |
        |                        | Y Register is added to the supplied    |
        |                        | two byte address. If the resulting     |
        |                        | address changes the page, an           |
        |                        | additional clock cycle is required     |
        +------------------------+----------------------------------------+

        :rtype: int
        :return: 16 bit address
        """
        o = self.nextWord()
        a = o + self.r.y

        special_op = [0x1B, 0x3B, 0x5B, 0x7B, 0x99, 0xDB, 0xFB]
        if o & 0xFF00 != a & 0xFF00 and (
            self.op and self.op.opcode not in special_op
        ):
            # self.cc += 1
            self.increment_extra_cycle()
        elif self.op and self.op.opcode in special_op:
            self.increment_cycle_count()

        return a & 0xFFFF

    def i_a(self) -> int:
        """
        Indirect - OPC ($LLHH) (Only used by indirect JMP)

        +--------------+--------------------------------------------------+
        | Address Mode | Description                                      |
        +--------------+--------------------------------------------------+
        | Indirect     | The supplied 16-bit address is read to get the   |
        |              | actual 16-bit address. This is instruction is    |
        |              | unusual in that it has a bug in the hardware!    |
        |              | To emulate its function accurately, we also need |
        |              | to emulate this bug. If the low byte of the      |
        |              | supplied address is 0xFF, then to read the high  |
        |              | byte of the actual address we need to cross a    |
        |              | page boundary. This doesn't actually work on the |
        |              | chip as designed, instead it wraps back around   |
        |              | in the same page, yielding an invalid actual     |
        |              | address                                          |
        +--------------+--------------------------------------------------+


        :rtype: int
        :return: 16 bit address
        """
        i = self.nextWord()
        # Doesn't carry, so if the low byte is in the XXFF position
        # Then the high byte will be XX00 rather than XY00
        if i & 0xFF == 0xFF:
            j = i - 0xFF
        else:
            j = i + 1

        return ((self.readByte(j) << 8) + self.readByte(i)) & 0xFFFF

    def ix_a(self) -> int:
        """
        Indirect X - OPC ($LL, X)

        +--------------+--------------------------------------------------+
        | Address Mode | Description                                      |
        +--------------+--------------------------------------------------+
        | Indirect X   | The supplied 8-bit address is offset by          |
        |              | X Register to index a location in page 0x00. The |
        |              | actual 16-bit address is read from this location |
        +--------------+--------------------------------------------------+

        :rtype: int
        :return: 16 bit address
        """
        # self.increment_extra_cycle()
        self.increment_cycle_count()
        i = (self.nextByte() + self.r.x) & 0xFF
        return (
            ((self.readByte((i + 1) & 0xFF) << 8) + self.readByte(i)) & 0xFFFF
        )

    def iy_a(self) -> int:
        """
        Indirect Y - OPC ($LL), Y

        +--------------+--------------------------------------------------+
        | Address Mode | Description                                      |
        +--------------+--------------------------------------------------+
        | Indirect Y   | The supplied 8-bit address is offset by          |
        |              | X Register to index a location in page 0x00. The |
        |              | actual 16-bit address is read, and the contents  |
        |              | of Y Register is added to it to offset it. If    |
        |              | the offset causes change in page then an         |
        |              | additional clock cycle is required.              |
        +--------------+--------------------------------------------------+

        :rtype: int
        :return: 16 bit address
        """
        i = self.nextByte()
        o = (self.readByte((i + 1) & 0xFF) << 8) + self.readByte(i)
        a = o + self.r.y

        special_op = [0x13, 0x33, 0x53, 0x73, 0x91, 0xD3, 0xF3]
        if o & 0xFF00 != a & 0xFF00 and (
            self.op and self.op.opcode not in special_op
        ):
            # self.cc += 1
            self.increment_extra_cycle()
        elif self.op and self.op.opcode in special_op:
            self.increment_cycle_count()

        return a & 0xFFFF

    # endregion Addressing modes

    # region Return values based on the addressing mode
    def im(self) -> int:
        """
        Alias method for :py:meth`py65emu.cpu.CPU.im_a`

        :return: 8 bit value
        :rtype: int
        """
        return self.im_a()

    def z(self) -> int:
        """
        Return memory value from Zero Page

        :return: 8 bit value
        :rtype: int
        """
        return self.readByte(self.z_a())

    def zx(self) -> int:
        """
        Return memory value from Zero Page, X

        :return: 8 bit value
        :rtype: int
        """
        return self.readByte(self.zx_a())

    def zy(self) -> int:
        """
        Return memory value from Zero Page, Y

        :return: 8 bit value
        :rtype: int
        """
        return self.readByte(self.zy_a())

    def a(self) -> int:
        """
        Return memory value from Absolute

        :return: 8 bit value
        :rtype: int
        """
        return self.readByte(self.a_a())

    def ax(self) -> int:
        """
        Return memory value from Absolute, X

        :return: 8 bit value
        :rtype: int
        """
        return self.readByte(self.ax_a())

    def ay(self) -> int:
        """
        Return memory value from Absolute, Y

        :return: 8 bit value
        :rtype: int
        """
        return self.readByte(self.ay_a())

    def i(self) -> int:
        """
        Return memory value from Indirect

        :return: 8 bit value
        :rtype: int
        """
        return self.readByte(self.i_a())

    def ix(self) -> int:
        """
        Return memory value from Indirect, X

        :return: 8 bit value
        :rtype: int
        """
        return self.readByte(self.ix_a())

    def iy(self) -> int:
        """
        Return memory value from Indirect, Y

        :return: 8 bit value
        :rtype: int
        """
        return self.readByte(self.iy_a())

    # endregion Return values based on the addressing mode

    # region Instructions Mnemonics

    def ADC(self, v2: int) -> None:
        """
        Add with Carry In

        +---------------+---------------+
        | Function      | `N Z C I D V` |
        +===============+===============+
        | A = A + M + C | `N Z C - - -` |
        +---------------+---------------+

        :param int v2: 8 bit value
        """
        v1 = self.r.a

        if self.r.getFlag("D") and self.bcd_disabled is False:  # decimal mode
            d1 = self.fromBCD(v1)
            d2 = self.fromBCD(v2)
            r = d1 + d2 + self.r.getFlag("C")
            self.r.a = self.toBCD(r % 100)

            self.r.setFlag("C", r > 99)
        else:
            r = v1 + v2 + self.r.getFlag("C")
            self.r.a = r & 0xFF

            self.r.setFlag("C", r > 0xFF)

        self.r.ZN(self.r.a)
        self.r.setFlag("V", ((~(v1 ^ v2)) & (v1 ^ r) & 0x80))

    def AND(self, v: int) -> None:
        """
        Bitwise Logic AND

        +-----------+---------------+
        | Function  | `N Z C I D V` |
        +===========+===============+
        | A = A & M | `N Z - - - -` |
        +-----------+---------------+

        :param int v: 8 bit value
        """
        self.r.a = (self.r.a & v) & 0xFF
        self.r.ZN(self.r.a)

    def ASL(self, a: str | int) -> None:
        """
        Arithmetic Shift Left

        +------------------------+---------------+
        | Function               | `N Z C I D V` |
        +========================+===============+
        | A = C <- (A << 1) <- 0 | `N Z - - - V` |
        +------------------------+---------------+

        :param a: 16 bit address, or accumulator
        :type a: int | str
        """
        if isinstance(a, str):
            v = self.r.a
            self.increment_cycle_count()
        else:
            v = self.readByte(a)
            self.writeByte(a, v)

        v = v << 1
        self.r.setFlag("C", v > 0xFF)
        self.r.ZN(v & 0xFF)

        if isinstance(a, str):
            self.r.a = v & 0xFF
        else:
            self.writeByte(a, v)

    def BIT(self, v: int) -> None:
        """
        Test Bits in Memory with Accumulator

        +----------+---------------+
        | Function | `N Z C I D V` |
        +==========+===============+
        | A & M    | `N Z - - - V` |
        +----------+               |
        | M6 -> V  |               |
        +----------+               |
        | M7 -> N  |               |
        +----------+---------------+

        :param int v: 8 bit value
        """
        self.r.setFlag("Z", self.r.a & v == 0)
        self.r.setFlag("N", v & 0x80)
        self.r.setFlag("V", v & 0x40)

    def B(self, v: tuple[str, bool]) -> None:
        """
        Aggregated method for all "Branch On"-operations
        For instance, BCC (Branch Carry Clear) will call B(('C', False))

        +------+------+---------------------------+----------+---------------+
        | Name | OPC  | Instruction               | Function | `N Z C I D V` |
        +======+======+===========================+==========+===============+
        | BCC  | 0x90 | Branch on Carry Clear     | C = 0    | `- - - - - -` |
        +------+------+---------------------------+----------+---------------+
        | BCS  | 0xB0 | Branch on Carry Set       | C = 1    | `- - - - - -` |
        +------+------+---------------------------+----------+---------------+
        | BEQ  | 0xF0 | Branch on Result Zero     | Z = 1    | `- - - - - -` |
        +------+------+---------------------------+----------+---------------+
        | BNE  | 0xD0 | Branch on Result not Zero | Z = 0    | `- - - - - -` |
        +------+------+---------------------------+----------+---------------+
        | BMI  | 0x30 | Branch on Result Minus    | N = 1    | `- - - - - -` |
        +------+------+---------------------------+----------+---------------+
        | BPL  | 0x10 | Branch on Result Plus     | N = 0    | `- - - - - -` |
        +------+------+---------------------------+----------+---------------+
        | BVC  | 0x50 | Branch on Overflow Clear  | V = 0    | `- - - - - -` |
        +------+------+---------------------------+----------+---------------+
        | BVS  | 0x70 | Branch on Overflow Set    | V = 1    | `- - - - - -` |
        +------+------+---------------------------+----------+---------------+

        :param v: v is a tuple of (flag, boolean)
        :type v: tuple[str, bool]
        """

        pc_rel = self.rel_a()
        if self.r.getFlag(v[0]) is v[1]:
            self.increment_extra_cycle()

            if pc_rel & 0xFF00 != self.r.pc & 0xFF00:
                self.increment_extra_cycle()
            self.r.pc = pc_rel

    def BRK(self, _: int) -> None:
        """
        Break

        +--------------+---------------------------+---------------+
        | Instruction  | Function                  | `N Z C I D V` |
        +==============+===========================+===============+
        | Break        | Program Sourced Interrupt | `- - - - - -` |
        +--------------+---------------------------+---------------+

        :param int _: Ignored
        """
        self.breakOperation("BRK")

    def CP(self, r: int, v: int) -> None:
        """
        Support method for CMP/CPX/CPY methods

        :param int r: 8 bit register value
        :param int v: 8 bit value
        """
        o = (r - v) & 0xFF
        self.r.setFlag("Z", (o & 0xFF) == 0)
        self.r.setFlag("C", r >= v)
        self.r.setFlag("N", o & 0x80)

    def CMP(self, v: int) -> None:
        """
        Compare Accumulator

        +---------------------+-------------------+---------------+
        | Instruction         | Function          | `N Z C I D V` |
        +=====================+===================+===============+
        | Compare Accumulator | C <- A >= M       | `N Z C - - -` |
        |                     +-------------------+               |
        |                     | Z <- (A - M) == 0 |               |
        +---------------------+-------------------+---------------+

        :param int v: 8 bit value
        """
        self.CP(self.r.a, v)

    def CPX(self, v: int) -> None:
        """
        Compare X Register

        +-------------------+---------------+
        | Function          | `N Z C I D V` |
        +===================+===============+
        | C <- X >= M       | `N Z C - - -` |
        +-------------------+               |
        | Z <- (X - M) == 0 |               |
        +-------------------+---------------+

        :param int v2: 8 bit value
        """
        self.CP(self.r.x, v)

    def CPY(self, v: int) -> None:
        """
        Compare Y Register

        +-------------------+---------------+
        | Function          | `N Z C I D V` |
        +===================+===============+
        | C <- Y >= M       | `N Z C - - -` |
        +-------------------+               |
        | Z <- (Y - M) == 0 |               |
        +-------------------+---------------+

        :param int v: 8 bit value
        """
        self.CP(self.r.y, v)

    def DEC(self, a: int) -> None:
        """
        Decrement Value at Memory Location

        +------------------------------------+-----------+---------------+
        | Instruction                        | Function  | `N Z C I D V` |
        +====================================+===========+===============+
        | Decrement Value at Memory Location | M = M - 1 | `N Z - - - -` |
        +------------------------------------+-----------+---------------+

        :param int a: 16 bit address location
        """
        v = self.readByte(a)
        self.writeByte(a, v & 0xFF)
        v = (v - 1) & 0xFF
        self.writeByte(a, v)
        self.r.ZN(v)

    def DEX(self, _) -> None:
        """
        Decrement X Register

        +-----------+---------------+
        | Function  | `N Z C I D V` |
        +===========+===============+
        | X = X - 1 | `N Z - - - -` |
        +-----------+---------------+

        :param int _: Ignored
        """
        self.r.x = (self.r.x - 1) & 0xFF
        self.r.ZN(self.r.x)
        self.increment_cycle_count()

    def DEY(self, _) -> None:
        """
        Decrement Y Register

        +-----------+---------------+
        | Function  | `N Z C I D V` |
        +===========+===============+
        | Y = Y - 1 | `N Z - - - -` |
        +-----------+---------------+

        :param int _: Ignored
        """
        self.r.y = (self.r.y - 1) & 0xFF
        self.r.ZN(self.r.y)
        self.increment_cycle_count()

    def EOR(self, v: int) -> None:
        """
        Bitwise Logic XOR

        +-----------+---------------+
        | Function  | `N Z C I D V` |
        +===========+===============+
        | A = A ^ M | `N Z - - - -` |
        +-----------+---------------+

        :param int v: 8 bit value
        """
        self.r.a = self.r.a ^ v
        self.r.ZN(self.r.a)

    """Flag Instructions."""

    def SE(self, v: str) -> None:
        """
        Aggregated method for all "Set Flag"-operations

        +------+------+------------------------+-----------+---------------+
        | Name | OPC  | Instruction            | Function  | `N Z C I D V` |
        +======+======+========================+===========+===============+
        | SEC  | 0x38 | Set Carry Flag         | 1 -> C    | `- - 1 - - -` |
        +------+------+------------------------+-----------+---------------+
        | SED  | 0xF8 | Set Decimal Flag       | 1 -> D    | `- - - - 1 -` |
        +------+------+------------------------+-----------+---------------+
        | SEI  | 0x78 | Set IRQ Disable Status | 1 -> I    | `- - - 1 - -` |
        +------+------+------------------------+-----------+---------------+

        :param v: One of `D`, `I`, `C`
        :type v: FlagBit or int or str
        """
        self.r.setFlag(v)
        self.increment_cycle_count()

    def CL(self, v: str) -> None:
        """
        Aggregated method for all "Clear Flag"-operations

        +------+------+--------------------------+-----------+---------------+
        | Name | OPC  | Instruction              | Function  | `N Z C I D V` |
        +======+======+==========================+===========+===============+
        | CLC  | 0x18 | Clear Carry Flag         | 0 -> C    | `- - 0 - - -` |
        +------+------+--------------------------+-----------+---------------+
        | CLD  | 0xD8 | Clear Decimal Flag       | 0 -> D    | `- - - - 0 -` |
        +------+------+--------------------------+-----------+---------------+
        | CLI  | 0x58 | Clear IRQ Disable Status | 0 -> I    | `- - - 0 - -` |
        +------+------+--------------------------+-----------+---------------+
        | CLV  | 0xB8 | Clear Overflow Flag      | 0 -> V    | `- - - - - 0` |
        +------+------+--------------------------+-----------+---------------+

        :param v: One of `V`, `D`, `I`, `C`
        :type v: FlagBit or int or str
        """
        self.r.clearFlag(v)
        self.increment_cycle_count()

    def INC(self, a: int) -> None:
        """
        Increment Value at Memory Location

        +------------------------------------+-----------+---------------+
        | Instruction                        | Function  | `N Z C I D V` |
        +====================================+===========+===============+
        | Increment Value at Memory Location | M = M + 1 | `N Z - - - -` |
        +------------------------------------+-----------+---------------+

        :param int a: 16 bit address location
        """
        v = self.readByte(a)
        self.writeByte(a, v & 0xFF)
        v = (v + 1) & 0xFF
        self.writeByte(a, v)
        self.r.ZN(v)

    def INX(self, _) -> None:
        """
        Increment X Register

        +----------------------+-----------+---------------+
        | Instruction          | Function  | `N Z C I D V` |
        +======================+===========+===============+
        | Increment X Register | X = X + 1 | `N Z - - - -` |
        +----------------------+-----------+---------------+

        :param int _: Ignored
        """
        self.r.x = (self.r.x + 1) & 0xFF
        self.r.ZN(self.r.x)
        self.increment_cycle_count()

    def INY(self, _) -> None:
        """
        Increment Y Register

        +-----------+---------------+
        | Function  | `N Z C I D V` |
        +===========+===============+
        | Y = Y + 1 | `N Z - - - -` |
        +-----------+---------------+

        :param int _: Ignored
        """
        self.r.y = (self.r.y + 1) & 0xFF
        self.r.ZN(self.r.y)
        self.increment_cycle_count()

    def JMP(self, a: int) -> None:
        """
        Jump to new Location

        +-----------------+---------------+
        | Function        | `N Z C I D V` |
        +=================+===============+
        | (PC + 1) -> PCL | `- - - - - -` |
        +-----------------+               |
        | (PC + 2) -> PCH |               |
        +-----------------+---------------+

        :param int a: Address to jump new PC
        """
        self.r.pc = a

    def JSR(self, a: int) -> None:
        """
        Jump To Sub-Routine

        +-----------------+---------------+
        | Function        | `N Z C I D V` |
        +=================+===============+
        | push (PC + 2)   | `- - - - - -` |
        +-----------------+               |
        | (PC + 1) -> PCL |               |
        +-----------------+               |
        | (PC + 2) -> PCH |               |
        +-----------------+---------------+

        :param int a: New PC Value
        """
        self.stackPushWord(self.r.pc - 1)
        self.r.pc = a
        self.increment_cycle_count()

    def LDA(self, v: int) -> None:
        """
        Load The Accumulator

        +-----------+---------------+
        | Function  | `N Z C I D V` |
        +===========+===============+
        | A = M     | `N Z - - - -` |
        +-----------+---------------+

        :param int v: 8 bit value
        """
        self.r.a = v
        self.r.ZN(self.r.a)

    def LDX(self, v: int) -> None:
        """
        Load X Register

        +-----------+---------------+
        | Function  | `N Z C I D V` |
        +===========+===============+
        | X = M     | `N Z - - - -` |
        +-----------+---------------+

        :param int v: 8 bit value
        """
        self.r.x = v
        self.r.ZN(self.r.x)

    def LDY(self, v: int) -> None:
        """
        Load Y Register

        +-----------+---------------+
        | Function  | `N Z C I D V` |
        +===========+===============+
        | Y = M     | `N Z - - - -` |
        +-----------+---------------+

        :param int v: 8 bit value
        """
        self.r.y = v
        self.r.ZN(self.r.y)

    def LSR(self, a: str | int) -> None:
        """
        Logical Shift Right

        +------------------------+---------------+
        | Function               | `N Z C I D V` |
        +========================+===============+
        | A = C <- (A << 1) <- 0 | `0 Z C - - -` |
        +------------------------+---------------+

        :param a: 16 bit address, or accumulator
        :type a: int | str
        """
        if isinstance(a, str):
            v = self.r.a
            self.increment_cycle_count()
        else:
            v = self.readByte(a)
            self.writeByte(a, v)

        self.r.setFlag("C", v & 0x01)
        v = v >> 1
        self.r.ZN(v)

        if isinstance(a, str):
            self.r.a = v
        else:
            self.writeByte(a, v)

    def NOP(self, _) -> None:
        """
        No Operation

        Sadly not all NOPs are equal, Ive added a few here
        based on `CPU_unofficial_opcodes`_ and will add more
        based on game compatibility, and ultimately I'd like
        to cover all illegal opcodes too

        ::
            **NOP (`$1A`, `$3A`, `$5A`, `$7A`, `$DA`, `$EA`, `$FA`)**

            Address Mode: Implied

            The official NOP (`$EA`) and six unofficial NOPs do nothing.

            **SKB - OPC (`$80`, `$82`, `$89`, `$C2`, `$E2`)**

            Address Mode: Immediate
            `$89` affects NVZ flags, like :py:meth:`.BIT`

            **IGN - OPC (`$0C`)**
            Address Mode: Absolute

            **IGN - OPC (`$1C`, `$3C`, `$5C`, `$7C`, `$DC`, `$FC`)
            Address Mode: Absolute, X

            **IGN - OPC (`$04`, `$44`, `$64`)
            Address Mode: Zero Page

            **IGN - OPC (`$14`, `$34`, `$54`, `$74`, `$D4`, `$F4`)
            Address Mode: Zero Page, X

        .. _CPU_unofficial_opcodes:
            https://www.nesdev.org/wiki/Programming_with_unofficial_opcodes

        +-----------+---------------+
        | Function  | `N Z C I D V` |
        +===========+===============+
        |           | `- - - - - -` |
        +-----------+---------------+

        :param int _: Ignored
        """
        if self.op and self.op.mode not in ['im']:
            self.increment_cycle_count()

    def ORA(self, v: int) -> None:
        """
        Bitwise Logic OR

        +-----------+---------------+
        | Function  | `N Z C I D V` |
        +===========+===============+
        | A = A | M | `N Z - - - -` |
        +-----------+---------------+

        :param int v: 8 bit value
        """
        self.r.a = self.r.a | v
        self.r.ZN(self.r.a)

    def P(self, v: tuple[str, str]) -> None:
        """
        Aggregated method for all Stack operations, PusH and PulL.

        v is a tuple where the first value is either PH or PL, specifying the
        action and the second is the source or target register, either A or P,
        meaning the Accumulator or the Processor status flag.

        +------+------+------------------+----------+---------------+
        | Name | OPC  | Instruction      | Function | `N Z C I D V` |
        +======+======+==================+==========+===============+
        | PHA  | 0x48 | Push Accumulator | push A   | `- - - - - -` |
        |      |      | on Stack         |          |               |
        +------+------+------------------+----------+---------------+
        | PHP  | 0x08 | Push Register P  | push SR  | `- - - - - -` |
        |      |      | on Stack         |          |               |
        +------+------+------------------+----------+---------------+
        | PLA  | 0x68 | Pull Accumulator | pull A   | `N Z - - - -` |
        |      |      | from Stack       |          |               |
        +------+------+------------------+----------+---------------+
        | PLP  | 0x28 | Pull Register P  | pull SR  | from    stack |
        |      |      | from Stack       |          |               |
        +------+------+------------------+----------+---------------+

        :param a: PHA would be `P((PH, A))`, and
                  PLP `P((PL, P))`
        :type a: tuple[str, str]
        """
        a, r = v

        self.increment_cycle_count()
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
                self.r.clearFlag(FlagBit.B)
                self.r.setFlag(FlagBit.U)
                # self.r.p = self.r.p | 0b00100000

            self.increment_cycle_count()

    def ROL(self, a: str | int) -> None:
        """
        Rotate Left

        +----------+---------------+
        | Function | `N Z C I D V` |
        +==========+===============+
        | (C << 1) | `N Z C - - -` |
        +----------+---------------+

        :param a: 16 bit address, or accumulator
        :type a: int | str
        """
        if isinstance(a, str):
            v_old = self.r.a
            self.increment_cycle_count()
        else:
            v_old = self.readByte(a)
            self.writeByte(a, v_old)

        v_new = ((v_old << 1) + self.r.getFlag("C")) & 0xFF
        self.r.setFlag("C", v_old & 0x80)
        self.r.ZN(v_new)

        if isinstance(a, str):
            self.r.a = v_new
        else:
            self.writeByte(a, v_new)

    def ROR(self, a: str | int) -> None:
        """
        Rotate Right

        +----------+---------------+
        | Function | `N Z C I D V` |
        +==========+===============+
        | (C >> 1) | `N Z C - - -` |
        +----------+---------------+

        :param a: 16 bit address, or accumulator
        :type a: int | str
        """
        if isinstance(a, str):
            v_old = self.r.a
            self.increment_cycle_count()
        else:
            v_old = self.readByte(a)
            self.writeByte(a, v_old)

        v_new = ((v_old >> 1) + self.r.getFlag("C") * 0x80) & 0xFF
        self.r.setFlag("C", v_old & 0x01)
        self.r.ZN(v_new)
        if isinstance(a, str):
            self.r.a = v_new
        else:
            self.writeByte(a, v_new)

    def RTI(self, _) -> None:
        """
        Return from Interrupt

        +-----------+---------------+
        | Function  | `N Z C I D V` |
        +===========+===============+
        | pull SR   | `from  stack` |
        +-----------+               |
        | pull PC   |               |
        +-----------+---------------+

        :param int _: Ignored
        """
        self.r.p = self.stackPop()
        self.r.setFlag(FlagBit.U)
        self.increment_cycle_count()

        self.r.pc = self.stackPopWord()
        self.increment_cycle_count()

    def RTS(self, _) -> None:
        """
        Return from Subroutine

        +--------------+---------------+
        | Function     | `N Z C I D V` |
        +==============+===============+
        | pull PC      | `- - - - - -` |
        +--------------+               |
        | PC + 1 -> PC |               |
        +--------------+---------------+

        :param int _: Ignored
        """
        self.increment_cycle_count(3)
        self.r.pc = (self.stackPopWord() + 1) & 0xFFFF

    def SBC(self, v2: int) -> None:
        """
        Subtraction with Borrow In

        +---------------------+---------------+
        | Function            | `N Z C I D V` |
        +=====================+===============+
        | A = A - M - (1 - C) | `N Z C - - V` |
        +---------------------+---------------+

        :param int v2: 8 bit value
        """
        v1 = self.r.a
        if self.r.getFlag("D") and self.bcd_disabled is False:
            d1 = self.fromBCD(v1)
            d2 = self.fromBCD(v2)
            r = d1 - d2 - (not self.r.getFlag("C"))
            self.r.a = self.toBCD(r % 100)
        else:
            """
            r = v1 + (v2 ^ 0xFF) + self.r.getFlag("C")

            self.r.a = r & 0xFF
            """
            r = v1 - v2 - (not self.r.getFlag('C'))
            self.r.a = r & 0xff

        self.r.setFlag("C", r >= 0)
        self.r.setFlag("V", ((v1 ^ v2) & (v1 ^ r) & 0x80))
        self.r.ZN(self.r.a)

    def STA(self, a: int) -> None:
        """
        Store Accumulator at Address

        +-----------+---------------+
        | Function  | `N Z C I D V` |
        +===========+===============+
        | M = A     | `- - - - - -` |
        +-----------+---------------+

        :param int a: 16 bit address
        """
        self.writeByte(a, self.r.a)

    def STX(self, a: int) -> None:
        """
        Store X Register at Address

        +-----------+---------------+
        | Function  | `N Z C I D V` |
        +===========+===============+
        | M = X     | `- - - - - -` |
        +-----------+---------------+

        :param int a: 16 bit address
        """
        self.writeByte(a, self.r.x)

    def STY(self, a: int) -> None:
        """
        Store Y Register at Address

        +-----------+---------------+
        | Function  | `N Z C I D V` |
        +===========+===============+
        | M = Y     | `- - - - - -` |
        +-----------+---------------+

        :param int a: 16 bit address
        """
        self.writeByte(a, self.r.y)

    def T(self, a: tuple[str, str]) -> None:
        """
        Aggregated method for all transfer register to register operations.

        Parameter is a 2-tuple with source and destination register,
        so a call to TAX-operation (`$AA`) would be:

        >>> cpu.T(('a', 'x'))

        +--------------------------------------+----------+---------------+
        | Instruction                          | Function | `N Z C I D V` |
        +======================================+==========+===============+
        | Transfer Accumulator to Register X   | A -> X   | `N Z - - - -` |
        +--------------------------------------+----------+---------------+
        | Transfer Accumulator to Register Y   | A -> Y   | `N Z - - - -` |
        +--------------------------------------+----------+---------------+
        | Transfer Stack Pointer to Register X | SP -> X  | `N Z - - - -` |
        +--------------------------------------+----------+---------------+
        | Transfer Register X to Accumulator   | X -> A   | `N Z - - - -` |
        +--------------------------------------+----------+---------------+
        | Transfer Register X to Stack Pointer | X -> SP  | `- - - - - -` |
        +--------------------------------------+----------+---------------+
        | Transfer Register Y to Accumulator   | Y -> A   | `N Z - - - -` |
        +--------------------------------------+----------+---------------+

        :param a: tuple with (source, destination)
        :type a: tuple[str, str]
        """
        s, d = a
        self.increment_cycle_count()
        setattr(self.r, d, getattr(self.r, s))
        if d != "s":
            self.r.ZN(getattr(self.r, d))

    # endregion Instructions Mnemonics

    # region Illegal/Undocumented Operations

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

    def AAC(self, v: int) -> None:  # ANC
        """
        AND oper + set C as ASL

        .. important:: Illegal Opcode

        +-------------+---------------+
        | Function    | `N Z C I D V` |
        +=============+===============+
        | A AND oper  | `N Z C - - -` |
        +-------------+               |
        | bit(7) -> C |               |
        +-------------+---------------+

        :param int v: 8 bit value
        """
        self.AND(v)
        self.r.setFlag("C", self.r.getFlag("N"))

    def AAX(self, a: int) -> None:  # SAX, AXS
        """
        A and X are put on the bus at the same time (resulting effectively in
        an AND operation) and stored in M

        .. important:: Illegal Opcode

        +--------------+---------------+
        | Function     | `N Z C I D V` |
        +==============+===============+
        | A AND X -> M | `- - - - - -` |
        +--------------+---------------+

        :param int a: 16 bit address
        """
        r = self.r.a & self.r.x
        self.writeByte(a, r)
        # There is conflicting information whether this effects P.
        # self.r.ZN(r)

    def ARR(self, v: int) -> None:
        """
        AND oper + ROR

        .. important:: Illegal Opcode

        +----------------------+---------------+
        | Function             | `N Z C I D V` |
        +======================+===============+
        | A AND oper           | `N Z C - - V` |
        +----------------------+               |
        | C -> [76543210] -> C |               |
        +----------------------+---------------+

        :param int v: 8 bit value
        """
        self.r.a = (self.r.getFlag('C') << 7) | (((self.r.a & v) & 0xFF) >> 1)
        self.r.setFlag("C", self.r.a & 0x40)
        self.r.setFlag("V", bool(self.r.a & 0x40) ^ bool(self.r.a & 0x20))
        self.r.ZN(self.r.a)
        # self.AND(v)
        # self.ROR("a")

    def ASR(self, v) -> None:  # ALR
        """
        AND oper + LSR

        .. important:: Illegal Opcode

        .. note:: Also known as: `ALR`

        +----------------------+---------------+
        | Function n n n n n   | `N Z C I D V` |
        +======================+===============+
        | A AND oper           | `N Z C - - -` |
        +----------------------+               |
        | 0 -> [76543210] -> C |               |
        +----------------------+---------------+

        :param int v: 8 bit value
        """
        self.r.a = (((self.r.a & v) & 0xFF) >> 1)
        self.r.setFlag("C", self.r.a & 0x0001)
        self.r.ZN(self.r.a)
        # self.AND(v)
        # self.LSR("a")

    def ATX(self, v: int) -> None:  # LXA, OAL
        """
        Store * AND oper in A and X

        .. important:: Illegal Opcode

        .. note:: Also known as: `LXA`, `OAL`

        .. DANGER:: Highly unstable
           Involves a "magic" constant

        .. seealso::
           :py:meth:`.XAA`

        +------------------+---------------+
        | Function         | `N Z C I D V` |
        +==================+===============+
        | (A OR CONST) AND | `N Z - - - -` |
        | oper -> A -> X   |               |
        +------------------+---------------+

        :param int v: 8 bit value
        """
        # self.AND(v)
        # self.T(("a", "x"))
        value = (self.r.a ^ self.magic) & v
        self.r.a = value
        self.r.x = value
        self.r.ZN(value)

    def AXA(self, a: int) -> None:  # SHA
        """
        Stores A AND X AND (high-byte of addr. + 1) at addr.

        .. important:: Illegal Opcode

        .. note:: Also known as: `SHA`

        .. Warning:: Unstable
           Sometimes `AND (H+1)` is dropped, page boundary crossings may not
           work (with the high-byte of the value used as the high-byte of the
           address)

        There are a few illegal opcodes which and the high
        bit of the address with registers and write the values
        back into that address.  These operations are
        particularly screwy.  These posts are used as reference
        but I am unsure whether they are correct.

        - http://forums.nesdev.com/viewtopic.php?f=3&t=3831&start=30#p113343

        - http://forums.nesdev.com/viewtopic.php?f=3&t=10698

        +------------------+---------------+
        | Function         | `N Z C I D V` |
        +==================+===============+
        | A AND X AND      | `- - - - - -` |
        | (H+1) -> M       |               |
        +------------------+---------------+

        :param int a: 16 bit address
        """
        o = (a - self.r.y) & 0xFFFF
        low = o & 0xFF
        high = o >> 8

        self.increment_cycle_count()
        if (low + self.r.y) > 0xFF:  # crossed page
            a = ((high & self.r.x) << 8) + low + self.r.y
        else:
            a = (high << 8) + low + self.r.y

        v = self.r.a & self.r.x & (high + 1)
        self.writeByte(a, v)

    def AXS(self, v: int) -> None:  # SBX, SAX
        """
        CMP and DEX at once, sets flags like CMP

        .. important:: Illegal Opcode

        .. note:: Also known as: `SBX`, `SAX`

        +-----------------------+---------------+
        | Function              | `N Z C I D V` |
        +=======================+===============+
        | (A AND X) - oper -> X | `- - - - - -` |
        +-----------------------+---------------+

        :param int v: 8 bit value
        """
        o = self.r.a & self.r.x
        self.r.x = (o - v) & 0xFF

        self.r.setFlag("C", v <= o)
        self.r.ZN(self.r.x)

    def DCP(self, a: int) -> None:  # DCM
        """
        DEC oper + CMP oper

        .. important:: Illegal Opcode

        .. note:: Also known as: `DCN`

        +------------+---------------+
        | Function   | `N Z C I D V` |
        +============+===============+
        | M - 1 -> M | `N Z C - - -` |
        +------------+               |
        | A - M      |               |
        +------------+---------------+

        :param int a: 16 bit address
        """
        self.DEC(a)
        self.CMP(self.mmu.read(a))

    def ISC(self, a: int) -> None:  # ISB, INS
        """
        INC oper + SBC oper

        .. important:: Illegal Opcode

        .. note:: Also known as: `ISB`, `INS`

        +----------------------+---------------+
        | Function             | `N Z C I D V` |
        +======================+===============+
        | M + 1 -> M           | `N Z C - - V` |
        +----------------------+               |
        | A - M - (C - 1) -> A |               |
        +----------------------+---------------+

        :param int a: 16 bit address
        """
        self.INC(a)
        self.SBC(self.mmu.read(a))

    def KIL(self, _) -> None:  # JAM, HLT
        """
        Freezes the CPU

        .. important:: Illegal Opcode

        .. note:: Also known as: `JAM`, `HLT`

        .. Warning:: This instruction freezes the CPU.
           The processor will be trapped infinitely in T1 phase with $FF on
           the data bus. **Reset required**

        :param int _: Ignored
        """
        self.running = False

    def LAR(self, v: int) -> None:  # LAE, LAS
        """
        LDA/TSX oper

        .. important:: Illegal Opcode

        .. note:: Also known as: `LAE`, `LAS`

        +----------------------+---------------+
        | Function             | `N Z C I D V` |
        +======================+===============+
        | M AND SP -> A, X, SP | `N Z - - - -` |
        +----------------------+---------------+

        :param int v: 8 bit value
        """
        self.r.a = self.r.x = self.r.s = self.r.s & v
        self.r.ZN(self.r.a)

    def LAX(self, v: int) -> None:
        """
        LDA oper + LDX oper

        .. important:: Illegal Opcode

        +-------------+---------------+
        | Function    | `N Z C I D V` |
        +=============+===============+
        | M -> A -> X | `N Z - - - -` |
        +-------------+---------------+

        :param int v: 8 bit value
        """
        self.r.a = v & 0xFF
        self.r.x = v & 0xFF
        self.r.ZN(self.r.a)

    def RLA(self, a: int) -> None:
        """
        ROL oper + AND oper

        .. important:: Illegal Opcode

        +--------------------------+---------------+
        | Function                 | `N Z C I D V` |
        +==========================+===============+
        | M = C <- [76543210] <- C | `N Z C - - -` |
        +--------------------------+               |
        | A AND M -> A             |               |
        +--------------------------+---------------+

        :param int a: 16 bit address
        """
        self.ROL(a)
        self.AND(self.mmu.read(a))

    def RRA(self, a: int) -> None:
        """
        ROL oper + ADC oper

        .. important:: Illegal Opcode

        +--------------------------+---------------+
        | Function                 | `N Z C I D V` |
        +==========================+===============+
        | M = C -> [76543210] -> C | `N Z C - - V` |
        +--------------------------+               |
        | A + M + C -> A, C        |               |
        +--------------------------+---------------+

        :param int a: 16 bit address
        """

        self.ROR(a)
        self.ADC(self.mmu.read(a))

    def SLO(self, a: int) -> None:  # ASO
        """
        ASL oper + ORA oper

        .. important:: Illegal Opcode

        .. note:: Also known as: `ASO`

        +--------------------------+---------------+
        | Function                 | `N Z C I D V` |
        +==========================+===============+
        | M = C <- [76543210] <- 0 | `N Z C - - -` |
        +--------------------------+               |
        | A OR M -> A              |               |
        +--------------------------+---------------+

        :param int a: 16 bit address
        """
        self.ASL(a)
        self.ORA(self.mmu.read(a))

    def SRE(self, a: int) -> None:  # LSE
        """
        LSR oper + EOR oper

        .. important:: Illegal Opcode

        .. note:: Also known as: `LSE`

        +--------------------------+---------------+
        | Function                 | `N Z C I D V` |
        +==========================+===============+
        | M = 0 -> [76543210] -> 0 | `N Z C - - -` |
        +--------------------------+               |
        | A EOR M -> A             |               |
        +--------------------------+---------------+

        :param int a: 16 bit address
        """
        self.LSR(a)
        self.EOR(self.mmu.read(a))

    def SXA(self, a: int) -> None:  # SHX, XAS
        """
        Stores X AND (high-byte of addr. + 1) at addr.

        .. important:: Illegal Opcode

        .. note:: Also known as: `SHX`, `XAS`

        +------------------+---------------+
        | Function         | `N Z C I D V` |
        +==================+===============+
        | X AND (H+1) -> M | `- - - - - -` |
        +------------------+---------------+

        .. seealso::
           :py:meth:`.AXA`

        :param int a: 16 bit address
        """
        o = (a - self.r.y) & 0xFFFF
        low = o & 0xFF
        high = o >> 8
        self.increment_cycle_count()
        if low + self.r.y > 0xFF:  # crossed page
            a = ((high & self.r.x) << 8) + low + self.r.y
        else:
            a = (high << 8) + low + self.r.y

        v = self.r.x & (high + 1)
        self.writeByte(a, v)

    def SYA(self, a: int) -> None:  # SHY, SAY
        """
        Stores Y AND (high-byte of addr. + 1) at addr.

        .. important:: Illegal Opcode

        .. note:: Also known as: `SHY`, `SAY`

        +------------------+---------------+
        | Function         | `N Z C I D V` |
        +==================+===============+
        | Y AND (H+1) -> M | `- - - - - -` |
        +------------------+---------------+

        .. seealso::
           :py:meth:`.AXA`

        :param int a: 16 bit address
        """
        o = (a - self.r.x) & 0xFFFF
        low = o & 0xFF
        high = o >> 8
        self.increment_cycle_count()
        if low + self.r.x > 0xFF:  # crossed page
            a = ((high & self.r.y) << 8) + low + self.r.x
        else:
            a = (high << 8) + low + self.r.x

        v = self.r.y & (high + 1)
        self.writeByte(a, v)

    def XAA(self, v: int) -> None:  # ANE
        """
        \\* AND X + AND oper

        .. important:: Illegal Opcode

        .. note:: Also known as: `ANE`

        | Another very wonky operation.
        | It's fully described here: `6502_Opcode_8B_XAA_ANE`_

        "magic" varies by version of the processor.
        `0xEE` seems to be common.

        The formula is: `A = (A | magic) & X & imm`

        .. DANGER:: Highly unstable
           Involves a "magic" constant

           A base value in A is determined based on the
           contets of A and a constant, which may be
           typically `$00`, `$FF`, `$EE`, etc.

           The value of this constant depends on temerature,
           the chip series, and maybe other factors, as well.

           In order to eliminate these uncertaincies from
           the equation, use either 0 as the operand or a
           value of `$FF` in the accumulator.

        .. _6502_Opcode_8B_XAA_ANE:
            http://visual6502.org/wiki/index.php?title=6502_Opcode_8B_%28XAA,_ANE%29  # noqa E501

        +--------------------+---------------+
        | Function           | `N Z C I D V` |
        +====================+===============+
        | (A OR CONST) AND X | `N Z - - - -` |
        | AND oper -> A      |               |
        +--------------------+---------------+

        :param int v: 8 bit value
        """
        self.r.a = (self.r.a | self.magic) & self.r.x & v
        self.r.ZN(self.r.a)

    def XAS(self, a: int) -> None:  # SHS, TAS
        """
        Puts A AND X in SP and stores A AND X AND (high-byte of addr. + 1)

        .. important:: Illegal Opcode

        .. note:: Also known as: `SHS`, `TAS`

        +--------------------------+---------------+
        | Function                 | `N Z C I D V` |
        +==========================+===============+
        | A AND X -> SP            | `- - - - - -` |
        +--------------------------+               |
        | A AND X AND (H + 1) -> M |               |
        +--------------------------+---------------+

        :param int a: 16 bit address
        """
        # First set the stack pointer's value
        self.r.s = self.r.a & self.r.x

        # Then write to memory using the new value of the stack pointer
        o = (a - self.r.y) & 0xFFFF
        low = o & 0xFF
        high = o >> 8
        self.increment_cycle_count()
        if low + self.r.y > 0xFF:  # crossed page
            a = ((high & self.r.s) << 8) + low + self.r.y
        else:
            a = (high << 8) + low + self.r.y

        v = self.r.s & (high + 1)
        self.writeByte(a, v)

    # endregion Illegal/Undocumented Operations

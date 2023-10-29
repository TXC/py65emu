import array
import io
from typing import Sequence


class MemoryRangeError(ValueError):
    pass


class ReadOnlyError(TypeError):
    pass


class Block:
    def __init__(
        self,
        start: int,
        length: int,
        readonly: bool,
        default: int = 0
    ):
        """
        :param int start: The starting address for this block
        :param int length: The length of this block in bytes
        :param bool readOnly: Whether this block should be read only
                              (such as ROM) (default False)
        :param int default: Default value to initialize the block with
        """
        self.start = start
        self.length = length
        self.readonly = readonly
        self.default = default
        self._memory = array.array("B", [default] * length)

    def reset(self) -> None:
        if not self.readonly:
            self._memory = array.array("B", [self.default] * self.length)

    @property
    def end(self) -> int:
        return self.start + self.length

    def __setitem__(self, index: int, value: int) -> None:
        """
        Set value, absolute to the block, by the address in the block.

        :param int index: Address/Position to write to
        :param int value: Value to write
        :raises ReadOnlyError: If block is readonly
        :raises IndexError: If address is out of bounds for block
        """

        """
        if self.readonly:
            raise ReadOnlyError(
                "Memory section is readonly "
                "([${start:0>4x}] - [${length:0>4x}])"
                .format(start=0, length=self.length)
            )
        """

        if index < 0 or index >= self.length:
            raise IndexError(
                "Index [${index:0>4x}] is outside of memory range "
                "([${start:0>4x}] - [${length:0>4x}])".format(
                    index=index,
                    start=0x0,
                    length=self.length
                )
            )

        self._memory[index] = value

    def __getitem__(self, index: int) -> int:
        """
        Get value, relative to the block, by the address in the block.

        :param int index: Position to read from
        :raises IndexError: If address is out of bounds for block
        :rtype: int
        :return: Value of address (8 bit)
        """
        if index < 0 or index >= self.length:
            raise IndexError(
                "Index [${index:0>4x}] is outside of memory range "
                "([${start:0>4x}] - [${length:0>4x}])".format(
                    index=index,
                    start=0x0,
                    length=self.length
                )
            )

        return self._memory[index]

    def __repr__(self) -> str:
        result = ""
        for i in self._memory:
            result += "{:0>2x} ".format(i)

        return result

    def set(self, addr: int, value: int) -> None:
        """
        Set value, by the address in the block.

        :param int index: Address/Position to write to
        :param int value: Value to write
        :raises ReadOnlyError: If block is readonly
        :raises IndexError: If address is out of bounds for block
        """
        if self.readonly:
            raise ReadOnlyError(
                "Memory section is readonly "
                "(0x{s.start:0>4x} - 0x{s.end:0>4x})".format(s=self)
                )

        if addr < self.start or addr >= self.start + self.length:
            raise IndexError(
                "Address 0x{addr:0>4x} is outside of memory range "
                "(0x{s.start:0>4x} - 0x{s.end:0>4x})".format(
                    addr=addr, s=self
                )
            )

        self[addr - self.start] = value

    def get(self, addr: int) -> int:
        """
        Get value, by the address in the block.

        :param int addr: Address to read from
        :raises ReadOnlyError: If block is readonly
        :raises IndexError: If address is out of bounds for block
        :rtype: int
        :return: Value of address (8 bit)
        """
        if addr < self.start or addr >= self.start + self.length:
            raise IndexError(
                "Address 0x{addr:0>4x} is outside of memory range "
                "(0x{s.start:0>4x} - 0x{s.end:0>4x})".format(
                    addr=addr, s=self
                )
            )

        return self[addr - self.start]


class MMU:
    def __init__(self, blocks: Sequence[tuple] = []):
        """
        Initialize the MMU with the blocks specified in blocks.  blocks
        is a list of 5-tuples, (start, length, readonly, value, valueOffset).

        See `addBlock` for details about the parameters.
        """

        # Different blocks of memory stored seperately so that they can
        # have different properties.  Stored as dict of "start", "length",
        # "readonly" and "memory"
        self.blocks: list[Block] = []

        for b in blocks:
            if isinstance(b, tuple):
                self.addBlock(*b)

    def reset(self) -> None:
        """
        In all writeable blocks reset all values to zero.
        """
        for b in self.blocks:
            b.reset()

    def addBlock(
        self,
        start: int,
        length: int,
        readonly: bool = False,
        value: list | tuple | io.BufferedIOBase | None = None,
        valueOffset: int = 0
    ) -> None:
        """
        Add a block of memory to the list of blocks with the given start
        address and length; whether it is readonly or not; and the starting
        value as either a file pointer, binary value or list of unsigned
        integers.  If the block overlaps with an existing block an exception
        will be thrown.

        :param int start: The starting address of the block of memory
        :param int length: The length of the block in bytes
        :param bool readOnly: Whether the block should be read only
                              (such as ROM) (default False)
        :param value: The intial value for the block of memory. Used for
                      loading program data. (Default None)
        :param int valueOffset: Used when copying the above `value` into the
                                block to offset the location it is copied
                                into. For example, to copy byte 0 in `value`
                                into location 1000 in the block, set
                                valueOffest=1000. (Default 0)
        :type value: TextIO | list[int]
        """

        for b in self.blocks:
            if (
                start + length > b.start
                and start + length < b.start + b.length
            ) or (
                b.start + b.length > start
                and b.start + b.length < start + length
            ):
                raise MemoryRangeError()

        newBlock = Block(
            start=start,
            length=length,
            readonly=readonly
        )

        # raise TypeError(type(value))

        if (
            isinstance(value, list) or
            isinstance(value, tuple) or
            isinstance(value, str)
        ):
            for i in range(len(value)):
                newBlock[i + valueOffset] = value[i]
                # newBlock.set(i + valueOffset, value[i])
        elif isinstance(value, io.BufferedIOBase):
            a = array.array("B")
            a.frombytes(value.read())
            for i in range(len(a)):
                newBlock[i + valueOffset] = a[i]
                # newBlock.set(i + valueOffset, a[i])

        self.blocks.append(newBlock)

    def getBlock(self, addr: int) -> Block:
        """
        Get the block associated with the given address.

        :param int addr: Memory address to locate
        """

        for b in self.blocks:
            if addr >= b.start and addr < b.start + b.length:
                return b

        raise IndexError("Unable to locate position %{:0>4x}".format(addr))

    def write(self, addr: int, value: int) -> None:
        """
        Write a value to the given address if it is writeable.

        :param int index: Address/Position to write to
        :param int value: Value to write
        :raises ReadOnlyError: If block is readonly
        :raises IndexError: If address is out of bounds for block
        """
        b = self.getBlock(addr)
        b.set(addr, value & 0xFF)

    def read(self, addr: int) -> int:
        """
        Return the value at the address.

        :param int index: Address to read from
        :param int value: Value to write
        :raises IndexError: If address is out of bounds for block
        :rtype: int
        :return: Value at address (8 bit)
        """
        b = self.getBlock(addr)
        return b.get(addr)

    def readWord(self, addr: int) -> int:
        """
        Return the value at the address.

        :param int index: Address to read from
        :raises IndexError: If address is out of bounds for block
        :rtype: int
        :return: Value at address (16 bit)
        """
        return (self.read(addr + 1) << 8) + self.read(addr)

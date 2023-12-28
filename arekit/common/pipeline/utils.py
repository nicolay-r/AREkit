class BatchIterator:

    def __init__(self, lst, batch_size):
        assert(isinstance(lst, list))
        assert(isinstance(batch_size, int) and batch_size > 0)
        self.__lst = lst
        self.__index = 0
        self.__batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.__index < len(self.__lst):
            batch = self.__lst[self.__index:self.__index + self.__batch_size]
            self.__index += 2
            return batch
        else:
            raise StopIteration

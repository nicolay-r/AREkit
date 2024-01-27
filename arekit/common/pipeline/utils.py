class BatchIterator:

    def __init__(self, data_iter, batch_size, end_value=None):
        assert(isinstance(batch_size, int) and batch_size > 0)
        assert(callable(end_value) or end_value is None)
        self.__data_iter = data_iter
        self.__index = 0
        self.__batch_size = batch_size
        self.__end_value = end_value

    def __iter__(self):
        return self

    def __next__(self):
        buffer = []
        while True:
            try:
                data = next(self.__data_iter)
            except StopIteration:
                break
            buffer.append(data)
            if len(buffer) == self.__batch_size:
                break

        if len(buffer) > 0:
            self.__index += 1
            return buffer

        if self.__end_value is None:
            raise StopIteration
        else:
            return self.__end_value()

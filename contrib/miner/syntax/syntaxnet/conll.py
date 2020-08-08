"""
(C) IINemo
Original: https://github.com/IINemo/syntaxnet_wrapper
"""
from StringIO import StringIO


class ConllFormatSentenceParser(object):
    def __init__(self, string_io):
        super(ConllFormatSentenceParser, self).__init__()
        self.string_ = string_io
        self.stop_ = False

    def __iter__(self):
        return self

    def next(self):
        if not self.stop_:
            line = self.string_.readline().rstrip('\n')
            if not line:
                self.stop_ = True
            else:
                return line.strip().split('\t')

        raise StopIteration()


class ConllFormatStreamParser(object):
    def __init__(self, string):
        super(ConllFormatStreamParser, self).__init__()
        self.string_io_ = StringIO(string)
        self.stop_ = False

    def __iter__(self):
        return self

    def next(self):
        if not self.stop_:
            sent_parser = ConllFormatSentenceParser(self.string_io_)
            result = list(sent_parser)
            if not result:
                self.stop_ = True
            else:
                return result

        raise StopIteration()

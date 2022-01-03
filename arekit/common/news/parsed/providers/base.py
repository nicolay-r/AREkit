class BaseParsedNewsServiceProvider(object):

    @property
    def Name(self):
        raise NotImplementedError()

    def init_parsed_news(self, parsed_news):
        raise NotImplementedError()

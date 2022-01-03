class BaseParsedNewsServiceProvider(object):

    @property
    def Name(self):
        raise NotImplementedError()

    # TODO. #245 utilize this method for parsed news assignation.
    def init_parsed_news(self, parsed_news):
        raise NotImplementedError()

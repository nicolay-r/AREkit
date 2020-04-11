# TODO. Add base implementations.
class DataIO(object):

    @property
    def Stemmer(self):
        raise NotImplementedError()

    @property
    def SynonymsCollection(self):
        raise NotImplementedError()

    @property
    def NeutralAnnotator(self):
        raise NotImplementedError()

    @property
    def KeepTokens(self):
        return True

    @property
    def CVFoldingAlgorithm(self):
        raise NotImplementedError()

    # TODO. Utilize this.
    @property
    def TermsPerContext(self):
        raise NotImplementedError()

    @property
    def OpinionFormatter(self):
        raise NotImplementedError()

    def get_data_root(self):
        raise NotImplementedError()

    def get_experiments_dir(self):
        raise NotImplementedError()

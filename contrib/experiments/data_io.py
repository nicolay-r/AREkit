# TODO. Add base implementations.
class DataIO(object):

    # region properties

    @property
    def Stemmer(self):
        raise NotImplementedError()

    @property
    def SynonymsCollection(self):
        raise NotImplementedError()

    @property
    def FramesCollection(self):
        raise NotImplementedError()

    @property
    def FrameVariantCollection(self):
        raise NotImplementedError()

    @property
    def NeutralAnnotator(self):
        raise NotImplementedError()

    @property
    def KeepTokens(self):
        return True

    @property
    def Evaluator(self):
        raise NotImplementedError()

    @property
    def CVFoldingAlgorithm(self):
        raise NotImplementedError()

    @property
    def OpinionFormatter(self):
        raise NotImplementedError()

    # TODO. In future Proposal to move from nn configs here.
    @property
    def TermsPerContext(self):
        raise NotImplementedError

    # endregion

    def get_data_root(self):
        raise NotImplementedError()

    def get_experiments_dir(self):
        raise NotImplementedError()

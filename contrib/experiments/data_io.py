from arekit.contrib.experiments.utils import \
    get_path_of_subfolder_in_experiments_dir, \
    rm_dir_contents


class DataIO(object):

    def __init__(self):
        self.__model_name = None

    # region Properties

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
    def ModelIO(self):
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

    @property
    def Callback(self):
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

    def set_model_name(self, value):
        self.__model_name = value

    def get_model_root(self):
        return get_path_of_subfolder_in_experiments_dir(
            subfolder_name=self.__model_name,
            experiments_dir=self.get_experiments_dir())

    def prepare_model_root(self, rm_contents=True):

        if not rm_contents:
            return

        rm_dir_contents(self.get_model_root())


class NeuralNetworkIO(object):

    # region public methods

    @property
    def ModelSavePathPrefix(self):
        raise NotImplementedError()

    @property
    def ModelSavePath(self):
        """ Is a path of the saved model during training process
        """
        raise NotImplementedError()

    # endregion

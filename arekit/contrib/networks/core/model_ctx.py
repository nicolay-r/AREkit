from arekit.common.experiment.data_type import DataType
from arekit.contrib.networks.core.feeding.batch.base import MiniBatch
from arekit.contrib.networks.tf_helpers.session import initialize_session


class TensorflowModelContext(object):

    FeedDictShow = False

    def __init__(self, network, config, nn_io, bags_collection_type, inference_ctx):

        self.__sess = None
        self.__optimiser = None
        self.__inference_ctx = inference_ctx
        self.__network = network
        self.__io = nn_io

        self.__config = config
        self.__bags_collection_type = bags_collection_type

    # region property

    @property
    def IO(self):
        return self.__io

    @property
    def Config(self):
        return self.__config

    @property
    def Network(self):
        return self.__network

    @property
    def Session(self):
        return self.__sess

    @property
    def Optimiser(self):
        return self.__optimiser

    @property
    def BagsCollectionType(self):
        return self.__bags_collection_type

    # endregion

    def __set_optimiser_value(self, value):
        self.__optimiser = value

    def get_bags_collection(self, data_type):
        return self.__inference_ctx.BagsCollections[data_type]

    def get_sample_id_label_pairs(self, data_type):
        return self.__inference_ctx.SampleIdAndLabelPairs[data_type]

    def set_optimiser(self):
        optimiser = self.Config.Optimiser.minimize(self.__network.Cost)
        self.__set_optimiser_value(optimiser)

    def initialize_session(self):

        if self.__sess is not None:
            return

        self.__sess = initialize_session()

    def dispose_session(self):
        """ Tensorflow session dispose method
        """
        self.__sess.close()

    def create_feed_dict(self, minibatch, data_type):
        """ Compose feeding dictionary from minibatch.
        """
        assert(isinstance(minibatch, MiniBatch))
        assert(isinstance(data_type, DataType))

        network_input = minibatch.to_network_input(provide_labels=data_type != DataType.Test)

        if self.FeedDictShow:
            MiniBatch.debug_output(network_input)

        return self.__network.create_feed_dict(input=network_input, data_type=data_type)

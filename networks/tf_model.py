import os
import numpy as np
import tensorflow as tf
from tensorflow.python.training.saver import Saver

from core.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from core.networks.callback import Callback
from core.networks.cancellation import OperationCancellation
from core.networks.context.debug import DebugKeys
from core.networks.context.training.batch import MiniBatch
from core.networks.context.training.embedding.offsets import TermsEmbeddingOffsets
from core.networks.network_io import NetworkIO
from core.networks.network import NeuralNetwork
from core.networks.context.training.data_type import DataType
from core.networks.predict_log import NetworkInputDependentVariables


class TensorflowModel(object):
    """
    Base model class, which provides api for
        - tensorflow model compilation
        - fitting
        - training
        - load/save states during fitting/training
        and more.
    """

    def __init__(self, io, network, callback=None):
        assert(isinstance(io, NetworkIO))
        assert(isinstance(network, NeuralNetwork))
        assert(isinstance(callback, Callback) or callback is None)
        self.__sess = None
        self.__saver = None
        self.__optimiser = None
        self.__io = io
        self.__network = network
        self.__callback = callback
        self.__current_epoch_index = 0

    # region Properties

    @property
    def CurrentEpochIndex(self):
        return self.__current_epoch_index

    @property
    def Config(self):
        raise NotImplementedError()

    @property
    def Session(self):
        return self.__sess

    @property
    def Callback(self):
        return self.__callback

    @property
    def Network(self):
        return self.__network

    @property
    def Optimiser(self):
        return self.__optimiser

    @property
    def IO(self):
        return self.__io

    # endregion

    # region public methods

    @staticmethod
    def display_log(log_names, log_values):
        pass

    def set_optimiser_value(self, value):
        self.__optimiser = value

    def load_model(self, save_path):
        assert(isinstance(self.__saver, Saver))
        save_dir = os.path.dirname(save_path)
        self.__saver.restore(sess=self.__sess,
                             save_path=tf.train.latest_checkpoint(save_dir))

    def save_model(self, save_path):
        assert(isinstance(self.__saver, Saver))
        self.__saver.save(self.__sess,
                          save_path=save_path,
                          write_meta_graph=False)

    def dispose_session(self):
        """
        Tensorflow session dispose method
        """
        self.__sess.close()

    def run(self, load_model=False):
        self.__network.compile(self.Config, reset_graph=True)
        self.set_optimiser()
        self.__notify_initialized()

        self.__initialize_session()

        if load_model:
            save_path = self.__io.create_model_state_filepath()
            print "Loading model: {}".format(save_path)
            self.load_model(save_path)

        self.fit()
        self.dispose_session()

    def before_labeling_func_application(self, text_opinions):
        assert(text_opinions.check_all_text_opinions_without_labels())

    def after_labeling_func_application(self, text_opinions):
        assert(text_opinions.check_all_text_opinions_has_labels())

    def before_evaluation(self, dest_data_type):
        pass

    def predict_core(self,
                     dest_data_type,
                     labeling_callback):
        assert(isinstance(dest_data_type, unicode))
        assert(callable(labeling_callback))

        text_opinions = self.get_text_opinions_collection(dest_data_type)
        assert(isinstance(text_opinions, LabeledLinkedTextOpinionCollection))

        text_opinions.reset_labels()

        # Predict.
        self.before_labeling_func_application(text_opinions)
        predict_log = labeling_callback(text_opinions, dest_data_type)
        self.after_labeling_func_application(text_opinions)

        self.before_evaluation(dest_data_type)

        eval_result = self.get_eval_helper().evaluate_model(
            data_type=dest_data_type,
            io=self.IO,
            doc_ids=text_opinions.iter_unique_news_ids(),
            epoch_index=self.__current_epoch_index)

        text_opinions.reset_labels()

        return eval_result, predict_log

    def iter_inner_input_vocabulary(self):
        word_iter = TermsEmbeddingOffsets.iter_words_vocabulary(
            words_embedding=self.Config.WordEmbedding,
            missed_words_embedding=self.Config.MissedWordEmbedding,
            tokens_embedding=self.Config.TokenEmbedding,
            frames_embedding=self.Config.FrameEmbedding)

        for word in word_iter:
            yield word

    # endregion

    # region Abstract

    def fit(self):
        assert(self.Session is not None)

        operation_cancel = OperationCancellation()
        minibatches = list(self.get_bags_collection(DataType.Train).iter_by_groups(self.Config.BagsPerMinibatch))
        print "Minibatches passing per epoch count: {}".format(len(minibatches))

        for epoch_index in xrange(self.Config.Epochs):

            if operation_cancel.IsCancelled:
                break

            e_cost, e_acc = self.__fit_epoch(minibatches=minibatches)

            if self.Callback is not None:
                self.Callback.on_epoch_finished(avg_cost=e_cost,
                                                avg_acc=e_acc,
                                                epoch_index=epoch_index,
                                                operation_cancel=operation_cancel)

            self.__current_epoch_index += 1

        if self.Callback is not None:
            self.Callback.on_fit_finished()

    def predict(self, dest_data_type=DataType.Test):
        eval_result, predict_log = self.predict_core(dest_data_type=dest_data_type,
                                                     labeling_callback=self.__text_opinions_labeling)
        return eval_result, predict_log

    def get_hidden_parameters(self):
        names = []
        tensors = []
        for name, tensor in self.Network.iter_hidden_parameters():
            names.append(name)
            tensors.append(tensor)

        result_list = self.Session.run(tensors)
        return names, result_list

    def set_optimiser(self):
        optimiser = self.Config.Optimiser.minimize(self.Network.Cost)
        self.set_optimiser_value(optimiser)

    def get_eval_helper(self):
        raise NotImplementedError()

    def get_labels_helper(self):
        raise NotImplementedError()

    def get_gpu_memory_fraction(self):
        raise NotImplementedError()

    def create_batch_by_bags_group(self, bags_group):
        raise NotImplementedError()

    def get_text_opinions_collection(self, data_type):
        raise NotImplementedError()

    def get_bags_collection(self, data_type):
        raise NotImplementedError()

    def create_feed_dict(self, minibatch, data_type):
        assert(isinstance(self.Network, NeuralNetwork))
        assert(isinstance(minibatch, MiniBatch))
        assert(isinstance(data_type, unicode))

        network_input = minibatch.to_network_input()
        if DebugKeys.FeedDictShow:
            MiniBatch.debug_output(network_input)

        return self.Network.create_feed_dict(network_input, data_type)

    # endregion

    # region Private

    def __fit_epoch(self, minibatches):
        assert(isinstance(minibatches, list))

        # self.get_bags_collection_helper(DataType.Train).print_log_statistics()

        total_cost = 0
        total_acc = 0
        groups_count = 0

        np.random.shuffle(minibatches)

        for bags_group in minibatches:

            minibatch = self.create_batch_by_bags_group(bags_group)
            feed_dict = self.create_feed_dict(minibatch, data_type=DataType.Train)

            hidden_list = list(self.Network.iter_hidden_parameters())
            hidden_names = [name for name, _ in hidden_list]
            fetches_default = [self.Optimiser, self.Network.Cost, self.Network.Accuracy]
            fetches_hidden = [tensor for _, tensor in hidden_list]

            result = self.Session.run(fetches_default + fetches_hidden,
                                      feed_dict=feed_dict)
            cost = result[1]

            if DebugKeys.FitBatchDisplayLog:
                self.display_log(hidden_names, result[len(fetches_default):])

            total_cost += np.mean(cost)
            total_acc += result[2]
            groups_count += 1

        if DebugKeys.FitSaveTensorflowModelState:
            self.save_model(save_path=self.IO.get_model_filepath())

        return total_cost / groups_count, total_acc / groups_count

    def __notify_initialized(self):
        if self.__callback is not None:
            self.__callback.on_initialized(self)

    def __initialize_session(self):
        """
        Tensorflow session initialization
        """
        init_op = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.get_gpu_memory_fraction())
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(init_op)
        self.__saver = tf.train.Saver(max_to_keep=2)
        self.__sess = sess

    def __text_opinions_labeling(self, text_opinions, dest_data_type):
        assert(isinstance(text_opinions, LabeledLinkedTextOpinionCollection))
        assert(isinstance(dest_data_type, unicode))

        predict_log = NetworkInputDependentVariables()
        var_names = []
        var_tensors = []
        for name, tensor in self.Network.iter_input_dependent_hidden_parameters():
            var_names.append(name)
            var_tensors.append(tensor)

        for bags_group in self.get_bags_collection(dest_data_type).iter_by_groups(self.Config.BagsPerMinibatch):

            minibatch = self.create_batch_by_bags_group(bags_group)
            feed_dict = self.create_feed_dict(minibatch, data_type=dest_data_type)

            result = self.Session.run([self.Network.Labels] + var_tensors, feed_dict=feed_dict)
            uint_labels = result[0]

            tensor_values = result[1:]
            if len(var_names) > 0 and len(tensor_values) > 0:
                predict_log.add(names=var_names,
                                tensor_values=tensor_values,
                                text_opinion_ids=[sample.TextOpinionID for sample in
                                                  minibatch.iter_by_samples()])

            if DebugKeys.PredictBatchDisplayLog:
                self.display_log(var_names, result[1:])

            # apply labeling
            for bag_index, bag in enumerate(minibatch.iter_by_bags()):

                label = self.get_labels_helper().create_label_from_uint(
                    label_uint=int(uint_labels[bag_index]))

                for sample in bag:
                    if sample.TextOpinionID < 0:
                        continue
                    text_opinions.apply_label(label, sample.TextOpinionID)

        return predict_log

    # endregion

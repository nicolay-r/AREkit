from arekit.common.experiment.io_utils import BaseIOUtils


class BertIOUtils(BaseIOUtils):

    @classmethod
    def get_target_dir(cls, experiment):
        """ Provides a main directory for input

            NOTE:
            We consider to save serialized results into model dir,
            rather than experiment dir in a base implementation,
            as model affects on text_b, entities representation, etc.
        """
        return experiment.DataIO.get_model_root(experiment.Name)

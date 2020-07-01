from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.formatters.opinions.base import BertOpinionsFormatter
from arekit.common.experiment.input.providers.opinions import OpinionProvider


class BertEncoder(object):

    @staticmethod
    def to_tsv(experiment, create_formatter_func):
        """
        Args:
            create_formatter_func: func(data_type) -> FormatterType
        """
        assert(isinstance(experiment, BaseExperiment))
        assert(callable(create_formatter_func))

        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():
            experiment.NeutralAnnotator.create_collection(data_type)

        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():
            opinion_provider = OpinionProvider.from_experiment(experiment=experiment, data_type=data_type)

            opnion_formatter = BertOpinionsFormatter(data_type=data_type)
            opnion_formatter.format(opinion_provider=opinion_provider)
            opnion_formatter.to_tsv_by_experiment(experiment=experiment)

            sampler = create_formatter_func(data_type=data_type)
            sampler.format(opinion_provider=opinion_provider)
            sampler.to_tsv_by_experiment(experiment=experiment)


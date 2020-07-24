from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.formatters.opinion import BaseOpinionsFormatter
from arekit.common.experiment.input.providers.opinions import OpinionProvider


class BaseInputEncoder(object):

    @staticmethod
    def to_tsv(out_dir, experiment, create_formatter_func, balance):
        """
        Args:
            out_dir: unicode
            experiment: BaseExperiment
            create_formatter_func: func(data_type) -> FormatterType
            balance: bool
        """
        assert(isinstance(out_dir, unicode))
        assert(isinstance(experiment, BaseExperiment))
        assert(callable(create_formatter_func))
        assert(isinstance(balance, bool))

        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():
            experiment.NeutralAnnotator.create_collection(data_type)

        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():
            opinion_provider = OpinionProvider.from_experiment(experiment=experiment, data_type=data_type)

            template = BaseInputEncoder.filename_template(data_type=data_type,
                                                          experiment=experiment)

            opnion_formatter = BaseOpinionsFormatter(data_type=data_type)
            opnion_formatter.format(opinion_provider=opinion_provider)
            opnion_formatter.save(out_dir=out_dir,
                                  filename_template=template)

            sampler = create_formatter_func(data_type=data_type)
            sampler.format(opinion_provider=opinion_provider)
            sampler.save(out_dir=out_dir,
                         filename_template=template,
                         balance=balance)

    @staticmethod
    def filename_template(data_type, experiment):
        assert(isinstance(data_type, DataType))
        return u"{data_type}-{cv_index}".format(
            data_type=data_type.name.lower(),
            cv_index=experiment.DataIO.CVFoldingAlgorithm.IterationIndex)


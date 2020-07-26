from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.formatters.base_row import BaseRowsFormatter
from arekit.common.experiment.input.formatters.opinion import BaseOpinionsFormatter
from arekit.common.experiment.input.formatters.sample import BaseSampleFormatter
from arekit.common.experiment.input.providers.opinions import OpinionProvider


class BaseInputEncoder(object):

    @staticmethod
    def to_tsv(out_dir, experiment, terms_per_context, create_formatter_func, balance, write_header_func):
        """
        Args:
            out_dir: unicode
            experiment: BaseExperiment
            terms_per_context: int
            create_formatter_func: func(data_type) -> FormatterType
            balance: bool
            write_header_func: func(data_type) -> bool
        """
        assert(isinstance(out_dir, unicode))
        assert(isinstance(experiment, BaseExperiment))
        assert(callable(create_formatter_func))
        assert(isinstance(balance, bool))
        assert(callable(write_header_func))

        # Create annotated collection per each type.
        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():
            experiment.NeutralAnnotator.create_collection(data_type)

        # Perform input serialization process.
        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():

            # crate opinion provider
            opinion_provider = OpinionProvider.from_experiment(experiment=experiment,
                                                               data_type=data_type,
                                                               terms_per_context=terms_per_context)

            # filepaths
            opinion_filepath, sample_filepath = BaseInputEncoder.get_filepaths(out_dir=out_dir,
                                                                               experiment=experiment,
                                                                               data_type=data_type)

            # Opinions
            opnion_formatter = BaseOpinionsFormatter(data_type)
            opnion_formatter.format(opinion_provider)
            opnion_formatter.save(opinion_filepath)

            # Samples
            sampler = create_formatter_func(data_type)
            sampler.format(opinion_provider)
            sampler.save(filepath=sample_filepath,
                         balance=balance,
                         write_header=write_header_func(data_type))

    @staticmethod
    def get_filepaths(out_dir, experiment, data_type):
        template = BaseInputEncoder.__filename_template(data_type=data_type,
                                                        experiment=experiment)

        opinions_filepath = BaseRowsFormatter.get_filepath_static(
            out_dir=out_dir,
            template=template,
            prefix=BaseOpinionsFormatter.formatter_type_log_name())

        samples_filepath = BaseRowsFormatter.get_filepath_static(
            out_dir=out_dir,
            template=template,
            prefix=BaseSampleFormatter.formatter_type_log_name())

        return opinions_filepath, samples_filepath

    @staticmethod
    def __filename_template(data_type, experiment):
        assert(isinstance(data_type, DataType))
        return u"{data_type}-{cv_index}".format(
            data_type=data_type.name.lower(),
            cv_index=experiment.DataIO.CVFoldingAlgorithm.IterationIndex)


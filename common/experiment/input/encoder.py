from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.formatters.opinion import BaseOpinionsFormatter
from arekit.common.experiment.input.providers.opinions import OpinionProvider


class BaseInputEncoder(object):

    @staticmethod
    def to_tsv(get_opinion_filepath,
               get_sample_filepath,
               experiment,
               terms_per_context,
               create_formatter_func,
               write_header_func):
        """
        Args:
            experiment: BaseExperiment
            terms_per_context: int
            create_formatter_func: func(data_type) -> FormatterType
            write_header_func: func(data_type) -> bool
        """
        assert(callable(get_opinion_filepath))
        assert(callable(get_sample_filepath))
        assert(isinstance(experiment, BaseExperiment))
        assert(callable(create_formatter_func))
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

            # Opinions
            opinion_filepath = get_opinion_filepath(data_type, experiment)
            opnion_formatter = BaseOpinionsFormatter(data_type)
            opnion_formatter.format(opinion_provider)
            opnion_formatter.save(opinion_filepath)

            # Samples
            sample_filepath = get_sample_filepath(data_type, experiment)
            sample_formatter = create_formatter_func(data_type)
            sample_formatter.format(opinion_provider)
            sample_formatter.save(filepath=sample_filepath,
                                  write_header=write_header_func(data_type))


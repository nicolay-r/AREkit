from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.input.providers.rows.opinions import BaseOpinionsRowProvider
from arekit.common.experiment.input.providers.rows.samples import BaseSampleRowProvider


class BaseInputProvider(object):

    @staticmethod
    def save(opinion_provider, opinion_row_provider, sample_row_provider):
        assert(isinstance(opinion_provider, OpinionProvider))
        assert(isinstance(opinion_row_provider, BaseOpinionsRowProvider))
        assert(isinstance(sample_row_provider, BaseSampleRowProvider))

        # Opinions
        with opinion_provider as orp:
            orp.format(opinion_provider, desc="opinion")
            orp.save()

        # Samples
        with sample_row_provider as srp:
            srp.format(opinion_provider, desc="sample")
            srp.save()

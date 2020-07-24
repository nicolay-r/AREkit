from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.readers.opinion import InputOpinionReader
from arekit.common.experiment.input.readers.sample import InputSampleReader
from arekit.common.experiment.output.base import BaseOutput
from arekit.common.experiment.output.converter import OutputToOpinionCollectionsConverter


def eval_output(
        source_dir,
        filename_template,
        opinions_reader,
        samples_reader,
        data_type,
        experiment,
        label_calculation_mode,
        output_instance):
    assert(isinstance(filename_template, unicode))
    assert(isinstance(opinions_reader, InputOpinionReader))
    assert(isinstance(samples_reader, InputSampleReader))
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(label_calculation_mode, unicode))
    assert(isinstance(output_instance, BaseOutput))

    experiment.OpinionOperations.create_opinion_collection()

    opinion_collections_iter = OutputToOpinionCollectionsConverter.iter_opinion_collections(
        source_dir=source_dir,
        filename_template=filename_template,
        opinions_reader=opinions_reader,
        samples_reader=samples_reader,
        experiment=experiment,
        label_calculation_mode=label_calculation_mode,
        output=output_instance)

    for news_id, collection in opinion_collections_iter:

        filepath = experiment.OpinionOperations.create_result_opinion_collection_filepath(
            data_type=data_type,
            doc_id=news_id,
            epoch_index=experiment.EPOCH_INDEX_PLACEHOLER)

        experiment.DataIO.OpinionFormatter.save_to_file(collection=collection,
                                                        filepath=filepath)


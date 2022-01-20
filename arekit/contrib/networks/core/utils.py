from arekit.contrib.networks.core.pipeline.item_base import EpochHandlingPipelineItem


def get_item_from_pipeline(pipeline, item_type):
    assert (isinstance(pipeline, list))
    assert (issubclass(item_type, EpochHandlingPipelineItem))

    for item in pipeline:
        if isinstance(item, item_type):
            return item

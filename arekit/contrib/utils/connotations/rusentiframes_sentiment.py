from arekit.common.frames.connotations.provider import FrameConnotationProvider
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection


class RuSentiFramesConnotationProvider(FrameConnotationProvider):
    """ This is a provider based on A0->A1 label type of RuSentiFrames collection.
        For a greater details, checkout the related collection at:
        https://github.com/nicolay-r/RuSentiFrames

        Papers:
            [1] Natalia Loukachevitch, Nicolay Rusnachenko: Sentiment Frames
                for Attitude Extraction in Russian, 2020
            [2] Distant Supervision for Sentiment Attitude Extraction, 2019
    """

    def __init__(self, collection):
        assert(isinstance(collection, RuSentiFramesCollection))
        self.__collection = collection

    def try_provide(self, frame_id):
        return self.__collection.try_get_frame_polarity(frame_id=frame_id,
                                                        role_src='a0',
                                                        role_dest='a1')
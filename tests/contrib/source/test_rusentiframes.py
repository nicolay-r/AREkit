import logging
import sys
import unittest

sys.path.append('../../../../')

from arekit.common.frames.variants.base import FrameVariant
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, \
    RuSentiFramesEffectLabelsFormatter
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions

from tests.contrib.source.labels import PositiveLabel, NegativeLabel


class TestRuSentiFrames(unittest.TestCase):

    def test_reading(self):

        # Initializing logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.DEBUG)

        frames = RuSentiFramesCollection.read(
            version=RuSentiFramesVersions.V20,
            labels_fmt=RuSentiFramesLabelsFormatter(
                neg_label_type=NegativeLabel, pos_label_type=PositiveLabel),
            effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(
                neg_label_type=NegativeLabel, pos_label_type=PositiveLabel))

        for frame_id in frames.iter_frames_ids():
            # id
            logger.info("Frame: {}".format(frame_id))
            # titles
            logger.info("Titles: {}".format(",".join(frames.get_frame_titles(frame_id))))
            # variants
            logger.info("Variants: {}".format(",".join(frames.get_frame_variants(frame_id))))
            # roles
            for role in frames.get_frame_roles(frame_id):
                logger.info("Role: {}".format(" -- ".join([role.Source, role.Description])))
            # states
            for state in frames.get_frame_states(frame_id):
                logger.info("State: {}".format(",".join([state.Role, state.Label.to_class_str(), str(state.Prob)])))
            # polarity
            for polarity in frames.get_frame_polarities(frame_id):
                logger.info("Polarity: {}".format(",".join([polarity.Source,
                                                            polarity.Destination,
                                                            polarity.Label.to_class_str()])))

            has_a0_a1_pol = frames.try_get_frame_polarity(frame_id, role_src="a0", role_dest="a1")
            logger.info("Has a0->a1 polarity: {}".format(has_a0_a1_pol is not None))

        # frame variants.
        frame_variants = FrameVariantsCollection()
        frame_variants.fill_from_iterable(variants_with_id=frames.iter_frame_id_and_variants(),
                                          overwrite_existed_variant=True,
                                          raise_error_on_existed_variant=False)

        frame_variant = frame_variants.get_variant_by_value("хвалить")

        assert(isinstance(frame_variant, FrameVariant))

        logger.info("FrameVariantValue: {}".format(frame_variant.get_value()))
        logger.info("FrameID: {}".format(frame_variant.FrameID))


if __name__ == '__main__':
    unittest.main()

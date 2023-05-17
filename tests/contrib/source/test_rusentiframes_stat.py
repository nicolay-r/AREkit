import argparse

from arekit.common.frames.variants.base import FrameVariant
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.labels.base import Label
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, \
    RuSentiFramesEffectLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions, RuSentiFramesVersionsService
from arekit.contrib.source.rusentiframes.polarity import RuSentiFramesFramePolarity
from arekit.contrib.source.rusentiframes.effect import FrameEffect
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.processing.pos.mystem_wrap import POSMystemWrapper
from tests.contrib.source.labels import PositiveLabel, NegativeLabel


def __iter_unique_frame_variants(frames_collection, frame_ids):
    v_set = set()
    for frame_id in frame_ids:
        for variant in frames_collection.get_frame_variants(frame_id):
            if variant in v_set:
                continue
            v_set.add(variant)
            yield variant


def __get_frame_effects(frames_collection, role, label):
    assert(isinstance(frames_collection, RuSentiFramesCollection))
    assert(isinstance(role, str))
    assert(isinstance(label, Label))

    frame_ids = []
    for frame_id in frames_collection.iter_frames_ids():
        for effect in frames_collection.get_frame_effects(frame_id):
            assert(isinstance(effect, FrameEffect))

            if effect is None:
                continue
            if effect.Role != role:
                continue
            if effect.Label != label:
                continue

            frame_ids.append(frame_id)
            break

    return list(__iter_unique_frame_variants(frames_collection, frame_ids))


def __get_variants_with_polarities(frames_collection, role_src, role_dest, label):
    assert(isinstance(frames_collection, RuSentiFramesCollection))
    assert(isinstance(role_dest, str))
    assert(isinstance(role_src, str))
    assert(isinstance(label, Label))

    frame_ids = []
    for frame_id in frames_collection.iter_frames_ids():
        polarity = frames_collection.try_get_frame_polarity(
            frame_id=frame_id,
            role_dest=role_dest,
            role_src=role_src)

        if polarity is None:
            continue

        assert(isinstance(polarity, RuSentiFramesFramePolarity))
        if polarity.Source != role_src:
            continue
        if polarity.Destination != role_dest:
            continue
        if polarity.Label != label:
            continue

        frame_ids.append(frame_id)

    return list(__iter_unique_frame_variants(frames_collection, frame_ids))


def __about(frames_collection, pos_tagger):
    all_frame_entries = list(frames_collection.iter_frame_id_and_variants())

    unique_frame_variants = FrameVariantsCollection()
    unique_frame_variants.fill_from_iterable(variants_with_id=all_frame_entries,
                                             overwrite_existed_variant=True,
                                             raise_error_on_existed_variant=False)

    assert(isinstance(frames_collection, RuSentiFramesCollection))
    unique_variants = list(unique_frame_variants.iter_variants())

    phrases = []
    nouns = []
    verbs = []
    other = []
    for frame_id, variant in unique_variants:
        assert(isinstance(variant, FrameVariant))

        terms = list(variant.iter_terms())
        if len(terms) > 1:
            phrases.append(variant.get_value())
            continue
        pos_type = pos_tagger.get_term_pos(terms[0])
        if pos_tagger.is_noun(pos_type):
            nouns.append(terms[0])
            continue
        if pos_tagger.is_verb(pos_type):
            verbs.append(terms[0])
            continue
        other.append(terms[0])

    titles = []
    for frame_id in frames_collection.iter_frames_ids():
        titles.extend(frames_collection.get_frame_titles(frame_id))

    print("Frames count:", len(list(frames_collection.iter_frames_ids())))
    print("---------------")
    print()

    print("Quantitative characteristics of the RuSentiFrames entries:")
    print("Verbs:", len(verbs))
    print("Nouns:", len(nouns))
    print("Phrases:", len(phrases))
    print("Other:", len(other))
    print("Unique entries:", len(unique_variants))
    print("Total entries: ", len(all_frame_entries))
    print()

    print("The distribution of RuSentiFrames text entries according to attitudes:")
    print("A0 to A1 Pos", len(__get_variants_with_polarities(frames_collection=frames_collection,
                                                             role_src='a0',
                                                             role_dest='a1',
                                                             label=PositiveLabel())))
    print("A0 to A1 Neg", len(__get_variants_with_polarities(frames_collection=frames_collection,
                                                             role_src='a0',
                                                             role_dest='a1',
                                                             label=NegativeLabel())))
    print("Author to A0 Pos", len(__get_variants_with_polarities(frames_collection=frames_collection,
                                                                 role_src='author',
                                                                 role_dest='a0',
                                                                 label=PositiveLabel())))
    print("Author to A0 Neg", len(__get_variants_with_polarities(frames_collection=frames_collection,
                                                                 role_src='author',
                                                                 role_dest='a0',
                                                                 label=NegativeLabel())))
    print("Author to A1 Pos", len(__get_variants_with_polarities(frames_collection=frames_collection,
                                                                 role_src='author',
                                                                 role_dest='a1',
                                                                 label=PositiveLabel())))
    print("Author to A1 Neg", len(__get_variants_with_polarities(frames_collection=frames_collection,
                                                                 role_src='author',
                                                                 role_dest='a1',
                                                                 label=NegativeLabel())))
    print()

    print("The distribution of RuSentiFrames text entries according to effects on main participants:")
    print("A0 Pos", len(__get_frame_effects(frames_collection=frames_collection, role='a0', label=PositiveLabel())))
    print("A0 Neg", len(__get_frame_effects(frames_collection=frames_collection, role='a0', label=NegativeLabel())))
    print("A1 Pos", len(__get_frame_effects(frames_collection=frames_collection, role='a1', label=PositiveLabel())))
    print("A1 Neg", len(__get_frame_effects(frames_collection=frames_collection, role='a1', label=NegativeLabel())))


def about_version(version=RuSentiFramesVersions.V20):
    stemmer = MystemWrapper()
    pos_tagger = POSMystemWrapper(stemmer.MystemInstance)
    frames_collection = RuSentiFramesCollection.read(
        version=version,
        labels_fmt=RuSentiFramesLabelsFormatter(neg_label_type=NegativeLabel,
                                                pos_label_type=PositiveLabel),
        effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(neg_label_type=NegativeLabel,
                                                             pos_label_type=PositiveLabel))

    print("Lexicon version:", version)
    return __about(frames_collection=frames_collection, pos_tagger=pos_tagger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Collection stat writer")

    default_name = RuSentiFramesVersionsService.get_name_by_type(RuSentiFramesVersions.V20)

    parser.add_argument('--version',
                        dest='version',
                        type=str,
                        default=default_name,
                        choices=list(RuSentiFramesVersionsService.iter_supported_names()),
                        nargs='?',
                        help='Version of RuSentiFrames collection (Default: {})'.format(default_name))

    # Parsing arguments.
    args = parser.parse_args()

    version = RuSentiFramesVersionsService.get_type_by_name(args.version)

    # Writing statistics.
    about_version(version)


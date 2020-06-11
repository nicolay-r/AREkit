## v0.20.5-rc

Collection of parsed news become dispatched from text opinions collection.
* Removed dependency from `RelatedParsedNewCollection` in TextOpinionCollection.
* Labeling now separated from LinkedTextOpinion collection.
* `ParsedText` class has been refactored, removed unused methods. Keep tokens has been discarded.
* BERT tsv-format-encoders are now in a Factory (at contrib directory)
* Fixed: `RuSentRelTextOpinion` depends on `TextOpinion` (not straightly from `OpinionRef`).

Minor changes (light refactoring):

* Bert now moved into separated folder from `contrib` directory.
* frame_variants moved to `frames` directory.
* Frame variants labeling in news now performed during `parse` operation.
* `DataType` now enumeration. List of Supported data-types now a part of experiment
* `RuSentRel` iter_wrapped_linked_text_opinions now does not provide special checks on entity positions. 
The latter were moved onto sample level.

## v0.20.4-rc
Updates:
* Labels conversion `to_str` and `from_str` now a part of external formatters (unique for each source, experiment, etc.).
* Added labels-scaler, and labels casing (to int or uint) now depends on scaler;
* Added bert exporter in contribution folder: with related formatters according to the following 
[paper](https://www.aclweb.org/anthology/N19-1035.pdf): 
    * **NLI** -- (Natural language inference) format, assumes to provide an additional sentence, which describes 
    attitude should be extracted
    * **QA** -- (Question answering) provides an additional question onto attitude sentiment.
    
   With Label encoding in following format:
   * **Multiple** -- all the supported sentiment labels (positive, negative, neutral)
   * **Binary** -- (YES, NO) according to mention (additional sentence), provided by **NLI** and **QA** formatters.

* Refactoring experiments in order to apply the latter also for classifiers (models from scikit-learn)
* Updated nn-engine API
* Refactoring tf-based neural network implementation.

## v0.20.3-rc (WIMS-2020 conference edition)

Updates:

* Experiments now supports two-scale and three-scale task representation with the related evaluation formats.


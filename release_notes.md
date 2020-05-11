## v0.20.4-rc
Updates:
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


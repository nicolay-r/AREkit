# AREkit 0.21.0

<p align="center">
    <img src="logo.png"/>
</p>

**AREkit** (Attitude and Relation Extraction Toolkit) -- is a python toolkit, devoted to 
document level Attitude and Relation Extraction between text objects from mass-media news 
and analytical articles with entity-linking (EL) API support for objects.

## Description

Is an open-source and extensible toolkit focused on data preparation for document-level relation extraction organization. 
In complements the OpenNRE functionality since *document-level RE setting is not widely explored* (2.4 [[paper]](https://aclanthology.org/D19-3029.pdf)).
The core functionality includes 
(1) API for document presentation with EL (Entity Linking, i.e. Object Synonymy) support 
for sentence level relations preparation (dubbed as contexts)
(2) API for contexts extraction
(3) relations transferring from sentence-level onto document-level, and more.
It providers contrib modules of 
[neural networks](https://github.com/nicolay-r/AREkit/tree/0.21.0-rc/arekit/contrib/networks) (like OpenNRE) and 
[BERT](https://github.com/nicolay-r/AREkit/tree/0.21.0-rc/arekit/contrib/bert),
both applicable for sentiment attitude extraction task.

## Installation 

```
pip install git+https://github.com/nicolay-r/AREkit.git@0.21.0-rc
```

## Download Resources
```python
from arekit.data import download_data
download_data()
```

## Deep-Learning Applications

* Frame-Based attitude extraction workflow for news processing [[code]](https://github.com/nicolay-r/frame-based-attitude-extraction-workflow)
    * Represents an attitude annotation workflow based on [RuSentiFrames](https://github.com/nicolay-r/RuSentiFrames) lexicon which is utilized for news processing;
* **AREnets** for analytical articles [[code]](https://github.com/nicolay-r/neural-networks-for-attitude-extraction/tree/0.21.0)
    * Neural Networks application for attitude extraction from analytical articles;
* **AREbert** for analytical articles processing [[code]](https://github.com/nicolay-r/bert-utils-for-attitude-extraction/tree/0.21.0)
    * Analytical news formatter for BERT-based models;

## Related Frameworks

*  **OpenNRE** [[github]](https://github.com/thunlp/OpenNRE) [[paper]](https://aclanthology.org/D19-3029.pdf)
*  **DeRE** [[github]](https://github.com/ims-tcl/DeRE) [[paper]](https://aclanthology.org/D18-2008/)

# AREkit 0.21.0

<p align="center">
    <img src="logo.png"/>
</p>

**AREkit** (Attitude and Relation Extraction Toolkit) -- is a python toolkit, devoted to 
document level Attitude and Relation Extraction for text objects with entity-linking (EL) API support.

## Description

Is an open-source and extensible toolkit focused on data preparation for document-level relation extraction organization. 
In complements the OpenNRE functionality since *document-level RE setting is not widely explored* (2.4 [[paper]](https://aclanthology.org/D19-3029.pdf)).
The core functionality includes 
(1) API for document presentation with EL (Entity Linking, i.e. Object Synonymy) support 
for sentence level relations preparation (dubbed as contexts)
(2) API for contexts extraction
(3) relations transferring from sentence-level onto document-level, and more.
It providers contrib modules of 
[neural networks](https://github.com/nicolay-r/AREkit/tree/0.21.0-rc/contrib/networks) (like OpenNRE) and 
[BERT](https://github.com/nicolay-r/AREkit/tree/0.21.0-rc/contrib/bert),
both applicable for sentiment attitude extraction task.

## Dependencies

* python == 2.7 (No doubts it will be updated to 3.4+)

* pymystem3 == 0.1.9

* pandas == 0.20.3

## Installation 

Using [virtualenv](https://www.pythoncentral.io/how-to-install-virtualenv-python/). 
Create virtual environment, suppose `my_env`, and activate it as follows:
```
virtualenv my_env
source my_env/bin/activate
```

Then install dependencies as follows:
```
pip install -r dependencies.txt
```

## Deep-Learning Applications

* Frame-Based attitude extraction workflow for news processing [[code]](https://github.com/nicolay-r/frame-based-attitude-extraction-workflow)
    * Represents an attitude annotation workflow based on [RuSentiFrames](https://github.com/nicolay-r/RuSentiFrames) lexicon which is utilized for news processing;
* **AREnets** for analytical articles [[code]](https://github.com/nicolay-r/neural-networks-for-attitude-extraction)
    * Neural Networks application for attitude extraction from analytical articles;
* **AREbert** for analytical articles processing [[code]](https://github.com/nicolay-r/bert-utils-for-attitude-extraction)
    * Analytical news formatter for BERT-based models;

## Related Frameworks

*  **OpenNRE** [[github]](https://github.com/thunlp/OpenNRE) [[paper]](https://aclanthology.org/D19-3029.pdf)

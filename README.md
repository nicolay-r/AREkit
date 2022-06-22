# AREkit 0.22.1

![](https://img.shields.io/badge/Python-3.6-brightgreen.svg)

<p align="center">
    <img src="logo.png"/>
</p>

**AREkit** (Attitude and Relation Extraction Toolkit) --
is a python toolkit, devoted to document level Attitude and Relation Extraction between text objects from mass-media news. 

## Description

This toolkit aims to solve data preparation problems in Relation Extraction related taks, considiering such factors as:
* ⛓️ EL (entity-linking) API support for objects, 
* ➰ avoidance of cyclic connections,
* 🔗 distance consideration between relation participants (in terms and sentences).
* 📑 relations annotations and filtering rules,
* *️⃣ entities formatting or masking, and more.

Using AREkit you may focus on preparation and experiments with your ML-models by shift all the data-preparation part  onto toolset of this project ([tutorial](https://nicolay-r.github.io/blog/articles/2022-05/process-mass-media-relations-with-arekit)).
In order to do so, we provide:
* 📚 API for external collection binding (native support of [BRAT](https://brat.nlplab.org/)-based exported annotations) 
* ➿ pipelines and iterators for handling large-scale collections serialization without out-of-memory issues.

AREkit complements the [OpenNRE](https://github.com/thunlp/OpenNRE) functionality since *document-level RE setting is not widely explored* (2.4 [[paper]](https://aclanthology.org/D19-3029.pdf)).
The core functionality includes 
(1) API for document presentation with EL (Entity Linking, i.e. Object Synonymy) support 
for sentence level relations preparation (dubbed as contexts)
(2) API for contexts extraction
(3) relations transferring from sentence-level onto document-level, and more.
It providers contrib modules of 
**neural networks** (like OpenNRE) applicable for sentiment attitude extraction task.

## Installation 

```
pip install git+https://github.com/nicolay-r/AREkit.git@0.22.1-rc
```

## Download Resources
```python
from arekit.data import download_data
download_data()
```

## Applications

* **ARElight** [[site]](https://nicolay-r.github.io/arelight-page/) [[github]](https://github.com/nicolay-r/ARElight)
    * **Infer attitudes** from large Mass-media documents or **sample texts** for your Machine Learning models applications

#### Papers
* Frame-Based attitude extraction workflow for news processing [[code]](https://github.com/nicolay-r/frame-based-attitude-extraction-workflow)
    * Represents an attitude annotation workflow based on [RuSentiFrames](https://github.com/nicolay-r/RuSentiFrames) lexicon which is utilized for news processing;
* Neural Networks Applications in Sentiment Attitude Extraction [[code]](https://github.com/nicolay-r/neural-networks-for-attitude-extraction)
    * Neural Networks application for attitude extraction from analytical articles;
* BERT-based model utils for Sentiment Attitude Extraction task [[code]](https://github.com/nicolay-r/bert-utils-for-attitude-extraction)
    * Analytical news formatter for BERT-based models;

## Related Frameworks

*  **OpenNRE** [[github]](https://github.com/thunlp/OpenNRE) [[paper]](https://aclanthology.org/D19-3029.pdf)
*  **DeRE** [[github]](https://github.com/ims-tcl/DeRE) [[paper]](https://aclanthology.org/D18-2008/)
*  **CREST** [[github]](https://github.com/phosseini/CREST) [[paper]](https://arxiv.org/abs/2103.13606)

# AREkit 0.23.0

![](https://img.shields.io/badge/Python-3.6-brightgreen.svg)

<p align="center">
    <img src="logo.png"/>
</p>

**AREkit** (Attitude and Relation Extraction Toolkit) --
is a python toolkit, devoted to document level Attitude and Relation Extraction between text objects from mass-media news. 

## Description

This toolkit aims to solve data preparation problems in Relation Extraction related taks, considiering such factors as:
* 🔗 EL (entity-linking) API support for objects, 
* ➰ avoidance of cyclic connections,
* :straight_ruler: distance consideration between relation participants (in `terms` or `sentences`),
* 📑 relations annotations and filtering rules,
* *️⃣ entities formatting or masking, and more.

Using AREkit you may focus on preparation and experiments with your ML-models by shift all the data-preparation part  onto toolset of this project ([tutorial](https://nicolay-r.github.io/blog/articles/2022-05/process-mass-media-relations-with-arekit)).
In order to do so, we provide:
* :file_folder: API for external collection binding (native support of [BRAT](https://brat.nlplab.org/)-based exported annotations) 
[[more]](https://nicolay-r.github.io/blog/articles/2022-08/arekit-collection-bind)
* ➿ pipelines and iterators for handling large-scale collections serialization without out-of-memory issues.
[[more]](https://nicolay-r.github.io/blog/articles/2022-08/arekit-text-opinion-annotation-pipeline)
* evaluators which allows you to assess your trained model.

AREkit is a very close to opensource framework [SeqIO](https://github.com/google/seqio) proposed by [Google](https://github.com/google) 
for data-preprocessing, evaluation, for sequence models.
While SeqIO dedicated for conversion/pre-processing of datasets of any type, 
this project proposes pipelines creation from the very raw or preannotated (BRAT-based) texts, including the solutions for problems mentioned above.

The core functionality includes 
(1) API for document presentation with EL (Entity Linking, i.e. Object Synonymy) support 
for sentence level relations preparation (dubbed as contexts)
(2) API for contexts extraction
(3) relations transferring from sentence-level onto document-level, and more.

## Installation 

1. Install required dependencies
```bash
pip install git+https://github.com/nicolay-r/AREkit.git@0.23.0-rc
```

2. Download Resources
```bash
python -m arekit.download_data
```

## Tutorials
Please follows th
[tutorials list](https://github.com/nicolay-r/AREkit/tree/master/tests/tutorials).

## Applications

* **AREnets** [[github]](https://github.com/nicolay-r/AREnets)
  * is an OpenNRE like project, but the kernel based on tensorflow library, with implementation of neural networks on top of it, designed for Attitude 
* **ARElight** [[site]](https://nicolay-r.github.io/arelight-page/) [[github]](https://github.com/nicolay-r/ARElight)
    * **Infer attitudes** from large Mass-media documents or **sample texts** for your Machine Learning models applications

## Papers
* Frame-Based attitude extraction workflow for news processing [[code]](https://github.com/nicolay-r/frame-based-attitude-extraction-workflow)
    * Represents an attitude annotation workflow based on [RuSentiFrames](https://github.com/nicolay-r/RuSentiFrames) lexicon which is utilized for news processing;
* Neural Networks Applications in Sentiment Attitude Extraction [[code]](https://github.com/nicolay-r/neural-networks-for-attitude-extraction)
    * Neural Networks application for attitude extraction from analytical articles;
* BERT-based model utils for Sentiment Attitude Extraction task [[code]](https://github.com/nicolay-r/bert-utils-for-attitude-extraction)
    * Analytical news formatter for BERT-based models;

## Related Frameworks

*  **SeqIO** [[github]](https://github.com/google/seqio)
*  **DeRE** [[github]](https://github.com/ims-tcl/DeRE) [[paper]](https://aclanthology.org/D18-2008/)
*  **CREST** [[github]](https://github.com/phosseini/CREST) [[paper]](https://arxiv.org/abs/2103.13606)

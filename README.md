# AREkit 0.24.0

![](https://img.shields.io/badge/Python-3.9+-brightgreen.svg)

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

Using AREkit you may focus on preparation and experiments with your ML-models by shift all the data-preparation part onto toolset of this project for:
[neural-networks](https://github.com/nicolay-r/AREkit/wiki/Sampling-for-Neural-Network), 
[language-models](https://github.com/nicolay-r/AREkit/wiki/Sampling-for-BERT), 
[ChatGPT](https://github.com/nicolay-r/AREkit/wiki/Sampling-for-ChatGPT).

In order to do so, we provide:
* :file_folder: API for external [collection binding](https://github.com/nicolay-r/AREkit/wiki/Binding-a-Custom-Source) (native support of [BRAT](https://brat.nlplab.org/)-based exported annotations)
* ➿ [pipelines](https://github.com/nicolay-r/AREkit/wiki/Pipelines:-Text-Opinion-Annotation) and iterators for handling large-scale collections serialization without out-of-memory issues.
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
pip install git+https://github.com/nicolay-r/AREkit.git@0.24.0-rc
```

2. Download Resources
```bash
python -m arekit.download_data
```

## Usage
Please follow the wiki page
[Tutorials List](https://github.com/nicolay-r/AREkit/wiki/Tutorials).

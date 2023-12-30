# AREkit 0.25.0

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
* ➿ [pipelines](https://github.com/nicolay-r/AREkit/wiki/Pipelines:-Text-Opinion-Annotation) and iterators for handling large-scale collections serialization without out-of-memory issues.

The core functionality includes 
(1) API for document presentation with EL (Entity Linking, i.e. Object Synonymy) support 
for sentence level relations preparation (dubbed as contexts)
(2) API for contexts extraction
(3) relations transferring from sentence-level onto document-level, and more.

## Installation 

```bash
pip install git+https://github.com/nicolay-r/AREkit.git@0.25.0-rc
```

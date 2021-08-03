## AREkit Core Documentation

Fundamental structures and types utilized in a Sentiment Attitudes Extraction Task

* Bound [[base-class]](common/bound.py) -- range in text;
* TextObject [[base-class]](common/text_object.py) -- any entry in text with related *Bound*;
* Entity [[base-class]](common/entities/base.py) -- same as TextObject but related to specific text entries;
* Opinion [[base-class]](common/opinions/base.py) -- actually text attitudes with 'source' and 'destination' ('X' -> 'Y');
* Label [[base-classes]](common/labels/base.py) -- sentiment label;
* Frame;
* FrameVariant [[base-class]](common/frame_variants/base.py);
* Embedding [[base-class]](common/embeddings/base.py) -- base class for Word2Vec-like embeddings;
* Synonyms [[base-class]](common/synonyms.py) -- storage for synonymous entries (words and phrases);
* Model [[base-class]](common/model/base.py) -- base model;
* **Evaluation** -- tools that allows to perform models quality assessment.
    * CmpOpinion [[API]](common/evaluation/cmp_opinions.py) -- structure describes pairs of opinions to compare;
    * BaseEvaluator [[API]](common/evaluation/evaluators/base.py);

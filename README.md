# 自然语言处理-词性标注

对比常见模型在词性标注任务上的效果，主要涉及以下几种模型：

- [GRN: Gated Relation Network to Enhance Convolutional Neural Network for Named Entity Recognition](https://arxiv.org/pdf/1907.05611.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

## Char-level 效果

### BiRNN

|-|Simple|CRF|Segment|Bigram|Segment + Bigram|CRF + Segment|CRF + Bigram|CRF + Segment + Bigram|CRF + Segment + Bigram + Fix Embedding|
|----|----|----|----|----|----|----|----|----|----|
|ctb8|0.785|0.84|0.893|0.813|0.902|0.913|0.858|0.915|<b>0.916</b>|
|pfr|0.742|0.811|0.896|0.803|0.915|0.921|0.85|<b>0.93</b>|0.929|
|ud|0.764|0.82|0.895|0.837|0.909|0.902|0.856|<b>0.913</b>|0.912|

### GRN

|-|Simple|CRF|Segment|Bigram|Segment + Bigram|CRF + Segment|CRF + Bigram|CRF + Segment + Bigram|CRF + Segment + Bigram + Fix Embedding|
|----|----|----|----|----|----|----|----|----|----|
|ctb8|0.808|0.858|0.9|0.85|0.909|0.916|0.889|<b>0.928</b>|<b>0.928</b>|
|pfr|0.773|0.842|0.899|0.831|0.912|0.922|0.887|<b>0.938</b>|<b>0.938</b>|
|ud|0.76|0.836|0.89|0.828|0.903|0.906|0.875|<b>0.921</b>|0.919|

### Bert

|-|Simple|CRF|Fix Embedding|CRF + Fix Embedding|
|----|----|----|----|----|
|ctb8|<b>0.949</b>|<b>0.949</b>|0.894|0.921|
|pfr|<b>0.968</b>|<b>0.968</b>|0.844|0.874|
|ud|<b>0.956</b>|<b>0.956</b>|0.814|0.852|

## Word-level 效果

### BiRNN

|-|Simple|CRF|CRF + Fix Embedding|
|----|----|----|----|
|ctb8|0.91|<b>0.916</b>|0.911|
|pfr|0.916|<b>0.929</b>|0.922|
|ud|0.897|<b>0.904</b>|0.903|

### GRN

|-|Simple|CRF|Char|CRF + Char|CRF + Char + Fix Embedding|
|----|----|----|----|----|----|
|ctb8|0.923|0.924|<b>0.938</b>|0.937|0.933|
|pfr|0.938|0.94|<b>0.95</b>|0.949|0.939|
|ud|0.905|0.906|<b>0.935</b>|<b>0.935</b>|0.931|

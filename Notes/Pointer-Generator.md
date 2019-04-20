## Pointer Generator Summary

Prelude:

We train a fixed-size network but map it over a variable length input to get variable length output. We fold the input into a fixed-size hidden state. Then unfold these hidden states into a series of soft pointers which are probability distributions over the input sequence. These pointer nets are known as content-based attention-using the values of the incoming data to decide dynamically where to attend to (or which indices to point to) - unlike location-based attention that keeps looking in position y for the area of interest.

ABS(abstractive text summarization) is the task of generating summaries from a source document which extract salient phrases/sentences/passages but retains the ability to create novel phrases and words unlike strictly extractive methods. 

Neural sequence models have enabled the successes in ABS but suffer from repetition and inaccuracy. See et. al (2017) augment Sequence to Sequence attentional models with a hybrid pointer-generator network that can copy words from the source by pointing (ensuring reproducibility and novel word generation) and coverage to keep track of what has been summarized already (ensuring less repetition). 

Traditional methods struggle to deal with Out-Of-Vocabulary (OOV) words. The pointer generator proposed here is able to copy words via a pointing mechanism which improves accuracy and overcomes OOV words, and the hybrid aspect enables the model to generate new words. They propose a novel variant of the coverage vector used in NMT, which is used to track and control what's been covered (coverage) in the source document to reduce repetition. 


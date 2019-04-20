## Pointer Generator Summary

Prelude:

We train a fixed-size network but map it over a variable length input to get variable length output. We fold the input into a fixed-size hidden state. Then unfold these hidden states into a series of soft pointers which are probability distributions over the input sequence. These pointer nets are known as content-based attention-using the values of the incoming data to decide dynamically where to attend to (or which indices to point to) - unlike location-based attention that keeps looking in position y for the area of interest.

Neural sequence models have enabled the successes in 

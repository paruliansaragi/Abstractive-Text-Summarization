## A Deep Reinforced Model For Abstractive Summarization

The authors of this paper point out that Attentional models suffer from repeating phrases. 
They overcome this via an intra-attentional encoder and sequential 
intra-attentional decoder that takes into account what words have been used already. 

They use a bidirectional LSTM encoder to compute hidden states from the input embeddings and an LSTM decoder 
to compute hidden states from output embeddings. Both input/output embeddings are taken from the same weight matrix.

The intra-attentional decoder attends to specific parts of the encoder input sequence, the decoders 
hidden state and the previously generated word to avoid repetition. Instead of using a dot product of 
vectors (query and key) they use a bilinear function to compute an attention score. They normalize the 
attention weights and penalize repeat input tokens to create new temporal scores. Then they compute the 
normalized attention scores from the temporal scores. Then use these normalized attention scores with the 
hidden input states to compute the input context vector. They also allow the decoder to intra-attention mechanism 
to look back at previous words to compute predictions (computes a context vector at each time step). 

To generate a new token they use a switch mechanism which determines whether or not (ut is = 1) to use a 
pointer mechanism or a token generation softmax layer or a copy mechanism. So computing the equations mentioned we get our output. 

What’s more, is that since the encoder/decoder/token generation share the same weight matrix the token generator 
can use the syntactic/semantic information in the embedding matrix. 

They also prevent the decoder from ever outputting the same trigram twice. 

They employ a hybrid learning objective of minimizing the maximum likelihood loss(which alone does not perform well 
on ROUGE due to exposure bias – where during test time the algorithm can't rely on teacher forcing) 
and a self-critical policy gradient training algorithm.  For the RL, they produce two 
outputs at each iteration, one from sampling the probability distribution at each decoding time step and the baseline 
output which is a greedy search (highest probability from each time step). The reward function compares the baseline 
output and sampling output allowing it to use any metric it wants. 

This guarantees nothing since ROUGE tends to overlap n-gram summaries whereas a LM gives better 
human-like summaries based on perplexity. The Maximum likelihood is a conditional LM. 

They thus combine the two learning objectives into one equation with a scaling factor to determine 
the difference in magnitude between the two learning objectives. They first run maximum likelihood training with and 
without intra-decoder attention and initialize the parameters with the best ML parameters then compare RL with the ML+RL 
mixed objective. Intra-attention improves performance on longer sequences but not on shorter. Both the RL and RL+ML models 
perform better than ML and outperform the SOTA as well as some of the extractive summarization baselines. They limit the 
generated summary length to the length of the ground-truth summary.  


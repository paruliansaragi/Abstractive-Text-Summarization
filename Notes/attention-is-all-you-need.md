## Attention Is All You Need

The work proposes a new architecture called the Transformer without the use of recurrence or convolutions but 
uses attention mechanisms that seeks to replace dominant transduction models. The sequential prerequisites of 
recurrent models prohibit parallelization and often attention mechanisms are used along side RNN's. The transformer
allows for significant increases in parallelization. Attempts at reducing sequential computation require the number 
of operations to relate signals from two arbitrary input and output positions grows in the distance between positions.
The transformer reduces this to constant number of ops, at the cost of reduced effective resolution due to averaging
attention-weighted positions (counteracted with Multi-head Attention). 

Self-attention (intra-attention) involves relating different positions of a single sequence to compute a representation
of the sequence. Paulus et al. 2017 use it for their work A deep reinforced model for abstractive summarization. 
Their architecture follows from the neural sequence trasnduction models with encoder-decoder structures. They use
stacked self-attention and point-wise, fully-connected layers for both encoder and decoder. 

The encoder is a stack of 6 identical layers, each layer has 2 sub-layers. There is a residual connection around 
each two sub-layers followed by a layer normalization. The output of each sub-layer is LayerNorm(x+Sublayer(x)). 
The first is a multi-head self-attention mechanism, and the second is a position-wise fully-connect FF net. 
The decoder also has 6 stacked identical layers but with a third sub-layer, which performs multi-head 
attention over the output of the encoder stack. Similarly, residual connections are around sub-layers 
followed by layer normalization. Self-attention is modified in the decoder to prevent positions 
from attending to subsequent positions. Masking and with the output embeddings being offset by one position
ensures predictions for position i can only depend on known outputs at positions less than i.

An attention function maps a query and a set of key-value pairs to an output (all vectors). 
The output is a weighted sum of values, where the weight is computed from a function of the 
query and corresponding key. This is called Scaled Dot-Product Attention and the multi-head attention consists
of several attention layers running in parallel.

Inputs are queries and keys of dimension dk, values of dv. This is a simple dot product of query with keys,
divided by each √dk then a softmax to obtain the weights on the values. This is done with matrices of Q, K, V.
Attention(Q, K, V ) = softmax(QK.T/√dk)V

They use dot-product attention over Bahdanau attention as its faster but with a scaling fact 1/√dk (because of
large values of dk, the dot product grows large and the softmax produces extremely small gradients). 

Multi-head attention allows the model to jointly attend to info from different representation subspaces at 
different positions. Multihead attention is a concatenation of attentions i.e. heads. 
The transformer allows every position in the decoder to attend over all positions in the input sequence 
and each positions in the decoder up to and including that position. Self-attention in the encoder allows 
each position in the encoder to attend to all positions in the previous layer encoder. To preserve the 
auto-regressive property we need to prevent leftward information flow in the decoder by masking out all 
values in the input of the softmax which are illegal connections. In addition to attention sub-layers 
each layer in the encoder-decoder has a fully-connected FF network applied to each position separately 
and identically. Since the model has no recurrence or convolution, to make the model use the order of 
the sequence, we inject some information about the relative or absolute position of the tokens in the sequence. 

## Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting
(https://arxiv.org/pdf/1805.11080.pdf)

Their method is to apply a hybrid extractive summarization then abstractive summarization model. 
The extractor selects salient sentences from a document and the abstractor rewrites (compresses and paraphrases) those sentences. 
This overcomes redundancy issues and speeds up the network as the salient sentences have already been extracted. 
The model produces more novel n-gram models and is SOTA. 

The hierarchical temporal convolutional encoder computes representations for each sentence. 
These are fed into a bi-LSTM to create global long-range context aware representations on the CNN output. 
The RNN decoder selects sentences and feeds the hidden states of the encoder to the RNN decoder. 

They add another LSTM-RNN which is a Pointer network that extracts sentences recurrently. 
At each time step the LSTM-RNN decoder does a 2-hop attention that first attends over the hidden states of the 
encoder to get a context vector and then attends to hidden states again to get extraction probabilities. 
The abstractor generates a sentence using a standard encoder-aligner method (Bahdanau 2015/ Luong 2015) 
and a copy mechanism to copy OOV words (See 2017), and compares the ground truth to the generated sentence for the reward function. 
Since extraction is non-differentiable (sentence selection is classification) they use a 
policy gradient method to link the extractor/abstractor. 

Since the dataset does not have saliency labels they use a similarity method to proxy a saliency label for the extractor. 
The extractor is trained to minimize the cross-entropy loss. The abstractor is trained as a usual seq-to-seq model to 
minimize cross-entropy loss of the decoder LM at each generation step. They make the extractor a Markov Decision Process 
at each extraction step, the agent observes the current state, samples an action to extract a document sentence and 
receive a reward after the abstractor summarizes the extracted sentences. To overcome the high variance problem with 
vanilla Reinforce algorithm they use critic network to predict the state-value function. The predicted value of the 
critic is used to estimate the advantage function and the goal is to maximize the advantage function with the policy 
gradient and the critic is trained to minimize the square loss. This is known as the Advantage-Actor Critic (A2C). 

If the extractor chooses a good sentence, after the abstractor rewrites it then, the ROUGE match would be high and 
the action encouraged. A bad sentence would not match the ground truth thus a low ROUGE and discouraged action. 
RL is the saliency guide without altering the abstractor’s LM. They also add a stop action. The model is encouraged to 
extract when there are remaining ground-truth summary sentences and learn to stop by optimizing a global ROUGE and 
avoid extraction. They also avoid tri-gram avoidance during beam search like Paulus et al 2018. They also re-rank 
combinations of generated summary sentence beams. Ranked by number of repeated N-grams. Their extractor and abstractor are SOTA.

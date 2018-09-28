#!/usr/bin/env python
# coding: utf-8

# # Learn to calculate with seq2seq model
# 
# In this assignment, you will learn how to use neural networks to solve sequence-to-sequence prediction tasks. Seq2Seq models are very popular these days because they achieve great results in Machine Translation, Text Summarization, Conversational Modeling and more.
# 
# Using sequence-to-sequence modeling you are going to build a calculator for evaluating arithmetic expressions, by taking an equation as an input to the neural network and producing an answer as it's output.
# 
# The resulting solution for this problem will be based on state-of-the-art approaches for sequence-to-sequence learning and you should be able to easily adapt it to solve other tasks. However, if you want to train your own machine translation system or intellectual chat bot, it would be useful to have access to compute resources like GPU, and be patient, because training of such systems is usually time consuming. 
# 
# ### Libraries
# 
# For this task you will need the following libraries:
#  - [TensorFlow](https://www.tensorflow.org) — an open-source software library for Machine Intelligence.
#  - [scikit-learn](http://scikit-learn.org/stable/index.html) — a tool for data mining and data analysis.
#  
# If you have never worked with TensorFlow, you will probably want to read some tutorials during your work on this assignment, e.g. [Neural Machine Translation](https://www.tensorflow.org/tutorials/seq2seq) tutorial deals with very similar task and can explain some concepts to you. 

# ### Data
# 
# One benefit of this task is that you don't need to download any data — you will generate it on your own! We will use two operators (addition and subtraction) and work with positive integer numbers in some range. Here are examples of correct inputs and outputs:
# 
#     Input: '1+2'
#     Output: '3'
#     
#     Input: '0-99'
#     Output: '-99'
# 
# *Note, that there are no spaces between operators and operands.*
# 
# 
# Now you need to implement the function *generate_equations*, which will be used to generate the data.

# In[ ]:


import random


# In[ ]:


def generate_equations(allowed_operators, dataset_size, min_value, max_value):
    """Generates pairs of equations and solutions to them.
    
       Each equation has a form of two integers with an operator in between.
       Each solution is an integer with the result of the operaion.
    
        allowed_operators: list of strings, allowed operators.
        dataset_size: an integer, number of equations to be generated.
        min_value: an integer, min value of each operand.
        max_value: an integer, max value of each operand.

        result: a list of tuples of strings (equation, solution).
    """
    sample = []
    for _ in range(dataset_size):
        ######################################
        ######### YOUR CODE HERE #############
        ######################################
    return sample


# To check the correctness of your implementation, use *test_generate_equations* function:

# In[ ]:


def test_generate_equations():
    allowed_operators = ['+', '-']
    dataset_size = 10
    for (input_, output_) in generate_equations(allowed_operators, dataset_size, 0, 100):
        if not (type(input_) is str and type(output_) is str):
            return "Both parts should be strings."
        if eval(input_) != int(output_):
            return "The (equation: {!r}, solution: {!r}) pair is incorrect.".format(input_, output_)
    return "Tests passed."


# In[ ]:


print(test_generate_equations())


# Finally, we are ready to generate the train and test data for the neural network:

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


allowed_operators = ['+', '-']
dataset_size = 100000
data = generate_equations(allowed_operators, dataset_size, min_value=0, max_value=9999)

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)


# ## Prepare data for the neural network
# 
# The next stage of data preparation is creating mappings of the characters to their indices in some vocabulary. Since in our task we already know which symbols will appear in the inputs and outputs, generating the vocabulary is a simple step.
# 
# #### How to create dictionaries for other task
# 
# First of all, you need to understand what is the basic unit of the sequence in your task. In our case, we operate on symbols and the basic unit is a symbol. The number of symbols is small, so we don't need to think about filtering/normalization steps. However, in other tasks, the basic unit is often a word, and in this case the mapping would be *word $\to$ integer*. The number of words might be huge, so it would be reasonable to filter them, for example, by frequency and leave only the frequent ones. Other strategies that your should consider are: data normalization (lowercasing, tokenization, how to consider punctuation marks), separate vocabulary for input and for output (e.g. for machine translation), some specifics of the task.

# In[ ]:


word2id = {symbol:i for i, symbol in enumerate('#^$+-1234567890')}
id2word = {i:symbol for symbol, i in word2id.items()}


# #### Special symbols

# In[ ]:


start_symbol = '^'
end_symbol = '$'
padding_symbol = '#'


# You could notice that we have added 3 special symbols: '^', '\$' and '#':
# - '^' symbol will be passed to the network to indicate the beginning of the decoding procedure. We will discuss this one later in more details.
# - '\$' symbol will be used to indicate the *end of a string*, both for input and output sequences. 
# - '#' symbol will be used as a *padding* character to make lengths of all strings equal within one training batch.
# 
# People have a bit different habits when it comes to special symbols in encoder-decoder networks, so don't get too much confused if you come across other variants in tutorials you read. 

# #### Padding

# When vocabularies are ready, we need to be able to convert a sentence to a list of vocabulary word indices and back. At the same time, let's care about padding. We are going to preprocess each sequence from the input (and output ground truth) in such a way that:
# - it has a predefined length *padded_len*
# - it is probably cut off or padded with the *padding symbol* '#'
# - it *always* ends with the *end symbol* '$'
# 
# We will treat the original characters of the sequence **and the end symbol** as the valid part of the input. We will store *the actual length* of the sequence, which includes the end symbol, but does not include the padding symbols. 

#  Now you need to implement the function *sentence_to_ids* that does the described job. 

# In[ ]:


def sentence_to_ids(sentence, word2id, padded_len):
    """ Converts a sequence of symbols to a padded sequence of their ids.
    
      sentence: a string, input/output sequence of symbols.
      word2id: a dict, a mapping from original symbols to ids.
      padded_len: an integer, a desirable length of the sequence.

      result: a tuple of (a list of ids, an actual length of sentence).
    """
    
    sent_ids = ######### YOUR CODE HERE #############
    sent_len = ######### YOUR CODE HERE #############
    
    return sent_ids, sent_len


# Check that your implementation is correct:

# In[ ]:


def test_sentence_to_ids():
    sentences = [("123+123", 7), ("123+123", 8), ("123+123", 10)]
    expected_output = [([5, 6, 7, 3, 5, 6, 2], 7), 
                       ([5, 6, 7, 3, 5, 6, 7, 2], 8), 
                       ([5, 6, 7, 3, 5, 6, 7, 2, 0, 0], 8)] 
    for (sentence, padded_len), (sentence_ids, expected_length) in zip(sentences, expected_output):
        output, length = sentence_to_ids(sentence, word2id, padded_len)
        if output != sentence_ids:
            return("Convertion of '{}' for padded_len={} to {} is incorrect.".format(
                sentence, padded_len, output))
        if length != expected_length:
            return("Convertion of '{}' for padded_len={} has incorrect actual length {}.".format(
                sentence, padded_len, length))
    return("Tests passed.")


# In[ ]:


print(test_sentence_to_ids())


# We also need to be able to get back from indices to symbols:

# In[ ]:


def ids_to_sentence(ids, id2word):
    """ Converts a sequence of ids to a sequence of symbols.
    
          ids: a list, indices for the padded sequence.
          id2word:  a dict, a mapping from ids to original symbols.

          result: a list of symbols.
    """
 
    return [id2word[i] for i in ids] 


# #### Generating batches

# The final step of data preparation is a function that transforms a batch of sentences to a list of lists of indices. 

# In[ ]:


def batch_to_ids(sentences, word2id, max_len):
    """Prepares batches of indices. 
    
       Sequences are padded to match the longest sequence in the batch,
       if it's longer than max_len, then max_len is used instead.

        sentences: a list of strings, original sequences.
        word2id: a dict, a mapping from original symbols to ids.
        max_len: an integer, max len of sequences allowed.

        result: a list of lists of ids, a list of actual lengths.
    """
    
    max_len_in_batch = min(max(len(s) for s in sentences) + 1, max_len)
    batch_ids, batch_ids_len = [], []
    for sentence in sentences:
        ids, ids_len = sentence_to_ids(sentence, word2id, max_len_in_batch)
        batch_ids.append(ids)
        batch_ids_len.append(ids_len)
    return batch_ids, batch_ids_len


# The function *generate_batches* will help to generate batches with defined size from given samples.

# In[ ]:


def generate_batches(samples, batch_size=64):
    X, Y = [], []
    for i, (x, y) in enumerate(samples, 1):
        X.append(x)
        Y.append(y)
        if i % batch_size == 0:
            yield X, Y
            X, Y = [], []
    if X and Y:
        yield X, Y


# To illustrate the result of the implemented functions, run the following cell:

# In[ ]:


sentences = train_set[0]
ids, sent_lens = batch_to_ids(sentences, word2id, max_len=10)
print('Input:', sentences)
print('Ids: {}\nSentences lengths: {}'.format(ids, sent_lens))


# ## Encoder-Decoder architecture
# 
# Encoder-Decoder is a successful architecture for Seq2Seq tasks with different lengths of input and output sequences. The main idea is to use two recurrent neural networks, where the first neural network *encodes* the input sequence into a real-valued vector and then the second neural network *decodes* this vector into the output sequence. While building the neural network, we will specify some particular characteristics of this architecture.

# In[ ]:


import tensorflow as tf


# Let us use TensorFlow building blocks to specify the network architecture.

# In[ ]:


class Seq2SeqModel(object):
    pass


# First, we need to create [placeholders](https://www.tensorflow.org/api_guides/python/io_ops#Placeholders) to specify what data we are going to feed into the network during the execution time. For this task we will need:
#  - *input_batch* — sequences of sentences (the shape will equal to [batch_size, max_sequence_len_in_batch]);
#  - *input_batch_lengths* — lengths of not padded sequences (the shape equals to [batch_size]);
#  - *ground_truth* — sequences of groundtruth (the shape will equal to [batch_size, max_sequence_len_in_batch]);
#  - *ground_truth_lengths* — lengths of not padded groundtruth sequences (the shape equals to [batch_size]);
#  - *dropout_ph* — dropout keep probability; this placeholder has a predifined value 1;
#  - *learning_rate_ph* — learning rate.

# In[ ]:


def declare_placeholders(self):
    """Specifies placeholders for the model."""
    
    # Placeholders for input and its actual lengths.
    self.input_batch = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_batch')
    self.input_batch_lengths = tf.placeholder(shape=(None, ), dtype=tf.int32, name='input_batch_lengths')
    
    # Placeholders for groundtruth and its actual lengths.
    self.ground_truth = ######### YOUR CODE HERE #############
    self.ground_truth_lengths = ######### YOUR CODE HERE #############
        
    self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])
    self.learning_rate_ph = ######### YOUR CODE HERE ############# 


# In[ ]:


Seq2SeqModel.__declare_placeholders = classmethod(declare_placeholders)


# Now, let us specify the layers of the neural network. First, we need to prepare an embedding matrix. Since we use the same vocabulary for input and output, we need only one such matrix. For tasks with different vocabularies there would be multiple embedding layers.
# - Create embeddings matrix with [tf.Variable](https://www.tensorflow.org/api_docs/python/tf/Variable). Specify its name, type (tf.float32), and initialize with random values.
# - Perform [embeddings lookup](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup) for a given input batch.

# In[ ]:


def create_embeddings(self, vocab_size, embeddings_size):
    """Specifies embeddings layer and embeds an input batch."""
     
    random_initializer = tf.random_uniform((vocab_size, embeddings_size), -1.0, 1.0)
    self.embeddings = ######### YOUR CODE HERE ############# 
    
    # Perform embeddings lookup for self.input_batch. 
    self.input_batch_embedded = ######### YOUR CODE HERE ############# 


# In[ ]:


Seq2SeqModel.__create_embeddings = classmethod(create_embeddings)


# #### Encoder
# 
# The first RNN of the current architecture is called an *encoder* and serves for encoding an input sequence to a real-valued vector. Input of this RNN is an embedded input batch. Since sentences in the same batch could have different actual lengths, we also provide input lengths to avoid unnecessary computations. The final encoder state will be passed to the second RNN (decoder), which we will create soon. 
# 
# - TensorFlow provides a number of [RNN cells](https://www.tensorflow.org/api_guides/python/contrib.rnn#Core_RNN_Cells_for_use_with_TensorFlow_s_core_RNN_methods) ready for use. We suggest that you use [GRU cell](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/GRUCell), but you can also experiment with other types. 
# - Wrap your cells with [DropoutWrapper](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/DropoutWrapper). Dropout is an important regularization technique for neural networks. Specify input keep probability using the dropout placeholder that we created before.
# - Combine the defined encoder cells with [Dynamic RNN](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn). Use the embedded input batches and their lengths here.
# - Use *dtype=tf.float32* everywhere.

# In[ ]:


def build_encoder(self, hidden_size):
    """Specifies encoder architecture and computes its output."""
    
    # Create GRUCell with dropout.
    encoder_cell = ######### YOUR CODE HERE #############
    
    # Create RNN with the predefined cell.
    _, self.final_encoder_state = ######### YOUR CODE HERE #############


# In[ ]:


Seq2SeqModel.__build_encoder = classmethod(build_encoder)


# #### Decoder
# 
# The second RNN is called a *decoder* and serves for generating the output sequence. In the simple seq2seq arcitecture, the input sequence is provided to the decoder only as the final state of the encoder. Obviously, it is a bottleneck and [Attention techniques](https://www.tensorflow.org/tutorials/seq2seq#background_on_the_attention_mechanism) can help to overcome it. So far, we do not need them to make our calculator work, but this would be a necessary ingredient for more advanced tasks. 
# 
# During training, decoder also uses information about the true output. It is feeded in as input symbol by symbol. However, during the prediction stage (which is called *inference* in this architecture), the decoder can only use its own generated output from the previous step to feed it in at the next step. Because of this difference (*training* vs *inference*), we will create two distinct instances, which will serve for the described scenarios.
# 
# The picture below illustrates the point. It also shows our work with the special characters, e.g. look how the start symbol `^` is used. The transparent parts are ignored. In decoder, it is masked out in the loss computation. In encoder, the green state is considered as final and passed to the decoder. 

# <img src="encoder-decoder-pic.png" style="width: 500px;">

# Now, it's time to implement the decoder:
#  - First, we should create two [helpers](https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Dynamic_Decoding). These classes help to determine the behaviour of the decoder. During the training time, we will use [TrainingHelper](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/TrainingHelper). For the inference we recommend to use [GreedyEmbeddingHelper](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/GreedyEmbeddingHelper).
#  - To share all parameters during training and inference, we use one scope and set the flag 'reuse' to True at inference time. You might be interested to know more about how [variable scopes](https://www.tensorflow.org/programmers_guide/variables) work in TF. 
#  - To create the decoder itself, we will use [BasicDecoder](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoder) class. As previously, you should choose some RNN cell, e.g. GRU cell. To turn hidden states into logits, we will need a projection layer. One of the simple solutions is using [OutputProjectionWrapper](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/OutputProjectionWrapper).
#  - For getting the predictions, it will be convinient to use [dynamic_decode](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode). This function uses the provided decoder to perform decoding.

# In[ ]:


def build_decoder(self, hidden_size, vocab_size, max_iter, start_symbol_id, end_symbol_id):
    """Specifies decoder architecture and computes the output.
    
        Uses different helpers:
          - for train: feeding ground truth
          - for inference: feeding generated output

        As a result, self.train_outputs and self.infer_outputs are created. 
        Each of them contains two fields:
          rnn_output (predicted logits)
          sample_id (predictions).

    """
    
    # Use start symbols as the decoder inputs at the first time step.
    batch_size = tf.shape(self.input_batch)[0]
    start_tokens = tf.fill([batch_size], start_symbol_id)
    ground_truth_as_input = tf.concat([tf.expand_dims(start_tokens, 1), self.ground_truth], 1)
    
    # Use the embedding layer defined before to lookup embedings for ground_truth_as_input. 
    self.ground_truth_embedded = ######### YOUR CODE HERE #############
     
    # Create TrainingHelper for the train stage.
    train_helper = tf.contrib.seq2seq.TrainingHelper(self.ground_truth_embedded, 
                                                     self.ground_truth_lengths)
    
    # Create GreedyEmbeddingHelper for the inference stage.
    # You should provide the embedding layer, start_tokens and index of the end symbol.
    infer_helper = ######### YOUR CODE HERE #############
    
  
    def decode(helper, scope, reuse=None):
        """Creates decoder and return the results of the decoding with a given helper."""
        
        with tf.variable_scope(scope, reuse=reuse):
            # Create GRUCell with dropout. Do not forget to set the reuse flag properly.
            decoder_cell = ######### YOUR CODE HERE #############
            
            # Create a projection wrapper.
            decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, vocab_size, reuse=reuse)
            
            # Create BasicDecoder, pass the defined cell, a helper, and initial state.
            # The initial state should be equal to the final state of the encoder!
            decoder = ######### YOUR CODE HERE #############
            
            # The first returning argument of dynamic_decode contains two fields:
            #   rnn_output (predicted logits)
            #   sample_id (predictions)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=max_iter, 
                                                              output_time_major=False, impute_finished=True)

            return outputs
        
    self.train_outputs = decode(train_helper, 'decode')
    self.infer_outputs = decode(infer_helper, 'decode', reuse=True)


# In[ ]:


Seq2SeqModel.__build_decoder = classmethod(build_decoder)


# In this task we will use [sequence_loss](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/sequence_loss), which is a weighted cross-entropy loss for a sequence of logits. Take a moment to understand, what is your train logits and targets. Also note, that we do not want to take into account loss terms coming from padding symbols, so we will mask them out using weights.  

# In[ ]:


def compute_loss(self):
    """Computes sequence loss (masked cross-entopy loss with logits)."""
    
    weights = tf.cast(tf.sequence_mask(self.ground_truth_lengths), dtype=tf.float32)
    
    self.loss = ######### YOUR CODE HERE #############


# In[ ]:


Seq2SeqModel.__compute_loss = classmethod(compute_loss)


# The last thing to specify is the optimization of the defined loss. 
# We suggest that you use [optimize_loss](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/optimize_loss) with Adam optimizer and a learning rate from the corresponding placeholder. You might also need to pass global step (e.g. as tf.train.get_global_step()) and clip gradients by 1.0.

# In[ ]:


def perform_optimization(self):
    """Specifies train_op that optimizes self.loss."""
    
    self.train_op = ######### YOUR CODE HERE #############


# In[ ]:


Seq2SeqModel.__perform_optimization = classmethod(perform_optimization)


# Congratulations! You have specified all the parts of your network. You may have noticed, that we didn't deal with any real data yet, so what you have written is just recipies on how the network should function.
# Now we will put them to the constructor of our Seq2SeqModel class to use it in the next section. 

# In[ ]:


def init_model(self, vocab_size, embeddings_size, hidden_size, 
               max_iter, start_symbol_id, end_symbol_id, padding_symbol_id):
    
    self.__declare_placeholders()
    self.__create_embeddings(vocab_size, embeddings_size)
    self.__build_encoder(hidden_size)
    self.__build_decoder(hidden_size, vocab_size, max_iter, start_symbol_id, end_symbol_id)
    
    # Compute loss and back-propagate.
    self.__compute_loss()
    self.__perform_optimization()
    
    # Get predictions for evaluation.
    self.train_predictions = self.train_outputs.sample_id
    self.infer_predictions = self.infer_outputs.sample_id


# In[ ]:


Seq2SeqModel.__init__ = classmethod(init_model)


# ## Train the network and predict output
# 
# [Session.run](https://www.tensorflow.org/api_docs/python/tf/Session#run) is a point which initiates computations in the graph that we have defined. To train the network, we need to compute *self.train_op*. To predict output, we just need to compute *self.infer_predictions*. In any case, we need to feed actual data through the placeholders that we defined above. 

# In[ ]:


def train_on_batch(self, session, X, X_seq_len, Y, Y_seq_len, learning_rate, dropout_keep_probability):
    feed_dict = {
            self.input_batch: X,
            self.input_batch_lengths: X_seq_len,
            self.ground_truth: Y,
            self.ground_truth_lengths: Y_seq_len,
            self.learning_rate_ph: learning_rate,
            self.dropout_ph: dropout_keep_probability
        }
    pred, loss, _ = session.run([
            self.train_predictions,
            self.loss,
            self.train_op], feed_dict=feed_dict)
    return pred, loss


# In[ ]:


Seq2SeqModel.train_on_batch = classmethod(train_on_batch)


# We implemented two prediction functions: *predict_for_batch* and *predict_for_batch_with_loss*. The first one allows only to predict output for some input sequence, while the second one could compute loss because we provide also ground truth values. Both these functions might be useful since the first one could be used for predicting only, and the second one is helpful for validating results on not-training data during the training.

# In[ ]:


def predict_for_batch(self, session, X, X_seq_len):
    feed_dict = ######### YOUR CODE HERE #############
    pred = session.run([
            self.infer_predictions
        ], feed_dict=feed_dict)[0]
    return pred

def predict_for_batch_with_loss(self, session, X, X_seq_len, Y, Y_seq_len):
    feed_dict = ######### YOUR CODE HERE #############
    pred, loss = session.run([
            self.infer_predictions,
            self.loss,
        ], feed_dict=feed_dict)
    return pred, loss


# In[ ]:


Seq2SeqModel.predict_for_batch = classmethod(predict_for_batch)
Seq2SeqModel.predict_for_batch_with_loss = classmethod(predict_for_batch_with_loss)


# ## Run your experiment
# 
# Create *Seq2SeqModel* model with the following parameters:
#  - *vocab_size* — number of tokens;
#  - *embeddings_size* — dimension of embeddings, recommended value: 20;
#  - *max_iter* — maximum number of steps in decoder, recommended value: 7;
#  - *hidden_size* — size of hidden layers for RNN, recommended value: 512;
#  - *start_symbol_id* — an index of the start token (`^`).
#  - *end_symbol_id* — an index of the end token (`$`).
#  - *padding_symbol_id* — an index of the padding token (`#`).
# 
# Set hyperparameters. You might want to start with the following values and see how it works:
# - *batch_size*: 128;
# - at least 10 epochs;
# - value of *learning_rate*: 0.001
# - *dropout_keep_probability* equals to 0.5 for training (typical values for dropout probability are ranging from 0.1 to 1.0); larger values correspond smaler number of dropout units;
# - *max_len*: 20.

# In[ ]:


tf.reset_default_graph()

model = ######### YOUR CODE HERE #############

batch_size = ######### YOUR CODE HERE #############
n_epochs = ######### YOUR CODE HERE #############
learning_rate = ######### YOUR CODE HERE #############
dropout_keep_probability = ######### YOUR CODE HERE #############
max_len = ######### YOUR CODE HERE #############

n_step = int(len(train_set) / batch_size)


# Finally, we are ready to run the training! A good indicator that everything works fine is decreasing loss during the training. You should account on the loss value equal to approximately 2.7 at the beginning of the training and near 1 after the 10th epoch.

# In[ ]:


session = tf.Session()
session.run(tf.global_variables_initializer())
            
invalid_number_prediction_counts = []
all_model_predictions = []
all_ground_truth = []

print('Start training... \n')
for epoch in range(n_epochs):  
    random.shuffle(train_set)
    random.shuffle(test_set)
    
    print('Train: epoch', epoch + 1)
    for n_iter, (X_batch, Y_batch) in enumerate(generate_batches(train_set, batch_size=batch_size)):
        ######################################
        ######### YOUR CODE HERE #############
        ######################################
        # prepare the data (X_batch and Y_batch) for training
        # using function batch_to_ids
        predictions, loss = ######### YOUR CODE HERE #############
        
        if n_iter % 200 == 0:
            print("Epoch: [%d/%d], step: [%d/%d], loss: %f" % (epoch + 1, n_epochs, n_iter + 1, n_step, loss))
                
    X_sent, Y_sent = next(generate_batches(test_set, batch_size=batch_size))
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    # prepare test data (X_sent and Y_sent) for predicting 
    # quality and computing value of the loss function
    # using function batch_to_ids
    
    predictions, loss = ######### YOUR CODE HERE #############
    print('Test: epoch', epoch + 1, 'loss:', loss,)
    for x, y, p  in list(zip(X, Y, predictions))[:3]:
        print('X:',''.join(ids_to_sentence(x, id2word)))
        print('Y:',''.join(ids_to_sentence(y, id2word)))
        print('O:',''.join(ids_to_sentence(p, id2word)))
        print('')

    model_predictions = []
    ground_truth = []
    invalid_number_prediction_count = 0
    # For the whole test set calculate ground-truth values (as integer numbers)
    # and prediction values (also as integers) to calculate metrics.
    # If generated by model number is not correct (e.g. '1-1'), 
    # increase invalid_number_prediction_count and don't append this and corresponding
    # ground-truth value to the arrays.
    for X_batch, Y_batch in generate_batches(test_set, batch_size=batch_size):
        ######################################
        ######### YOUR CODE HERE #############
        ######################################
    
    all_model_predictions.append(model_predictions)
    all_ground_truth.append(ground_truth)
    invalid_number_prediction_counts.append(invalid_number_prediction_count)
            
print('\n...training finished.')


# ## Evaluate results
# 
# Because our task is simple and the output is straight-forward, we will use [MAE](https://en.wikipedia.org/wiki/Mean_absolute_error) metric to evaluate the trained model during the epochs. Compute the value of the metric for the output from each epoch.

# In[ ]:


from sklearn.metrics import mean_absolute_error


# In[ ]:


for i, (gts, predictions, invalid_number_prediction_count) in enumerate(zip(all_ground_truth,
                                                                            all_model_predictions,
                                                                            invalid_number_prediction_counts), 1):
    mae = ######### YOUR CODE HERE #############
    print("Epoch: %i, MAE: %f, Invalid numbers: %i" % (i, mae, invalid_number_prediction_count))


#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import data_reader
import codecs
# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 220, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.2, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 200, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
#input data
tf.flags.DEFINE_string("data_path", "./data/", "data_path")
tf.flags.DEFINE_string("load_word2vec", "true", "Whether to load existing word2vec dictionary file.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")

if FLAGS.load_word2vec == "true":
#     allWordsList, sList, labelsList,count = data_reader._read_words(os.path.join(FLAGS.data_path, 'test.txt'))
#     for i in range(10):
#         print (sList[i])
#         print (labelsList[i])
    word2id, word_embeddings = data_reader.load_word2vec("/search/odin/data/liuqi/vector.skip.win2.100.float.for_python") #vector.skip.win2.100.float.for_python
    vocabulary_size = len(word2id) + 1
    (train_idsList, train_lList), (test_idsList, test_lList), vocabulary, train_count, test_count,max_sentence_length = data_reader.get_data_by_word2vec(word2id, FLAGS.data_path)
    print("Vocabulary Size: {:d}".format(vocabulary_size)) 
    for i in range(10):
        print (train_idsList[i])
    x_train =  np.array(train_idsList)
    for i in range(10):
        print (x_train[i])
    x_test = np.array(test_idsList)
    y_train = train_lList
    y_test = test_lList
if FLAGS.load_word2vec == "false":    
    x_text, y = data_helpers.load_data_and_labels("test")
    x_t, y_t = data_helpers.load_data_and_labels("test")
#     for i in range(10):
#         print (x_text[i])
#         print (y[i])
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(7000000)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
#     for i in range(10):
#         print (x_shuffled[i])
#         print (y_shuffled[i])

    xt = np.array(list(vocab_processor.fit_transform(x_t)))
    np.random.seed(9000)
    shuffle_indices = np.random.permutation(np.arange(len(y_t)))
    x_test = xt[shuffle_indices]
    y_test = y_t[shuffle_indices]
    
    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    x_train, x_dev = x_shuffled[:-1], x_shuffled[-1:]
    y_train, y_dev = y_shuffled[:-1], y_shuffled[-1:]
    for i in range(10):
        print (x_train[i])
        print (y_train[i])

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================
fo = codecs.open("//search//odin//data//liuqi//cnn//pr_re","w","utf-8")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with tf.device('/cpu:0'): ##sess.as_default():
        if FLAGS.load_word2vec == "false":
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=2,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
        if FLAGS.load_word2vec == "true":
             cnn = TextCNN(
                sequence_length=max_sentence_length,
                num_classes=2,
                vocab_size=vocabulary_size,
                embedding_size=100,
                wordembedding=word_embeddings,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
#              sess.run(cnn.embedding.assign(word_embeddings))
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)
        
        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())
        
        # Write vocabulary
#         vocab_processor.save(os.path.join(out_dir, "vocab"))
        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy, prediction, target_y, embedded_chars  = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.predictions, cnn.target_y, cnn.embedded_chars],
                feed_dict)
#             for i in range(15):
#                 print (embedded_chars[i])
            true_positive = 0.0
            false_positve = 0.0
            for i in range(len(prediction)):
                if prediction[i] == 1 and target_y[i] == 1:
                    true_positive += 1
                if prediction[i] == 1 and target_y[i] == 0:
                    false_positve += 1
            recovery =  true_positive / (true_positive + false_positve)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, rec {:g}".format(time_str, step, loss, accuracy, recovery))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy, prediction, target_y = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions, cnn.target_y],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            true_positive = 0.0
            false_positve = 0.0
            for i in range(len(prediction)):
                if prediction[i] == 1 and target_y[i] == 1:
                    true_positive += 1
                if prediction[i] == 1 and target_y[i] == 0:
                    false_positve += 1
#             print (true_positive)
#             print (false_positve)
            recovery =  true_positive / (true_positive + false_positve)
            F1 = (2 * recovery * accuracy ) / (recovery + accuracy)
            fo.write(str(accuracy))
            fo.write("\t")
            fo.write(str(recovery))
            fo.write("\n")
            print("{}: step {}, loss {:g}, acc {:g}, rec {:g}".format(time_str, step, loss, accuracy, recovery))
            if writer:
                writer.add_summary(summaries, step)
            return F1

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        F1_s = 0.0
        for d in ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']:
	    with tf.device(d):
		for batch in batches:
            	    x_batch, y_batch = zip(*batch)
            	    train_step(x_batch, y_batch)
            	    current_step = tf.train.global_step(sess, global_step)
            	    if current_step % FLAGS.evaluate_every == 0:
                	print("\nEvaluation:")
                	F1 = dev_step(x_test, y_test, writer=dev_summary_writer)
                	print("")
#                 		if current_step % FLAGS.checkpoint_every == 0:
                	if F1 > F1_s:
                    	    F1_s = F1
                    	    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    	    print("Saved model checkpoint to {}\n".format(path))

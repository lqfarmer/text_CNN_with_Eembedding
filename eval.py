#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import codecs
from text_cnn import TextCNN
from tensorflow.contrib import learn
import data_reader
# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "/data1/my_tensorflow/zhangxiao/liuqi/cnn/runs/1474199634/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("data_path", "./data/test/", "data_path")
tf.flags.DEFINE_string("load_word2vec", "true", "Whether to load existing word2vec dictionary file.")
tf.flags.DEFINE_string("load_random", "false", "Whether to load existing word2vec dictionary file.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
start = time.clock()

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    if FLAGS.load_word2vec:
        word2id, word_embeddings = data_reader.load_word2vec("/data1/my_tensorflow/vector.skip.win2.100.float.for_python") #vector.skip.win2.100.float.for_python
        (train_idsList, train_lList), (test_idsList, test_lList), vocabulary, train_count, test_count,max_sentence_length = data_reader.get_data_by_word2vec(word2id, FLAGS.data_path)
        x_test, y_test = test_idsList, test_lList
    else:
        x_raw, y_test = data_helpers.load_data_and_labels("test")
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
# if FLAGS.load_random:
#     vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
#     vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
#     x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#         saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        fo = codecs.open("//data1//my_tensorflow//zhangxiao//liuqi//cnn//predict_result.txt","w","utf-8")
        print ("predicting .....")
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
        print ("writing out data>>>>>")
        for i in range(len(all_predictions)):
            fo.write(str(all_predictions[i]))
            fo.write("\n") 
        end = time.clock()
        print("The function run time is : %.03f seconds" %(end-start))
# Print accuracy if y_test is defined
# if y_test is not None:
#     correct_predictions = float(sum(all_predictions == y_test))
#     print("Total number of test examples: {}".format(len(y_test)))
#     print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))


r"""
使用tensorflow进行有限的语音关键词识别。

语音关键词识别主程序:train

语音文件结构：

dataset >
  1 >
    audio_0.wav
    audio_1.wav
  2 >
    audio_2.wav
    audio_3.wav
  3>
    audio_4.wav
    audio_5.wav

使用--wanted_words解析参数来告诉系统你自己需要识别的关键词，如果你只需要识别两个关键词，那么剩下的关键词将全部
用来作为“unknown"类进行训练
解析命令为：
#python E:\speech_rocognition_demo\method3\tf-keywords\train.py 
--data_dir=F:\speech_rocognition_demo\method3\tf-keywords\speech_dataset 
--wanted_words=yonghudenglu,yonghutuichu,dakaicaidan,huidaoshouye

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange  
import tensorflow as tf

import input_data
import models
from tensorflow.python.platform import gfile

FLAGS = None


def main(_):
  # 可看到所有日志消息
  tf.logging.set_verbosity(tf.logging.INFO)

  # 开始一个新的tensorflow对话
  sess = tf.InteractiveSession()

  
  model_settings = models.prepare_model_settings(
      len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess)
  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir,
      FLAGS.silence_percentage, FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings, FLAGS.summaries_dir)
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
  
  #计算每个训练阶段的学习率。由于在训练开始时设置较高的学习率，然后在训练结束时设置较低的学习率通常是有效的，
  #因此将步骤数和学习率指定为逗号分隔的列表，以定义每个阶段的学习率。
  #例如--how_many_training_steps=10000，3000--learning_rate=0.001，0.0001
  #将总共运行13000个训练循环，前10000个循环的学习速率为0.001，最后3000个循环的学习速率为0.0001
  
  training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))

  input_placeholder = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')
  if FLAGS.quantize:
    
    if FLAGS.preprocess == 'average':
      fingerprint_min = 0.0
      fingerprint_max = 2048.0
    elif FLAGS.preprocess == 'mfcc':
      fingerprint_min = -247.0
      fingerprint_max = 30.0
    else:
      raise Exception('Unknown preprocess mode "%s" (should be "mfcc" or'
                      ' "average")' % (FLAGS.preprocess))
    fingerprint_input = tf.fake_quant_with_min_max_args(
        input_placeholder, fingerprint_min, fingerprint_max)
  else:
    fingerprint_input = input_placeholder

  logits, dropout_prob = models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      is_training=True)

  #定义loss值和优化器
  ground_truth_input = tf.placeholder(
      tf.int64, [None], name='groundtruth_input')

  #或者，我们可以添加运行时检查，以确定在训练期间何时开始出现NaNs或其他数值错误症状。
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

  # 在图中创建反向传播和训练评估机制。
  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
        labels=ground_truth_input, logits=logits)
  if FLAGS.quantize:
    tf.contrib.quantize.create_training_graph(quant_delay=0)
  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    learning_rate_input = tf.placeholder(
        tf.float32, [], name='learning_rate_input')
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate_input).minimize(cross_entropy_mean)
  predicted_indices = tf.argmax(logits, 1)
  correct_prediction = tf.equal(predicted_indices, ground_truth_input)
  confusion_matrix = tf.confusion_matrix(
      ground_truth_input, predicted_indices, num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  with tf.get_default_graph().name_scope('eval'):
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    tf.summary.scalar('accuracy', evaluation_step)

  global_step = tf.train.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)

  saver = tf.train.Saver(tf.global_variables())

  # 合并所有摘要并将其写入/tmp/retrain_logs（默认情况下）
  merged_summaries = tf.summary.merge_all(scope='eval')
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                       sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

  tf.global_variables_initializer().run()

  start_step = 1

  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)

  tf.logging.info('Training from step: %d ', start_step)

  #保存 graph.pbtxt
  tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                       FLAGS.model_architecture + '.pbtxt')

  # 保存词列表。
  with gfile.GFile(
      os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'),
      'w') as f:
    f.write('\n'.join(audio_processor.words_list))

  # 训练循环.
  training_steps_max = np.sum(training_steps_list)
  for training_step in xrange(start_step, training_steps_max + 1):
    
      # 找出当前的学习率。
    training_steps_sum = 0
    for i in range(len(training_steps_list)):
      training_steps_sum += training_steps_list[i]
      if training_step <= training_steps_sum:
        learning_rate_value = learning_rates_list[i]
        break
   
    # 把我们需要用于训练的音频样本拉出来。
    
    train_fingerprints, train_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
        FLAGS.background_volume, time_shift_samples, 'training', sess)
    
    #运行这一批训练数据的图表
    
    train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
        [
            merged_summaries,
            evaluation_step,
            cross_entropy_mean,
            train_step,
            increment_global_step,
        ],
        feed_dict={
            fingerprint_input: train_fingerprints,
            ground_truth_input: train_ground_truth,
            learning_rate_input: learning_rate_value,
            dropout_prob: 0.5
        })
    train_writer.add_summary(train_summary, training_step)
    tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                    (training_step, learning_rate_value, train_accuracy * 100,
                     cross_entropy_value))
    
    
    
    traintxt=str(train_accuracy * 100)          # data是前面运行出的数据，先将其转为字符串才能写入
    with open('E:\\speech_rocognition_demo\\method3\\tf-keywords\\result\\result_original\\train_low_latency_conv.txt','a') as file_handle:
        file_handle.write(traintxt)     # 写入
        file_handle.write('\n')         # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
    
    
    
    
    is_last_step = (training_step == training_steps_max)
    if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
      set_size = audio_processor.set_size('validation')
      total_accuracy = 0
      total_conf_matrix = None
      for i in xrange(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                     0.0, 0, 'validation', sess))

        # 运行验证步骤并且使用’merged‘方法来获取tensorboard的训练摘要
        
        validation_summary, validation_accuracy, conf_matrix = sess.run(
            [merged_summaries, evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: validation_fingerprints,
                ground_truth_input: validation_ground_truth,
                dropout_prob: 1.0
            })
        validation_writer.add_summary(validation_summary, training_step)
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
          total_conf_matrix = conf_matrix
        else:
          total_conf_matrix += conf_matrix
      tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
      tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                      (training_step, total_accuracy * 100, set_size))
      
      validationtxt=str(total_accuracy * 100)          # data是前面运行出的数据，先将其转为字符串才能写入
      with open('E:\\speech_rocognition_demo\\method3\\tf-keywords\\result\\result_original\\validation_low_latency_conv.txt','a') as file_handle:
            file_handle.write(validationtxt)     # 写入
            file_handle.write('\n')         # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
      

    #定期保存模型checkpoint
    if (training_step % FLAGS.save_step_interval == 0 or
        training_step == training_steps_max):
      checkpoint_path = os.path.join(FLAGS.train_dir,
                                     FLAGS.model_architecture + '.ckpt')
      tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
      saver.save(sess, checkpoint_path, global_step=training_step)

  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in xrange(0, set_size, FLAGS.batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    test_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_input: test_ground_truth,
            dropout_prob: 1.0
        })
    batch_size = min(FLAGS.batch_size, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,
                                                           set_size))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str, 
      default='',
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='E:/speech_rocognition_demo/method3/tf-keywords/dataset/original',
      help="""\
      Where is the speech training data.
      """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=2200,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is.',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How far to move in time between spectogram timeslices.',)
  parser.add_argument(
      '--feature_bin_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='2000,2000',
      help='How many training loops to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=20,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='How many items to train with at once',)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='E:/speech_rocognition_demo/method3/tf-keywords/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yonghudenglu,yonghutuichu,dakaicaidan,huidaoshouye',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='E:/speech_rocognition_demo/method3/tf-keywords/tmp/speech_commands_train_low_latency_conv',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=200,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='low_latency_conv',
      help='What model architecture to use, "single_fc", "conv","low_latency_conv", "low_latency_svdf","tiny_conv"')
  parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')
  parser.add_argument(
      '--quantize',
      type=bool,
      default=False,
      help='Whether to train the model for eight-bit deployment')
  parser.add_argument(
      '--preprocess',
      type=str,
      default='mfcc',
      help='Spectrogram processing mode. Can be "mfcc" or "average"')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

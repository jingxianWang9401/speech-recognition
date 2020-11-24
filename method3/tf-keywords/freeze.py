"""

将经过训练的检查点转换为用于移动推断的冻结模型。

使用“train.py”脚本训练模型后，可以使用此脚本将其转换为二进制GraphDef文件，
该文件可以加载到Android、iOS或Raspberry Pi中使用。

命令行运行示例：

bazel run tensorflow/examples/speech_commands/freeze -- \
--sample_rate=16000 --dct_coefficient_count=40 --window_size_ms=20 \
--window_stride_ms=10 --clip_duration_ms=1000 \
--model_architecture=conv \
--start_checkpoint=/tmp/speech_commands_train/conv.ckpt-1300 \
--output_file=/tmp/my_frozen_graph.pb

需要注意：
需要传入与训练脚本相同的“sample_rate”参数和其他命令行变量参数。              
结果图有一个WAV编码数据的输入名为“WAV_data”，
一个用于原始PCM数据（浮动范围为-1.0到1.0），称为“decoded_sample_data”，输出名为“labels_softmax”。

"""

#python E:\speech_rocognition_demo\method3\tf-keywords\freeze.py --sample_rate=16000 
#--clip_duration_ms=2200 --wanted_words=yonghudenglu,yonghutuichu,dakaicaidan,huidaoshouye
#--start_checkpoint=E:/speech_rocognition_demo/method3/tf-keywords/tmp/speech_commands_train_conv/conv.ckpt-4000 
#--output_file=E:/speech_rocognition_demo/method3/tf-keywords/tmp/speech_commands_train_conv/model/speechcommands_recognition.pb

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import input_data
import models
from tensorflow.python.framework import graph_util

FLAGS = None


def create_inference_graph(wanted_words, sample_rate, clip_duration_ms,
                           clip_stride_ms, window_size_ms, window_stride_ms,
                           feature_bin_count, model_architecture, preprocess):
  """
  创建包含推理所需节点的音频模型。
  
  使用提供的参数创建模型，并插入使用图形进行推理所需的输入和输出节点。
  
  以下参数和模型训练时的参数是一致的
  Args:
    wanted_words: 
    sample_rate: 
    clip_duration_ms: 
    clip_stride_ms: 
    window_size_ms: 
    window_stride_ms: 
    feature_bin_count: 
    model_architecture: 
    preprocess: How the spectrogram is processed to produce features, for
      example 'mfcc' or 'average'.

  """

  words_list = input_data.prepare_words_list(wanted_words.split(','))
  model_settings = models.prepare_model_settings(
      len(words_list), sample_rate, clip_duration_ms, window_size_ms,
      window_stride_ms, feature_bin_count, preprocess)
  runtime_settings = {'clip_stride_ms': clip_stride_ms}

  wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
  decoded_sample_data = contrib_audio.decode_wav(
      wav_data_placeholder,
      desired_channels=1,
      desired_samples=model_settings['desired_samples'],
      name='decoded_sample_data')
  spectrogram = contrib_audio.audio_spectrogram(
      decoded_sample_data.audio,
      window_size=model_settings['window_size_samples'],
      stride=model_settings['window_stride_samples'],
      magnitude_squared=True)

  if preprocess == 'average':
    fingerprint_input = tf.nn.pool(
        tf.expand_dims(spectrogram, -1),
        window_shape=[1, model_settings['average_window_width']],
        strides=[1, model_settings['average_window_width']],
        pooling_type='AVG',
        padding='SAME')
  elif preprocess == 'mfcc':
    fingerprint_input = contrib_audio.mfcc(
        spectrogram,
        sample_rate,
        dct_coefficient_count=model_settings['fingerprint_width'])
  else:
    raise Exception('Unknown preprocess mode "%s" (should be "mfcc" or'
                    ' "average")' % (preprocess))

  fingerprint_size = model_settings['fingerprint_size']
  reshaped_input = tf.reshape(fingerprint_input, [-1, fingerprint_size])

  logits = models.create_model(
      reshaped_input, model_settings, model_architecture, is_training=False,
      runtime_settings=runtime_settings)

  # 创建用于推理的输出
  tf.nn.softmax(logits, name='labels_softmax')


def main(_):

  # 创建模型并加载其权重
  sess = tf.InteractiveSession()
  create_inference_graph(
      FLAGS.wanted_words, FLAGS.sample_rate, FLAGS.clip_duration_ms,
      FLAGS.clip_stride_ms, FLAGS.window_size_ms, FLAGS.window_stride_ms,
      FLAGS.feature_bin_count, FLAGS.model_architecture, FLAGS.preprocess)
  if FLAGS.quantize:
    tf.contrib.quantize.create_eval_graph()
  models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)

  # 将所有变量转换为图形中的内联常量并保存它
  frozen_graph_def = graph_util.convert_variables_to_constants(
      sess, sess.graph_def, ['labels_softmax'])
  tf.train.write_graph(
      frozen_graph_def,
      os.path.dirname(FLAGS.output_file),
      os.path.basename(FLAGS.output_file),
      as_text=False)
  tf.logging.info('Saved frozen graph to %s', FLAGS.output_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
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
      '--clip_stride_ms',
      type=int,
      default=30,
      help='How often to run recognition. Useful for models with cache.',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long the stride is between spectrogram timeslices',)
  parser.add_argument(
      '--feature_bin_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',
  )
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='E:/speech_rocognition_demo/method3/tf-keywords/tmp/speech_commands_train_low_latency_conv/low_latency_conv.ckpt-4000',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='low_latency_conv',
      help='What model architecture to use')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yonghudenglu,yonghutuichu,dakaicaidan,huidaoshouye',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--output_file', 
      type=str,
      default='E:/speech_rocognition_demo/method3/tf-keywords/tmp/speech_commands_train_low_latency_conv/model/speechcommands_recognition.pb',
      help='Where to save the frozen graph.')
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

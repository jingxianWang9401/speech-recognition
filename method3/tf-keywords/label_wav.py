
r"""
对WAVE文件运行经过训练的音频图并打印结果。

参数中指定的模型、标签和.wav文件将被加载，然后针对音频数据运行模型的预测并打印到控制台。

用于检查经过训练的模型。 

命令行示例：

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""

#python E:\speech_rocognition_demo\method3\tf-keywords\label_wav.py --graph=./tmp/speechcommands_recognition.pb
#--labels=./tmp/speech_commands_train/conv_labels.txt --wav=F:\speech_rocognition_demo\method3\tf-keywords\test_dataset\yonghudenglu14.wav

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

FLAGS = None


def load_graph(filename):
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """读标签，每行一个标签"""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  """通过图表运行音频数据并打印预测标签"""
  with tf.Session() as sess:
    # 将音频数据作为图形的输入。预测将包含一个二维数组，其中一个维度表示输入图像计数，另一个维度具有每个类的预测。
   
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    # 按可信度排序以显示标签
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

    return 0


def label_wav(wav, labels, graph, input_name, output_name, how_many_labels):
  """加载模型和标签，并打印预测"""
  if not wav or not tf.gfile.Exists(wav):
    tf.logging.fatal('Audio file does not exist %s', wav)

  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)

  # 加载图形，存储在默认会话中
  load_graph(graph)

  with open(wav, 'rb') as wav_file:
    wav_data = wav_file.read()

  run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)


def main(_):
  """脚本的入口点，将标志转换为参数"""
  label_wav(FLAGS.wav, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
            FLAGS.output_name, FLAGS.how_many_labels)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wav', type=str, default='E:/speech_rocognition_demo/method3/tf-keywords/dataset/original_test/zhoudan_huidaoshouye_1.wav', help='Audio file to be identified.')
  parser.add_argument(
      '--graph', type=str, default='E:/speech_rocognition_demo/method3/tf-keywords/tmp/speech_commands_train_low_latency_conv/model/speechcommands_recognition.pb', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='E:/speech_rocognition_demo/method3/tf-keywords/tmp/speech_commands_train_low_latency_conv/low_latency_conv_labels.txt', help='Path to file containing labels.')
  parser.add_argument(
      '--input_name',
      type=str,
      default='wav_data:0',
      help='Name of WAVE data input node in model.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')
  parser.add_argument(
      '--how_many_labels',
      type=int,
      default=3,
      help='Number of results to show.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

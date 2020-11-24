
"""
简单语音识别的模型定义
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import math
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185


def prepare_words_list(wanted_words):
  """
  在自定义单词列表前添加常用标记。

  Args:
    wanted_words: 包含自定义单词的字符串列表。

  Returns:
    添加了标准静音和未知标记的列表。
  """
  return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
  """
  确定文件应属于哪个数据分区。
  
  将文件保存在相同的训练、验证或测试集中，
  即使后续添加了新的文件。
  测试样本不太可能在重新启动运行时在训练中重用。
  为了保持这种稳定性，将获取文件名的哈希值，并用于确定它应该属于哪个集合。
  这个决定只取决于名称和训练、验证、测试设置的比例，所以不会随着其他文件的添加而改变。
  
  将特定文件关联为相关文件（例如同一个人所说的单词），
  因此文件名中的任何内容都将被忽略以进行数据集划分。
  

  Args:
    filename: 数据样本的文件路径
    validation_percentage: 验证集比例
    testing_percentage: 测试集比例

  Returns:
    字符串, 训练，验证和测试
  """
  base_name = os.path.basename(filename)
  
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  
  """
  决定这个文件是否应该进入训练、测试或验证集，
  并且即使在随后添加更多文件的情况下也将现有文件保存在同一个集合中。              
  要做到这一点，需要一种基于文件名本身的稳定的方法,
  所以对其进行散列，然后使用它生成一个用于分配它的概率值。
  """
  
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result


def load_wav_file(filename):
  with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    return sess.run(
        wav_decoder,
        feed_dict={wav_filename_placeholder: filename}).audio.flatten()


def save_wav_file(filename, wav_data, sample_rate):
  with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    sample_rate_placeholder = tf.placeholder(tf.int32, [])
    wav_data_placeholder = tf.placeholder(tf.float32, [None, 1])
    wav_encoder = contrib_audio.encode_wav(wav_data_placeholder,
                                           sample_rate_placeholder)
    wav_saver = io_ops.write_file(wav_filename_placeholder, wav_encoder)
    sess.run(
        wav_saver,
        feed_dict={
            wav_filename_placeholder: filename,
            sample_rate_placeholder: sample_rate,
            wav_data_placeholder: np.reshape(wav_data, (-1, 1))
        })


class AudioProcessor(object):

  def __init__(self, data_url, data_dir, silence_percentage, unknown_percentage,
               wanted_words, validation_percentage, testing_percentage,
               model_settings, summaries_dir):
    self.data_dir = data_dir
    self.maybe_download_and_extract_dataset(data_url, data_dir)
    self.prepare_data_index(silence_percentage, unknown_percentage,
                            wanted_words, validation_percentage,
                            testing_percentage)
    self.prepare_background_data()
    self.prepare_processing_graph(model_settings, summaries_dir)

  def maybe_download_and_extract_dataset(self, data_url, dest_directory):
    if not data_url:
      return
    if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

      def _progress(count, block_size, total_size):
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%' %
            (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

      try:
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
      except:
        tf.logging.error('Failed to download URL: %s to folder: %s', data_url,
                         filepath)
        tf.logging.error('Please make sure you have enough free space and'
                         ' an internet connection')
        raise
      print()
      statinfo = os.stat(filepath)
      tf.logging.info('Successfully downloaded %s (%d bytes)', filename,
                      statinfo.st_size)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

  def prepare_data_index(self, silence_percentage, unknown_percentage,
                         wanted_words, validation_percentage,
                         testing_percentage):
    """
    准备按集合和标签组织的样本列表:
    训练循环需要一个所有可用数据的列表，
    按照它应该属于的数据集划分情况进行组织，
    并附上真实标签。
    
    此函数分析“data-dir”下面的文件夹，
    根据文件所属子目录的名称为每个文件指定正确的标签，
    并使用稳定的哈希将其分配给数据集分区。 

    Args:
      silence_percentage: 结果数据中应该有多少是静音数据。
      unknown_percentage: 多少设为未知数据。
      wanted_words: 我们想要识别的标签类别
      validation_percentage: 多少数据集用于验证
      testing_percentage: 多少数据集用于测试
    Returns:
      包含每个集合分区的文件信息列表的字典，以及用于确定其数值索引的每个类的查找映射。
    """
    random.seed(RANDOM_SEED)
    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
      wanted_words_index[wanted_word] = index + 2
    self.data_index = {'validation': [], 'testing': [], 'training': []}
    unknown_index = {'validation': [], 'testing': [], 'training': []}
    all_words = {}
    
    # 查看所有子文件夹以查找音频样本
    
    search_path = os.path.join(self.data_dir, '*', '*.wav')
    
    for wav_path in gfile.Glob(search_path):
      _, word = os.path.split(os.path.dirname(wav_path))
      word = word.lower()
      # 添加背景噪音
      if word == BACKGROUND_NOISE_DIR_NAME:
        continue
      all_words[word] = True
      set_index = which_set(wav_path, validation_percentage, testing_percentage)
     
      # 如果是已知标签，保存它的细节，否则将使用它用于训练未知的标签。
      if word in wanted_words_index:
        self.data_index[set_index].append({'label': word, 'file': wav_path})
      else:
        unknown_index[set_index].append({'label': word, 'file': wav_path})
    if not all_words:
      raise Exception('No .wavs found at ' + search_path)
    for index, wanted_word in enumerate(wanted_words):
      if wanted_word not in all_words:
        raise Exception('Expected to find ' + wanted_word +
                        ' in labels but only found ' +
                        ', '.join(all_words.keys()))
    
    # 我们需要一个任意文件作为静音文件的输入来加载。它后来乘以零，所以内容无关紧要都能变为静音数据集。
    
    silence_wav_path = self.data_index['training'][0]['file']
    for set_index in ['validation', 'testing', 'training']:
      set_size = len(self.data_index[set_index])
      silence_size = int(math.ceil(set_size * silence_percentage / 100))
      for _ in range(silence_size):
        self.data_index[set_index].append({
            'label': SILENCE_LABEL,
            'file': silence_wav_path
        })
      
    # 选择一些未知项添加到数据集的训练、验证和测试部分
    
      random.shuffle(unknown_index[set_index])
      unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
      self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
   
    # 确保训练顺序是随机的
    
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])
   
    # 准备结果数据结构的其余部分
    
    self.words_list = prepare_words_list(wanted_words)
    self.word_to_index = {}
    for word in all_words:
      if word in wanted_words_index:
        self.word_to_index[word] = wanted_words_index[word]
      else:
        self.word_to_index[word] = UNKNOWN_WORD_INDEX
    self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

  def prepare_background_data(self):
    """
    在文件夹中搜索背景噪音音频，并将其加载到内存中。
    
    Returns:
     背景噪声的原始PCM编码音频样本列表
     
    """
    self.background_data = []
    background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
    if not os.path.exists(background_dir):
      return self.background_data
    with tf.Session(graph=tf.Graph()) as sess:
      wav_filename_placeholder = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
      search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME,
                                 '*.wav')
      for wav_path in gfile.Glob(search_path):
        wav_data = sess.run(
            wav_decoder,
            feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
        self.background_data.append(wav_data)
      if not self.background_data:
        raise Exception('No background wav files were found in ' + search_path)

  def prepare_processing_graph(self, model_settings, summaries_dir):
    """
    建立张量流图以应用输入失真。
    创建一个图形，加载一个WAVE文件，对其进行解码、缩放体积、平移，
    添加背景噪声，计算一个声谱图，然后从中生成MFCC特征。
    必须在TensorFlow会话运行时调用它，它会创建多个占位符输入和一个输出：:

      - wav_filename_placeholder_: 音频文件名
      - foreground_volume_placeholder_: 主剪辑的声音应该有多大
      - time_shift_padding_placeholder_: 在哪个位置剪辑
      - time_shift_offset_placeholder_: 在剪辑上移动多少
      - background_data_placeholder_: 背景噪声的PCM采样数据
      - background_volume_placeholder_: 背景中混音的响度
      - output_: 经过处理后的二维输出

    Args:
      model_settings: 正在训练的当前模型信息
      summaries_dir: 保存训练摘要信息的路径
      
    """
    with tf.get_default_graph().name_scope('data'):
      desired_samples = model_settings['desired_samples']
      self.wav_filename_placeholder_ = tf.placeholder(
          tf.string, [], name='wav_filename')
      wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
      wav_decoder = contrib_audio.decode_wav(
          wav_loader, desired_channels=1, desired_samples=desired_samples)
     
      #允许调整音频样本的音量
      
      self.foreground_volume_placeholder_ = tf.placeholder(
          tf.float32, [], name='foreground_volume')
      scaled_foreground = tf.multiply(wav_decoder.audio,
                                      self.foreground_volume_placeholder_)
      
      # 移动样本的起始位置，并用零填充任何间隙
      
      self.time_shift_padding_placeholder_ = tf.placeholder(
          tf.int32, [2, 2], name='time_shift_padding')
      self.time_shift_offset_placeholder_ = tf.placeholder(
          tf.int32, [2], name='time_shift_offset')
      padded_foreground = tf.pad(
          scaled_foreground,
          self.time_shift_padding_placeholder_,
          mode='CONSTANT')
      sliced_foreground = tf.slice(padded_foreground,
                                   self.time_shift_offset_placeholder_,
                                   [desired_samples, -1])
      # 混入背景噪音
      self.background_data_placeholder_ = tf.placeholder(
          tf.float32, [desired_samples, 1], name='background_data')
      self.background_volume_placeholder_ = tf.placeholder(
          tf.float32, [], name='background_volume')
      background_mul = tf.multiply(self.background_data_placeholder_,
                                   self.background_volume_placeholder_)
      background_add = tf.add(background_mul, sliced_foreground)
      background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
      
      # 运行频谱图和MFCC节点来获取音频的二维特征
      
      spectrogram = contrib_audio.audio_spectrogram(
          background_clamp,
          window_size=model_settings['window_size_samples'],
          stride=model_settings['window_stride_samples'],
          magnitude_squared=True)
      tf.summary.image(
          'spectrogram', tf.expand_dims(spectrogram, -1), max_outputs=1)
     
      #频谱图中每个FFT行中的桶数将取决于每个窗口中有多少输入样本。
      #不需要详细分类，希望缩小它们以产生更小的结果。
      #一种方法是使用平均法来遍历相邻的bucket，更复杂的方法是应用MFCC算法来缩小表示。
      
      if model_settings['preprocess'] == 'average':
        self.output_ = tf.nn.pool(
            tf.expand_dims(spectrogram, -1),
            window_shape=[1, model_settings['average_window_width']],
            strides=[1, model_settings['average_window_width']],
            pooling_type='AVG',
            padding='SAME')
        tf.summary.image('shrunk_spectrogram', self.output_, max_outputs=1)
      elif model_settings['preprocess'] == 'mfcc':
        self.output_ = contrib_audio.mfcc(
            spectrogram,
            wav_decoder.sample_rate,
            dct_coefficient_count=model_settings['fingerprint_width'])
        tf.summary.image(
            'mfcc', tf.expand_dims(self.output_, -1), max_outputs=1)
      else:
        raise ValueError('Unknown preprocess mode "%s" (should be "mfcc" or'
                         ' "average")' % (model_settings['preprocess']))

      # 合并所有摘要并将其写入/tmp/retrain_日志
      
      self.merged_summaries_ = tf.summary.merge_all(scope='data')
      self.summary_writer_ = tf.summary.FileWriter(summaries_dir + '/data',
                                                   tf.get_default_graph())

  def set_size(self, mode):
    """
    计算数据集每部分的样本数量

    Args:
      mode: 'training', 'validation', or 'testing'，训练、验证和测试三部分

    Returns:
      每部分的样本数量
    """
    return len(self.data_index[mode])

  def get_data(self, how_many, offset, model_settings, background_frequency,
               background_volume_range, time_shift, mode, sess):
    """
    从数据集中收集样本，根据需要应用转换
    当模式为“training”时，将返回随机选择的样本，否则将使用分区中的前N个剪辑。
    这样可以确保验证始终使用相同的样本，从而减少度量中的噪声
    Args:
      how_many: 要返回的所需样本数，-1表示此分区的全部内容
      offset: 确定获取时从何处开始
      model_settings: 目前正在被训练模型的信息
      background_frequency: 有多少音频剪辑将带有背景噪音，可以是0.0-1.0
      background_volume_range: 背景噪音的音量多大
      time_shift: 随机移动音频数据的时间
      mode: 使用训练、验证或者测试模式
      sess: 创建处理器时处于活动状态的TensorFlow会话

    Returns:
      转换样本的样本数据列表，以及标签索引列表
    """
    # 选择样本模式，训练、验证、测试
    
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = max(0, min(how_many, len(candidates) - offset))
    
    # 将填充并返回数据和标签
    
    data = np.zeros((sample_count, model_settings['fingerprint_size']))
    labels = np.zeros(sample_count)
    desired_samples = model_settings['desired_samples']
    use_background = self.background_data and (mode == 'training')
    pick_deterministically = (mode != 'training')
   
    #使用先前创建的处理图重复生成将在训练中使用的最终输出示例数据
    
    for i in xrange(offset, offset + sample_count):
      
        # 选择使用哪个音频样本
      
      if how_many == -1 or pick_deterministically:
        sample_index = i
      else:
        sample_index = np.random.randint(len(candidates))
      sample = candidates[sample_index]
      
      # 如果是时间偏移，设置此样本的偏移量。
      
      if time_shift > 0:
        time_shift_amount = np.random.randint(-time_shift, time_shift)
      else:
        time_shift_amount = 0
      if time_shift_amount > 0:
        time_shift_padding = [[time_shift_amount, 0], [0, 0]]
        time_shift_offset = [0, 0]
      else:
        time_shift_padding = [[0, -time_shift_amount], [0, 0]]
        time_shift_offset = [-time_shift_amount, 0]
      input_dict = {
          self.wav_filename_placeholder_: sample['file'],
          self.time_shift_padding_placeholder_: time_shift_padding,
          self.time_shift_offset_placeholder_: time_shift_offset,
      }
      # 选择要混入的背景噪声区域
      if use_background or sample['label'] == SILENCE_LABEL:
        background_index = np.random.randint(len(self.background_data))
        background_samples = self.background_data[background_index]
        if len(background_samples) <= model_settings['desired_samples']:
          raise ValueError(
              'Background sample is too short! Need more than %d'
              ' samples but only %d were found' %
              (model_settings['desired_samples'], len(background_samples)))
        background_offset = np.random.randint(
            0, len(background_samples) - model_settings['desired_samples'])
        background_clipped = background_samples[background_offset:(
            background_offset + desired_samples)]
        background_reshaped = background_clipped.reshape([desired_samples, 1])
        if sample['label'] == SILENCE_LABEL:
          background_volume = np.random.uniform(0, 1)
        elif np.random.uniform(0, 1) < background_frequency:
          background_volume = np.random.uniform(0, background_volume_range)
        else:
          background_volume = 0
      else:
        background_reshaped = np.zeros([desired_samples, 1])
        background_volume = 0
      input_dict[self.background_data_placeholder_] = background_reshaped
      input_dict[self.background_volume_placeholder_] = background_volume
      
      # 如果是静音模式，将主样本设置为静音，保持背景噪音不变。
      
      if sample['label'] == SILENCE_LABEL:
        input_dict[self.foreground_volume_placeholder_] = 0
      else:
        input_dict[self.foreground_volume_placeholder_] = 1
      
        # 运行图表以生成输出音频
        
        summary, data_tensor = sess.run(
          [self.merged_summaries_, self.output_], feed_dict=input_dict)
      self.summary_writer_.add_summary(summary)
      data[i - offset, :] = data_tensor.flatten()
      label_index = self.word_to_index[sample['label']]
      labels[i - offset] = label_index
    return data, labels

  def get_unprocessed_data(self, how_many, model_settings, mode):
    """
    检索给定分区的示例数据，不进行转换。

    Args:
      how_many: 要返回的所需样本数，-1表示此分区（训练、验证、测试）的全部内容。
      model_settings: 有关正在训练的当前模型的信息。
      mode: 使用哪部分数据, 必须是'training', 'validation', or 'testing'.

    Returns:
      样本数据列表，以及one-hot标签列表。
    """
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = how_many
    desired_samples = model_settings['desired_samples']
    words_list = self.words_list
    data = np.zeros((sample_count, desired_samples))
    labels = []
    with tf.Session(graph=tf.Graph()) as sess:
      wav_filename_placeholder = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = contrib_audio.decode_wav(
          wav_loader, desired_channels=1, desired_samples=desired_samples)
      foreground_volume_placeholder = tf.placeholder(tf.float32, [])
      scaled_foreground = tf.multiply(wav_decoder.audio,
                                      foreground_volume_placeholder)
      for i in range(sample_count):
        if how_many == -1:
          sample_index = i
        else:
          sample_index = np.random.randint(len(candidates))
        sample = candidates[sample_index]
        input_dict = {wav_filename_placeholder: sample['file']}
        if sample['label'] == SILENCE_LABEL:
          input_dict[foreground_volume_placeholder] = 0
        else:
          input_dict[foreground_volume_placeholder] = 1
        data[i, :] = sess.run(scaled_foreground, feed_dict=input_dict).flatten()
        label_index = self.word_to_index[sample['label']]
        labels.append(words_list[label_index])
    return data, labels

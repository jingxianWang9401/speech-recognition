
"""
简单的语音识别模型定义

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


def _next_power_of_two(x):  
  return 1 if x == 0 else 2**(int(x) - 1).bit_length()


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, feature_bin_count,
                           preprocess):
  """
    计算所有模型所需的通用参数设置。              
    参数：              
    label_count：要识别多少个类的音频数据。              
    sample_rate:：每秒音频采样数。              
    clip_duration_ms：要分析的每个音频剪辑的长度。              
    window_size_ms：频率分析窗口的持续时间。              
    window_stride_ms：频率窗口间的时间移动距离。              
    feature_bin_count：用于分析的频率库的数量。              
    preprocess：如何处理声谱图以产生特征。              
    返回：              
    包含常用设置的字典。              
    
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  if preprocess == 'average':
    fft_bin_count = 1 + (_next_power_of_two(window_size_samples) / 2)
    average_window_width = int(math.floor(fft_bin_count / feature_bin_count))
    fingerprint_width = int(math.ceil(fft_bin_count / average_window_width))
  elif preprocess == 'mfcc':
    average_window_width = -1
    fingerprint_width = feature_bin_count
  else:
    raise ValueError('Unknown preprocess mode "%s" (should be "mfcc" or'
                     ' "average")' % (preprocess))
  fingerprint_size = fingerprint_width * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'fingerprint_width': fingerprint_width,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
      'preprocess': preprocess,
      'average_window_width': average_window_width,
  }

  """
    
    构建与设置兼容的模型体系结构：
    
    有许多可能的方法可以从声谱图输入中导出预测，因此此函数提供了一个抽象接口，
    用于以黑盒方式创建不同类型的模型。需要传入一个TensorFlow节点作为“指纹”输入，
    这将输出一批描述音频的1D特征。通常这将从通过MFCC运行的声谱图中导出，
    但理论上它可以是model_settings['fingerprint_size']中指定大小的任意特征向量。
    
    函数将在当前TensorFlow图中生成所需的图，并将包含“logits”输入的TensorFlow输出返回到softmax预测过程。
    如果training flag为on，它还将返回一个占位符节点，可用于控制dropout的数量。
    
    参数：
    fingerprint_input:输出音频特征向量的tensorflow节点
    model_settings:模型的信息词典
    model_architecture:指定建立哪种模型的字符串
    is_training:该模型是否将用于训练
    runtime_settings:有关运行库的信息字典
    
    返回：
    TensorFlow节点输出logits结果，还可以选择一个dropout占位符。  
  
  """
def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None):
  
    
  if model_architecture == 'single_fc':
    return create_single_fc_model(fingerprint_input, model_settings,
                                  is_training)
  elif model_architecture == 'conv':
    return create_conv_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'low_latency_conv':
    return create_low_latency_conv_model(fingerprint_input, model_settings,
                                         is_training)
  elif model_architecture == 'low_latency_svdf':
    return create_low_latency_svdf_model(fingerprint_input, model_settings,
                                         is_training, runtime_settings)
  elif model_architecture == 'tiny_conv':
    return create_tiny_conv_model(fingerprint_input, model_settings,
                                  is_training)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "single_fc", "conv",' +
                    ' "low_latency_conv, "low_latency_svdf",' +
                    ' or "tiny_conv"')



def load_variables_from_checkpoint(sess, start_checkpoint):
  
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)


def create_single_fc_model(fingerprint_input, model_settings, is_training):
 
  """
  (fingerprint_input)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

    构建只含有一个隐含的全连接层。
    参数：              
    fingerprint_input：输出音频特征向量的TensorFlow节点。              
    model_settings：关于模型的信息字典。              
    is_training：该模型是否将用于训练。              
    返回：
    TensorFlow节点输出logits结果，还可以选择一个dropout占位符。   
  """
  
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  weights = tf.get_variable(
      name='weights',
      initializer=tf.truncated_normal_initializer(stddev=0.001),
      shape=[fingerprint_size, label_count])
  bias = tf.get_variable(
      name='bias', initializer=tf.zeros_initializer, shape=[label_count])
  logits = tf.matmul(fingerprint_input, weights) + bias
  if is_training:
    return logits, dropout_prob
  else:
    return logits


def create_conv_model(fingerprint_input, model_settings, is_training):
  
  """
  建立标准卷积层：

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
  涉及大量的权重参数和计算。             
  在训练过程中，每个relu之后都会引入dropout节点，由一个占位符控制。              
  参数：              
  fingerprint_input：输出音频特征向量的TensorFlow节点。              
  model_settings：关于模型的信息字典。              
  is_training：该模型是否将用于训练。
              
  返回：              
  TensorFlow节点输出logits结果，还可以选择一个dropout占位符。
  
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_weights = tf.get_variable(
      name='first_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_filter_height, first_filter_width, 1, first_filter_count])
  first_bias = tf.get_variable(
      name='first_bias',
      initializer=tf.zeros_initializer,
      shape=[first_filter_count])
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_weights = tf.get_variable(
      name='second_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[
          second_filter_height, second_filter_width, first_filter_count,
          second_filter_count
      ])
  second_bias = tf.get_variable(
      name='second_bias',
      initializer=tf.zeros_initializer,
      shape=[second_filter_count])
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                             'SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu
  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(second_dropout,
                                     [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.get_variable(
      name='final_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[second_conv_element_count, label_count])
  final_fc_bias = tf.get_variable(
      name='final_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[label_count])
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_low_latency_conv_model(fingerprint_input, model_settings,
                                  is_training):
  """
  建立低计算要求的卷积模型。

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
  
  比标准“conv”模型稍低的结果，但需要较少的权重参数和计算。              
  在训练过程中，dropout节点在relu之后引入，由占位符控制。

  参数:
    fingerprint_input: 将输出音频特征向量的TensorFlow节点。
    model_settings: 模型的信息字典。
    is_training: 模型是否用于训练。

  返回:
    TensorFlow节点输出logits结果，还可以选择一个dropout占位符。
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = input_time_size
  first_filter_count = 186
  first_filter_stride_x = 1
  first_filter_stride_y = 1
  first_weights = tf.get_variable(
      name='first_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_filter_height, first_filter_width, 1, first_filter_count])
  first_bias = tf.get_variable(
      name='first_bias',
      initializer=tf.zeros_initializer,
      shape=[first_filter_count])
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
      1, first_filter_stride_y, first_filter_stride_x, 1
  ], 'VALID') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  first_conv_output_width = math.floor(
      (input_frequency_size - first_filter_width + first_filter_stride_x) /
      first_filter_stride_x)
  first_conv_output_height = math.floor(
      (input_time_size - first_filter_height + first_filter_stride_y) /
      first_filter_stride_y)
  first_conv_element_count = int(
      first_conv_output_width * first_conv_output_height * first_filter_count)
  flattened_first_conv = tf.reshape(first_dropout,
                                    [-1, first_conv_element_count])
  first_fc_output_channels = 128
  first_fc_weights = tf.get_variable(
      name='first_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_conv_element_count, first_fc_output_channels])
  first_fc_bias = tf.get_variable(
      name='first_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[first_fc_output_channels])
  first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias
  if is_training:
    second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    second_fc_input = first_fc
  second_fc_output_channels = 128
  second_fc_weights = tf.get_variable(
      name='second_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_fc_output_channels, second_fc_output_channels])
  second_fc_bias = tf.get_variable(
      name='second_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[second_fc_output_channels])
  second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
  if is_training:
    final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
  else:
    final_fc_input = second_fc
  label_count = model_settings['label_count']
  final_fc_weights = tf.get_variable(
      name='final_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[second_fc_output_channels, label_count])
  final_fc_bias = tf.get_variable(
      name='final_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[label_count])
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_low_latency_svdf_model(fingerprint_input, model_settings,
                                  is_training, runtime_settings):
  """
  建立一个计算需求低的SVDF模型。              
  基于“压缩深层神经”中的拓扑结构。

  (fingerprint_input)
          v
        [SVDF]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

  这个模型产生的识别精度比上面的“conv”模型低，但需要的权重参数更少，计算量也更少。

  在训练过程中，dropout节点在relu之后引入，由占位符控制。

  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']

  # 验证
  input_shape = fingerprint_input.get_shape()
  if len(input_shape) != 2:
    raise ValueError('Inputs to `SVDF` should have rank == 2.')
  if input_shape[-1].value is None:
    raise ValueError('The last dimension of the inputs to `SVDF` '
                     'should be defined. Found `None`.')
  if input_shape[-1].value % input_frequency_size != 0:
    raise ValueError('Inputs feature dimension %d must be a multiple of '
                     'frame size %d', fingerprint_input.shape[-1].value,
                     input_frequency_size)

  # 设置节点数和等级
  rank = 2
  num_units = 1280
  
  # 过滤器数量：特征和时间过滤器对。
  num_filters = rank * num_units
  
  batch = 1
  memory = tf.get_variable(
      initializer=tf.zeros_initializer,
      shape=[num_filters, batch, input_time_size],
      trainable=False,
      name='runtime-memory')
  if is_training:
    num_new_frames = input_time_size
  else:
    window_stride_ms = int(model_settings['window_stride_samples'] * 1000 /
                           model_settings['sample_rate'])
    num_new_frames = tf.cond(
        tf.equal(tf.count_nonzero(memory), 0),
        lambda: input_time_size,
        lambda: int(runtime_settings['clip_stride_ms'] / window_stride_ms))
  new_fingerprint_input = fingerprint_input[
      :, -num_new_frames*input_frequency_size:]
  # 添加输入通道
  new_fingerprint_input = tf.expand_dims(new_fingerprint_input, 2)

  # 创建频率滤波器
  weights_frequency = tf.get_variable(
      name='weights_frequency',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[input_frequency_size, num_filters])
 
  #添加输入通道维度
    
  weights_frequency = tf.expand_dims(weights_frequency, 1)
  
  #卷积在时间维上滑动的一维特征滤波器
  
  activations_time = tf.nn.conv1d(
      new_fingerprint_input, weights_frequency, input_frequency_size, 'VALID')
  
  activations_time = tf.transpose(activations_time, perm=[2, 0, 1])

  # 运行时内存优化
  if not is_training:
   
      # 删除与最旧帧对应的激活，然后添加与新帧对应的激活
    new_memory = memory[:, :, num_new_frames:]
    new_memory = tf.concat([new_memory, activations_time], 2)
    tf.assign(memory, new_memory)
    activations_time = new_memory

  # 创建时间过滤器
  weights_time = tf.get_variable(
      name='weights_time',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[num_filters, input_time_size])
  
  # 对特征滤波器的输出应用时间滤波器。              
  
  weights_time = tf.expand_dims(weights_time, 2)
  outputs = tf.matmul(activations_time, weights_time)
  
  #将num_units拆分并排列成单独的维度（剩余维度是输入的形状[0]-批大小）。
  # [num_filters, batch, 1] => [num_units, rank, batch]
  
  outputs = tf.reshape(outputs, [num_units, rank, -1])
 
    # 对每个单元的列组输出求和=>[num_units，batch]
    
  units_output = tf.reduce_sum(outputs, axis=1)
  
  # 转换为形状 [batch, num_units]
  
  units_output = tf.transpose(units_output)

  # 应用bias参数
  bias = tf.get_variable(
      name='bias', initializer=tf.zeros_initializer, shape=[num_units])
  first_bias = tf.nn.bias_add(units_output, bias)

  # Relu
  first_relu = tf.nn.relu(first_bias)

  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu

  first_fc_output_channels = 256
  first_fc_weights = tf.get_variable(
      name='first_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[num_units, first_fc_output_channels])
  first_fc_bias = tf.get_variable(
      name='first_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[first_fc_output_channels])
  first_fc = tf.matmul(first_dropout, first_fc_weights) + first_fc_bias
  if is_training:
    second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
  else:
    second_fc_input = first_fc
  second_fc_output_channels = 256
  second_fc_weights = tf.get_variable(
      name='second_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_fc_output_channels, second_fc_output_channels])
  second_fc_bias = tf.get_variable(
      name='second_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[second_fc_output_channels])
  second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
  if is_training:
    final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
  else:
    final_fc_input = second_fc
  label_count = model_settings['label_count']
  final_fc_weights = tf.get_variable(
      name='final_fc_weights',
      initializer=tf.truncated_normal(stddev=0.01),
      shape=[second_fc_output_channels, label_count])
  final_fc_bias = tf.get_variable(
      name='final_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[label_count])
  final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_tiny_conv_model(fingerprint_input, model_settings, is_training):
  """
  针对微控制器建立卷积模型
  
  像DSP和微控制器这样的设备可以有非常小的内存和有限的处理能力。
  这种型号的设计使用小于20KB的工作RAM，并且适合32KB的只读（flash）内存。

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v

 不能产生特别精确的结果，它被设计成作为管道的第一级，
 运行在一个始终可以打开的低能耗硬件上，
 然后在发现声音时唤醒功率更高的芯片。
    
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['fingerprint_width']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = 10
  first_filter_count = 8
  first_weights = tf.get_variable(
      name='first_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_filter_height, first_filter_width, 1, first_filter_count])
  first_bias = tf.get_variable(
      name='first_bias',
      initializer=tf.zeros_initializer,
      shape=[first_filter_count])
  first_conv_stride_x = 2
  first_conv_stride_y = 2
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights,
                            [1, first_conv_stride_y, first_conv_stride_x, 1],
                            'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  first_dropout_shape = first_dropout.get_shape()
  first_dropout_output_width = first_dropout_shape[2]
  first_dropout_output_height = first_dropout_shape[1]
  first_dropout_element_count = int(
      first_dropout_output_width * first_dropout_output_height *
      first_filter_count)
  flattened_first_dropout = tf.reshape(first_dropout,
                                       [-1, first_dropout_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.get_variable(
      name='final_fc_weights',
      initializer=tf.truncated_normal_initializer(stddev=0.01),
      shape=[first_dropout_element_count, label_count])
  final_fc_bias = tf.get_variable(
      name='final_fc_bias',
      initializer=tf.zeros_initializer,
      shape=[label_count])
  final_fc = (
      tf.matmul(flattened_first_dropout, final_fc_weights) + final_fc_bias)
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc

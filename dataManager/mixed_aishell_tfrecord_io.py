import tensorflow as tf
import numpy as np
import librosa
import os
import shutil
import time
import multiprocessing
import copy
import scipy.io
import datetime
import wave
import utils
from utils import audio_tool
from utils import spectrum_tool
from numpy import linalg
from FLAGS import PARAM

FILE_NAME = __file__[max(__file__.rfind('/')+1, 0):__file__.rfind('.')]


def _ini_data(wave_dir, noise_dir, out_dir):
  data_dict_dir = out_dir
  if os.path.exists(data_dict_dir):
    shutil.rmtree(data_dict_dir)
  os.makedirs(data_dict_dir)
  clean_wav_speaker_set_dir = wave_dir
  os.makedirs(data_dict_dir+'/train')
  os.makedirs(data_dict_dir+'/validation')
  os.makedirs(data_dict_dir+'/test_cc')
  cwl_train_file = open(data_dict_dir+'/train/clean_wav_dir.list', 'a+')
  cwl_validation_file = open(
      data_dict_dir+'/validation/clean_wav_dir.list', 'a+')
  cwl_test_cc_file = open(data_dict_dir+'/test_cc/clean_wav_dir.list', 'a+')
  clean_wav_list_train = []
  clean_wav_list_validation = []
  clean_wav_list_test_cc = []
  speaker_list = os.listdir(clean_wav_speaker_set_dir)
  speaker_list.sort()
  for speaker_name in speaker_list:
    speaker_dir = clean_wav_speaker_set_dir+'/'+speaker_name
    if os.path.isdir(speaker_dir):
      speaker_wav_list = os.listdir(speaker_dir)
      speaker_wav_list.sort()
      for wav in speaker_wav_list[:PARAM.UTT_SEG_FOR_MIX[0]]:
        # 清洗长度为0的数据
        if wav[-4:] == ".wav" and os.path.getsize(speaker_dir+'/'+wav) > 2048:
          cwl_train_file.write(speaker_dir+'/'+wav+'\n')
          clean_wav_list_train.append(speaker_dir+'/'+wav)
      for wav in speaker_wav_list[PARAM.UTT_SEG_FOR_MIX[0]:PARAM.UTT_SEG_FOR_MIX[1]]:
        if wav[-4:] == ".wav" and os.path.getsize(speaker_dir+'/'+wav) > 2048:
          cwl_validation_file.write(speaker_dir+'/'+wav+'\n')
          clean_wav_list_validation.append(speaker_dir+'/'+wav)
      for wav in speaker_wav_list[PARAM.UTT_SEG_FOR_MIX[1]:]:
        if wav[-4:] == ".wav" and os.path.getsize(speaker_dir+'/'+wav) > 2048:
          cwl_test_cc_file.write(speaker_dir+'/'+wav+'\n')
          clean_wav_list_test_cc.append(speaker_dir+'/'+wav)

  cwl_train_file.close()
  cwl_validation_file.close()
  cwl_test_cc_file.close()
  print('train clean: '+str(len(clean_wav_list_train)))
  print('validation clean: '+str(len(clean_wav_list_validation)))
  print('test_cc clean: '+str(len(clean_wav_list_test_cc)))

  # NOISE LIST
  noise_wav_list = os.listdir(noise_dir)
  noise_wav_list = [os.path.join(noise_dir, noise) for noise in noise_wav_list]

  dataset_names = PARAM.DATASET_NAMES
  dataset_mixedutt_num = PARAM.DATASET_SIZES
  all_mixed = 0
  all_stime = time.time()
  for (clean_wav_list, j) in zip((clean_wav_list_train, clean_wav_list_validation, clean_wav_list_test_cc), range(3)):
    print('\n'+dataset_names[j]+" data preparing...")
    s_time = time.time()
    mixed_wav_list_file = open(
        data_dict_dir+'/'+dataset_names[j]+'/mixed_wav_dir.list', 'a+')
    mixed_wave_list = []
    len_wav_list = len(clean_wav_list)
    len_noise_wave_list = len(noise_wav_list)
    # print(len_wav_list,len_noise_wave_list)
    generated_num = 0
    while generated_num < dataset_mixedutt_num[j]:
      uttid = np.random.randint(len_wav_list)
      noiseid = np.random.randint(len_noise_wave_list)
      utt1_dir = clean_wav_list[uttid]
      utt2_dir = noise_wav_list[noiseid]
      generated_num += 1
      mixed_wav_list_file.write(utt1_dir+' '+utt2_dir+'\n')
      mixed_wave_list.append([utt1_dir, utt2_dir])
    # for i_utt in range(len_wav_list): # n^2混合，数据量巨大
    #   for j_utt in range(i_utt,len_wav_list):
    #     utt1_dir=clean_wav_list[i_utt]
    #     utt2_dir=clean_wav_list[j_utt]
    #     speaker1 = utt1_dir.split('/')[-2]
    #     speaker2 = utt2_dir.split('/')[-2]
    #     if speaker1 == speaker2:
    #       continue
    #     mixed_wav_list_file.write(utt1_dir+' '+utt2_dir+'\n')
    #     mixed_wave_list.append([utt1_dir, utt2_dir])
    mixed_wav_list_file.close()
    scipy.io.savemat(
        data_dict_dir+'/'+dataset_names[j]+'/mixed_wav_dir.mat', {"mixed_wav_dir": mixed_wave_list})
    all_mixed += len(mixed_wave_list)
    print(dataset_names[j]+' data preparation over, Mixed num: ' +
          str(len(mixed_wave_list))+(', Cost time %dS.') % (time.time()-s_time))
  print('\nData preparation over, all mixed num: %d,cost time: %dS' %
        (all_mixed, time.time()-all_stime))


def _get_padad_waveData(file):
  waveData, sr = audio_tool.read_audio(file)
  waveData *= 32767.0
  if(sr != PARAM.FS):
    print("Audio samplerate error.")
    exit(-1)

  while len(waveData) < PARAM.LEN_WAWE_PAD_TO:
    waveData = np.tile(waveData, 2)

  len_wave = len(waveData)
  wave_begin = np.random.randint(len_wave-PARAM.LEN_WAWE_PAD_TO+1)
  waveData = waveData[wave_begin:wave_begin+PARAM.LEN_WAWE_PAD_TO]

  return waveData


def _mix_wav_by_SNR(waveData, noise):
  # S = (speech+alpha*noise)/(1+alpha)
  snr = np.random.randint(PARAM.MIN_SNR, PARAM.MAX_SNR+1)
  As = linalg.norm(waveData)
  An = linalg.norm(noise)

  alpha = As/(An*(10**(snr/20))) if An != 0 else 0
  waveMix = (waveData+alpha*noise)/(1.0+alpha)
  return waveMix


def _mix_wav_LINEAR(waveData, noise):
  coef = np.random.random()*(PARAM.MAX_COEF-PARAM.MIN_COEF)+PARAM.MIN_COEF
  waveMix = (waveData+coef*noise)/(1.0+coef)
  return waveMix


def _extract_mag_spec(data):
  # 幅度谱
  mag_spec = spectrum_tool.magnitude_spectrum_librosa_stft(
      data, PARAM.NFFT, PARAM.OVERLAP)
  return mag_spec


def _extract_phase(data):
  theta = spectrum_tool.phase_spectrum_librosa_stft(data,
                                                    PARAM.NFFT,
                                                    PARAM.OVERLAP)
  return theta


def _extract_feature_x_y_xtheta_ytheta(utt_dir1, utt_dir2):
  waveData1 = _get_padad_waveData(utt_dir1)
  waveData2 = _get_padad_waveData(utt_dir2)
  # utt2作为噪音
  if PARAM.MIX_METHOD == 'SNR':
    mixedData = _mix_wav_by_SNR(waveData1, waveData2)
  if PARAM.MIX_METHOD == 'LINEAR':
    mixedData = _mix_wav_LINEAR(waveData1, waveData2)

  # write mixed wav
  # name1 = utt_dir1[utt_dir1.rfind('/')+1:utt_dir1.rfind('.')]
  # name2 = utt_dir2[utt_dir2.rfind('/')+1:]
  # utils.audio_tool.write_audio('mixwave/mixed_'+name1+"_"+name2,
  #                              mixedData,16000,16,'wav')

  X = _extract_mag_spec(mixedData)
  Y = _extract_mag_spec(waveData1)
  x_theta = _extract_phase(mixedData)
  y_theta = _extract_phase(waveData1)

  return [X, Y, x_theta, y_theta]


def parse_func(example_proto):
  sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(shape=[PARAM.INPUT_SIZE],
                                           dtype=tf.float32),
      'labels': tf.FixedLenSequenceFeature(shape=[PARAM.OUTPUT_SIZE],
                                           dtype=tf.float32),
      'xtheta': tf.FixedLenSequenceFeature(shape=[PARAM.INPUT_SIZE],
                                           dtype=tf.float32),
      'ytheta': tf.FixedLenSequenceFeature(shape=[PARAM.OUTPUT_SIZE],
                                           dtype=tf.float32),
  }
  _, sequence = tf.parse_single_sequence_example(
      example_proto, sequence_features=sequence_features)
  length = tf.shape(sequence['inputs'])[0]
  return sequence['inputs'], sequence['labels'], 0, 0, length


def parse_func_with_theta(example_proto):
  sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(shape=[PARAM.INPUT_SIZE],
                                           dtype=tf.float32),
      'labels': tf.FixedLenSequenceFeature(shape=[PARAM.OUTPUT_SIZE],
                                           dtype=tf.float32),
      'xtheta': tf.FixedLenSequenceFeature(shape=[PARAM.INPUT_SIZE],
                                           dtype=tf.float32),
      'ytheta': tf.FixedLenSequenceFeature(shape=[PARAM.OUTPUT_SIZE],
                                           dtype=tf.float32),
  }
  _, sequence = tf.parse_single_sequence_example(
      example_proto, sequence_features=sequence_features)
  length = tf.shape(sequence['inputs'])[0]
  return sequence['inputs'], sequence['labels'], sequence['xtheta'], sequence['ytheta'], length


def _gen_tfrecord_minprocess(
        dataset_index_list, s_site, e_site, dataset_dir, i_process):
  tfrecord_savedir = os.path.join(dataset_dir, ('%08d.tfrecords' % i_process))
  with tf.python_io.TFRecordWriter(tfrecord_savedir) as writer:
    for i in range(s_site, e_site):
      index_ = dataset_index_list[i]
      X_Y_Xtheta_Ytheta = _extract_feature_x_y_xtheta_ytheta(index_[0],
                                                             index_[1])
      X = np.reshape(np.array(X_Y_Xtheta_Ytheta[0], dtype=np.float32),
                     newshape=[-1, PARAM.INPUT_SIZE])
      Y = np.reshape(np.array(X_Y_Xtheta_Ytheta[1], dtype=np.float32),
                     newshape=[-1, PARAM.OUTPUT_SIZE])
      Xtheta = np.reshape(np.array(X_Y_Xtheta_Ytheta[2], dtype=np.float32),
                          newshape=[-1, PARAM.INPUT_SIZE])
      Ytheta = np.reshape(np.array(X_Y_Xtheta_Ytheta[3], dtype=np.float32),
                          newshape=[-1, PARAM.OUTPUT_SIZE])
      # print(np.mean(X),np.sqrt(np.var(X)),np.median(X),np.max(X),np.min(X))
      # print(np.mean(X),np.sqrt(np.var(X)),np.median(X),np.max(Y),np.min(Y))
      input_features = [
          tf.train.Feature(float_list=tf.train.FloatList(value=input_))
          for input_ in X]
      label_features = [
          tf.train.Feature(float_list=tf.train.FloatList(value=label))
          for label in Y]
      xtheta_features = [
          tf.train.Feature(float_list=tf.train.FloatList(value=xtheta))
          for xtheta in Xtheta]
      ytheta_features = [
          tf.train.Feature(float_list=tf.train.FloatList(value=ytheta))
          for ytheta in Ytheta]
      feature_list = {
          'inputs': tf.train.FeatureList(feature=input_features),
          'labels': tf.train.FeatureList(feature=label_features),
          'xtheta': tf.train.FeatureList(feature=xtheta_features),
          'ytheta': tf.train.FeatureList(feature=ytheta_features),
      }
      feature_lists = tf.train.FeatureLists(feature_list=feature_list)
      record = tf.train.SequenceExample(feature_lists=feature_lists)
      writer.write(record.SerializeToString())
    writer.flush()
    # print(dataset_dir + ('/%08d.tfrecords' % i), 'write done')


def generate_tfrecord(gen=True):
  tfrecords_dir = PARAM.TFRECORDS_DIR
  train_tfrecords_dir = os.path.join(tfrecords_dir, 'train')
  val_tfrecords_dir = os.path.join(tfrecords_dir, 'validation')
  testcc_tfrecords_dir = os.path.join(tfrecords_dir, 'test_cc')
  dataset_dir_list = [train_tfrecords_dir,
                      val_tfrecords_dir,
                      testcc_tfrecords_dir]

  if gen:
    _ini_data(PARAM.RAW_DATA, PARAM.NOISE_DIR, PARAM.DATA_DICT_DIR)
    if os.path.exists(train_tfrecords_dir):
      shutil.rmtree(train_tfrecords_dir)
    if os.path.exists(val_tfrecords_dir):
      shutil.rmtree(val_tfrecords_dir)
    if os.path.exists(testcc_tfrecords_dir):
      shutil.rmtree(testcc_tfrecords_dir)
    os.makedirs(train_tfrecords_dir)
    os.makedirs(val_tfrecords_dir)
    os.makedirs(testcc_tfrecords_dir)

    gen_start_time = time.time()
    pool = multiprocessing.Pool(PARAM.PROCESS_NUM_GENERATE_TFERCORD)
    for dataset_dir in dataset_dir_list:
      # start_time = time.time()
      dataset_index_list = None
      if dataset_dir[-2:] == 'in':
        # continue
        dataset_index_list = scipy.io.loadmat(
            '_data/mixed_aishell/train/mixed_wav_dir.mat')["mixed_wav_dir"]
      elif dataset_dir[-2:] == 'on':
        dataset_index_list = scipy.io.loadmat(
            '_data/mixed_aishell/validation/mixed_wav_dir.mat')["mixed_wav_dir"]
      elif dataset_dir[-2:] == 'cc':
        dataset_index_list = scipy.io.loadmat(
            '_data/mixed_aishell/test_cc/mixed_wav_dir.mat')["mixed_wav_dir"]

      # 使用.mat，字符串长度会强制对齐，所以去掉空格
      dataset_index_list = [[index_[0].replace(' ', ''),
                             index_[1].replace(' ', '')] for index_ in dataset_index_list]
      len_dataset = len(dataset_index_list)
      minprocess_utt_num = int(
          len_dataset/PARAM.TFRECORDS_NUM)
      for i_process in range(PARAM.TFRECORDS_NUM):
        s_site = i_process*minprocess_utt_num
        e_site = s_site+minprocess_utt_num
        if i_process == (PARAM.TFRECORDS_NUM-1):
          e_site = len_dataset
        # print(s_site,e_site)
        pool.apply_async(_gen_tfrecord_minprocess,
                         (dataset_index_list,
                          s_site,
                          e_site,
                          dataset_dir,
                          i_process))
        # _gen_tfrecord_minprocess(dataset_index_list,
        #                          s_site,
        #                          e_site,
        #                          dataset_dir,
        #                          i_process)

      # print(dataset_dir+' set extraction over. cost time %06dS' %
      #       (time.time()-start_time))
    pool.close()
    pool.join()
    print('Generate TFRecord over. cost time %06dS' %
          (time.time()-gen_start_time))

  train_set = os.path.join(train_tfrecords_dir, '*.tfrecords')
  val_set = os.path.join(val_tfrecords_dir, '*.tfrecords')
  testcc_set = os.path.join(testcc_tfrecords_dir, '*.tfrecords')
  return train_set, val_set, testcc_set


def get_batch_use_tfdata(tfrecords_list, get_theta=False):
  files = tf.data.Dataset.list_files(tfrecords_list)
  files = files.take(PARAM.MAX_TFRECORD_FILES_USED)
  if PARAM.SHUFFLE:
    files = files.shuffle(PARAM.PROCESS_NUM_GENERATE_TFERCORD)
  if not PARAM.SHUFFLE:
    dataset = files.interleave(tf.data.TFRecordDataset,
                               cycle_length=1,
                               block_length=PARAM.batch_size,
                               #  num_parallel_calls=1,
                               )
  else:  # shuffle
    dataset = files.interleave(tf.data.TFRecordDataset,
                               cycle_length=PARAM.batch_size*3,
                               #  block_length=1,
                               num_parallel_calls=PARAM.num_threads_processing_data,
                               )
  if PARAM.SHUFFLE:
    dataset = dataset.shuffle(PARAM.batch_size*3)
  # region
  # !tf.data with tf.device(cpu) OOM???
  # dataset = dataset.map(
  #     map_func=parse_func,
  #     num_parallel_calls=NNET_PARAM.num_threads_processing_data)
  # dataset = dataset.padded_batch(
  #     NNET_PARAM.batch_size,
  #     padded_shapes=([None, NNET_PARAM.INPUT_SIZE],
  #                    [None, NNET_PARAM.OUTPUT_SIZE],
  #                    [None, NNET_PARAM.OUTPUT_SIZE],
  #                    []))
  # endregion
  # !map_and_batch efficient is better than map+paded_batch
  dataset = dataset.apply(tf.data.experimental.map_and_batch(
      map_func=parse_func_with_theta if get_theta else parse_func,
      batch_size=PARAM.batch_size,
      num_parallel_calls=PARAM.num_threads_processing_data,
      # num_parallel_batches=2,
  ))
  # dataset = dataset.prefetch(buffer_size=NNET_PARAM.batch_size) # perfetch 太耗内存，并没有明显的速度提升
  dataset_iter = dataset.make_initializable_iterator()
  x_batch, y_batch, xtheta, ytheta, lengths_batch = dataset_iter.get_next()
  return x_batch, y_batch, xtheta, ytheta, lengths_batch, dataset_iter


def _get_batch_use_tfdata(tfrecords_list, get_theta=False):
  files = os.listdir(tfrecords_list[:-11])
  files = files[:min(PARAM.MAX_TFRECORD_FILES_USED, len(files))]
  files = [os.path.join(tfrecords_list[:-11], file) for file in files]
  dataset_list = [tf.data.TFRecordDataset(file).map(parse_func_with_theta if get_theta else parse_func,
                                                    num_parallel_calls=PARAM.num_threads_processing_data) for file in files]

  num_classes = PARAM.MAX_TFRECORD_FILES_USED
  num_classes_per_batch = PARAM.batch_size
  num_utt_per_class = PARAM.batch_size//num_classes_per_batch

  def generator(_):
    # Sample `num_classes_per_batch` classes for the batch
    sampled = tf.random_shuffle(tf.range(num_classes))[:num_classes_per_batch]
    # Repeat each element `num_images_per_class` times
    batch_labels = tf.tile(tf.expand_dims(sampled, -1), [1, num_utt_per_class])
    return tf.to_int64(tf.reshape(batch_labels, [-1]))

  selector = tf.contrib.data.Counter().map(generator)
  selector = selector.apply(tf.contrib.data.unbatch())

  dataset = tf.data.experimental.choose_from_datasets(dataset_list, selector)
  dataset = dataset.batch(num_classes_per_batch * num_utt_per_class)
  # dataset = dataset.prefetch(buffer_size=NNET_PARAM.batch_size) # perfetch 太耗内存，并没有明显的速度提升
  dataset_iter = dataset.make_initializable_iterator()
  x_batch, y_batch, xtheta, ytheta, lengths_batch = dataset_iter.get_next()
  return x_batch, y_batch, xtheta, ytheta, lengths_batch, dataset_iter

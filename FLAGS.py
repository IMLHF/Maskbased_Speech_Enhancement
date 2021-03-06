from losses import loss


class base_config:
  NFFT = 512
  OVERLAP = 256
  FS = 16000
  AUDIO_BITS = 16
  MASK_TYPE = "PSM"  # "PSM" or "IRM"
  LOSS_FUNC = loss.reduce_sum_frame_batchsize_MSE_LOW_FS_IMPROVE
  MODEL_TYPE = "BLSTM"  # "BLSTM" OR "BGRU"
  INPUT_SIZE = 257
  OUTPUT_SIZE = 257
  LSTM_num_proj = 128
  RNN_SIZE = 512
  LSTM_ACTIVATION = 'tanh'
  KEEP_PROB = 0.8
  RNN_LAYER = 2
  CLIP_NORM = 5.0
  SAVE_DIR = 'exp/rnn_speech_enhancement'
  CHECK_POINT = None

  GPU_RAM_ALLOW_GROWTH = True
  batch_size = 128
  learning_rate = 0.001
  start_halving_impr = 0.0003
  resume_training = 'false'  # set start_epoch = final model ID
  start_epoch = 0
  min_epochs = 10  # Min number of epochs to run trainer without halving.
  max_epochs = 50  # Max number of epochs to run trainer totally.
  halving_factor = 0.7  # Factor for halving.
  # Halving when ralative loss is lower than start_halving_impr.
  start_halving_impr = 0.003
  # Stop when relative loss is lower than end_halving_impr.
  end_halving_impr = 0.0005
  # The num of threads to read tfrecords files.
  num_threads_processing_data = 16
  RESTORE_PHASE = 'GRIFFIN_LIM'  # 'MIXED','GRIFFIN_LIM'.
  GRIFFIN_ITERNUM = 50
  minibatch_size = 400  # batch num to show
  RAW_DATA = '/home/room/work/lhf/alldata/aishell2_speaker_list'
  # NOISE_DIR = '/all_data/many_noise' # for docker
  NOISE_DIR = '/home/room/work/lhf/alldata/many_noise'
  DATA_DICT_DIR = '_data/mixed_aishell'
  # TFRECORDS_DIR = '/all_data/feature_tfrecords' # for docker
  TFRECORDS_DIR = '/home/room/work/lhf/alldata/irm_data/feature_tfrecords_utt03s_irm_span32767'
  GENERATE_TFRECORD = False
  PROCESS_NUM_GENERATE_TFERCORD = 16
  TFRECORDS_NUM = 160  # 提多少，后面设置MAX_TFRECORD_FILES_USED表示用多少
  MAX_TFRECORD_FILES_USED = 160  # <=TFRECORDS_NUM
  SHUFFLE = False

  LEN_WAWE_PAD_TO = FS*3  # Mixed wave length (FS*3 is 3 seconds)
  # [260,290] Separate utt to [0:260],[260,290],[290:end]
  UTT_SEG_FOR_MIX = [400, 460]
  DATASET_NAMES = ['train', 'validation', 'test_cc']
  DATASET_SIZES = [600000, 18000, 100000]

  INPUT_TYPE = None  # 'mag' or 'logmag'
  LABEL_TYPE = None  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = None  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = None  # should be same to TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 0
  LOG_BIAS_TRAINABEL = False
  #LOG_NORM_MAX = log(LOG_BIAS+MAG_NORM_MAX)
  #LOG_NORM_MIN = log(LOG_BIAS)
  # MAG_NORM_MAX = 70
  MAG_NORM_MAX = 5e5
  # MAG_NORM_MIN = 0

  AUDIO_VOLUME_AMP = False

  MIX_METHOD = 'LINEAR'  # "LINEAR" "SNR"
  MAX_SNR = 9  # 以不同信噪比混合
  MIN_SNR = -6
  #MIX_METHOD = "LINEAR"
  MAX_COEF = 1.0  # 以不同系数混合
  MIN_COEF = 0

class CXX12(base_config): # 原名C12，使用对数谱输入和作为LOSS，增加了log_bias （因为有logbias 所以效果好，待验证） TODO
  CHECK_POINT = 'nnet_C12_autobias'
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'logmag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 50-1e-10
  '''
  iter15 PESQ:
    avg_score_raw: 1.202100,
    avg_score_en: 2.341708,
    imp: 1.139608
  '''


class C001(base_config): #
  CHECK_POINT = 'nnet_C001'
  INPUT_TYPE = 'mag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  '''
  iter4 PESQ:
    avg_score_raw: 1.202100,
    avg_score_en: 2.237535,
    imp: 1.035436
  iter10 PESQ:
    avg_score_raw: 1.202100,
    avg_score_en: 2.324778,
    imp: 1.122679
  '''


class C002(base_config):  #
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'log_mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 0


class C003(base_config):
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'logmag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 0

# 002和003对比看mask最佳位置，001和003对比看对数谱和幅度谱哪个做loss较好

class CXX(base_config):  #
  INPUT_TYPE = 'logmag'  # 'mag' or 'logmag'
  LABEL_TYPE = 'mag'  # 'mag' or 'logmag'
  TRAINING_MASK_POSITION = 'mag'  # 'mag' or 'logmag'
  DECODING_MASK_POSITION = TRAINING_MASK_POSITION
  INIT_LOG_BIAS = 0
  LOG_BIAS_TRAINABEL = True


PARAM = C001
# print(PARAM.TRAINING_MASK_POSITION != PARAM.LABEL_TYPE)

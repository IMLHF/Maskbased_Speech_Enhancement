import time
import tensorflow as tf
import numpy as np
import sys
import utils
import utils.audio_tool
import os
from models.lstm_SE import SE_MODEL
import wave
import gc
from utils import spectrum_tool
from FLAGS import PARAM
from utils import audio_tool


def build_session(ckpt_dir=PARAM.CHECK_POINT):
  g = tf.Graph()
  with g.as_default():
    with tf.device('/cpu:0'):
      with tf.name_scope('input'):
        x_batch = tf.placeholder(tf.float32,shape=[1,None,PARAM.INPUT_SIZE],name='x_batch')
        lengths_batch = tf.placeholder(tf.int32,shape=[1],name='lengths_batch')
    with tf.name_scope('model'):
      model = SE_MODEL(x_batch,
                       lengths_batch,
                       infer=True)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(
        os.path.join(PARAM.SAVE_DIR, ckpt_dir))
    if ckpt and ckpt.model_checkpoint_path:
      tf.logging.info("Restore from " + ckpt.model_checkpoint_path)
      model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      tf.logging.fatal("checkpoint not found.")
      sys.exit(-1)
  g.finalize()
  return sess,model


def decode_one_wav(sess, model, wavedata):
  x_spec_t = spectrum_tool.magnitude_spectrum_librosa_stft(wavedata,
                                                           PARAM.NFFT,
                                                           PARAM.OVERLAP)
  length = np.shape(x_spec_t)[0]
  x_spec = np.array([x_spec_t], dtype=np.float32)
  lengths = np.array([length], dtype=np.int32)
  cleaned, mask, x_mag, norm_x_mag, norm_logmag = sess.run(
      [model.cleaned, model.mask,
       model._x_mag_spec, model._norm_x_mag_spec, model._norm_x_logmag_spec],
      feed_dict={
        model.inputs: x_spec,
        model.lengths: lengths,
      })

  cleaned = np.array(cleaned)
  cleaned_spec = utils.spectrum_tool.griffin_lim(cleaned, wavedata,
                                                 PARAM.NFFT,
                                                 PARAM.OVERLAP,
                                                 PARAM.GRIFFIN_ITERNUM)

  reY = utils.spectrum_tool.librosa_istft(
      cleaned_spec, PARAM.NFFT, PARAM.OVERLAP)

  # print(np.shape(mask), np.max(mask), np.min(mask))
  # print(np.shape(x_mag), np.max(x_mag), np.min(x_mag))
  # print(np.shape(norm_x_mag), np.max(norm_x_mag), np.min(norm_x_mag))
  # print(np.shape(norm_logmag), np.max(norm_logmag), np.min(norm_logmag))
  # spectrum_tool.picture_spec(mask[0],"233")

  return reY

if __name__=='__main__':
  waveData, sr = audio_tool.read_audio('../IRM_Speech_Enhancement/_decode_index/speech0_16k.wav')
  waveData *= 32767.0

  sess, model = build_session()
  reY = decode_one_wav(sess,model,waveData)

  reY /= 32767.0
  utils.audio_tool.write_audio('restore_audio2.wav',
                               reY,
                               sr,
                               PARAM.AUDIO_BITS, 'wav')

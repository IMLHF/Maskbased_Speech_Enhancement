import os
import soundfile as sf
import numpy as np
import shutil
from pypesq import pesq
from decoder import build_session, decode_one_wav
from utils import spectrum_tool

def mix_wav_by_SNR(waveData, noise, snr):
  # S = (speech+alpha*noise)/(1+alpha)
  # snr = np.random.randint(DATA_PARAM.MIN_SNR, DATA_PARAM.MAX_SNR+1)
  As = np.mean(waveData**2)
  An = np.mean(noise**2)

  alpha_pow = As/(An*(10**(snr/10))) if An != 0 else 0
  alpha = np.sqrt(alpha_pow)
  waveMix = (waveData+alpha*noise)/(1.0+alpha)
  return waveMix

def repeat_to_len(wave, repeat_len):
  while len(wave) < repeat_len:
    wave = np.tile(wave, 2)
  wave = wave[0:repeat_len]
  return wave


def gen_clean_speech(sess,model_):
  if os.path.exists('exp/data_for_ac/speech_for_ac_en'):
    shutil.rmtree('exp/data_for_ac/speech_for_ac_en')
    os.makedirs('exp/data_for_ac/speech_for_ac_en')

  wav_list = os.listdir('exp/data_for_ac/speech_for_ac')
  wav_list = [os.path.join('exp/data_for_ac/speech_for_ac',wav_name) for wav_name in wav_list]
  wav_list.sort()

  for wav_dir in wav_list:
    wave, sr = sf.read(wav_dir)
    enhanced_ = decode_one_wav(sess, model_, wave*32767)/32767
    en_dir = wav_dir.replace('speech_for_ac','speech_for_ac_en')
    if en_dir.rfind('.') - en_dir.rfind('_') <= 2:
      en_dir = en_dir[::-1].replace('_','0_',1)[::-1]
    sf.write(en_dir,enhanced_,sr,subtype="PCM_16",format='wav')


def en_speech(noise_dir_name, speech_dir_name, name, snr, sess,model_):
  noises_dir = 'exp/data_for_ac/'+noise_dir_name
  noise_list = os.listdir(noises_dir)
  noise_dir_list = [os.path.join(noises_dir, noise_file) for noise_file in noise_list]
  noise_dir_list.sort()

  speeches_dir = 'exp/data_for_ac/'+speech_dir_name
  speech_list = os.listdir(speeches_dir)
  speech_dir_list = [os.path.join(speeches_dir, speech_file) for speech_file in speech_list]
  speech_dir_list.sort()

  # print(noise_dir_list)
  # print(speech_dir_list)

  mixed_waves_dir = 'exp/data_for_ac/'+name+'/mixed_wav'
  if not os.path.exists(mixed_waves_dir):
    os.makedirs(mixed_waves_dir)
  else:
    shutil.rmtree(mixed_waves_dir)
    os.makedirs(mixed_waves_dir)

  enhanced_waves_dir = 'exp/data_for_ac/'+name+'/enhanced_wav'
  if not os.path.exists(enhanced_waves_dir):
    os.makedirs(enhanced_waves_dir)
  else:
    shutil.rmtree(enhanced_waves_dir)
    os.makedirs(enhanced_waves_dir)

  for speech_dir in speech_dir_list:
    for noise_dir in noise_dir_list:
      speech_name = speech_dir[speech_dir.rfind('/')+1:speech_dir.find('.')]
      noise_name = noise_dir[noise_dir.rfind('/')+1:noise_dir.find('.')]
      speech_wave, sr_s = sf.read(speech_dir)
      noise_wave, sr_n = sf.read(noise_dir)
      noise_wave = repeat_to_len(noise_wave, len(speech_wave))
      if sr_s != sr_n:
        print('sr error', sr_s, sr_n)
        exit(-1)
      mixed_wave = mix_wav_by_SNR(speech_wave, noise_wave, snr)
      # mixed_wave = (speech_wave + noise_wave) /2
      sf.write(os.path.join(mixed_waves_dir, speech_name+'_MIX_'+noise_name+'.wav'),
               mixed_wave,
               samplerate=sr_n,
               subtype='PCM_16',
               format='wav')
      enhanced_wave = decode_one_wav(sess, model_, mixed_wave*32767)/32767
      sf.write(os.path.join(enhanced_waves_dir, speech_name+'_MIX_'+noise_name+'_enhanced.wav'),
               enhanced_wave,
               samplerate=sr_n,
               subtype='PCM_16',
               format='wav')
      print(enhanced_waves_dir+'/'+speech_name+'_MIX_'+noise_name)
  return len(noise_dir_list)


def getPESQ(name,clean_speech_dir,noise_num):
  clean_waves_dir = 'exp/data_for_ac/'+clean_speech_dir
  clean_list = os.listdir(clean_waves_dir)
  clean_dir_list = [os.path.join(clean_waves_dir, clean_file) for clean_file in clean_list]
  clean_dir_list.sort()
  enhanced_waves_dir = 'exp/data_for_ac/'+name+'/enhanced_wav'
  enhanced_list = os.listdir(enhanced_waves_dir)
  enhanced_dir_list = [os.path.join(enhanced_waves_dir, enhanced_file) for enhanced_file in enhanced_list]
  enhanced_dir_list.sort()
  mixed_waves_dir = 'exp/data_for_ac/'+name+'/mixed_wav'
  mixed_list = os.listdir(mixed_waves_dir)
  mixed_dir_list = [os.path.join(mixed_waves_dir, mixed_file) for mixed_file in mixed_list]
  mixed_dir_list.sort()

  clean_dir_list_long = []
  for clean_dir in clean_dir_list:
    for i in range(noise_num):
      clean_dir_list_long.append(clean_dir)

  avg_score_raw = 0.0
  avg_score_en = 0.0
  avg_score_imp = 0.0
  i=0
  for clean_wave, enhanced_wave, mixed_wave in zip(clean_dir_list_long, enhanced_dir_list, mixed_dir_list):
    ref, sr = sf.read(clean_wave)
    mixed, sr = sf.read(mixed_wave)
    enhanced, sr = sf.read(enhanced_wave)

    # spec = spectrum_tool.magnitude_spectrum_librosa_stft(enhanced,512,256)
    # angle = spectrum_tool.phase_spectrum_librosa_stft(enhanced,512,256)
    # spec = spec ** 1.3
    # enhanced = spectrum_tool.librosa_istft(spec*np.exp(angle*1j),512,256)

    # score_raw = pesq(ref/np.max(np.abs(ref)), mixed/np.max(np.abs(mixed)), sr)
    # score_en = pesq(ref/np.max(np.abs(ref)), enhanced/np.max(np.abs(enhanced)), sr)
    score_raw = pesq(ref, mixed, sr)
    score_en = pesq(ref, enhanced, sr)
    print(str(i % noise_num + 1)+"_score_raw, score_en:",score_raw, score_en)
    i+=1
    avg_score_raw += score_raw
    avg_score_en += score_en
    avg_score_imp += (score_en - score_raw)


  avg_score_raw /= len(clean_dir_list_long)
  avg_score_en /=len(clean_dir_list_long)
  avg_score_imp /=len(clean_dir_list_long)
  print('avg_score_raw: %f,\navg_score_en: %f,\nimp: %f' % (avg_score_raw,
                                                            avg_score_en,
                                                            avg_score_imp))

if __name__ == '__main__':
  name = 'nnet_C001'
  sess, model_ = build_session(name)
  # gen_clean_speech(sess,model_) # 先去除refer语音的底噪
  # print('clean_speech_done')
  noise_num = en_speech('noise_for_ac','speech_for_ac_en',name,0,sess,model_) # 生成增强结果
  getPESQ(name,'speech_for_ac_en', noise_num) # 计算PESQ Imp


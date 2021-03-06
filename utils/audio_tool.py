import soundfile
import numpy as np

'''
soundfile.info(file, verbose=False)
soundfile.available_formats()
soundfile.available_subtypes(format=None)
soundfile.read(file, frames=-1, start=0, stop=None, dtype='float64', always_2d=False, fill_value=None, out=None, samplerate=None, channels=None, format=None, subtype=None, endian=None, closefd=True)
soundfile.write(file, data, samplerate, subtype=None, endian=None, format=None, closefd=True)
'''


def write_audio(file, data, sr, bits, _format='wav'):
  subtype = {
      8: 'PCM_S8',
      16: 'PCM_16',
      24: 'PCM_24'
  }[bits]
  # -1.0 < data < 1.0, data.type=float
  return soundfile.write(file, data, sr, subtype=subtype, format=_format)


def read_audio(file):
  data, sr = soundfile.read(file)
  return data, sr

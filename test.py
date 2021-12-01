from a4.utils import pad_sents as f1
import importlib
import time

def testUtilsPadents():
  f = importlib.import_module("[finished]Assignment_4_neural_machine_translation_s2s_attention.utils")
  f2 = f.pad_sents 
  sents1 = [['a','a','a','a'],['a'],['a','a'],['a','a','a','a'],['a'],['a','a'],['a','a','a','a'],['a'],['a','a'],['a','a','a','a'],['a'],['a','a'],['a','a','a','a'],['a'],['a','a'],['a','a','a','a'],['a'],['a','a'],['a','a','a','a'],['a'],['a','a']]
  sents2 = [['a','a','a','a'],['a'],['a','a'],['a','a','a','a'],['a'],['a','a'],['a','a','a','a'],['a'],['a','a'],['a','a','a','a'],['a'],['a','a'],['a','a','a','a'],['a'],['a','a'],['a','a','a','a'],['a'],['a','a'],['a','a','a','a'],['a'],['a','a']]
  n = 10000

  start = time.process_time()
  for i in range(n):
    f2(sents2,'b')
  end = time.process_time()
  print(end-start)

  start = time.process_time()
  for i in range(n):
    f1(sents1,'b')
  end = time.process_time()
  print(end-start)

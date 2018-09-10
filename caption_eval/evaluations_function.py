#!/usr/bin/env python3.6
# encoding: utf-8

import argparse
import getopt
import hashlib
import io
import json
import os
import pylab
import sys

sys.path.append('./coco-caption/')
from pycocotools.coco import COCO
from .pycocoevalcap.eval import COCOEvalCap

class CocoResFormat:
  def __init__(self):
    self.res = []
    self.caption_dict = {}

  def read_multiple_files(self, filelist, hash_img_name):
    for filename in filelist:
      print ('In file %s\n' % filename)
      self.read_file(filename, hash_img_name)

  def read_file(self, filename, hash_img_name):
    count = 0
    with open(filename,'r') as opfd:
      for line in opfd:
        count +=1
        id_sent = line.strip().split('\t')
        if len(id_sent)>2:
          id_sent = id_sent[-2:]
        assert len(id_sent) == 2
        sent = id_sent[1]# remove incase of python2 -- .decode('ascii', 'ignore')

        if hash_img_name:
          img_id = int(int(hashlib.sha256(id_sent[0].encode('utf-8','ignore')).hexdigest(),
                           16) % sys.maxsize)
        else:
          img = id_sent[0].split('_')[-1].split('.')[0]
          img_id = int(img)
        imgid_sent = {}

        if img_id in self.caption_dict:
          assert self.caption_dict[img_id] == sent
        else:
          self.caption_dict[img_id] = sent
          imgid_sent['image_id'] = img_id
          imgid_sent['caption'] = sent
          self.res.append(imgid_sent)
        if count%500 == 0:
          print ('Processed %d ...' % count)

  def dump_json(self, outfile):
    res = self.res
    with io.open(outfile, 'w', encoding='utf-8') as fd:
      fd.write(str(json.dumps(res,
         ensure_ascii=False,sort_keys=True,indent=2,separators=(',', ': '))))

def evaluate(prediction_file, reference_file):
  HASH_IMG_NAME = True
  pylab.rcParams['figure.figsize'] = (10.0, 8.0)
  json.encoder.FLOAT_REPR = lambda o: format(o, '.3f')
  json_predictions_file = '{0}.json'.format(prediction_file)

  crf = CocoResFormat()
  crf.read_file(prediction_file, HASH_IMG_NAME)
  crf.dump_json(json_predictions_file)

  # create coco object and cocoRes object.
  coco = COCO(reference_file)
  cocoRes = coco.loadRes(json_predictions_file)

  # create cocoEval object.
  cocoEval = COCOEvalCap(coco, cocoRes)

  # evaluate results
  cocoEval.evaluate()

  # print output evaluation scores
  results = dict()
  for metric, score in cocoEval.eval.items():
    results[metric] = score
    print ('%s: %.3f'%(metric, score))
  return results
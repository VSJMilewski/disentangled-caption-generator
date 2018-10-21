#!/usr/bin/env python

import argparse
import hashlib
import io
import json
import sys


class CocoAnnotations:
    def __init__(self, dataset):
        self.dataset = dataset
        self.images = [];
        self.annotations = [];
        self.img_dict = {};
        if dataset == "flickr30k":
            info = {
                "year": 2016,
                "version": '1',
                "description": 'CaptionEval_Flickr30k',
                "contributor": '...',
                "url": '...',
                "date_created": '',
            }
            licenses = [{
                "id": 1,
                "name": "Flick30k",
                "url": "...",
            }]
        else:
            info = {
                "year": 2016,
                "version": '1',
                "description": 'CaptionEval_Flickr8k',
                "contributor": '...',
                "url": '...',
                "date_created": '',
            }
            licenses = [{
                "id": 1,
                "name": "Flick8k",
                "url": "...",
            }]
        self.res = {"info": info,
                    "type": 'captions',
                    "images": self.images,
                    "annotations": self.annotations,
                    "licenses": licenses,
                    }

    def read_multiple_files(self, filelist):
        for filename in filelist:
            print('In file %s\n' % filename)
            self.read_file(filename)

    def get_image_dict(self, img_name):
        image_hash = int(int(hashlib.sha256(img_name).hexdigest(), 16) % sys.maxint)
        if image_hash in self.img_dict:
            assert self.img_dict[image_hash] == img_name, 'hash colision: {0}: {1}'.format(image_hash, img_name)
        else:
            self.img_dict[image_hash] = img_name

        if self.dataset == "flickr30k":
            image_dict = {"id": image_hash,
                          "width": 224,
                          "height": 224,
                          "file_name": img_name,
                          "license": '',
                          "url": 'data/flickr30k/flickr30k_images/%s' % img_name,
                          "date_captured": '',
                          }
        else:
            image_dict = {"id": image_hash,
                          "width": 224,
                          "height": 224,
                          "file_name": img_name,
                          "license": '',
                          "url": 'data/flickr8k/Flicker8k_Dataset/%s' % img_name,
                          "date_captured": '',
                          }
        return image_dict, image_hash

    def read_file(self, filename):
        count = 0
        with open(filename, 'r') as opfd:
            for line in opfd:
                count += 1
                id_sent = line.strip().split('\t')
                assert len(id_sent) == 2
                sent = id_sent[1].decode('ascii', 'ignore')
                image_dict, image_hash = self.get_image_dict(id_sent[0][:-2])  # last two char are caption num
                self.images.append(image_dict)

                self.annotations.append({
                    "id": len(self.annotations) + 1,
                    "image_id": image_hash,
                    "caption": sent,
                })

                if count % 1000 == 0:
                    print('Processed %d ...' % count)

    def read_file_restricted(self, filename, restriction_list):
        count = 0
        with open(filename, 'r') as opfd:
            for line in opfd:
                count += 1
                id_sent = line.strip().split('\t')
                assert len(id_sent) == 2
                if id_sent[0][:-2] in restriction_list:  # last two char are caption num
                    sent = id_sent[1].decode('ascii', 'ignore')
                    image_dict, image_hash = self.get_image_dict(id_sent[0][:-2])  # last two char are caption num
                    self.images.append(image_dict)

                    self.annotations.append({
                        "id": len(self.annotations) + 1,
                        "image_id": image_hash,
                        "caption": sent,
                    })

                if count % 1000 == 0:
                    print('Processed %d ...' % count)

    def dump_json(self, outfile):
        self.res["images"] = self.images
        self.res["annotations"] = self.annotations
        res = self.res
        with io.open(outfile, 'w', encoding='utf-8') as fd:
            fd.write(unicode(json.dumps(res, ensure_ascii=True, sort_keys=True, indent=2, separators=(',', ': '))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_references", type=str, required=True,
                        help='File containing reference sentences.')
    parser.add_argument("-o", "--outputfile", type=str,
                        help='Filename for the JSON references.')
    parser.add_argument("-r", "--restrictionfile", type=str,
                        help='Filename for file with restriced images.')
    parser.add_argument("-d", "--dataset", type=str,
                        help='which datatset to use.')
    args = parser.parse_args()

    input_file = args.input_references
    output_file = '{0}.json'.format(input_file)
    if args.outputfile:
        output_file = args.outputfile
    res_file = None
    if args.restrictionfile:
        res_file = args.restrictionfile
    dataset = "flickr8k"
    if args.dataset:
        dataset = args.dataset

    crf = CocoAnnotations(dataset)
    if res_file:
        restriction_list = None
        with open(res_file) as f:
            restriction_list = f.read().splitlines()
        crf.read_file_restricted(input_file, restriction_list)
    else:
        crf.read_file(input_file)
    crf.dump_json(output_file)
    print('Created json references in %s' % output_file)


if __name__ == "__main__":
    main()

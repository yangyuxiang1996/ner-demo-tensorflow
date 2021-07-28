#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-07-25 10:26:19
LastEditors: Yuxiang Yang
LastEditTime: 2021-07-25 11:05:26
FilePath: /NER-demo/data.py
Description: 
'''
import os


class DataManager(object):
    def __init__(self, config, logger):
        self.config = config
        pass

    def build_vocab(self):
        """构造词典，一种直接从bert的vocab获取，一种是直接从train_file获取
        """
        pass

    def load_vocab(self):
        """加载vocab
        """
        if not os.path.isfile(self.token2id_file):
            self.build_vocab(self.train_file)
        pass

    def padding(self, sample):
        pass

    def prepare(self, tokens, labels, is_padding=True):
        pass

    def prepare_bert_embedding(self, df):
        pass

    def get_training_set(self, training_val_ratio=0.9):
        pass

    def get_valid_set(self):
        pass

    def map_func(self, x, tokens2id):
        pass

    def prepare_single_sentence(self, sentence):
        pass

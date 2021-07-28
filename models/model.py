#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-07-25 10:26:01
LastEditors: Yuxiang Yang
LastEditTime: 2021-07-25 12:05:48
FilePath: /NER-demo/models/model.py
Description:
'''
import tensorflow as tf
from transformers import TFBertModel
import tensorflow_addons as tfa


class NerModel(tf.keras.Model):
    """NER Model"""
    def __init__(self, configs, vocab_size, num_classes):
        super(NerModel, self).__init__
        self.use_bert = configs.use_bert
        self.finetune = configs.finetune
        self.hidden_dim = configs.hidden_dim
        self.dropout_p = configs.dropout_p
        self.use_bilstm = configs.use_bilstm
        self.embeddings_size = configs.embeddings_size
        if self.use_bert and self.finetune:
            self.bert_model = TFBertModel.from_pretrained('bert-base-chinese')
        self.embeddings = tf.keras.layers.Embedding(vocab_size, self.embeddings_size, mask_zero=True)
        self.dropout = tf.keras.layers.Dropout(self.dropout_p)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True))
        self.dense = tf.keras.layers.Dense(num_classes)
        self.transition_params = tf.Variable(tf.random.uniform(shape=(num_classes, num_classes)))  # 学习crf转移矩阵，tf.contrib.crf.crf_log_lokelihood()


    @tf.function
    def __call__(self, inputs, inputs_length, targets, training=None):
        if self.use_bert:
            if self.finetune: # finetune模式会改变bert模型的参数
                embeddings = self.bert_model(inputs[0], attention_mask=inputs[1])[0]
            else:  # 不使用finetune模式只使用bert的编码embedding
                embeddings = inputs
        else:
            embeddings = self.embeddings(inputs)

        outputs = self.dropout(embeddings)

        if self.use_bilstm:
            outputs = self.bilstm(outputs)

        logits = self.dense(outputs)

        tensor_target = tf.convert_to_tensor(targets, dtype=tf.int32)
        log_likelihood, transition_params = tfa.text.crf_log_likelihood(
            logits,
            tensor_target,
            inputs_length,
            transition_params=self.transition_params)
        
        return logits, log_likelihood, transition_params

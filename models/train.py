#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-07-28 20:57:22
LastEditors: Yuxiang Yang
LastEditTime: 2021-07-28 20:58:07
FilePath: /NER-demo-tensorflow/models/train.py
Description:
'''
import tensorflow as tf
import numpy as np
import math
import time
from tensorflow.python.ops.gen_state_ops import variable
import tokenizers
from tqdm import tqdm
from models.model import NerModel
from models.utils.metrics import metrics
from tensorflow_addons.text.crf import crf_decode
from transformers import TFBertModel, BertTokenizer


def train(configs, data_manager, logger):
    vocab_size = data_manager.max_token_size
    num_classes = data_manager.max_label_number
    learning_rate = configs.learning_rate
    max_to_keep = configs.checkpoint_max_to_keep
    checkpoint_dir = configs.checkpoint_dir
    checkpoint_name = configs.checkpoint_name
    best_f1_score = 0.0
    best_at_epoch = 0
    unprocessed  = 0
    very_start_time = time.time()
    epoch = configs.epoch
    batch_size = configs.batch_size

    # 优化器大致效果Adagrad>Adam>RMSprop>SGD
    if configs.optimizer == 'Adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif configs.optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif configs.optimizer == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    tokenizer = BertTokenizer.from_pretrained(
        configs.hf_bert_name, cache_dir=configs.model_cache_dir)
    if configs.use_bert and not configs.finetune:
        bert_model = TFBertModel.from_pretrained(
            configs.hf_bert_name, cache_dir=configs.model_cache_dir)
    else:
        bert_model = None

    train_dataset, val_dataset = data_manager.get_training_set()
    ner_model = NerModel(configs, vocab_size, num_classes)

    checkpoint = tf.train.Checkpoint(ner_model=ner_model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        max_to_keep=max_to_keep)

    if checkpoint_manager.latest_checkpoint:
        print("Restore from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    num_val_iterations = int(math.ceil(1.0 * len(val_dataset) / batch_size))
    logger.info(('+' * 20) + 'training starting' + ('+' * 20))
    for i in range(epoch):
        start_time = time.time()
        logger.info('epoch:{}/{}'.format(i + 1, epoch))
        for step, batch in tqdm(train_dataset.shuffle(len(train_dataset)).batch(batch_size).enumerate()):
            if configs.use_bert:
                X_train_batch, y_train_batch, att_mask_batch = batch
                if configs.finetune:
                    model_inputs = (X_train_batch, att_mask_batch)
                else:
                    model_inputs = bert_model(X_train_batch, attention_mask=att_mask_batch)[0]
            else:
                X_train_batch, y_train_batch = batch
                model_inputs = X_train_batch

        inputs_length = tf.math.count_nonzero(X_train_batch, 1)
        with tf.GradientTape() as tape:
            logits, log_likehood, transition_params = ner_model(
                inputs=model_inputs,
                inputs_length=inputs_length,
                targets=y_train_batch,
                training=1)
            loss = -tf.reduce_mean(log_likehood)

        variables = ner_model.trainable_variables
        variables = [var for var in variables if 'pooler' not in var.name]  # 注意，列表式可以一次从把数据加载进内存，当数据量比较大时，比使用for循环速度快很多
        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))
        if step % configs.print_per_batch == 0 and step != 0:
            batch_pred_sequence, _ = crf_decode(logits, transition_params,
                                                inputs_length)
            measures = metrics(X_train_batch, y_train_batch,
                               batch_pred_sequence, configs, data_manager,
                               tokenizer)
            res_str = ''
            for k, v in measures.items():
                res_str += (k + ':%.3f' % v)

            logger.info('training batch:%d, loss: %.5f, %s' % (step, loss, res_str))
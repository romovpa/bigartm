# This example demonstrates multimodal topics using paired russian-english documents
# -*- coding: utf-8 -*-

import string
import uuid

import artm.messages_pb2
import artm.library

ens = []
rus = []

ens.append(u"Topic model statistical analysis text collection LDA PLSA ARTM")
rus.append(u"Тематическая модель статистический анализ текст коллекция")

ens.append(u"LDA statistical topic model text collection")
rus.append(u"LDA статистический тематическая модель текст документ коллекция")

ens.append(u"PLSA statistical analysis text model")
rus.append(u"PLSA статистический анализ документ текст модель")

ens.append(u"ARTM analysis topic model")
rus.append(u"ARTM анализ документ topic model")

ens.append(u"Pinnipeds seal marine mammal order")
rus.append(u"Тюлень семейство млекопитающие моржовых отряд ласто ногие")

ens.append(u"Eared seal marine predatory mammal")
rus.append(u"Ушастый тюлень семейство млекопитающие отряд хищный семейство моржовых ласто ногие")

ens.append(u"Eared Crocodilia predatory reptile not mammal")
rus.append(u"Ушастый крокодил гена отряд хищный не млекопитающие коротко ногие")

ru_dic = {}  # mapping from russian token to its index in batch.token list
en_dic = {}  # mapping from english token to its index in batch.token list
batch = artm.messages_pb2.Batch()  # batch representing the entire collection
batch.id = str(uuid.uuid1())
unique_tokens = artm.messages_pb2.DictionaryConfig()  # BigARTM dictionary to initialize model


def append(tokens, dic, item, class_id):
    for token in tokens:
        if not dic.has_key(token):              # New token discovered:
            dic[token] = len(batch.token)       # 1. update ru_dic or en_dic
            batch.token.append(token)           # 2. update batch.token and batch.class_id
            batch.class_id.append(class_id)
            entry = unique_tokens.entry.add()   # 3. update unique_tokens
            entry.key_token = token
            entry.class_id = class_id

        # Add token to the item.
        item.field[0].token_id.append(dic[token])
        item.field[0].token_count.append(1)     # <- replace '1' with the actual number of token occupancies in the item


# Iterate through all items and populate the batch
for (en, ru) in zip(ens, rus):
    next_item = batch.item.add()
    next_item.id = len(batch.item) - 1
    next_item.field.add()
    append(string.split(ru.lower()), ru_dic, next_item, '@russian')
    append(string.split(en.lower()), en_dic, next_item, '@english')

# Create master component and infer topic model
with artm.library.MasterComponent() as master:
    dictionary = master.CreateDictionary(unique_tokens)

    # Create one top-token score per each class_id
    ru_top_tokens_score = master.CreateTopTokensScore(class_id='@russian')
    en_top_tokens_score = master.CreateTopTokensScore(class_id='@english')

    # Populate class_id and class_weight in ModelConfig
    config = artm.messages_pb2.ModelConfig()
    config.class_id.append('@russian')
    config.class_weight.append(1.00)
    config.class_id.append('@english')
    config.class_weight.append(1.00)

    # Create and initialize model, enable scores. Our expert knowledge says we need 2 topics ;)
    model = master.create_model(topics_count=2, inner_iterations_count=10, config=config)
    model.initialize(dictionary)  # Setup initial approximation for Phi matrix.
    model.enable_score(ru_top_tokens_score)
    model.enable_score(en_top_tokens_score)

    # Infer the model in 10 passes over the batch
    for iteration in range(0, 10):
        master.add_batch(batch=batch)
        master.wait_idle()  # wait for all batches are processed
        model.synchronize()  # synchronize model

    # Retrieve and visualize top tokens in each topic
    artm.library.Visualizers.print_top_tokens_score(ru_top_tokens_score.get_value(model))
    artm.library.Visualizers.print_top_tokens_score(en_top_tokens_score.get_value(model))

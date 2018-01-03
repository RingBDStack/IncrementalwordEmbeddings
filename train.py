import logging
import time
import sys
import word2vec_weight
import word2vec

context_model_cbow = "output/context/global/2g/CBOW.model"
context_model_sg = "output/context/global/2g/skip_gram.model"
origin_model_cbow = "output/origin/global/2g/CBOW.model"
origin_model_sg = "output/origin/global/2g/skip_gram.model"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename="log/train.log", filemode="w")
data_type_list = ["1g", "100m", "10m", "1m", "100k", "10k"]
model_type_list = ["cbow", "sg"]


def global_context(data_type, train_type="global", tree_type="context"):
    sentences = "../2g+%s" % data_type
    name = "2g+%s_%s_%s" % (data_type, train_type, tree_type)
    save_model_cbow = "output/%s/%s/%s/CBOW.model" % (tree_type, train_type, data_type)
    save_model_sg = "output/%s/%s/%s/skip_gram.model" % (tree_type, train_type, data_type)

    logging.info(name)

    sentences = word2vec_weight.Text8Corpus(sentences)

    logging.info("cbow")
    a = time.time()
    model = word2vec_weight.Word2Vec(sentences, size=500, hs=1, sg=0, workers=32)
    b = time.time()
    logging.info("time consume:" + str((b - a)))
    model.save(save_model_cbow)

    logging.info("sg")
    a = time.time()
    model = word2vec_weight.Word2Vec(sentences, size=500, hs=1, sg=1, workers=32)
    b = time.time()
    logging.info("time consume:" + str((b - a)))
    model.save(save_model_sg)


def global_origin(data_type, train_type="global", tree_type="origin"):
    sentences = "../2g+%s" % data_type
    name = "2g+%s_%s_%s" % (data_type, train_type, tree_type)
    save_model_cbow = "output/%s/%s/%s/CBOW.model" % (tree_type, train_type, data_type)
    save_model_sg = "output/%s/%s/%s/skip_gram.model" % (tree_type, train_type, data_type)

    logging.info(name)

    sentences = word2vec.Text8Corpus(sentences)

    logging.info("cbow")
    a = time.time()
    model = word2vec.Word2Vec(sentences, size=500, hs=1, sg=0, workers=32)
    b = time.time()
    logging.info("time consume:" + str((b - a)))
    model.save(save_model_cbow)

    logging.info("sg")
    a = time.time()
    model = word2vec.Word2Vec(sentences, size=500, hs=1, sg=1, workers=32)
    b = time.time()
    logging.info("time consume:" + str((b - a)))
    model.save(save_model_sg)


def incremental_context(data_type, train_type="incremental", tree_type="context"):
    sentences = "../%s" % data_type
    global_sent = "../2g+%s" % data_type
    sentences_2g = "../2g"
    name = "2g+" + data_type + "_" + train_type + "_" + tree_type
    save_model_cbow = "output/%s/%s/%s/CBOW.model" % (tree_type, train_type, data_type)
    save_model_sg = "output/%s/%s/%s/skip_gram.model" % (tree_type, train_type, data_type)

    logging.info(name)

    sentences_a = word2vec_weight.Text8Corpus(sentences)
    sentences_2g = word2vec_weight.Text8Corpus(sentences_2g)
    global_sent = word2vec_weight.Text8Corpus(global_sent)

    for model_type in model_type_list:
        logging.info(model_type)
        a = time.time()
        if model_type == "cbow":
            model = word2vec_weight.Word2Vec.load(context_model_cbow)
        else:
            model = word2vec_weight.Word2Vec.load(context_model_sg)
        b = time.time()
        logging.info("time consume:" + str((b - a)))
        logging.info("incremental train")
        b = time.time()
        model.build_vocab(global_sent, update=True)
        model.train(sentences_2g, update=True)
        model.train(sentences_a)
        c = time.time()
        logging.info("incremental train:" + str((c - b)))
        logging.info("total time:" + str((c - a)))
        if model_type == "cbow":
            model.save(save_model_cbow)
        else:
            model.save(save_model_sg)


def incremental_origin(data_type, train_type="incremental", tree_type="origin"):
    sentences = "../%s" % data_type
    global_sent = "../2g+%s" % data_type
    sentences_2g = "../2g"

    name = "2g+" + data_type + "_" + train_type + "_" + tree_type
    save_model_cbow = "output/%s/%s/%s/CBOW.model" % (tree_type, train_type, data_type)
    save_model_sg = "output/%s/%s/%s/skip_gram.model" % (tree_type, train_type, data_type)

    logging.info(name)

    sentences_a = word2vec.Text8Corpus(sentences)
    sentences_2g = word2vec.Text8Corpus(sentences_2g)
    global_sent = word2vec.Text8Corpus(global_sent)

    for model_type in model_type_list:
        logging.info(model_type)
        a = time.time()
        if model_type == "cbow":
            model = word2vec.Word2Vec.load(origin_model_cbow)
        else:
            model = word2vec.Word2Vec.load(origin_model_sg)
        b = time.time()
        logging.info("time consume:" + str((b - a)))
        logging.info("incremental train")
        b = time.time()
        model.build_vocab(global_sent, update=True)
        model.train(sentences_2g, update=True)
        model.train(sentences_a)
        c = time.time()
        logging.info("incremental train:" + str((c - b)))
        logging.info("total time:" + str((c - a)))
        if model_type == "cbow":
            model.save(save_model_cbow)
        else:
            model.save(save_model_sg)


def origin_2g():
    data_type = "2g"
    train_type = "global"
    tree_type = "origin"

    sentences = "../%s" % data_type
    name = "2g_%s_%s" % (train_type, tree_type)
    save_model_cbow = "output/%s/%s/%s/CBOW.model" % (tree_type, train_type, data_type)
    save_model_sg = "output/%s/%s/%s/skip_gram.model" % (tree_type, train_type, data_type)

    logging.info(name)

    sentences = word2vec.Text8Corpus(sentences)

    logging.info("cbow")
    model = word2vec.Word2Vec(sentences, size=500, hs=1, sg=0, workers=32)
    model.save(save_model_cbow)

    logging.info("sg")
    model = word2vec.Word2Vec(sentences, size=500, hs=1, sg=1, workers=32)
    model.save(save_model_sg)


def context_2g():
    data_type = "2g"
    train_type = "global"
    tree_type = "context"

    sentences = "../%s" % data_type
    name = "2g_%s_%s" % (train_type, tree_type)
    save_model_cbow = "output/%s/%s/%s/CBOW.model" % (tree_type, train_type, data_type)
    save_model_sg = "output/%s/%s/%s/skip_gram.model" % (tree_type, train_type, data_type)

    logging.info(name)

    sentences = word2vec_weight.Text8Corpus(sentences)

    logging.info("cbow")
    model = word2vec_weight.Word2Vec(sentences, size=500, hs=1, sg=0, workers=32)
    model.save(save_model_cbow)

    logging.info("sg")
    model = word2vec_weight.Word2Vec(sentences, size=500, hs=1, sg=1, workers=32)
    model.save(save_model_sg)


def bd15():
    logging.info("bd15--global origin")
    for data_type in data_type_list:
        logging.info(data_type)
        global_origin(data_type)


def bd16():
    logging.info("bd16--global context")
    for data_type in data_type_list:
        logging.info(data_type)
        global_context(data_type)


def bd17():
    logging.info("bd17--incremental origin")
    origin_2g()
    for data_type in data_type_list:
        logging.info(data_type)
        incremental_origin(data_type)


def bd18():
    logging.info("bd18--incremental context")
    context_2g()
    for data_type in data_type_list:
        logging.info(data_type)
        incremental_context(data_type)


def bd57():
    logging.info("bd57--2g+1g global origin")
    for data_type in ["1g"]:
        logging.info(data_type)
        global_origin(data_type)


def bd54():
    logging.info("bd54--2g+1g global context")
    for data_type in ["1g"]:
        logging.info(data_type)
        global_context(data_type)


if __name__ == '__main__':
    bd = sys.argv[1]
    print bd
    if bd == '15':
        bd15()
    elif bd == '16':
        bd16()
    elif bd == '17':
        bd17()
    elif bd == '18':
        bd18()
    elif bd == '54':
        incremental_context("1g")
    elif bd == '55':
        incremental_origin("1g")

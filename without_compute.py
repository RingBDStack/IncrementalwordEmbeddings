import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename="without.log", filemode="w")

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
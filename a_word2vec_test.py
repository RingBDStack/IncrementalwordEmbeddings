import logging
import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename="output/test.log", filemode="w")
text2 = word2vec.Text8Corpus("test_data/text2")
text6 = word2vec.Text8Corpus("test_data/text6")
text8 = word2vec.Text8Corpus("test_data/text8")
print "CBOW"
model = word2vec.Word2Vec(text6, size=500, hs=1, sg=0, workers=32, iter=1)
model.save("output/CBOW_text6.model")
#
model = word2vec.Word2Vec(text8, size=500, hs=1, sg=0, workers=32, iter=1)
model.save("output/CBOW_text8.model")

print "incremental"
model = word2vec.Word2Vec.load("output/CBOW_text6.model")
model.build_vocab(text8, update=True)
model.train(text6, update=True)
model.train(text2)
model.save("output/CBOW_incremental.model")

# print "skip gram"
# model = word2vec.Word2Vec(text6, size=500, hs=1, sg=1, workers=32)
# model.save("output/sg_text6.model")
#
# model = word2vec.Word2Vec(text8, size=500, hs=1, sg=1, workers=32)
# model.save("output/sg_text8.model")

# print "incremental"
# model = word2vec.Word2Vec.load("output/sg_text6.model")
# model.build_vocab(text8, update=True)
# model.iter = 1
# model.train(text6, update=True)
# # model.train(text2)
# model.save("output/sg_incremental.model")



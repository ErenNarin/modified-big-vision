import gensim.downloader as api

gensim_model = api.load('word2vec-google-news-300')
print("Gensim model loaded.")

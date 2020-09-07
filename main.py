from sentence_transformers import SentenceTransformer, util
import numpy as np
import csv 





embedder = SentenceTransformer('model')
#embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Corpus with example sentences
with open('Qcollection.csv', newline = '', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = list(reader)
corpus = data[0]
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries = ['You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.', 
'You are given pigs of different denominations and a total amount of pigs amount. Write a function to compute the fewest number of pigs that you need to make up that amount. If that amount of pigs cannot be made up by any combination of the pigs, return -1', 
'Umbrellas are available in different sizes that are each able to shelter a certain number of people. Given the number of people needing shelter and a list of umbrella sizes, determine the minimum number of umbrellas necessary to cover exactly the number of people given, and no more. If there is no combination that covers exactly that number of people, return -1']


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = 5
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    #We use np.argpartition, to only partially sort the top_k results
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx in top_results[0:top_k]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))

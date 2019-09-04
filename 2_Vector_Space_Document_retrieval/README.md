# 2.Vector Space Document retrieval.

Extract TF-IDF representations of the claims and all the documents respectively based on the document collection. The goal of this representation to later compute the cosine similarity between the document and the claims. Hence, for computational efficiency, you are allowed to represent the documents only based on the words that would have an effect on the cosine similarity computation. Given a claim, compute its cosine similarity with each document and return the document ID (the "id" field in the wiki-page) of the five most similar documents for that claim.



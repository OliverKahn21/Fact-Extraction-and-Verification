# 4.Sentence Relevance.

For a claim in the training subset and the retrieved five documents for this claim (either based on cosine similarity or the query likelihood model), represent the claim and sentences in these documents based on a word embedding method, (such as Word2Vec, GloVe, FastText or ELMo). With these embeddings as input, implement a logistic regression model trained on the training subset. Use the verifiable claims in the development dataset to evaluate the performance of the logistic regression model. Report the performance of your system in this dataset using an evaluation metric you think would fit to this task. Analyze the effect of the learning rate on the model training loss. Instead of using Python sklearn or other packages, the implementations of the logistic regression algorithm should be your own.

# 5.Relevance Evaluation.

Implement methods to compute recall, precision and F1 metrics. Analyze the sentence retrieval performance of your model using the labelled data in the development subset.

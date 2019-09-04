# FEVER
An increasing amount of online misinformation has motivated research in automated fact checking. Your task in this project is to develop information retrieval and data mining methods to assess the veracity of a claim. Specifically, the automated fact checking project consists of three steps: relevant document retrieval, evidence sentence selection, and claim veracity prediction.

We will be using the publicly available Fact Extraction and Verification (FEVER) dataset 1. It consists of 185,445 claims manually verified against Wikipedia pages and classified as Supported, Refuted and NotEnoughInfo. For the first two classes, there is a combination of sentences forming the necessary evidence supporting or refuting the claim. This dataset consists of a collection of documents (wiki-pages), a labeled training subset (train.jsonl), a labeled development subset (dev.jsonl) and a reserved testing subset (test.jsonl). For a claim in the train.jsonl file, the value of the "evidence" field is a list of relevant sentences in the format of [_, _, wiki document ID, sentence ID]. More details about this dataset can be found in [http://fever.ai/resources.html] and this website [http://fever.ai/2018/task.html]. A demo for reading this dataset is on the website [https://github.com/QiangAIResearcher/Fact-Extraction-and-Verification].

## Involved Subtasks

The project involves several subtasks that are required to be solved. This is a research oriented project so you are expected to be creative and coming up with your own solutions is strongly encouraged for any part of the project.

### 1.Text Statistics.
Count frequencies of every term in the collection of documents, plot the curve of term frequencies and verify Zipf’s Law. Report the values of the parameters for Zipf’s law for this corpus.

### 2.Vector Space Document retrieval.
Extract TF-IDF representations of the claims and all the documents respectively based on the document collection. The goal of this representation to later compute the cosine similarity between the document and the claims. Hence, for computational efficiency, you are allowed to represent the documents only based on the words that would have an effect on the cosine similarity computation. Given a claim, compute its cosine similarity with each document and return the document ID (the "id" field in the wiki-page) of the five most similar documents for that claim.

### 3.Probabilistic Document Retrieval.
Establish a query-likelihood unigram language model based on the document collection, and return the five most similar documents for each one of the claims. Implement and apply Laplace Smoothing, Jelinek-Mercer Smoothing and Dirichlet Smoothing to the query-likelihood language model, return the five most similar documents for the 10 claims.

### 4.Sentence Relevance.
For a claim in the training subset and the retrieved five documents for this claim (either based on cosine similarity or the query likelihood model), represent the claim and sentences in these documents based on a word embedding method, (such as Word2Vec, GloVe, FastText or ELMo). With these embeddings as input, implement a logistic regression model trained on the training subset. Use the verifiable claims in the development dataset to evaluate the performance of the logistic regression model. Report the performance of your system in this dataset using an evaluation metric you think would fit to this task. Analyze the effect of the learning rate on the model training loss. Instead of using Python sklearn or other packages, the implementations of the logistic regression algorithm should be your own.

### 5.Relevance Evaluation.
Implement methods to compute recall, precision and F1 metrics. Analyze the sentence retrieval performance of your model using the labelled data in the development subset.

### 6.Truthfulness of Claims.
Filter out the ’NOT ENOUGH INFO’ claims and only keep the ’SUPPORTS’ or ’REFUTES’ claims in the train.jsonl and dev.jsonl datasets. Using the relevant sentences specified in the ’evidence’ field as your training data and using their cor- responding truthfulness labels in the train.jsonl file, build a neural network based model to assess the truthfulness of a claim in the training subset. No need to retrieve documents and select sentences for this part, just use the sentences specified in the ’evidence’ field in the train.jsonl and dev.jsonl.You may use existing packages like Tensorflow or PyTorch in this subtask. You are expected to propose your own network architecture for the neural network. Report the performance of your system in the labelled development subset using evaluation metrics you have implemented. Furthermore, describe the motivation behind your proposed neural architecture. The marks you get will be based on the quality and novelty of your proposed neural network architecture, as well as its performance.

### 7.Literature Review.
Do a literature review regarding fact checking and misinformation detec- tion, identify pros and cons of existing models/methods and provide critical analysis. Explain what you think the drawback of each of these models are (if any).

### 8.Propose ways to improve the machine learning models you have implemented.
You can either propose new machine learning models, new ways of sampling/using the training data, or propose new neural architectures. You are allowed to use existing libraries/packages for this part. Explain how your proposed method(s) differ from existing work in the literature. The marks you get will be based on the quality and novelty of your proposed methods, as well as the performance of your final model.

## This project including the following files:

1.Project_Description.pdf: This is the original project description including the project descritption of 8 subtask.

2.Report.pdf: The report about this information retrieval and data mining project.

3.instructions.txt: The exlaination of how to run the code to reproduce the result shown in the report.

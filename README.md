# FEVER
Text Statistics, Vector Space Document retrieval, Probabilistic Document Retrieval, Sentence Relevance,Truthfulness of Claims on Fact Extraction and Verification (FEVER) dataset

An increasing amount of online misinformation has motivated research in automated fact checking. Your task in this project is to develop information retrieval and data mining methods to assess the veracity of a claim. Specifically, the automated fact checking project consists of three steps: relevant document retrieval, evidence sentence selection, and claim veracity prediction.

We will be using the publicly available Fact Extraction and Verification (FEVER) dataset 1. It consists of 185,445 claims manually verified against Wikipedia pages and classified as Supported, Refuted and NotEnoughInfo. For the first two classes, there is a combination of sentences forming the necessary evidence supporting or refuting the claim. This dataset consists of a collection of documents (wiki-pages), a labeled training subset (train.jsonl), a labeled development subset (dev.jsonl) and a reserved testing subset (test.jsonl). For a claim in the train.jsonl file, the value of the "evidence" field is a list of relevant sentences in the format of [_, _, wiki document ID, sentence ID]. More details about this dataset can be found in [http://fever.ai/resources.html] and this website [http://fever.ai/2018/task.html]. A demo for reading this dataset is on the website [https://github.com/QiangAIResearcher/Fact-Extraction-and-Verification].

This project including the following files:

1.Project_Description.pdf: This is the original project description including the project descritption of 8 subtask.

2.Report.pdf: The report about this information retrieval and data mining project.

3.instructions.txt: The exlaination of how to run the code to reproduce the result shown in the report.

# Self-supervised-Information-Extraction

CSCI-GA.2590 - Course Project for Natural Language Processing under Prof. Ralph Grishman

Based on http://aiweb.cs.washington.edu/research/projects/aiweb/media/papers/tmpYZBSTp.pdf


Our methodology primarily consists of four important modules Relation Extraction, Building Classifier, Single Pass Extractor, Query
Module. Using Stanford Parser’s dependency graph we form tuples. A Naive Bayes classifier is trained to label tuples
extracted from a corpus as either trustworthy or untrustworthy. During querying, all similar tuples with a certain
overlap are grouped together and ranked. The tuple with the highest rank is returned as the answer to the query. We
then report our experiments on different domain-diverse paragraphs taken from various sources over the web


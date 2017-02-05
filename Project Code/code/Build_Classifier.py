#!/usr/bin/env python
import nltk
import pickle
from Dijkstras import *

def BuildFeatures(tup, tup_indices, graph, POS):
	#print '-'*60
	#print POS
	#for tup, tup_indices in zip(trusted_tuples, trusted_tuples_indices):
	features = {}
	ei = tup[0]
	ei_ind = tup_indices[0][2]
	rij = tup[1]
	rij_ind = tup_indices[1][1]
	ej = tup[2]
	ej_ind = tup_indices[2][2]
	#print ei, rij, ej
	#print ei_ind, rij_ind, ej_ind
	#d_ei_rij = len(shortestPath(graph, ei_ind, rij_ind))
	#d_rij_ej = len(shortestPath(graph, rij_ind, ej_ind))
	d_ei_rij = abs(tup_indices[0][1]-1-tup_indices[1][0])
	d_rij_ej = abs(tup_indices[1][0]-tup_indices[2][0])
	d_ei_ej = abs(tup_indices[0][1]-1-tup_indices[2][0])

	features['e_i_NN'] = False
	features['e_i_NNS'] = False
	features['e_i_NNP'] = False
	features['e_i_NNPS'] = False

	features['e_j_NN'] = False
	features['e_j_NNS'] = False
	features['e_j_NNP'] = False
	features['e_j_NNPS'] = False

	features['VB'] = False
	features['VBD'] = False
	features['VBG'] = False
	features['VBN'] = False
	features['VBP'] = False
	features['VBZ'] = False
	

	features['dist_ei_rij'] = d_ei_rij
	features['dist_rij_ej'] = d_rij_ej
	features['dist_ei_ej'] = d_ei_ej



	for token_ind in range(tup_indices[0][0], tup_indices[0][1]):
		#print POS[token_ind]
		if POS[token_ind] in ['NN','NNS','NNP','NNPS']:
			features['e_i_'+POS[token_ind]] = True


	for token_ind in range(tup_indices[2][0], tup_indices[2][1]):
		#print POS[token_ind]
		if POS[token_ind] in ['NN','NNS','NNP','NNPS']:
			features['e_j_'+POS[token_ind]] = True

	#print POS[rij_ind-1]
	if POS[rij_ind-1] in ['VB','VBD','VBG','VBN','VBP','VBZ']:
		features[POS[rij_ind-1]] = True

	if tup_indices[0][1]-1 < tup_indices[1][0]:
		features['order_ei_rij'] = 'before'
	else:
		features['order_ei_rij'] = 'after'

	if tup_indices[1][0] < tup_indices[2][0]:
		features['order_rij_ej'] = 'before'
	else:
		features['order_rij_ej'] = 'after'

	if tup_indices[0][1]-1 < tup_indices[2][0]:
		features['order_ei_ej'] = 'before'
	else:
		features['order_ei_ej'] = 'after'

	return features

with open('vars.pickle') as f:
    trusted_tuples_sen, untrusted_tuples_sen, trusted_tuples_sen_indices, untrusted_tuples_sen_indices, POS_sen, graph_sen = pickle.load(f)

for x, y in zip(trusted_tuples_sen, trusted_tuples_sen_indices):
	print x, y

for x,y in zip(untrusted_tuples_sen, untrusted_tuples_sen_indices):
	print x, y


'''featureSets_trusted = [(BuildFeatures(tup, tup_indices, graph), 'trusted') for tup, tup_indices in zip(trusted_tuples, trusted_tuples_indices)]
featureSets_untrusted = [(BuildFeatures(tup, tup_indices, graph), 'untrusted') for tup, tup_indices in zip(untrusted_tuples, untrusted_tuples_indices)]
featureSets = featureSets_trusted + featureSets_untrusted
classifier = nltk.NaiveBayesClassifier.train(featureSets)'''
featureSets_trusted, featureSets_untrusted = [], []
for trusted_tuples, trusted_tuples_indices, graph, POS in zip(trusted_tuples_sen, trusted_tuples_sen_indices, graph_sen, POS_sen):
	for tup, tup_indices in zip(trusted_tuples, trusted_tuples_indices):
		features = BuildFeatures(tup, tup_indices, graph, POS)
		featureSets_trusted.append((features, 'trusted'))

for untrusted_tuples, untrusted_tuples_indices, graph, POS in zip(untrusted_tuples_sen, untrusted_tuples_sen_indices, graph_sen, POS_sen):
	for tup, tup_indices in zip(untrusted_tuples, untrusted_tuples_indices):
		features = BuildFeatures(tup, tup_indices, graph, POS)
		featureSets_untrusted.append((features, 'untrusted'))

featureSets = featureSets_trusted + featureSets_untrusted
classifier = nltk.NaiveBayesClassifier.train(featureSets)
print type(classifier)

with open('classifier.dump', 'w') as f:
	pickle.dump(classifier, f)

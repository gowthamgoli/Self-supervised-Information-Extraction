#input: query_answer.py <ei> <rij> <ej>		for empty use <_>

import sys
import nltk
from nltk.stem.lancaster import LancasterStemmer
#from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.corpus import wordnet as wn
from collections import OrderedDict

def assignProbs_1(list2, i):
	prob = {}
	numTuples = len(list2)
	for tup1 in list2:
		counter = 0
		for tup2 in list2:
			if tup1[i] in tup2[i] or tup2[i] in tup1[i]:
				counter = counter + 1
		prob[tup1] = 1.0*counter/numTuples;
	return prob;

def assignProbs_2(list2):
	prob = {}
	numTuples = len(list2)
	for tup1 in list2:
		#print tup1
		counter1 = 0
		counter2 = 0
		for tup2 in list2:
			if tup1[0] in tup2[0] or tup2[0] in tup1[0]:
				counter1 = counter1 + 1
				
			if tup1[2] in tup2[2] or tup2[2] in tup1[2]:
				counter2 = counter2 + 1
		#print counter1
		#print counter2
		prob[tup1] = (1.0*counter1/numTuples)*(1.0*counter2/numTuples)
	return prob

lines = [line.rstrip('\n') for line in open('queries.txt')]

st = LancasterStemmer()

for line in lines:

	sentence, query = line.split(':::')
	if len(query.split(' ')) == 3:
		q1, q2, q3 = query.split(' ')

		mypath = 'trusted_tuples.txt'
		file = open(mypath,'r').readlines()

		trusted_tuple=[]
		for line in file:
			trusted_tuple.append(eval(line))

		#for i in trusted_tuple:
		#	print i

		e_i_search=[]
		r_i_j_search=[]
		e_j_search=[]

		e_i_search_stem=[]
		r_i_j_search_stem=[]
		e_j_search_stem=[]

		q1_stem = st.stem(q1)
		q2_stem = st.stem(q2)
		q3_stem = st.stem(q3)

		#print 'q1_stem: ',q1_stem
		#print 'q2_stem: ',q2_stem
		#print 'q3_stem: ',q3_stem

		q2_syn=[]

		syns = wn.synsets(q2)
		#print syns

		#for s in syns:
		#	print s.lemmas

		for s in syns:
			for l_name in s.lemma_names():
				if l_name not in q2_syn:
					q2_syn.append(l_name)

			arr = s.hypernyms()
			for hyn in arr:
				for l1_name in hyn.lemma_names():
					if l1_name not in q2_syn:
						q2_syn.append(l1_name)
		#print q2_syn

		'''for s in syns:
			for l in s.lemmas:
				if l.name not in q2_syn:
					q2_syn.append(l.name)
			arr =  s.hypernyms()
			for hyn in arr:
				for l in hyn.lemmas:
					if l.name not in q2_syn:
						q2_syn.append(l.name)'''

			

		#print q2_syn

		q2_syn_stem = [st.stem(x) for x in q2_syn]

		#print 'q2_syn_stem: ',q2_syn_stem

		for ei,rij,ej in trusted_tuple:
			exitFlag = False
			#print ''
			#print (ei, rij, ej)
			#print '-'*50
			ei_stem = [st.stem(x) for x in nltk.word_tokenize(ei)]
			rij_stem = [st.stem(x) for x in nltk.word_tokenize(rij)]
			ej_stem = [st.stem(x) for x in nltk.word_tokenize(ej)]

			#print 'ei_stem: ', ei_stem
			#print 'rij_stem: ', rij_stem
			#print 'ej_stem: ', ej_stem

			if q1_stem in ei_stem and q1 != '_':
				e_i_search.append((ei,rij,ej));
				e_i_search_stem.append((ei_stem,rij_stem,ej_stem));

			for lp in q2_syn_stem:
				for x in rij_stem:
					if (lp in x and q2 != '_') or (q2 in x and q2 != '_'):
						#print 'found rij: ', lp
						if (ei, rij, ej) not in r_i_j_search:
							r_i_j_search.append((ei,rij,ej));
							r_i_j_search_stem.append((ei_stem,rij_stem,ej_stem));
						#exitFlag = True
		        		#break
		        #if exitFlag:
		        	#break

			if q3_stem in ej_stem and q3 != '_':
				e_j_search.append((ei,rij,ej));
				e_j_search_stem.append((ei_stem,rij_stem,ej_stem));


			#print (e_i_search,r_i_j_search,e_j_search)
			#print ''

		query_result = e_i_search + r_i_j_search + e_j_search

		#print 'query_result: ', query_result
		#print ''
		query_result_stem = e_i_search_stem + r_i_j_search_stem + e_j_search_stem

		set_ei = set(e_i_search)
		set_rij = set(r_i_j_search)
		set_ej = set(e_j_search)

		#print 'set_ei: ',set_ei
		#print 'set_rij: ', set_rij
		#print 'set_ej: ', set_ej

		list3 = set.intersection(set_ej,set_rij,set_ei)
		#print 'list3: ', list3
		#print ''

		prob = {}
		maxProb = 0
		maxProb_tups = []
		list2 = set([''])

		if q1=='_' and q2!='_' and q3!='_':
			list2 = set.intersection(set_ej,set_rij)
			prob = assignProbs_1(list2, 0)
			if len(prob) > 0:
				maxProb = max(prob.values())
				maxProb_tups = [k for k,v in prob.items() if v == maxProb]
		elif q1!='_' and q2=='_' and q3!='_':
			list2 = set.intersection(set_ei,set_ej)
			prob = assignProbs_1(list2, 1)
			if len(prob) > 0:
				maxProb = max(prob.values())
				maxProb_tups = [k for k,v in prob.items() if v == maxProb]
		elif q1!='_' and q2!='_' and q3=='_':
			list2 = set.intersection(set_ei,set_rij)
			prob = assignProbs_1(list2, 2)
			if len(prob) > 0:
				maxProb = max(prob.values())
				maxProb_tups = [k for k,v in prob.items() if v == maxProb]
		elif q1=='_' and q2!='_' and q3=='_':
			list2 = set_rij
			prob = assignProbs_2(list2)
			if len(prob) > 0:
				maxProb = max(prob.values())
				maxProb_tups = [k for k,v in prob.items() if v == maxProb]

		max_len = 0
		max_k = 0
		for k in maxProb_tups:
			if len(k[0]+k[1]+k[2]) > max_len:
				max_len = len(k[0]+k[1]+k[2])
				max_k = k

		#print sentence
		if len(prob) > 0:
			print q1 + ' ' + q2 + ' ' + q3 + ':\t'+ max_k[0]+ " " + max_k[1] + " " + max_k[2]

		else:
			print q1 + ' ' + q2 + ' ' + q3 + ':\t'+ 'NULL'
		print ''

		list1 = set.union(set_ei,set_ej,set_rij)
		list1 = set.difference(list1,set.union(list2,list3))
		prob_sorted = OrderedDict(sorted(prob.items(), key=lambda x: x[1], reverse=True))
		#print list3
		'''print 'possible asnwers: ', list2
		if len(prob_sorted) > 0:
			print q1 + ' ' + q2 + ' ' + q3 + ':\t'+ prob_sorted.keys()[0][0]+ " " + prob_sorted.keys()[0][1] + " " + prob_sorted.keys()[0][2] + '\t' + "prob = " + str(prob_sorted.values()[0])

		else:
			print q1 + ' ' + q2 + ' ' + q3 + ':\t'+ 'NULL'''

	#print maxProb_tups

	#print list1	
## To run: jython collect_training_data_with_chunker.py englishPCFG.ser.gz
import progressbar
import sys, pickle, random
from time import sleep
from stanford import StanfordParser, PySentence
from Dijkstras import *
import pickle
#from BuildFeatures import BuildFeatures
#sys.path.append('/usr/lib/python2.7')
#sys.path.append('/usr/local/lib/python2.7/dist-packages/')
#from nltk.corpus import brown
#nltk.data.path.append('/home/manbearpig/acads/NLP/project/open_ie/codes/nltk_data')

parser_file = sys.argv[1]
PARSER = StanfordParser(parser_file)


#input = 'The American Civil War, also known as the War between the States or simply the Civil War, was a civil war fought from 1861 to 1865 in the United States after several Southern slave states declared their secession and formed the Confederate States of America.'
#input = 'Tendulkar won the 2010 Sir Garfield Sobers Trophy for cricketer of the year at the ICC awards.'
#input = 'Tendular is born on April 24, 1973 in Mumbai.'
#input = 'Lincoln won, but before his inauguration on March 4, 1861, seven slave states with cotton-based economies formed the Confederacy'
#input = 'Sachin Ramesh Tendulkar is an Indian cricketer widely acknowledged as the greatest batsman of his generation.'
#input = "The jury further said in term-end presentments that the City Executive Committee , which had over-all charge of the election , `` deserves the praise and thanks of the City of Atlanta '' for the manner in which the election was conducted ."

#print 'Input Sentence: ', input

def get_index(id_str):
            return '-' if '.' in id_str else id_str

def get_noun_phrases(sentence):
	#print type(sentence.parse)
	#sentence.parse.indexLeaves()
	#print sentence.node
	NP_subtrees = []
	for subtree in sentence.parse.subTrees():
		label = subtree.label()
		if label.value() == 'NP':# or label.value() == 'PP':
			NP_subtrees.append(subtree)
	        	#print subtree, subtree.getLeaves()
	to_remove = []
	for np_parent in NP_subtrees:
		for np_child in NP_subtrees:
			parent_leaves = [str(np_parent.getLeaves().get(i)) for i in range(np_parent.getLeaves().size())]
			child_leaves = [str(np_child.getLeaves().get(i)) for i in range(np_child.getLeaves().size())]
			#print child_leaves, parent_leaves
			#print type(np_child.getLeaves()), np_parent.getLeaves().size(), np_parent.getLeaves().contains(np_child.getLeaves())
			if set(child_leaves).issubset(set(parent_leaves)) and child_leaves != parent_leaves and np_child not in to_remove:
				to_remove.append(np_child)

	#for rem in to_remove:
	#	NP_subtrees.remove(rem)

	#print NP_subtrees
	Noun_Phrases = []
	for x in NP_subtrees:
		#print x.getLeaves().get(0).label().beginPosition()
		#print x.getLeaves().get(0).nodeNumber()
		List = []
		List.append(str(x.getLeaves().get(0).label().beginPosition()))
		List.append(str(x.getLeaves().get(x.getLeaves().size()-1).label().endPosition()))
		for i in range(x.getLeaves().size()):
			List.append(str(x.getLeaves().get(i)))
		#Noun_Phrases.append([str(x.getLeaves().get(i)) for i in range(x.getLeaves().size())])
		Noun_Phrases.append(List)
	
	return Noun_Phrases


def clean1(untrusted_tuples, untrusted_tuples_indices, trusted_tuples, Noun_Phrases):	# to make sure that after the mapping to noun phrases, some of the trusted ones did not show up in the untrusted ones.
	#print [' '.join(np[2:]) for np in Noun_Phrases]
	np = [' '.join(np[2:]) for np in Noun_Phrases]

	to_remove = []
	for ind in range(len(untrusted_tuples)):
		for x in trusted_tuples:
			if untrusted_tuples[ind][0] in x[0] and untrusted_tuples[ind][2] in x[2] and untrusted_tuples[ind][1] == x[1] and \
			   untrusted_tuples[ind][0] in np and untrusted_tuples[ind][2] in np and ind not in to_remove:
				to_remove.append(ind)
	
	untrusted_tuples_new = [untrusted_tuples[ind] for ind in range(len(untrusted_tuples)) if ind not in to_remove]
	untrusted_tuples_indices_new = [untrusted_tuples_indices[ind] for ind in range(len(untrusted_tuples)) if ind not in to_remove]
	return [untrusted_tuples_new, untrusted_tuples_indices_new]


def clean2(tuples, tuples_indices): # We don't want the tuples where atleast one of ei,rij,ej is superset of other.
	tuples_new, tuples_indices_new = [], []
	for ind in range(len(tuples)):
		x = tuples[ind]
		if x[0] not in x[2] and x[2] not in x[0] and x[0] not in x[1] and x[1] not in x[0] and x[1] not in x[2] and x[2] not in x[1]:
			tuples_new.append(x)
			tuples_indices_new.append(tuples_indices[ind])

	return [tuples_new, tuples_indices_new]





POS, words = [], []
tolerance = 2	# set this accordingly
def get_tuples(input):
	global PARSER, POS, words, tolerance

	for sentence, prob in PARSER.get_most_probable_parses(input,kbest=1):
		Noun_Phrases = get_noun_phrases(sentence)
		#print 'NP: ',Noun_Phrases
		#print ''

		words = PARSER.tokenize(input)
		word_ids = range(len(words))
		letter_indices_begin = [w.beginPosition() for w in words]
		letter_indices_end = [w.endPosition() for w in words]
		#print 'lib: ',letter_indices_begin
		#print 'lie: ', letter_indices_end
		#print 'Parse Probability: ', prob
		#sentence.print_tree()
		#sentence.print_table()

		#print sentence.word

		POS = []
		DEP = []
		for idx in sorted(sentence.word):
		    POS.append(sentence.tag.get(idx,''))
		    DEP.append(sentence.dep.get(idx,''))

		#print 'words: ', words
		#print 'POS: ', POS
		#print 'DEP: ', DEP

		Noun_ids = [idx+1 for idx in range(len(words)) if POS[idx] in [u'NNP',u'NN',u'NNPS',u'NNS',u'CD']]#NNP(S)-> Proper Noun,Singular(Plural); #NN(S)->Noun(Plural)
		Verb_ids = [idx+1 for idx in range(len(words)) if POS[idx] in [u'VBN',u'VBP',u'VB',u'VBD',u'VBG',u'VBZ']]
		Nouns = [words[x-1] for x in Noun_ids]
		Verbs = [words[x-1] for x in Verb_ids]
		tuples_2D = [(i,j) for i in Noun_ids for j in Noun_ids if i < j]

		#print 'tup2d: ', tuples_2D

		tuples_3D = [(tup2[0],i,tup2[1]) for tup2 in tuples_2D for i in Verb_ids]
		#print 'tup3d: ', tuples_3D
		#print 'nouns, verbs: ', Nouns, Verbs
		#for e_i,e_j in 

		#print 'Collapsed Dependencies:'
		tmpl = 'Head: %s (%d); dependent: %s (%d); relation: %s'
		'''for td in sentence.gs.typedDependenciesCollapsed():
		    head = td.gov()
		    head_idx = head.index()
		    dep = td.dep()
		    dep_idx = dep.index()
		    rel = td.reln()
		    '''
		#print '-' * 80

		#weight = {'acomp':1,'advcl':1,'advmod':1,'agent':1,'amod':1,'appos':1,'attr':1,'aux':1,'auxpass':1,'cc':1,'ccomp':1,\
		#	  'conj':1,'cop':1,'csubj':1,'csubjpass':1,'dep':1}
		weight = 1
		graph = {}
		appendix = {}
		for td in sentence.gs.typedDependenciesCollapsed():
			head = td.gov()
			head_idx = head.index()
			dep = td.dep()
			dep_idx = dep.index()
			rel = td.reln()
			#print type(str(rel))
			#print tmpl % (head.value(), head_idx, dep.value(), dep_idx, rel)
			#print rel
			if 'prep_' in str(rel):# and (head in Verbs or dep in Verbs):
				#print 'prep relation\n'
				rel_temp = str(rel).strip('prep_')
				appendix = dict(appendix.items() + {(head_idx,dep_idx):rel_temp}.items())
				#appendix.append((head_idx,dep_idx,rel))
			try:
				graph[head_idx] = dict(graph[head_idx].items() + {dep_idx:weight}.items())
			except KeyError:
				graph[head_idx] = {dep_idx:weight}
			try:
				graph[dep_idx] = dict(graph[dep_idx].items() + {head_idx:weight}.items())
			except KeyError:
				graph[dep_idx] = {head_idx:weight}
	
		#print 'graph: ', graph
		#print 'appendix: ', appendix

		trusted_tuples, untrusted_tuples = [], []
		trusted_tuples_indices, untrusted_tuples_indices = [], []
		#print tuples_3D
		for e_i,r_i_j,e_j in tuples_3D:
			#print words[e_i-1], words[r_i_j-1], words[e_j-1]
			try:
				shortest_path1 = shortestPath(graph,e_i,r_i_j)
			except KeyError:
				continue
			try:
				shortest_path2 = shortestPath(graph,r_i_j,e_j)
			except KeyError:
				continue
			#print 'paths: ', shortest_path1, shortest_path2
			prefix, suffix = '', ''
			if tuple(shortest_path1[-2:]) in appendix.keys():
				#print 'prefix: ',shortest_path1[-2:]
				prefix = appendix[tuple(shortest_path1[-2:])] + ' '
			if tuple(shortest_path2[0:2]) in appendix.keys():
				#print 'suffix: ',shortest_path2[0:2]
				#print shortest_path1[0:2:]
				suffix = ' ' + appendix[tuple(shortest_path2[0:2])]
			#if len(shortest_path1)-1 <= 1 and len(shortest_path2)-1 <= 1:
			e1 = str(words[e_i-1])
			e2 = str(words[e_j-1])
			e1_candidates, e2_candidates = [], []
			e1_candidates_indices, e2_candidates_indices = [], []
			for np in Noun_Phrases:
				#print np
				if (str(words[e_i-1]) in np) and (int(np[0])<=letter_indices_begin[e_i-1]<=letter_indices_end[e_i-1]<=int(np[1])):
					e1 = ' '.join(np[2:])
					e1_candidates.append(e1)
					e1_candidates_indices.append((letter_indices_begin.index(int(np[0])),letter_indices_end.index(int(np[1]))+1, e_i))
					#print e1_candidates[-1], e1_candidates_indices[-1]
				if (str(words[e_j-1]) in np) and (int(np[0])<=letter_indices_begin[e_j-1]<=letter_indices_end[e_j-1]<=int(np[1])):
					e2 = ' '.join(np[2:])
					e2_candidates.append(e2)
					e2_candidates_indices.append((letter_indices_begin.index(int(np[0])),letter_indices_end.index(int(np[1]))+1, e_j))

				for e1_ind in range(len(e1_candidates)):
					e1 = e1_candidates[e1_ind]
					e1_span = e1_candidates_indices[e1_ind]
					for e2_ind in range(len(e2_candidates)):
						e2 = e2_candidates[e2_ind]
						e2_span = e2_candidates_indices[e2_ind]
						if e1==e2:
							tup = (str(words[e_i-1]),str(prefix)+str(words[r_i_j-1])+str(suffix),str(words[e_j-1]))
							#print 'tup: ', tup
							if len(shortest_path1)-1 + len(shortest_path2)-1 <= tolerance:
								if tup not in trusted_tuples:
									#print words[e_i-1],letter_indices[e_i-1],words[e_j-1],letter_indices[e_j-1]
									#print '= ', tup
									trusted_tuples.append(tup)
									trusted_tuples_indices.append([(e_i-1,e_i, e_i),(r_i_j-1,r_i_j),(e_j-1,e_j,e_j)])
							else:
								if tup not in untrusted_tuples:
									untrusted_tuples.append(tup)
									untrusted_tuples_indices.append([(e_i-1,e_i, e_j),(r_i_j-1,r_i_j),(e_j-1,e_j,e_j)])
						else:
							tup = (e1,str(prefix)+str(words[r_i_j-1])+str(suffix),e2)
							#print 'tup: ', tup
							if len(shortest_path1)-1 + len(shortest_path2)-1 <= tolerance:
								if tup not in trusted_tuples:
									#print words[e_i-1],letter_indices[e_i-1],words[e_j-1],letter_indices[e_j-1]
									#print '!= ', tup
									trusted_tuples.append(tup)
									trusted_tuples_indices.append([e1_span,(r_i_j-1,r_i_j),e2_span])
							else:
								if tup not in untrusted_tuples:
									untrusted_tuples.append(tup)
									untrusted_tuples_indices.append([e1_span,(r_i_j-1,r_i_j),e2_span])

		'''print '-'*60
		print 'Trusted tuples:\n'
		for ind in range(len(trusted_tuples)):
			x = trusted_tuples[ind]
			x_ind = trusted_tuples_indices[ind]
			print x
			#print x, [POS[i[0]:i[1]] for i in x_ind]
		print '-'*60
		print 'Untrusted tuples:\n'
		for x in untrusted_tuples[0:len(trusted_tuples)]:
			print x'''


		trusted_tuples, trusted_tuples_indices = clean2(trusted_tuples, trusted_tuples_indices)
		untrusted_tuples, untrusted_tuples_indices = clean2(untrusted_tuples, untrusted_tuples_indices)
		untrusted_tuples, untrusted_tuples_indices = clean1(untrusted_tuples, untrusted_tuples_indices, trusted_tuples, Noun_Phrases)


		#print '-'*60
		#print 'Trusted tuples:\n'
		#for x in trusted_tuples:
		#	print x
		#print '-'*60
		#print 'Untrusted tuples:\n'
		#for x in untrusted_tuples:
		#	print x
		#print 'Shortest Path:', shortest_path, 'Shortest Distance:', len(shortest_path)-1

		#common_node, shortest_path = sentence.get_least_common_node(node_i_idx, node_j_idx)
		#print 'Shortest Path:', shortest_path, 'Shortest Distance:', len(shortest_path)-1, 'Common Node:', common_node

		return [trusted_tuples, untrusted_tuples, trusted_tuples_indices, untrusted_tuples_indices, graph, POS]


def BuildFeatures(tup, tup_indices, graph):
	print '-'*60
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
	d_ei_rij = len(shortestPath(graph, ei_ind, rij_ind))
	d_rij_ej = len(shortestPath(graph, rij_ind, ej_ind))

	features['NN'] = False
	features['NNS'] = False
	features['NNP'] = False
	features['NNPS'] = False

	features['VB'] = False
	features['VBD'] = False
	features['VBG'] = False
	features['VBN'] = False
	features['VBP'] = False
	features['VBZ'] = False
	

	features['dist_ei_rij'] = d_ei_rij
	features['dist_rij_ej'] = d_rij_ej
	features['dist_ei_ej'] = d_ei_rij + d_rij_ej

	for token_ind in range(tup_indices[0][0], tup_indices[0][1]):
		#print POS[token_ind]
		if POS[token_ind] in ['NN','NNS','NNP','NNPS']:
			features[POS[token_ind]] = True

	print POS[rij_ind-1]
	if POS[rij_ind-1] in ['VB','VBD','VBG','VBN','VBP','VBZ']:
		features[POS[rij_ind-1]] = True

	if ei_ind < rij_ind:
		features['order_ei_rij'] = 'before'
	else:
		features['order_ei_rij'] = 'after'

	if rij_ind < ej_ind:
		features['order_rij_ej'] = 'before'
	else:
		features['order_rij_ej'] = 'after'

	return features

corpus_path = '/home/manbearpig/acads/NLP/project/open_ie/codes/test_corpus.txt'
#corpus_path = '/home/manbearpig/acads/NLP/project/open_ie/codes/brown_corpus.txt'
trusted_tuples_path = '/home/manbearpig/acads/NLP/project/open_ie/codes/trusted_tuples.txt'
untrusted_tuples_path = '/home/manbearpig/acads/NLP/project/open_ie/codes/untrusted_tuples.txt'
untrusted_tuples_path_short = '/home/manbearpig/acads/NLP/project/open_ie/codes/untrusted_tuples_short.txt'
features_path = '/home/manbearpig/acads/NLP/project/open_ie/codes/features.txt'
feature_lables_path = '/home/manbearpig/acads/NLP/project/open_ie/codes/feature_labels.txt'

corpus_f_large = open(corpus_path,'r').readlines()
#random.shuffle(corpus_f_large)
corpus_f = corpus_f_large[0:4500]
trusted_tuples_f = open(trusted_tuples_path,'w')
untrusted_tuples_f = open(untrusted_tuples_path,'w')
untrusted_tuples_f_short = open(untrusted_tuples_path_short,'w')


sentence_number = 0
total_sentences_in_corpus = len(corpus_f)
trusted_tuples_sen, untrusted_tuples_sen = [], []
trusted_tuples_sen_indices, untrusted_tuples_sen_indices = [], []
graph_sen, POS_sen = [], []

lencorpus = len(corpus_f)

bar = progressbar.ProgressBar(maxval=lencorpus, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for i, line in enumerate(corpus_f):#[0:3]:
	#print 'Sentences to go: ', total_sentences_in_corpus - sentence_number
	#print line
	sentence_number += 1
	if line != '\n':
		#input = ' '.join(brown.sents()[i])
		input = line
		#input = "The jury further said in term-end presentments that the City Executive Committee , which had over-all charge of the election , `` deserves the praise and thanks of the City of Atlanta '' for the manner in which the election was conducted ."
		#print 'Input Sentence: ', input
		[trusted_tuples, untrusted_tuples, trusted_tuples_indices, untrusted_tuples_indices, graph, POS1] = get_tuples(input)
		#print 'size: ', len(trusted_tuples), len(untrusted_tuples)
		
		###### Comment out this section #########
		if len(untrusted_tuples)!=0:
			z = zip(untrusted_tuples, untrusted_tuples_indices)
			random.shuffle(z)
			untrusted_tuples, untrusted_tuples_indices = zip(*z)

		min_len = min(len(trusted_tuples), len(untrusted_tuples))
		#print '-'*60
		#print 'Positive tuples:\n'
		for ind in range(len(trusted_tuples)):
			x = trusted_tuples[ind]
			x_ind = trusted_tuples_indices[ind]
			print x
			#print x, [POS[i[0]:i[1]] for i in x_ind]
		'''print '-'*60
		print 'Negative tuples:\n'
		for x in untrusted_tuples[0:min_len]:
			print x'''

			

		trusted_tuples_sen.append(trusted_tuples)
		untrusted_tuples_sen.append(untrusted_tuples[0:min_len])
		trusted_tuples_sen_indices.append(trusted_tuples_indices)
		untrusted_tuples_sen_indices.append(untrusted_tuples_indices[0:min_len])
		graph_sen.append(graph)
		POS_sen.append(POS1)
		#########################################

		'''for x in trusted_tuples:
			print>>trusted_tuples_f, x
			#trsuted_tuples_f.write(x[0])
		print>>trusted_tuples_f, '\n'
		for x in untrusted_tuples:
			print>>untrusted_tuples_f, x
			#untrsuted_tuples_f.write(x)
		print>>untrusted_tuples_f, '\n'
		for x in untrusted_tuples[0:len(trusted_tuples)]:
			print>>untrusted_tuples_f_short, x
		print>>untrusted_tuples_f_short, '\n'''

	bar.update(i+1)
	sleep(0.1)
bar.finish()

with open('vars.pickle', 'w') as f:
	pickle.dump([trusted_tuples_sen, untrusted_tuples_sen, trusted_tuples_sen_indices, untrusted_tuples_sen_indices, POS_sen, graph_sen], f)


		#featureSets_trusted = [(BuildFeatures(tup, tup_indices, graph), 'trusted') for tup, tup_indices in zip(trusted_tuples, trusted_tuples_indices)]
		#featureSets_untrusted = [(BuildFeatures(tup, tup_indices, graph), 'untrusted') for tup, tup_indices in zip(untrusted_tuples, untrusted_tuples_indices)]
		#featureSets = featureSets_trusted + featureSets_untrusted
		#classifier = nltk.NaiveBayesClassifier.train(featureSets)
# "1:NN?,2:NNS?,3:NNP?,4:NNPS?,5:VB,VBD,VBG,VBN,VBP,VBZ,WDT,WH,WRB,WP,WP,dist btwn(e1,rij),is order(ei,rij,ej)?,scaled lenghts of ei,ej,rij,$"

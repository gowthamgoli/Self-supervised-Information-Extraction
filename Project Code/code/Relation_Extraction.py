import sys, random
from stanford import StanfordParser, PySentence
from Dijkstras import *

parser_file = sys.argv[1]
PARSER = StanfordParser(parser_file)


#input = 'The American Civil War, also known as the War Between the States or simply the Civil War, was a civil war fought from 1861 to 1865 in the United States after several Southern slave states declared their secession and formed the Confederate States of America.'

def get_index(id_str):
            return '-' if '.' in id_str else id_str

input = 'Messi has won seven La Liga titles and four UEFA Champions League titles, as well as three Copa del Rey titles.'


corpus_path = '/home/manbearpig/acads/NLP/project/open_ie/codes/test_corpus.txt'
corpus_f_large = open(corpus_path,'r').readlines()

for line in corpus_f_large:
	input = line
	#print '-'*150
	for sentence, prob in PARSER.get_most_probable_parses(input,kbest=1):
		words = PARSER.tokenize(input)
		word_ids = range(len(words))
		#print 'Parse Probability: ', prob
		#sentence.print_table()
		#print ''

		#print sentence.word
		#print '-'*50
		#print ''

		POS = []
		DEP = []
		for idx in sorted(sentence.word):
			#print 'idx: ', idx
			#print 'tag: ', sentence.tag.get(idx,'')
			#print 'dep: ', sentence.dep.get(idx,'')
			#print ''
			POS.append(sentence.tag.get(idx,''))
			DEP.append(sentence.dep.get(idx,''))

		#print 'words: ', words
		#print POS
		#print DEP

		Noun_ids = [idx+1 for idx in range(len(words)) if POS[idx] in [u'NNP',u'NN',u'NNPS',u'NNS']]#NNP(S)-> Proper Noun,Singular(Plural); #NN(S)->Noun(Plural)
		Verb_ids = [idx+1 for idx in range(len(words)) if POS[idx] in [u'VBN',u'VBP',u'VB',u'VBD',u'VBG',u'VBZ']]
		Nouns = [words[x-1] for x in Noun_ids]
		Verbs = [words[x-1] for x in Verb_ids]
		#print 'Nouns:', Nouns
		#print 'Verbs:', Verbs
		tuples_2D = [(i,j) for i in Noun_ids for j in Noun_ids if i < j]

		#print 'tuples_2D: ',tuples_2D
		#print ''

		tuples_3D = [(tup2[0],i,tup2[1]) for tup2 in tuples_2D for i in Verb_ids]
		#print 'tuples_3D:' ,tuples_3D
		#print ''
		
		#for e_i,e_j in 

		'''print 'Collapsed Dependencies:'
		tmpl = 'Head: %s (%d); dependent: %s (%d); relation: %s'
		for td in sentence.gs.typedDependenciesCollapsed():
		    head = td.gov()
		    head_idx = head.index()
		    dep = td.dep()
		    dep_idx = dep.index()
		    rel = td.reln()
		    print tmpl % (head.value(), head_idx, dep.value(), dep_idx, rel)
		print '-' * 80'''

		#weight = {'acomp':1,'advcl':1,'advmod':1,'agent':1,'amod':1,'appos':1,'attr':1,'aux':1,'auxpass':1,'cc':1,'ccomp':1,\
		#	  'conj':1,'cop':1,'csubj':1,'csubjpass':1,'dep':1}
		weight = 1
		graph = {}
		appendix = {}
		#tmpl = 'Head: %s (%d); dependent: %s (%d); relation: %s'
		for td in sentence.gs.typedDependenciesCollapsed():
			head = td.gov()
			head_idx = head.index()
			dep = td.dep()
			dep_idx = dep.index()
			rel = td.reln()
			#print tmpl % (head.value(), head_idx, dep.value(), dep_idx, rel)
			#print type(str(rel))
			if 'prep_' in str(rel):# and (head in Verbs or dep in Verbs):
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
		
		print 'graph:', graph
		#print 'appendix:', appendix

		trusted_tuples, untrusted_tuples = [], []
		for e_i,r_i_j,e_j in tuples_3D:
			shortest_path1 = shortestPath(graph,e_i,r_i_j)
			shortest_path2 = shortestPath(graph,r_i_j,e_j)
			#print shortest_path1, shortest_path2
			prefix, suffix = '', ''
			if tuple(shortest_path1) in appendix.keys():
				prefix = appendix[tuple(shortest_path1)] + ' '
			if tuple(shortest_path2) in appendix.keys():
				suffix = ' ' + appendix[tuple(shortest_path2)]
			#if len(shortest_path1)-1 <= 1 and len(shortest_path2)-1 <= 1:
			if len(shortest_path1)-1 + len(shortest_path2)-1 <= 2:
				trusted_tuples.append((words[e_i-1],str(prefix)+str(words[r_i_j-1])+str(suffix),words[e_j-1]))
				
			else:
				untrusted_tuples.append((words[e_i-1],str(prefix)+str(words[r_i_j-1])+str(suffix),words[e_j-1]))

		#print trusted_tuples
		#print untrusted_tuples

		random.shuffle(untrusted_tuples)
		print '-'*60
		print 'Positive tuples:\n'
		for x in trusted_tuples:
			print x
		print '-'*60
		min_len = min(len(trusted_tuples), len(untrusted_tuples))
		print 'Negative tuples:\n'
		for x in untrusted_tuples[0:min_len]:
			print x

		#print 'Shortest Path:', shortest_path, 'Shortest Distance:', len(shortest_path)-1

		#common_node, shortest_path = sentence.get_least_common_node(node_i_idx, node_j_idx)
	        #print 'Shortest Path:', shortest_path, 'Shortest Distance:', len(shortest_path)-1, 'Common Node:', common_node


		
		


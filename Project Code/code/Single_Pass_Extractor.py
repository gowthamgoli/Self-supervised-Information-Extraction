import nltk
from nltk.translate.ribes_score import position_of_ngram
import pickle
#a="Tendulkar won the golden trophy at Wisden cricket awards in 2011."
#a="Tendulkar won the ICC trophy at Wisden cricket awards in 2011."
#a="The American Civil War, also known as the War between the States or simply the Civil War, was a civil war fought from 1861 to 1865 in the United States after several Southern slave states declared their secession and formed the Confederate States of America."
a="The corner store was robbed last night."

sent=nltk.word_tokenize(a)
words = nltk.word_tokenize(a)
sent_pos=nltk.pos_tag(sent)

grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        
    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    VP:
        {<VBD><PP>?}
        {<VBZ><PP>?}
        {<VB><PP>?}
        {<VBN><PP>?}
        {<VBG><PP>?}
        {<VBP><PP>?}
"""

cp = nltk.RegexpParser(grammar)
result = cp.parse(sent_pos)

nounPhrases = []
nounPhrases_POS = []
nounPhrases_span = []

lastFoundpos_end = -1

for subtree in result.subtrees(filter=lambda t: t.label() == 'NP'):
  np = ''
  pos = ''
  for x in subtree.leaves():
    np = np + ' ' + x[0]
    pos = pos + ' ' + x[1]
  np = np.strip()

  foundPos =  position_of_ngram(tuple(np.split()), words[lastFoundpos_end+1:])
  foundPos = foundPos + lastFoundpos_end + 1
  foundPos_end = foundPos + len(np.split()) - 1
  lastFoundpos_end = foundPos_end
  nounPhrases_span.append((foundPos, foundPos_end))
  nounPhrases.append(np)
  nounPhrases_POS.append(pos.strip())

print nounPhrases
print nounPhrases_span
print nounPhrases_POS

verbPhrases = []
verbPhrases_POS = []
verbPhrases_span = []

lastFoundpos_end = -1

for subtree in result.subtrees(filter=lambda t: t.label() == 'VP'):

  vp = ''
  pos = ''
  for x in subtree.leaves():
    vp = vp + ' ' + x[0]
    pos = pos + ' ' + x[1]

  vp = vp.strip()

  foundPos = position_of_ngram(tuple(vp.split()), words[lastFoundpos_end+1:])
  foundPos = foundPos + lastFoundpos_end + 1
  foundPos_end = foundPos + len(vp.split()) - 1
  lastFoundpos_end = foundPos_end
  verbPhrases_span.append((foundPos, foundPos_end))
  verbPhrases.append(vp)
  verbPhrases_POS.append(pos.strip())

print verbPhrases
print verbPhrases_span
print verbPhrases_POS



def BuildFeatures(nounPhrases, nounPhrases_POS, nounPhrases_span, verbPhrases, verbPhrases_POS, verbPhrases_span):

  with open('classifier.dump') as f:
    classifier = pickle.load(f)

  for i in range(len(nounPhrases)):
    for j in range(i+1, len(nounPhrases)):
      for k in range(len(verbPhrases)):

        if nounPhrases[i].lower() not in nounPhrases[j].lower() and nounPhrases[j].lower() not in nounPhrases[i].lower():
          tup = (nounPhrases[i], verbPhrases[k], nounPhrases[j])
          
          features = {}
          pos_i = nounPhrases_POS[i].strip(' ')
          pos_j = nounPhrases_POS[j].strip(' ')
          pos_k = verbPhrases_POS[k].strip(' ')

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

          features['dist_ei_rij'] = abs(nounPhrases_span[i][1] - verbPhrases_span[k][0])
          features['dist_rij_ej'] = abs(verbPhrases_span[k][0] - nounPhrases_span[j][0])
          features['dist_ei_ej'] = abs(nounPhrases_span[i][1] - nounPhrases_span[j][0])

          if nounPhrases_span[i][1] < verbPhrases_span[k][0]:
            features['order_ei_rij'] = 'before'
          else:
            features['order_ei_rij'] = 'after'

          if verbPhrases_span[k][0] < nounPhrases_span[j][0]:
            features['order_rij_ej'] = 'before'
          else:
            features['order_rij_ej'] = 'after'

          if nounPhrases_span[i][1] < nounPhrases_span[j][0]:
            features['order_ei_ej'] = 'before'
          else:
            features['order_ei_ej'] = 'after'

          for pos in pos_i:
            if pos in ['NN','NNS','NNP','NNPS']:
              features['e_i_'+pos] = True

          for pos in pos_j:
            if pos in ['NN','NNS','NNP','NNPS']:
              features['e_j_'+pos] = True

          for pos in pos_k:
            if pos in ['VB','VBD','VBG','VBN','VBP','VBZ']:
              featues[pos] = True

          print tup, classifier.classify(features)

#BuildFeatures(nounPhrases, nounPhrases_POS, nounPhrases_span, verbPhrases, verbPhrases_POS, verbPhrases_span)
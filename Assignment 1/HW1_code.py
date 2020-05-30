import nltk
from nltk.corpus import brown,wordnet
from nltk.wsd import lesk

mapping = {'VB':['v'],
           'JJ':['a','s'],
           'ADV':['r']}
# Prepare a collection of intensity modifiers and Brown tagged sentences
intensifiers = ['very','highly','too','completely','extremely','especially']
brown_tagged_sents = brown.tagged_sents()
POS = {'VB':[(0,0),(1,1),(0,0),(1,1),(1,1),(1,1)],
       'JJ':[(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)],  ## (0,1) means very+JJ and highly+JJ, not JJ+very or JJ+highly
       'ADV':[(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]
      }           

pairs = [] #list of tuples ##An element of the form (word,syn,common_synset,intensifier_index,location(word 0:before; 1:after intensifier]           


## Create a "closest" synonym list for a word within the context of the given sentence with Lesk algorithm. 
def create_thesaurus(word,pos,sent):
    thesaurus = {}
    synset = lesk(sent,word,pos) #WSD synset of closest meaning
    
    if synset is not None:
        for lemma in synset.lemmas():
            thesaurus[lemma.name().lower()] = synset
        del thesaurus[word]  
    return thesaurus

## Make a list of synonyms of a given word (probably multiple POS)
def synonyms_finder(word,pos,sent):
    word = word.lower()
    pos = mapping[pos]
    synonyms = {} #Each set value is list
            
    for each in pos:
        thesaurus = create_thesaurus(word,each,sent)
        for syn in thesaurus:
            if syn not in synonyms: # If the word not in synonyms, them add it along with its synset
                synonyms[syn] = [thesaurus[syn]]
            elif thesaurus[syn] not in synonyms[syn]: # If the word already in synonyms but with another synset, append synset
                synonyms[syn].append(thesaurus[syn])
    return synonyms


## Store pairs information for retrieval after
def form_pair(word,pos,sent,intensifer_no,loc):
    #Use global var 'pairs'
    synonyms = synonyms_finder(word,pos,sent)
    
    for syn in synonyms:
        for common_synset in synonyms[syn]:
            pair = (word,syn,common_synset,intensifer_no,loc)
            pairs.append(pair)
    return

## Forward iteration to list out potential pairs
def create_pairs():
    for sent in brown_tagged_sents[:]:
        for idx,(token,pos) in enumerate(sent): 
            if token in intensifiers:

                # Index 0: word, Index 1: POS
                intensifier_no = intensifiers.index(token) #very:1, highly:2 etc
                if (idx-1>=0) and (sent[idx-1][1] in POS) and (POS[sent[idx-1][1]][intensifier_no][0]==1):
                    if not ((sent[idx-1][1] == 'VB') and (idx+1<len(sent) and (sent[idx+1][1] in POS))):
                        form_pair(sent[idx-1][0],sent[idx-1][1],sent,intensifier_no,0)

                if (idx+1<len(sent)) and (sent[idx+1][1] in POS) and (POS[sent[idx+1][1]][intensifier_no][1]==1):
                    form_pair(sent[idx+1][0],sent[idx+1][1],sent,intensifier_no,1)
    return

## Forward iteration to check if the pairs are valid
def trace_synonyms():
    ## Use global var 'pairs'
    syn_shortlist = set([each[1] for each in pairs])
    synset_shortlist = set([each[2] for each in pairs])
    word_shortlist = set(tuple([pair[0],pair[2]]) for pair in pairs)
    collections = [] ## Final output 
    
    for sent in brown_tagged_sents[:]:
        for idx,(token,pos) in enumerate(sent): 
            if (pos in POS) and (token in syn_shortlist): # Check if token is verb/adj/adv and is one of synonyms.
                pos_list = mapping[pos] # Convert to wordnet-styled POS
                for cv_pos in pos_list:
                    synset = lesk(sent,token,cv_pos) # Closest synset of the potential word
                    
                    # Check if the closest synset ever occurs + the token does not occur with an intensifier and the same synset 
                    if (synset in synset_shortlist) and (tuple([token, synset]) not in word_shortlist):
                        for pair in pairs:
                            if pair[1:3] == (token,synset):
                                if pair[4] == 0:
                                    item = [token,pair[0]+' '+intensifiers[pair[3]]]
                                    if item not in collections:
                                        collections.append(item)
                                else:
                                    item = [token,intensifiers[pair[3]]+' '+pair[0]]
                                    if item not in collections:
                                        collections.append(item)
    
    return collections


create_pairs()
collections = trace_synonyms()  
for p in collections[:]:
	print(p[0]+', '+p[1])

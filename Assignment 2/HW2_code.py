import nltk
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.wsd import lesk
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords 
import math
from collections import defaultdict
import string
  
UNI_WN_MAPPING = {'NOUN': 'n',
             'ADV': 'r',
             'ADJ': 's'}  

def createBigrams(sent,stopset):
	# '''
	# Create bigram of interested types from a sentence
	# sent: given sentence
	# stopset: excluded characters and words
	# '''
    tagged_sent = [(w.lower(),tag) for (w,tag) in sent if len(w)>2 and w not in stopset]
    length = len(tagged_sent)
    grammar = [['ADJ','NOUN'],['ADV','ADJ']] 

    bigrams = []
    for i in range(length):
        if i+1 < length:
            # Check non-English word and grammar satisfaction
            if (wn.synsets(tagged_sent[i][0])) and (wn.synsets(tagged_sent[i+1][0])) and ([tagged_sent[i][1],tagged_sent[i+1][1]] in grammar):
                pos_1 = UNI_WN_MAPPING[tagged_sent[i][1]]
                synsets = []
                for pos in pos_1:
                    synset = lesk(tagged_sent,tagged_sent[i][0],pos)
                    if synset:
                        synsets.append(synset)
                bigrams.append((tagged_sent[i][0] + ' ' + tagged_sent[i+1][0], 
                                tagged_sent[i][1] + ' ' + tagged_sent[i+1][1],
                                synsets))
    return bigrams

def createFineDist():
	# '''
	# Create Frequency Distribution for candidate bigrams
	# '''
    fine_dist = []
    for (p,pos,_) in fine_bigrams:
        w = p.split()
        tag = pos.split()
        fine_dist.append((w[0],tag[0]))
        fine_dist.append((w[1],tag[1]))

    return nltk.FreqDist(fine_dist)

def createSynset():
	# '''
	# Make a dictionary of Lesk synset for the collocates
	# '''
    fine_synsets = {p:synset for (p,_,synset) in fine_bigrams}
    return fine_synsets

def PMI_scoring(bigram,word_dist,fine_bigrams_dist,num_word,num_bigram):
	# '''
	# Compute point mutual information score for a given bigram
	# '''
    pmi_score = 0
    p = bigram.split()
    joint_occurence_prob = fine_bigrams_dist[bigram]/num_bigram
    prob_word_1 = word_dist[p[0]]/num_word
    prob_word_2 = word_dist[p[1]]/num_word
    raw_pmi = math.log(joint_occurence_prob,2) - math.log(prob_word_1,2) - math.log(prob_word_2,2)
    pmi_score = (raw_pmi/math.log(joint_occurence_prob,2) + 1)/2 ##Normalize into [0,1]
    return pmi_score

def freq_scoring(bigram,word_dist,fine_bigrams_dist,fine_bigrams_pos,num_word):
	# '''
	# Compute frequency score for a given bigram
	# '''
    p = bigram.split()
    pos_1 = fine_bigrams_pos[bigram].split()[0]
    pos_2 = fine_bigrams_pos[bigram].split()[1]
    
    bigram_freq_score = fine_bigrams_dist[bigram]/len(fine_bigrams_dist)
    word_freq_score = (word_dist[p[0]]+word_dist[p[1]])/(2*num_word)
    
    freq_score = bigram_freq_score/word_freq_score
    return freq_score

def swn_scoring(bigram,fine_bigrams_pos):
	# '''
	# Compute sentiment score for a given bigram
	# '''
    w1 = bigram.split()[0]
    w2 = bigram.split()[1]
    pos_1 = UNI_WN_MAPPING[fine_bigrams_pos[bigram].split()[0]]
    pos_2 = UNI_WN_MAPPING[fine_bigrams_pos[bigram].split()[1]]
    score_1 = 0
    score_2 = 0
    
    word_synsets = list(swn.senti_synsets(w1))
    size_1 = len(word_synsets)
    for synset in word_synsets:
        score_1 += synset.neg_score() + synset.pos_score()
  
    word_synsets = list(swn.senti_synsets(w2))
    size_2 = len(word_synsets)
    for synset in word_synsets:
        score_2 += synset.neg_score() + synset.pos_score()
    
    if size_1 != 0:
        score_1 = score_1/size_1 
    if size_2 != 0:
        score_2 = score_2/size_2
    return max(score_1,score_2)

def intensification_scoring(bigram,fine_synsets,intensifier_dict,stemmer):
	# '''
	# Compute intensification score for a given bigram
	# '''
    w1 = bigram.split()[0]
    w2 = bigram.split()[1]
    lesk_synsets = fine_synsets[bigram]
    score = 0
    synsets = wn.synsets(w1)
    count = 0
    w2_stem = stemmer.stem(w2)
    for sense in synsets:
        definition = sense.definition().translate(str.maketrans('','', string.punctuation))
        for w  in definition.split():
            if intensifier_dict[w] > 0:
                score += intensifier_dict[w]
                count += 1
                if stemmer.stem(w) == w2_stem: # Large bonus if collocate definition similar to base
                    score += 20
            if sense in lesk_synsets: # Bonus if lesk_synsets has intensifier signals
                score += 2*intensifier_dict[w]
    weighted = score
    if count != 0:
        weighted = weighted/count
    return weighted

def uniqueness_scoring(bigram,fine_word_dist,fine_bigrams_pos):
	# '''
	# Compute uniqueness score for a given bigram
	# '''
    w1 = bigram.split()[0]
    w2 = bigram.split()[1]
    pos_1 = fine_bigrams_pos[bigram].split()[0]
    pos_2 = fine_bigrams_pos[bigram].split()[1]

    # Better score if collocate and base has low frequency as separate words among candidate bigrams
    if fine_word_dist[(w1,pos_1)] * fine_word_dist[(w2,pos_2)] == 1:
        score = 1
    else:
        score = 1/math.log( fine_word_dist[(w1,pos_1)] * (fine_word_dist[(w2,pos_2)]) ,2)
        
    # Better score if collocate is not 'ly' adverb and adjective
    if (pos_1 == 'ADV') and (pos_2 == 'ADJ') and (len(w1) > 2) and (w1[-2:] == 'ly'):
        score = score/9
    elif (pos_1 == 'ADJ'):
        adv = w1+'ly'
        if fine_word_dist[(adv,'ADV')] > 1:
            score = score/fine_word_dist[(adv,'ADV')]
    
    # Better score if base has broader word family
    if (pos_2 != 'NOUN'):
        n_synsets = wn.synsets(w2)
        pos_list = []
        related_pos_list = []
        for sense in n_synsets:
            pos = sense.name().split('.')[1]
            if pos not in pos_list:
                pos_list.append(pos)
            
            lemmas = sense.lemmas()
            for lemma in lemmas:
                for form in lemma.derivationally_related_forms():
                    form_pos = form.synset().name().split('.')[1]
                    if form_pos not in related_pos_list:
                        related_pos_list.append(form_pos)
        score = score*(len(pos_list)*len(related_pos_list))
        
    return score

def overall_scoring(word_dist,bigram_dist,num_word,num_bigram,fine_bigrams,fine_bigrams_dist,fine_bigrams_pos):
	# '''
	# Compute overall score for candidate bigrams
	# '''
	stemmer = PorterStemmer()
	word_score = {}
	intensifier_dict = defaultdict(int) # Intensifier signal words
	intensifier_dict.update({
		'intensifier':3,
		'intensifiers':3,
		'intensifying':3,
		'intensification':3,
		'intense':2,
		'extreme':2,
		'extremely':2,
		'completely':0.5,
		'complete':0.5,
		'very':0.5,
		'highly':0.5,
		'degree':0.5,
		'extent':0.5,
		'amount':0.5
		})

	word_score = {};
	for word in list(fine_bigrams_dist):
		swn_score = swn_scoring(word,fine_bigrams_pos)
		pmi_score = PMI_scoring(word,word_dist,fine_bigrams_dist,num_word,num_bigram)
		freq_score = freq_scoring(word,word_dist,fine_bigrams_dist,fine_bigrams_pos,num_word)
		intensification_score = intensification_scoring(word,fine_synsets,intensifier_dict,stemmer)
		uniqueness_score = uniqueness_scoring(word,fine_word_dist,fine_bigrams_pos)
		word_score[word] = swn_score*pmi_score*freq_score*intensification_score*uniqueness_score
	return word_score


# Get corpus material and define pos mapping
sents = brown.tagged_sents(tagset = 'universal')
stopset = set(stopwords.words('english')) 

# Prapare word and bigram dictionaries
words = [w.lower() for w in brown.words() if len(w)>2 and w not in stopset]
word_dist = nltk.FreqDist(words)
all_bigrams = nltk.bigrams(nltk.word_tokenize(' '.join(words)))
bigram_dist = {' '.join(k):v for k,v in nltk.FreqDist(nltk.bigrams(words)).items()}
num_word = len(words)
num_bigram = len(list(all_bigrams))

# Filter potential bigrams
fine_bigrams = []
for sent in sents:
    fine_bigrams += (createBigrams(sent,stopset))
for w in fine_bigrams:
    if len(w[0].split()) > 2:
         del fine_bigrams
fine_bigrams_pos =  {bigram:pos for (bigram,pos,_) in fine_bigrams}
fine_bigrams_dist = nltk.FreqDist([w[0] for w in fine_bigrams])  
fine_word_dist = createFineDist()
fine_synsets = createSynset()  

# Evaluation
sorted_rank = [k for k,v in sorted(overall_scoring(word_dist,bigram_dist,num_word,num_bigram,fine_bigrams,fine_bigrams_dist,fine_bigrams_pos).items(), key=lambda item: item[1], reverse=True) if v>0.05]
print(sorted_rank[:100])




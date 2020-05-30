from wiktionaryparser import WiktionaryParser
import re
import nltk
from nltk.corpus import brown,cmudict,stopwords,treebank,semcor,nps_chat
from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np
import spacy
import string
import warnings
from pywsd import disambiguate
warnings.filterwarnings("ignore") ## Ignore warning for empty word vector
joke_data_dir = '.\joke_dataset\\' ## Directory for the Joke dataset

## The function below used for step (1) as in report
def get_het_from_corpus(corpus):
    '''
    Create list of heteronyms from corpus based on number of CMU's pronunciation and wn.synset's part-of-speech
    '''
    
    pos = ['NOUN','VERB','ADJ']
    heteronyms = {}
    
    for (word,tag) in corpus:
        if (word.isalpha()) and (word not in stopset) and (tag in pos) and (word in pron_dict) and (len(pron_dict[word]) > 1):            
            if word not in heteronyms:
                heteronyms[word.lower()] = [tag]
            elif (word in heteronyms) and (tag not in heteronyms[word]):
                heteronyms[word].append(tag)
                
    heteronyms = {k:v for k,v in heteronyms.items() if len(v) > 1} 
    heteronyms = {k:v for k,v in heteronyms.items() if len(k)>=3 and k[-3:] != 'ing'}
    
    
    keys_to_remove = []
    for het in heteronyms:
        wn_pos = set(sense.pos() for sense in wn.synsets(het))
        if len(wn_pos) <= 1:
            keys_to_remove.append(het)
    for k in keys_to_remove:
        heteronyms.pop(k,None)
    
    
    data = pd.DataFrame(columns=['word'], dtype='object')
    data['word'] = pd.Series([k for k,v in heteronyms.items()])
    data.sort_values(['word'], ascending = [1], inplace = True)
    data.reset_index(drop = True, inplace = True)
    return data


## The following 4 functions used for step (2)
def init_wikparser():
    '''
    Initialize wiktionary parser
    '''
    parser = WiktionaryParser()
    RELATIONS = [
        "synonyms", "antonyms", "hypernyms", "hyponyms",
        "meronyms", "holonyms", "troponyms", "related terms",
        "coordinate terms",
    ]
    for rel in RELATIONS:
        parser.exclude_relation(rel)
        
    return parser

def skip_helper(each,SKIP_MARK,SKIP_MARK_EXCEP):
    '''
    Return 1 if take the ipa (not skipped)
    '''
    skip = [mark for mark in SKIP_MARK if mark in each]
    skip_excep = [mark for mark in SKIP_MARK_EXCEP if mark in each]
    return not (skip and not skip_excep)

def merge_def_example(item):
    return item['text'][1:] + item['examples']

def get_pronunciation(parser,data):
    '''
    Aggregate POS, definition and pronunciation from Wiktionary
    '''
    ## Below are signal for pronunciation collection and halting. However, Wiktionary is extremely non-machine-readable,
    ## we need to make a list of exceptions (but they'll never be sufficiennt)
    STOP_MARK = ['Rhymes', 'Hyphenation','Received Pronunciation']
    SKIP_MARK = ['archaic','obsolete','UK','also','General Australian','Received Pronunciation', 'Dialects', 'Dialect', 'Indian English','non-standard','accents','NYC','instrusive','New Zealand','cot–caught merger','Conservative RP']
    SKIP_MARK_EXCEP = ['UK, US', 'US, UK']
    POS_MARK = ['Noun:','Adjective:','Verb:']
    POS_TAKEN = ['adjective', 'noun', 'verb']
    
    wordlist = list(data['word'])
    data['pos'] = pd.Series(dtype='object')
    data['pronunciation'] = pd.Series(dtype='object')
    data['definition'] = pd.Series(dtype='object')
    
    for het in wordlist:
        word = parser.fetch(het)
        
        if not (word):
            continue
            
        pos_list = []
        def_list = []
        
        for sense in word:
            etym_pos = [item['partOfSpeech'] for item in sense['definitions'] if item['partOfSpeech'] in POS_TAKEN]
            etym_def = [merge_def_example(item) for item in sense['definitions']]

            if etym_pos and etym_def:
                pos_list.append(etym_pos)
                def_list.append(etym_def)
                
        for i,row in data[data['word'] == het].iterrows():
            data['pos'][i] = pos_list
            data['definition'][i] = def_list
            

        pron_text = word[0]['pronunciations']['text']
        prons = []
        etym_pron = []
        

        for i,each in enumerate(pron_text):
            temp = re.findall(r'IPA.*?/.*?/', each)


            if (temp) and (skip_helper(each,SKIP_MARK,SKIP_MARK_EXCEP)):
                ipa = temp[0][temp[0].find('/')+1:-1]

                etym_pron.append(ipa)
    
            if (([mark for mark in STOP_MARK if mark in each]) or (i+1 == len(pron_text))) and (len(etym_pron) > 0):
                prons.append(etym_pron)
                etym_pron = []


        for i,row in data[data['word'] == het].iterrows():
            data['pronunciation'][i] = prons
            
    data[['pos','pronunciation','definition']].replace('  ', np.nan, inplace=True)
    data = data.dropna(subset=['pos','pronunciation','definition'])
    data.reset_index(drop = True, inplace = True)
    
    return data

## The next function filters out true heteronyms from potential ones (still in step (2))
def fine_graining(data):
    fine_data = pd.DataFrame(columns = ['word','pos','pronunciation','definition'])
    
    ## Traverse throughout data and parse into fine_data with structure adjusted
    for i,row in data.iterrows():
        
        ## Assert to have same number of etymologies
        if len(row['pos']) != len(row['pronunciation']):
            continue
        
        ## Traverse through etyms
        for etym_idx,etym in enumerate(row['pos']):
            #print('traverse etym', etym)
            
            ## Copy pairs (pos,pron,def) in order if number of pos <= number of prons
            if len(etym) <= len(row['pronunciation'][etym_idx]):
                for pos_idx, pos in enumerate(etym):
                    new_row = {'word':row['word'], 'pos':pos,
                               'pronunciation':row['pronunciation'][etym_idx][pos_idx], 'definition':row['definition'][etym_idx][pos_idx]}
                    fine_data = fine_data.append(new_row, ignore_index=True)
                    
            else:
                adj_vrb_idx = -1
                adj_vrb = ['adjective', 'verb']
                last_idx = -1
                
                for pos_idx, pos in enumerate(etym):
                    if (pos_idx < len(row['pronunciation'][etym_idx])):
                    
                        if ((pos in adj_vrb) and (adj_vrb_idx == -1)) or (pos not in adj_vrb):
                            new_row = {'word':row['word'], 'pos':pos,
                                       'pronunciation':row['pronunciation'][etym_idx][pos_idx], 'definition':row['definition'][etym_idx][pos_idx]}
                            fine_data = fine_data.append(new_row, ignore_index=True)
                            last_idx = pos_idx
                            
                            if (pos in adj_vrb):
                                adj_vrb_idx = pos_idx
                        
                        ## Use pronunciation of verb to assign for adj, and vice versa
                        elif (pos in adj_vrb) and (adj_vrb_idx > -1):
                            new_row = {'word':row['word'], 'pos':pos,
                                       'pronunciation':row['pronunciation'][etym_idx][adj_vrb_idx], 'definition':row['definition'][etym_idx][adj_vrb_idx]}
                            fine_data = fine_data.append(new_row, ignore_index=True)
                            
                            
                    ## Run out of available pronunciation        
                    else:
                        if (pos in adj_vrb) and (adj_vrb_idx > -1):
                            new_row = {'word':row['word'], 'pos':pos,
                                       'pronunciation':row['pronunciation'][etym_idx][adj_vrb_idx], 'definition':row['definition'][etym_idx][adj_vrb_idx]}   
                            fine_data = fine_data.append(new_row, ignore_index = True)
                            
                        else:
                            new_row = {'word':row['word'], 'pos':pos,
                                       'pronunciation':row['pronunciation'][etym_idx][last_idx], 'definition':row['definition'][etym_idx][last_idx]} 
                            fine_data = fine_data.append(new_row, ignore_index = True)
                            
    
    ## Trim out redundant note in definition
    for i,row in fine_data.iterrows():
        text = row['definition']
        new_text = []
        
        for line in text:
            new_line = line
            
            ## Strip out content in parentheses
            paren = [new_line.find('('), new_line.find(')')]
            if paren[0] > -1 and paren[1] > -1:
                new_line = new_line[:paren[0]] + new_line[paren[1]+1:]
              
            ## Strip out content in brackets
            bracket = [new_line.find('['), new_line.find(']')]
            if bracket[0] > -1 and bracket[1] > -1:
                new_line = new_line[:bracket[0]] + new_line[bracket[1]+1:] 
                
            new_line = new_line.strip()
            new_text.append(new_line)
            
        fine_data['definition'][i] = new_text
        
        
    ## Remove words with one pronunciation (homograph) to make complete list of heteronyms
    wordset = set(fine_data['word'])
    word_del = []
    
    for word in wordset:
        pronset = set(fine_data[fine_data['word'] == word]['pronunciation'])
        if len(pronset) <= 1:
            fine_data = fine_data.drop(fine_data[fine_data['word'] == word].index)
            
    fine_data.reset_index(drop = True, inplace = True)                        
    return fine_data


## The next three functions cluster wordnet senses to a set of coarser Wiktionary sense as in step (3)
def eval_similarity(def_text,sense):
    sense_def = nltk.word_tokenize(sense.definition())
    sense_def = ' '.join([str(t) for t in sense_def if (t not in stopset) and (t not in string.punctuation)])
    main_def = nlp(sense_def)
    max_score = -1
    
    for line in def_text:
        line = ' '.join([str(t) for t in nltk.word_tokenize(line)  if (t not in stopset) and (t not in string.punctuation)])
        search_def = nlp(line)
        score = main_def.similarity(search_def)
        max_score = max(max_score,score)
        
    return max_score 

def flatten_data(fine_data):
    '''
    Flatten out to make one row for one synset
    '''
    ## Drop row with no sense
    fine_copy = fine_data.copy()
    fine_copy = fine_copy.dropna()     
    fine_copy.reset_index(drop = True, inplace = True)   
    
    
    flattened_data = pd.DataFrame(columns = ['word','pos','pronunciation','definition','sense'])  
    for i,row in fine_copy.iterrows():
        for sense in row['sense']:
            new_row = {'word':row['word'],
                      'pos':row['pos'],
                      'pronunciation':row['pronunciation'],
                      'definition':row['definition'],
                      'sense':sense}
            flattened_data = flattened_data.append(new_row, ignore_index = True)
    
    return flattened_data


def cluster_wn_sense(fine_data,nlp):
    '''
    Cluster and assign wn senses of words
    '''
    pos_mapping = {'noun':['n'],
              'verb':['v'],
              'adjective':['a','s']}
    
    fine_data['sense'] = pd.Series(dtype='object')#np.NaN #fine_data.apply(lambda x: [], axis=1)
    wordset = set(fine_data['word'])
    
    for word in wordset:
        sense_score_list = []
        
        for i,row in fine_data[fine_data['word'] == word].iterrows():
            
            ## def_text and synsets: list of definitions and wn.synsets of a row
            def_text = row['definition']
            synsets = []
            
            wn_pos = pos_mapping[row['pos']]
            for pos in wn_pos:
                synsets += wn.synsets(word, pos)
                
            for sense in synsets:
                sense_score_list.append((sense,i,eval_similarity(def_text, sense)))
                
                      
        for synset in wn.synsets(word):
            max_score = -1
            max_score_idx = -1
            
            ## Update highest score for each synset
            for (sense,row_idx,score) in sense_score_list:
                if (sense == synset) and (score > max_score):
                    max_score = score
                    max_score_idx = row_idx
                    
    
            ## Assign synset to row of highest similarity 
            if (max_score > 0.6) and (max_score_idx > -1):
                if type(fine_data['sense'][max_score_idx]) is list:
                    fine_data['sense'][max_score_idx].append(synset)
                    
                else:
                    fine_data['sense'][max_score_idx] = [synset]                
          
    return flatten_data(fine_data)

## Extract heteronyms for SemCor dataset
def process_semcor(ref_dict):
    '''
    Return a DataFrame that contrains sentences along with citations and information of detected heteronyms
    '''
    sents = semcor.sents()
    tagged_sents = semcor.tagged_sents(tag = 'sem')
    
    sense_list = list(ref_dict['sense'])
    semcor_sents = pd.DataFrame(columns = ['sentence', 'citation', 'heteronym'])
    word_duplicate_sense = set(ref_dict[ref_dict.duplicated(['sense'])]['word'])
    
    for sent_idx,sent in enumerate(tagged_sents):
        het_in_sent = []
        for token_idx,token in enumerate(sent):
            
            if type(token) == nltk.Tree:          
                lemma = token.label()
                chunk = token.leaves()
                
                ## Check whether token is a heteronym
                if (type(lemma) == nltk.corpus.reader.wordnet.Lemma) and (lemma.synset() in sense_list) and (len(chunk) == 1):
                    
                    synset = lemma.synset()
                    word = chunk[0]
                    
                    
                    ## Take care of sense-duplcated heteronyms (rare),
                    ## e.g. project and projects can have same sense but different pronunciations.
                    if word.lower() in word_duplicate_sense:
                        pron = list(ref_dict[(ref_dict['word'] == word.lower()) & (ref_dict['sense'] == synset)]['pronunciation'])
                        if pron:
                            het_in_sent.append((word.lower(), synset, pron[0]))

                    
                    ## If sense if not duplicated, mapping to pron is one-to-one
                    else:
                        pron = list(ref_dict[ref_dict['sense'] == synset]['pronunciation'])[0]
                        word_in_ref = list(ref_dict[ref_dict['sense'] == synset]['word'])[0]
                        if word.lower() == word_in_ref:
                            het_in_sent.append((word_in_ref, synset, pron))
                            


        if het_in_sent:
            new_row = {'sentence':  "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in sents[sent_idx]]).strip(),
                        'citation':'SemCor',
                        'heteronym':het_in_sent}
            semcor_sents = semcor_sents.append(new_row, ignore_index = True)
            
    return semcor_sents

## Read and extract heteronyms for Joke dataset
def read_joke_data():
    '''
    Read text from three sources of jokes and merge together
    '''
    reddit = 'reddit_jokes.json'
    stupidstuff = 'stupidstuff.json'
    wocka = 'wocka.json'
    
    ## Reddit Jokes
    temp_1 = pd.read_json(joke_data_dir + reddit)
    temp_1['body'] = temp_1['title'].map(str) + '. ' + temp_1['body']   
    temp_1.drop(['id','score', 'title'], axis=1, inplace = True)
    temp_1['citation'] = 'Reddit'
    
    ## Stupidstuff
    temp_2 = pd.read_json(joke_data_dir + stupidstuff)
    temp_2.drop(['id','category','rating'], axis=1, inplace=True)
    temp_2['citation'] = 'Stupidstuff'
    
    ## Wocka
    temp_3 = pd.read_json(joke_data_dir + wocka)
    temp_3.drop(['id','category','title'], axis=1, inplace=True)
    temp_3['citation'] = 'Wocka'
    
    joke_data = pd.concat([temp_1,temp_2,temp_3], ignore_index = True)   
    joke_data.rename({'body':'sentence'}, axis='columns', inplace = True)
    return joke_data

def process_jokes(joke_data, ref_dict):
    '''
    Return a DataFrame that contrains sentences along with citations and information of detected heteronyms
    '''
    
    sense_list = list(ref_dict['sense'])
    het_list = set(ref_dict['word'])
    word_duplicate_sense = set(ref_dict[ref_dict.duplicated(['sense'])]['word'])
    
    joke_sents = pd.DataFrame(columns = ['sentence', 'citation', 'heteronym'])
    
    for i,row in joke_data.iterrows():
        ## List of sentence in 1 joke. Sometimes jokes do not have proper punctuation.
        ## We may see 2-3 sentence in 1 row in the out result if it fails to decompose text to separate sentences.
        sents = nltk.sent_tokenize(row['sentence']) 
        
        for sent in sents:
            
            het_in_row = []
            text_token = [w.lower() for w in nltk.word_tokenize(sent) if (w not in string.punctuation) and (w.lower() not in stopset)]
            het_occur = set(text_token).intersection(het_list)
            
            if not het_occur:
                continue
        
        
            for (word,synset) in disambiguate(sent):
                if (word in het_list) and (synset) and (synset in sense_list):

                    ## Take care of sense-duplcated heteronyms (rare),
                    ## e.g. project and projects can have same sense but different pronunciations.
                    if word.lower() in word_duplicate_sense:
                        pron = list(ref_dict[(ref_dict['word'] == word.lower()) & (ref_dict['sense'] == synset)]['pronunciation'])
                        if pron:
                            het_in_row.append((word.lower(), synset, pron[0]))

                    
                    ## If sense if not duplicated, mapping to pron is one-to-one
                    else:
                        pron = list(ref_dict[ref_dict['sense'] == synset]['pronunciation'])[0]
                        word_in_ref = list(ref_dict[ref_dict['sense'] == synset]['word'])[0]
                        if word.lower() == word_in_ref:
                            het_in_row.append((word_in_ref, synset, pron))
                    
          
            if het_in_row:
                new_row = {'sentence':sent,
                            'citation':row['citation'],
                            'heteronym':het_in_row}
                joke_sents = joke_sents.append(new_row, ignore_index = True)
            
    
    return joke_sents



## The next three functions help rank the sentences obtained
def back_mapping(ref_dict):
    '''
    Map wordnet synset back to reference dictionary
    '''

    ## Construct senses for reference dictionary
    ## Create mapping from wordnet synset to the new senses, this mapping is many-to-one
    back_mapping = {}
    
    for word in set(ref_dict['word']):
        for pos in set(ref_dict[ref_dict['word'] == word]['pos']):
            
            def_set = set(ref_dict[(ref_dict['word'] == word) & (ref_dict['pos'] == pos)]['definition'].apply(lambda x:tuple(x)))
            for index,definition in enumerate(def_set):
                sense_name = word + '.' + pos + '.' + str(index + 1)
                wn_sense_list = list(ref_dict[(ref_dict['word']==word) & (ref_dict['pos']==pos) & (ref_dict['definition'].apply(lambda x:x==list(definition)))]['sense'])
                for wn_synset in wn_sense_list:
                    back_mapping[wn_synset] = sense_name

    return back_mapping

def count_num_of_pos(synset_list):
    '''
    Count the number of POS of a word given its synsets (for criterion 3)
    '''
    pos_set = set([synset.split('.')[1] for synset in synset_list])
    return len(pos_set)

def rank_sent(table, ref_dict):
    '''
    Sort sentences following five homograph-involved criteria:
    1. Sentences with more occurences of homographs > Sentences with less occurences of homographs
    2. Sentences with multiple occurences of homographs > Sentences with heteronyms but not homographs
    3. Sentences with heteronyms of the same part-of-speech > Sentences with heteronyms of different part-of-speechs
    4. (Supplementary) Sentences with more occurrences of pure heteronyms > Sentences with less occurrences of pure heteronyms
    5. (Supplementary) Sentences with more pronunciations of heteronyms > Sentences with less pronunciations of heteronyms
    '''
    ## Homographs only count when occuring in pairs. But a single heteronym already count.
    ## Homographs differ in wn.synset but haves same spelling (regardless of pronunciation)
    
    table['single_heteronym'] = [[]] * len(table)
    table['duplicate_heteronym'] = [[]] * len(table)
    table['homograph'] = [[]] * len(table)
    table['number_of_single_heteronym'] = 0
    table['number_of_duplicate_heteronym'] = 0
    table['number_of_homograph'] = 0 
    
    ## For criterion 3, only need to count number of homographs's pos when 1. and 2. satisfied
    table['number_of_pos'] = 0 
    table['number_of_pronunciation'] = 0
    
    sense_backmap = back_mapping(ref_dict)
    
    for row_idx, row in table.iterrows():
        het_list = row['heteronym'] ## List of tuples
        het_list = [(spelling, sense_backmap[synset], pronunciation) for (spelling,synset,pronunciation) in het_list]
        
        homograph_lookup = pd.DataFrame(het_list,columns = ['spelling', 'synset', 'pronunciation'])         
        word_list = set(homograph_lookup['spelling'])
        
        
        for word in word_list:
            num_synset = homograph_lookup[homograph_lookup['spelling'] == word]['synset'].nunique()
            num_pronunciation = homograph_lookup[homograph_lookup['spelling'] == word]['pronunciation'].nunique()
            
            if num_synset == 1:
                ## A spelling is pure heteronym if it has only 1 synset
                
                num_occur = len(homograph_lookup[homograph_lookup['spelling'] == word])
                
                if num_occur == 1:
                    ## Once-occuring pure heteronym
                    
                    num_of_pos = 1
                    table['single_heteronym'][row_idx] = table['single_heteronym'][row_idx] + [(word,num_occur,num_of_pos)]
                    table['number_of_pronunciation'][row_idx] += num_pronunciation
                    
                else:
                    ## Many-occuring pure heteronym
                    
                    num_of_pos = 1
                    table['duplicate_heteronym'][row_idx] = table['duplicate_heteronym'][row_idx] + [(word,num_occur,num_of_pos)]
                    table['number_of_pronunciation'][row_idx] += num_pronunciation
                    
            else:
                ## A spelling contains homographs if it has more than 1 synset
                
                num_of_pos = count_num_of_pos(list(homograph_lookup[homograph_lookup['spelling'] == word]['synset']))
                table['homograph'][row_idx] = table['homograph'][row_idx] + [(word,1,num_of_pos)]
                table['number_of_pos'][row_idx] +=  num_of_pos
                table['number_of_pronunciation'][row_idx] += num_pronunciation
                
                
        table['number_of_single_heteronym'] = table['single_heteronym'].apply(lambda x:len(x))
        table['number_of_duplicate_heteronym'] = table['duplicate_heteronym'].apply(lambda x:len(x)) 
        table['number_of_homograph'] = table['homograph'].apply(lambda x:len(x))  
                 
        table.sort_values(['number_of_homograph','number_of_pos','number_of_duplicate_heteronym','number_of_pronunciation'], ascending=[False,True,False,False], inplace=True)  
    return table
          

## Save .csv and display on console
def save_output(output_data,number,filename='HW3_output.csv'):
    '''
    Convert data format and save into .csv
    '''
    output_data = output_data.copy(deep=True).drop_duplicates(subset=['sentence'], keep='first')[:number]
    output_data.reset_index(drop = True, inplace = True)
    format_output = pd.DataFrame()
    format_output['sentence'] = output_data['sentence']
    format_output['citation'] = output_data['citation']
    
    for row_idx,row in output_data.iterrows():
        tag_list = row['heteronym']
        sent = row['sentence']
        current_loc = 0
        
        for i,tag in enumerate(tag_list):
            spelling,sense,pron = tag
            info = '(' + str(sense) + ',' + pron + ')'
            loc = sent.find(spelling,current_loc)
            sent = sent[:loc+len(spelling)] + info + sent[loc+len(spelling):]
            current_loc = loc + len(spelling) + len(info)
        format_output['sentence'][row_idx] = sent
        print(sent, '--- Source: ', row['citation'])
    
    format_output.to_csv(filename, index=False, header=0)
    return



## Set up basic corpora
pron_dict = cmudict.dict()
brown_words = brown.tagged_words(tagset = 'universal')
treebank_words = treebank.tagged_words(tagset = 'universal')
nps_words = nps_chat.tagged_words(tagset = 'universal')
corpus = brown_words + treebank_words + nps_words
corpus = [(word.lower(),tag) for (word,tag) in corpus]
stopset = set(stopwords.words('english'))  
## Set up pretrained spaCy's word vector
nlp = spacy.load('en_core_web_lg') 



## Collect potential heteronyms
data = get_het_from_corpus(corpus)


## Assign Wiktionary data to the potential heteronyms
parser = init_wikparser()
data = get_pronunciation(parser,data)
fine_data = fine_graining(data)

## Create reference dictionary for heteronyms
ref_dict = cluster_wn_sense(fine_data, nlp)


## Extract heteronyms from some datasets and merge together
semcor_sents = process_semcor(ref_dict)
joke_data = read_joke_data()
joke_sents = process_jokes(joke_data,ref_dict)
combined_sents = semcor_sents.append(joke_sents, ignore_index = True)

## Rank the sentence based on given criteria
ranked_sents = combined_sents.copy()
ranked_sents = rank_sent(ranked_sents,ref_dict)

## Save and display
save_output(ranked_sents,30)





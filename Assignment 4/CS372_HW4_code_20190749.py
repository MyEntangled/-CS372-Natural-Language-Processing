import pandas as pd
import re
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.tokenize import TreebankWordTokenizer
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
from nltk.tag import ClassifierBasedTagger,RegexpTagger
from nltk.chunk import ChunkParserI
from collections import Iterable
import string
import warnings

def extract_annotation(data_train):
	'''
	Extract relation annotations of training data store them in machine accessible way
	'''

	columns = ['word','sentence','annotation','is_train']
	data = pd.DataFrame(columns = columns)
	for row_idx, row in data_train.iterrows():
	    anno_str = row['annotation']
	    annos = re.findall('\<(.*?)\>',anno_str)
	    annos = [tuple(map(str.strip, each.split(','))) for each in annos]
	    for anno in annos:
	        new_row = {'word':row['word'], 'sentence':row['sentence'], 'annotation':anno, 'is_train':row['is_train']}
	        data = data.append(new_row, ignore_index=True)
	return data

def reassign_head_helper(anno):
	'''
	Reassign headword that signifies the action
	This executes after a sentence with many annotations get duplicated
	'''
	HEADWORDS = ['activate','inhibit','bind','accelerate','decelerate']
	main = anno[1]
	for i in HEADWORDS:
	    if i in main:
	        return i

def preprocess(dataset):
    ''' 
    Duplicate a sentence correspond to the number of annotations it has.
    Decompress complex annotations with "and" and "or" in subject and object into separate annotations.
    After this, each sentence has one relation annotation.
    '''

    dataset['word'] = dataset['annotation'].apply(lambda anno:reassign_head_helper(anno))
    
    ## Decompress complex annotation
    temp = pd.DataFrame(columns = dataset.columns)
    for i,row in dataset.iterrows():
        sub,act,obj = row['annotation']
        list_of_subs = re.split(' and | or ',sub)
        list_of_objs = re.split(' and | or ',obj)

        for s in list_of_subs:
            for o in list_of_objs:
                new_row = {'word':row['word'], 'sentence':row['sentence'], 'annotation':(s,act,o)}
                temp = temp.append(new_row, ignore_index=True)
    dataset = temp
    return dataset
    
def upsample(dataset):
	'''
	Upsample training data by generating new sentences by replacing their old SUBJECT and OBJECT
	with subjects and objects of another sentence. (Those 2 sentences must have the same ACTION)
	'''
	temp = pd.DataFrame(columns = dataset.columns)
	groups = dataset.groupby('word')

	for name,group in groups:
	    for idx_1,row_1 in group.iterrows():
	        for idx_2,row_2 in group.iterrows():
	            sent = row_1['sentence']
	            
	            sub_1,act_1,obj_1 = row_1['annotation']                
	            sub_2,act_2,obj_2 = row_2['annotation']
	            
	            sent = sent.replace(sub_1,sub_2).replace(act_1,act_2).replace(obj_1,obj_2)
	            
	            new_row = {'word':row_1['word'], 'sentence':sent, 'annotation':row_2['annotation']}
	            temp = temp.append(new_row, ignore_index=True)

	dataset = temp.drop_duplicates(subset=None, keep='first', inplace = False)
	return dataset

def refine_pos(token,pos):
    '''
    Fix POS for 5 ACTION words
    '''
    forms = set(('activate','activates','activated','inhibit','inhitbits','inhibited','bind','binds','binded','accelerate','accelerates','accelerated','decelerate','decelerates','decelerated'))
    if token not in forms:
        return tuple((token,pos))
    else:
        if token[-1] == 's':
            return tuple((token,'VBZ'))
        elif token[-1] == 'ed':
            return tuple((token,'VBD'))
        else:
            return tuple((token,'VBP'))
        
def tag_token(token, pos, iob):
    line = ((token,pos),iob)
    return line

def IOB_tag(dataset):
	'''
	Produce IOB tags for training set based on the annotations.
	IOB tags include B-SUB, I-SUB, ACTION, B-OBJ, I-OBJ, O
	'''
	tokenizer = TreebankWordTokenizer()
	dataset['IOB'] = pd.Series()

	for row_idx,row in dataset.iterrows():
	    
	    sub = row['annotation'][0]
	    obj = row['annotation'][2]
	    action = row['annotation'][1]
	    head = row['word'][:3].upper()
	    
	    ## Get list of all SUB/OBJ/ACTION of the sentence 
	    list_of_subs = tokenizer.tokenize(sub)
	    list_of_objs = tokenizer.tokenize(obj)
	    list_of_act = tokenizer.tokenize(action)
	    sent = row['sentence']
	    

	    ## Adjoin proper noun with a dot in between
	    temp = tokenizer.tokenize(sent)
	    token_list = []
	    for i,token in enumerate(temp):
	        if (token.isalpha()) and (token[0].islower()) and (i-1>=0) and (temp[i-1][-1] == '.'):
	            token_list[-1] = token_list[-1] + token
	        else:
	            token_list.append(token)
	        
	        
	    pos_list = [refine_pos(t,pos)[1] for (t,pos) in pos_tag(token_list)]
	    iob_txt = []
	    inside_sub = False
	    inside_obj = False
	    inside_act = False
	    end_sub = -1
	    end_obj = -1
	    end_act = -1
	    
	    for token_idx,token in enumerate(token_list):

	    	## Assign IOB tags for SUBs
	        if (token == list_of_subs[0]):
	            
	            if (token_idx+len(list_of_subs) < len(token_list)) and (token_list[token_idx : token_idx+len(list_of_subs)] == list_of_subs):
	                if len(list_of_subs) == 1:
	                    line =((token, pos_list[token_idx]), 'I-SUB')
	                else:
	                    line = ((token, pos_list[token_idx]), 'B-SUB')

	                iob_txt.append(line)
	                inside_sub = True
	                end_sub = token_idx + len(list_of_subs)
	                continue
	            else:
	                line = ((token, pos_list[token_idx]), 'O')
	                iob_txt.append(line)
	                continue
	                
	        elif (token in list_of_subs) and (inside_sub == True):
	            line = ((token, pos_list[token_idx]), 'I-SUB')
	            iob_txt.append(line)
	            if token_idx == end_sub:
	                inside_sub = False
	            continue
	        
	        
	        ## ASSIGN IOB tags for OBJs
	        elif (token == list_of_objs[0]):
	            if (token_idx+len(list_of_objs) < len(token_list)) and (token_list[token_idx : token_idx+len(list_of_objs)] == list_of_objs):
	                if len(list_of_objs) == 1:
	                    line = ((token, pos_list[token_idx]), 'I-OBJ')
	                else:
	                    line = ((token, pos_list[token_idx]), 'B-OBJ')

	                iob_txt.append(line)
	                inside_obj = True
	                end_obj = token_idx + len(list_of_objs)
	                continue
	                
	            else:
	                line = ((token, pos_list[token_idx]), 'O')
	                iob_txt.append(line)
	                continue
	        elif (token in list_of_objs) and (inside_obj == True):
	            line = ((token, pos_list[token_idx]), 'I-OBJ')
	            iob_txt.append(line)
	            if token_idx == end_obj:
	                inside_obj = False
	            continue
	            
	            
	        ## Assign IOB tags for ACTIONs
	        elif (token == list_of_act[0]):
	            if (token_idx+len(list_of_act) < len(token_list)) and (token_list[token_idx : token_idx+len(list_of_act)] == list_of_act):
	                if len(list_of_act) == 1:
	                    line = ((token, pos_list[token_idx]), 'ACTION')
	                else:
	                    line = ((token, pos_list[token_idx]), 'ACTION')

	                iob_txt.append(line)
	                inside_act = True
	                end_act = token_idx + len(list_of_act)
	                continue
	                
	            else:
	                line = ((token, pos_list[token_idx]), 'O')
	                iob_txt.append(line)
	                continue
	        elif (token in list_of_act) and (inside_act == True):
	            line = ((token, pos_list[token_idx]), 'ACTION')
	            iob_txt.append(line)
	            if token_idx == end_act:
	                inside_act = False
	            continue
	            
	        ## Other words get 'O' 
	        else:
	            line = ((token, pos_list[token_idx]), 'O')
	            iob_txt.append(line)
	            
	    dataset['IOB'][row_idx] = iob_txt
	return dataset



class NamedEntityChunker(ChunkParserI):
	'''
	Named Entity Chunker using ClassiferBasedTagger() and features generated by features()
	'''
	def __init__(self, train_sents, **kwargs):
	    assert isinstance(train_sents, Iterable)

	    self.feature_detector = features
	    self.tagger = ClassifierBasedTagger(feature_detector = features, train = train_sents, **kwargs)
 
	def parse(self, tagged_sent):
	    chunks = self.tagger.tag(tagged_sent)

	    # Transform [((w1, t1), iob1), ...] to triplets [(w1, t1, iob1), ...]
	    iob_triplets = [(w, t, c) for ((w, t), c) in chunks]
	    
	    return iob_triplets

def generate_placeholder(n,window_size, TYPE):
    '''
    Generate placeholders START/END at the beginning and at the end of a sentence
    '''

    if TYPE == 'START':
        placeholder = '[START]'
    else:
        placeholder = '[END]'
    
    sequence = []

    for i in range(window_size): 
        n_placeholder = placeholder[:-1] + str(i+1) + placeholder[-1]
        tup = (n_placeholder,)*n
        sequence.append(tup)
        
    if TYPE == 'START':
        sequence.reverse()
    return sequence

def is_inside_parenthesis(WINDOW_SIZE,prev_words,post_words):
	'''
	(for features())
	Check the word is bounded by a valid pair parentheses within the window size.
	'''

	open_paren = 0
	closed_paren = 0
	for i in range(WINDOW_SIZE):
	    if (prev_words[i][0] == '('):
	        open_paren = 1
	    if (post_words[i][0] == ')'):
	        closed_paren = 1
	return (open_paren and closed_paren)

def is_inside_comma(WINDOW_SIZE,prev_words,post_words):
	'''
	(for features())
	Check the word is bounded by a valid pair commas within the window size.
	'''

	open_comma = 0
	closed_comma = 0
	for i in range(WINDOW_SIZE):
	    if (prev_words[i] == ','):
	        open_comma = 1
	    if (post_words[i] == ','):
	        closed_comma = 1
	return (open_comma and closed_comma)

def check_prev_occurrence(items, prev_iob, target, allowed):
	'''
	(for features())
	Check if a target (certain iob or pos) that occurs in prior to the word within the window
	and all in-between words are under alloweance
	'''
	
	for i,item in enumerate(items):
	    if item == target:
	        return True
	    if (prev_iob[i][-3:] not in allowed):
	        return False
	return False

def check_post_occurrence(items, target, allowed):
	'''
	(for features())
	Check if a target (certain iob or pos) that occurs after the word within the window
	and all in-between words are under alloweance
	'''

	for i,item in enumerate(items):
	    if item == target:
	        return True
	    if items[i] not in allowed:
	        return False
	return False

def count_entity(history,entity):
	'''
	Count the number of SUB/OBJ sequences that were predicted in history
	''' 
	seq_marker = ['B','I']
	in_sequence = False
	count = 0

	for i,iob in enumerate(history):
	    if (iob[0] in seq_marker) and (iob[-3:] == entity):
	        if in_sequence == False:
	            count += 1
	        in_sequence = True
	    elif in_sequence:
	        in_sequence = False
	return count

def count_action(history):
	'''
	Count the number of ACTION in history
	'''
	count = 0
	for i,iob in enumerate(history):
	    if iob == 'ACTION':
	        count += 1
	return count

def features(tokens, index, history):
    """
    Produce a dictionary of features for every token in a sentence 
    `tokens`  = a POS-tagged sentence [(w1, pos1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    See the final dictionary for more information
    """
    NUM_OF_ELEMENT = 2
    WINDOW_SIZE = 3
    
    # Pad the sequence with placeholders
    tokens = generate_placeholder(NUM_OF_ELEMENT, WINDOW_SIZE,'START') + list(tokens) + generate_placeholder(NUM_OF_ELEMENT,WINDOW_SIZE,'END')
    history = [w[0] for w in generate_placeholder(1,WINDOW_SIZE,'START')] + list(history)
    # shift the index with WINDOW_SIZE, to accommodate the padding
    index += WINDOW_SIZE

    word, pos = tokens[index]
    
    prev_words = [w for (w,_) in tokens[index-WINDOW_SIZE : index][::-1]]
    prev_pos = [p for (_,p) in tokens[index-WINDOW_SIZE : index][::-1]]
    post_words = [w for (w,_) in tokens[index+1 : index+WINDOW_SIZE+1]]
    post_pos = [p for (_,p) in tokens[index+1 : index+WINDOW_SIZE+1]]
    prev_iobs = history[index-WINDOW_SIZE : index][::-1]
    
    contains_dash = '-' in word
    contains_dot = '.' in word
    allascii = all([True for c in word if c in string.ascii_lowercase])
 
    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase
 
    rightprev, rightprev_pos = tokens[index-1]
    rightpost, rightpost_pos = tokens[index+1]
    rightprev_iob = history[index-1]

    prevprev, prevprev_pos = tokens[index-2]
    postpost, postpost_pos = tokens[index+2]
    prevprev_iob = history[index-2]

    prevallcaps = prev_words[0] == prev_words[0].capitalize()
    prevcapitalized = prev_words[0] in string.ascii_uppercase
 
    nextallcaps = prev_words[0] == prev_words[0].capitalize()
    nextcapitalized = prev_words[0] in string.ascii_uppercase
    
    in_iob_sequence = (prev_iobs[0][0] == 'B') or (prev_iobs[0][0] == 'I')
    
    allowed_pos = set(('NN','NNS','NNP','NNPS','JJ','CC','IN'))
    #allowed_iob = set(('SUB','OBJ','ACT','INH','BIN','ACC','DEC'))
    allowed_iob = set(('SUB','OBJ','ACTION'))

    after_cc = check_prev_occurrence(prev_pos, prev_iobs,'CC',allowed_iob)
    before_cc = check_post_occurrence(post_pos,'CC', allowed_pos)
    
    after_to = check_prev_occurrence(prev_pos, prev_iobs,'TO',allowed_iob)
    before_to = check_post_occurrence(post_pos,'TO', allowed_pos)
    
    after_in = check_prev_occurrence(prev_pos, prev_iobs,'IN',allowed_iob)
    before_in = check_post_occurrence(post_pos,'IN', allowed_pos)
    
    after_modal = check_prev_occurrence(prev_pos, prev_iobs,'MD',allowed_iob)
    before_modal = check_post_occurrence(post_pos,'MD', allowed_pos)
    
    after_det = check_prev_occurrence(prev_pos, prev_iobs,'DET',allowed_iob)
    before_det = check_post_occurrence(post_pos,'DET', allowed_pos)
    
    after_wdt = check_prev_occurrence(prev_pos, prev_iobs,'WDT',allowed_iob)
    before_wdt = check_post_occurrence(post_pos,'WDT', allowed_pos)

    inside_parenthesis = is_inside_parenthesis(WINDOW_SIZE,prev_words,post_words)
    inside_comma = is_inside_comma(WINDOW_SIZE,prev_words,post_words)
    

    
    num_SUB = count_entity(history,'SUB'),
    num_OBJ = count_entity(history,'OBJ'),
    num_ACTION = count_action(history),
    
    forms = set(('activate','activates','activated','inhibit','inhitbits','inhibited','bind','binds','binded','accelerate','accelerates','accelerated','decelerate','decelerates','decelerated'))
    noun_pos = set(('NN','NNS','NNP','NNPS'))
    verb_pos = set(('VB','VBD','VBN'))
    

    
    after_action = False
    before_action = False
    for i,(word,pos) in enumerate(tokens[WINDOW_SIZE:index]):
        if (word in forms) and ((pos in verb_pos)):
            ater_action = True
            
    for i,(word,pos) in enumerate(tokens[index:len(tokens)-WINDOW_SIZE]):
        if (word in forms) and ((pos in verb_pos)):
            after_action = True  

    after_unim_verb = False
    for i,(word,pos) in enumerate(tokens[WINDOW_SIZE:index][::-1]):
        if (pos in verb_pos) and (word not in forms):
            after_unim_verb = True
            break
        if (pos in verb_pos) and (word in forms):
            break
            
    before_unim_verb = False
    for i,(word,pos) in enumerate(tokens[index+1:index+WINDOW_SIZE]):
        if (pos in verb_pos) and (word not in forms):
            before_unim_verb = True
            break
        if (pos in verb_pos) and (word in forms):
            break
    
    
    passive_sign = set(('is','are','was','were'))
    is_english = word in english_words
    is_verb_form = word in forms
    is_passive = ((prev_words[0] in passive_sign) or (prev_words[1] in passive_sign)) and (len(word)>1) and (word[-2:] == 'ed')
    is_noun = pos in noun_pos
    is_verb = pos in verb_pos
    is_adj = pos in set(('JJ'))
    is_punctuation = word in string.punctuation
    is_modal = pos == 'MD'
    is_wdt = pos == 'WDT'
    is_to_be = pos in passive_sign
    is_prep = pos == 'IN'
    has_num = any(char.isdigit() for char in word)
    has_symbol = any(char in string.punctuation for char in word)
    
    wn_synsets = wn.synsets(word)
    in_wn = len(wn_synsets) > 0
    if in_wn:
        wn_pos = wn_synsets[0].pos()
    else:
        wn_pos = 'None'
    

    feature_set = {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'all-ascii': allascii,
        'is-english': is_english,
        
        'is-verb-form': is_verb_form,
        'is-passive': is_passive,
        'is-noun': is_noun,
        'is-verb': is_verb,
        'is-punctuation': is_punctuation,
        'is-modal': is_modal,
        'is-wdt': is_wdt,
        'is-be': is_to_be,
        'is-prep': is_prep,
        'has_num': has_num,
        'has-symbol': has_symbol,
        
        'in-wn': in_wn, # In wordnet corpus
        'wn-pos': wn_pos, # First POS of wordnet corpus
        
        'after-action': after_action, # behind an ACTION in window
        'before-action': before_action, # before an ACTION in window
        'after-unim-verb': after_unim_verb, # before an non-ACTION verb in window
        'before-unim-verb': before_unim_verb, # before a non-ACTION verb in window
 
 
        'contains-dash': contains_dash,
        'contains-dot': contains_dot,
 
        'all-caps': allcaps,
        'capitalized': capitalized,
 		
 		#'rightprev': rightprev,
 		#'rightpost': rightpost,
 		'rightprev-pos': rightprev_pos,
 		'rightpost-pos': rightpost_pos,
 		'rightprev-iob': rightprev_iob,

 		'prevprev-pos': prevprev_pos,
 		'postpost-pos': postpost_pos,
 		'prevprev-iob': prevprev_iob,

        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,
 
        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
        
        'in_iob_sequence': in_iob_sequence,
        
        'after_cc': after_cc, ## Behind/Before a coordinator conjunction window
        'before_cc':before_cc,
        
        'after_to': after_to, ## Behind/Before 'to' in window
        'before_to': before_to,
        
        'after_in': after_in, ## Behind/Before preposition in window
        'before_in': before_in,
        
        'after_modal': after_modal, ## Behind/Before modal verb in window
        'before_modal': before_modal,
        
        'after_det': after_det, ## Behind/Before determiner in window
        'before_det': before_det,
        
        'after_wdt': after_wdt, ## Behind/Before WH-determiner in window
        'before_wdt': before_wdt,
        
        'inside_parenthesis': inside_parenthesis,
        'inside_comma': inside_comma,
        
        'num-sub': num_SUB, ## Number of predicted SUB/OBJ/ACTION until this index
        'num-obj': num_OBJ,
        'num-action': num_ACTION
    }
    
    ## Get all pos and iob within window
    for i in range(WINDOW_SIZE):
        key_prev_word = 'prev-word-' + str(i+1)
        key_prev_pos = 'prev-pos-' + str(i+1)
        key_post_word = 'post-word-' + str(i+1)
        key_post_pos = 'post-pos-' + str(i+1)
        key_prev_iob = 'prev-iob-' + str(i+1)
        #feature_set[key_prev_word] = prev_words[i]
        feature_set[key_prev_pos] = prev_pos[i]
        #feature_set[key_post_word] = post_words[i]
        feature_set[key_post_pos] = post_pos[i]
        feature_set[key_prev_iob] = prev_iobs[i]
        
    return feature_set
 


def re_parse_relation(chunk):
	'''
	## Parse SUB, ACTION, OBJ chunks to relations
	'''
	grammar = r'''
	REL: {<SUB>+<ACTION>+<OBJ>+}
	'''
	re_parser = nltk.RegexpParser(grammar)
	return re_parser.parse(chunk)

def re_parse_npChunk(chunk):
	'''
	Parse raw SUB and OBJ into noun phrases
	'''
	grammar = r"""
	NP: {<JJ|VBG>* <NN.?>+ <CC>+ <JJ|VBG>* <NN.?>+}
	    {<JJ|VBG>* <NN.?>+}
	"""

	re_parser = nltk.RegexpParser(grammar, loop=5)
	return re_parser.parse(chunk)

def get_continuous_chunk(tokens,action_loc):
	'''
	Form continuous SUB and OBJ chunks (remove B- and I- signals)
	'''

	marker = set(('B-SUB','I-SUB','B-OBJ','I-OBJ'))
	current_chunk = []
	chunks = []
	for i,(word,pos,iob) in enumerate(tokens):
	    if (i in action_loc):
	        if current_chunk:
	            chunks.append(current_chunk)
	            current_chunk = []
	        chunks.append([iob,(word,pos,i)])
	    elif (i == 0) and (iob in marker):
	        current_chunk.append(iob[-3:])
	        current_chunk.append((word,pos,i))
	    elif (i>0):
	        if (iob in marker) and (iob[-3:] == tokens[i-1][2][-3:]):
	            current_chunk.append((word,pos,i))
	        elif (iob in marker) and (not current_chunk):
	            current_chunk.append(iob[-3:])
	            current_chunk.append((word,pos,i))
	        else:
	            if current_chunk:
	                chunks.append(current_chunk)
	            current_chunk = []
	return chunks

def refine(tokens):
    '''
    Strip out redundant words of subject/object noun phrases
    '''
    forms = set(('activate','activates','activated','inhibit','inhitbits','inhibited','bind','binds','binded','accelerate','accelerates','accelerated','decelerate','decelerates','decelerated'))
    verb_pos = set(('VB','VBD','VBN','VBP','VBZ'))
    action_loc = []
    
    for i,(word,pos,iob) in enumerate(tokens):
        if  (word in forms) and ((pos in verb_pos) or ((i+1 < len(tokens)) and tokens[i+1][1] == 'TO')):
            tokens[i] = (word,pos,'ACTION')
            action_loc.append(i)
        elif (word not in forms) and (pos in verb_pos) and (word in english_words):
            tokens[i] = (word,pos,'UNIM_VERB')
            action_loc.append(i)
    return get_continuous_chunk(tokens,action_loc)
           

def merge_relation(relations):
    '''
    Merge single relations into complex relations with 'and' and 'or' if possible
    '''
    set_sub = set([rel[0] for rel in relations])
    set_obj = set([rel[2] for rel in relations])
    set_action = set([rel[1] for rel in relations])
    
    for action in set_action:
        for sub in set_sub:
            
            ## Merge relations with same action and subject
            merge_list = list(filter(lambda rel: (rel[0] == sub) and (rel[1] == action), relations))
            
            if len(merge_list) > 1:
                merge_rel = ( sub, action, ' and '.join([rel[2] for rel in merge_list]).strip() )
                relations.add(merge_rel)
            
                for rel in merge_list:
                    relations.remove(rel)
                
        for obj in set_sub:
            ## Merge relations with same action and object
            merge_list = list(filter(lambda rel: (rel[2] == obj) and (rel[1] == action), relations))
                
            if len(merge_list) > 1:
                merge_rel = ( ' and '.join([rel[2] for rel in merge_list]).strip(), action,obj )
                relations.add(merge_rel)
            
                for rel in merge_list:
                    relations.remove(rel)
    return relations

def generate_relation(raw_relations,clusters):
	'''
	Produce complete relations from raw_relations and clusters (chunker's parse)
	'''
	verb_loc = [i for i,item in enumerate(raw_relations) if (item[0] == 'ACTION') or (item[0] == 'UNIM_VERB')]
	action_loc = [i for i,item in enumerate(raw_relations) if item[0] == 'ACTION']
	action_seq = 0
	last_verb_loc = 0

	single_relations = set()

	for i,loc in enumerate(verb_loc):
	    if loc in action_loc:
	        sub = None
	        obj = None

	        ## Find a SUB lying between the last verb with the current ACTION
	        sub = next((x[1] for x in raw_relations[last_verb_loc:loc] if x[0] == 'SUB'), None)
	        if not sub:
	            sub = next((x[1] for x in raw_relations[last_verb_loc:loc] if x[0] == 'OBJ'), None)
	        if not sub:
	            sub = 'NAN'

	        ## Find a OBJ lying between the current ACTION and the next verb  
	        next_loc_marker = next((x for x in verb_loc[i+1:] if x not in action_loc), None)
	        if not next_loc_marker:
	            next_loc_marker = len(raw_relations)

	        obj = next((x[1] for x in raw_relations[loc+1:next_loc_marker] if x[0] == 'OBJ'), None)
	        if not obj:
	            obj = next((x[1] for x in raw_relations[loc+1:next_loc_marker] if x[0] == 'SUB'), None)
	        if not obj:
	            obj = 'NAN'

	        single_relations.add((sub, raw_relations[loc][1], obj))
	    else:
	        last_verb_loc = i
	 
	return(merge_relation(single_relations))

def annotate_test_data(df,test_set,chunker):
	'''
	Compute, predict and tag relation annotations to test data
	'''

	df['predicted_relation'] = ''
	test_set['predicted_relation'] = ''
	all_relations = []
	for i,row in test_set.iterrows():
	    txt = row['sentence']

	    ## Tokenize and adjoin words that has a dot in between
	    temp = word_tokenize(txt)
	    tokens = []
	    for i,token in enumerate(temp):
	        if (token.isalpha()) and (token[0].islower()) and (i-1>=0) and (temp[i-1][-1] == '.'):
	            tokens[-1] = tokens[-1] + token
	        else:
	            tokens.append(token)

	    ## Parse the sentence
	    sent = [refine_pos(token,pos) for (token,pos) in pos_tag(tokens)]
	    parse = chunker.parse(sent)
	    parse_clusters = refine(parse)


	    ## Make raw relations from parse
	    relation_items = []
	    raw_relations = []

	    for cluster in parse_clusters:
	        label = cluster[0]
	        content = cluster[1:]
	        token_pos_list = [(token,pos) for (token,pos,_) in content]

	        new_content = [label]

	        if token_pos_list:
	            tree = re_parse_npChunk(token_pos_list)
	        for sub_tree in tree.subtrees(): 
	            if sub_tree.label()  in ['NP']:
	                for leave in sub_tree.leaves():
	                    new_content.append(leave)


	        if (label == 'ACTION') or (label == 'UNIM_VERB'):
	            relation_items.append(cluster)
	        if (len(new_content) > 1):
	            relation_items.append(new_content)

	    for i,item in enumerate(relation_items):
	        label = item[0]
	        words = item[1:]

	        text = ' '.join([word[0] for word in words])
	        raw_relations.append((label,text))

	    ## Process raw relations
	    complete_relations = generate_relation(raw_relations,parse_clusters)
	    rel_str = ''


	    ## Add string tags to dataframe
	    for rel in complete_relations:
	        rel_str += '<' + rel[0] + ', ' + rel[1] + ', ' + rel[2] + '>, '
	    if len(rel_str) > 0:
	        index = list(df[df['sentence'] == txt].index)
	        df['predicted_relation'][index] = rel_str[:-2]
	        
	    all_relations += list(complete_relations)
	    
	return all_relations



def evaluate(df,test_set,prediction):
	'''
	Computer precision, recall, and F1 score
	'''
	annotations = list(test_set['annotation'])
	true_relations = []
	for anno in annotations:
	    anno = anno.lower()
	    annos = re.findall('\<(.*?)\>',anno)
	    annos = [tuple(map(str.strip, each.split(','))) for each in annos]
	    true_relations += annos
	    
	predicted_relations = [(w1.lower(),w2.lower(),w3.lower()) for (w1,w2,w3) in prediction]

	precision = sum(rel in true_relations for rel in predicted_relations) / len(predicted_relations) # TP/(TP + FP)
	recall = sum(rel in true_relations for rel in predicted_relations) / len(true_relations) # TP/(TP + FN)
	F1_score = 2/(1/precision + 1/recall)

	print("PRECISION: {:.2f}".format(precision) + ', ' + "RECALL: {:.2f}".format(recall) + ', ' + "F1-SCORE: {:.2f}".format(F1_score))
	return 




global english_words, stemmer
english_words = set(nltk.corpus.words.words())
stemmer = SnowballStemmer('english')   
warnings.filterwarnings("ignore") 

## Read and split data into train and test sets
columns = ['word','sentence','annotation','is_train', 'year','journal','affiliation','DOI/PMID', 'expected_relation']
filename = 'CS372_HW4_output.csv'

df = pd.read_csv(filename,names=columns,index_col = False)
df.drop(['expected_relation'], axis = 1)
train_set = df[df['is_train'] == 1]
test_set = df[df['is_train'] == 0]

## Process training data
train_set = extract_annotation(train_set)

train_set = train_set.drop('is_train',1)
test_set = test_set.drop('is_train',1)

train_set.reset_index(inplace=True)

train_set = preprocess(train_set)
train_set = upsample(train_set)
train_set = IOB_tag(train_set)

## Train parser for SUB/OBJ/ACTION
chunker = NamedEntityChunker(train_sents = list(train_set['IOB']))

## Predict relations
all_relations = annotate_test_data(df,test_set,chunker)


## Print sentence along with prediction for test set
for i,row in df[df['is_train'] == 0].iterrows():
	print(row['sentence'])
	print(row['annotation']) # True relation
	print(row['predicted_relation']) # Prediction
	print('\n')

## Performance evaluation
evaluate(df,test_set,all_relations)

#df.to_csv('CS372_HW4_output_20190749.csv', header=False, index=False)

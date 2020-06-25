import pandas as pd
import nltk
import wikipedia
import sys
import nltk
from nltk.corpus import names
from nltk import Tree
from queue import Queue
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk import RegexpParser
from nltk import edit_distance
import re
import gap_scorer

def format_data(data):
	'''
	Reformat data to make one row contain a pair of pronoun and coreference
	'''
	columns = ['ID','Text','Pronoun','Antecedent','Coref','URL']
	new_data = pd.DataFrame(columns = columns)
	for i,row in data.iterrows():
	    new_row_A = {'ID':row['ID'],
	               'Text':row['Text'],
	               'Pronoun':(row['Pronoun'],row['Pronoun-offset']),
	               'Antecedent':(row['A'], row['A-offset']),
	                'Coref':row['A-coref'],
	                'URL':row['URL']}

	    new_row_B = {'ID':row['ID'],
	               'Text':row['Text'],
	               'Pronoun':(row['Pronoun'],row['Pronoun-offset']),
	               'Antecedent':(row['B'], row['B-offset']),
	                'Coref':row['B-coref'],
	                'URL':row['URL']}

	    new_data = new_data.append(new_row_A,ignore_index = True)
	    new_data = new_data.append(new_row_B,ignore_index = True)

	return new_data


def get_token_loc(token_list, PRP_MARKER):
    """
    Find the location of the marked pronoun in the list of tokens.
    token_list: list of list, each a tokenized sentence
    PRP_MARKER: marker attached to pronoun prior to this
    """
    loc = -1
    for i,tokens in enumerate(token_list):

        for j,t in enumerate(tokens):
            loc += 1

            if (len(t) > 6) and (PRP_MARKER in t):
                l = t.index(PRP_MARKER)
                token_list[i][j] = t[:l] + t[l+len(PRP_MARKER):] 
                return token_list,loc
    return
def get_tree_loc(tree,loc):
    """
    Find the location in tree of the pronoun by the given location in tokens
    """
    tree_loc = ()
    count = -1
    for i,sub in enumerate(tree):
        if isinstance(sub,nltk.Tree):
            count += len(sub)
            if count >= loc:
                count -= len(sub)
                node_loc = loc - count - 1
                tree_loc = (i,node_loc)
                return tree_loc
        else:
            count += 1
    return
def create_np_tree(text,str_loc):
    """
    Chunk NP for text in the given text
    Return:
    tree: parsed tree for the given text
    token_list: list of (word,pos) for the given text
    sent_token: list of list of (word,pos), each sublist for a sentence in the given text
    token_loc: location of the token corresponding to str_loc
    sent_loc: the order of the sentence (in the whole text) that contains the word given by str_loc
    tree_loc: the location of the tree that contains the word given by str_loc
    """
    PRP_MARKER = 'MARKER_'
    grammar =   """
            Name: {<NNP>+}
            NP: {<DT|PRP\$>?<JJ>*<NN.*>+}
            NP: {<PRP|PRP\$>}
            """
    parser = RegexpParser(grammar)
    
    ## Add marker for the word we want to keep track
    text = text[:str_loc] + PRP_MARKER + text[str_loc:]
    sents = sent_tokenize(text)

    ## Make list of tokenized sentence plus keep track of the word
    loc = -1
    token_list = [] # [[(w,pos),...],[...]] for whole text
    for sent in sents:
        sent_tokens = word_tokenize(sent)
        temp_tokens = sent_tokens.copy()
        extra = 0
        
        for i,token in enumerate(temp_tokens):
            if ('/' in token):
                split = re.split("/",token)
                split = list(filter(None, split))
                if len(split) > 1:
                    if i == 0:
                        sent_tokens = split + sent_tokens[1:]
                    else:
                        sent_tokens = sent_tokens[:i+extra] + split + sent_tokens[i+extra+1:]
                        extra += len(split) - 1
                
        token_list.append(sent_tokens)  

        

    temp = token_list.copy()

    ## Get location of the word of interest in the token list
    token_list, token_loc = get_token_loc(temp, PRP_MARKER)

    temp = token_list.copy()
    token_list = []
    sent_token = []   
    
    for sent in temp:
        token_list += pos_tag(sent)
        sent_token.append(pos_tag(sent))
        
    ## Chunk NP and HUman Name    
    tree = parser.parse(token_list)
    
    ## Get sent_location
    count = -1
    sent_loc = -1
    for j,sent in enumerate(sent_token):
        for (token,_) in sent:
            count += 1
            if count == token_loc:
                sent_loc = j
        if sent_loc != -1:
            break
            
    ## Get tree location
    tree_loc = get_tree_loc(tree,loc)
    return token_list,sent_token,tree,token_loc,sent_loc,tree_loc


def split(txt, seps):
	'''
	Split the wiki page into small textual sections
	'''
	default_sep = seps[0]

	# we skip seps[0] because that's the default separator
	for sep in seps[1:]:
	    txt = txt.replace(sep, default_sep)
	return [i.strip() for i in txt.split(default_sep)]
def new_text_from_wiki(row,sent_token,prp_sent_loc,ante_sent_loc):
    '''
    Create new text from wikipedia that is relevant with the initial text
    Return:
    new_text: new text, may or may not be different from the former text
    new_prp_strloc: new pronoun offset
    new_ante_strloc: new antecedent offset
    URL_keyword: title of the wiki page
    instruct: special instruction for what to do next 
    '''
    
    text = row['Text']
    prp,prp_strloc = row['Pronoun']
    ante,ante_strloc = row['Antecedent']
    URL_keyword = row['URL'][row['URL'].rfind('/')+1:].replace('_',' ')

    sents = sent_tokenize(text)
    prp_sent = sents[prp_sent_loc]
    ante_sent = sents[ante_sent_loc]

    prp_sent_head_loc = text.find(prp_sent)
    ante_sent_head_loc = text.find(ante_sent)
    prp_loc_in_sent = prp_strloc - prp_sent_head_loc
    ante_loc_in_sent = ante_strloc - ante_sent_head_loc
    
    ## Load wiki page
    try:
        page_content = wikipedia.page(URL_keyword).content
    except:
        return (text, prp_strloc, ante_strloc, URL_keyword, 'NO CHANGE')
    
    spliter = re.findall('=+.*?=+',page_content)
    paragraphs = split(page_content,spliter)

    ## Extract the paragraph that contains relevant information
    extract = ''
    for para in paragraphs:
        if prp_sent in para:
            extract = para
            break

    ## Exact relevant sentence from the paragraph above
    new_text = ''
    current_len = 0
    if (prp_sent not in extract) or (text in new_text): #or (ante_sent not in extract):
        return (text, prp_strloc, ante_strloc, URL_keyword, 'NO CHANGE')
    else:
        new_prp_strloc = -1
        new_ante_strloc = -1
        extract_sents = sent_tokenize(extract)
        
        for sent in sents:
            for each in extract_sents:
                #distance = edit_distance(sent,each)
                #if distance < 80:
                sent_tok = set(word_tokenize(sent))
                each_tok = set(word_tokenize(each))
                common = sent_tok.intersection(each_tok)
                if (len(common) / len(sent_tok) > 0.7) or (len(common) / len(each_tok) > 0.7):
                    new_text = new_text + ' ' + sent
                    break
                else:
                    pass
                    #print(sent,each)
                    
            if sent == prp_sent:
                new_prp_strloc = current_len + prp_loc_in_sent
            if sent == ante_sent:
                new_ante_strloc = current_len + ante_loc_in_sent
            current_len = len(new_text)
    new_text = new_text.strip()
    


    if new_text == text:
        return (text, prp_strloc, ante_strloc, URL_keyword, 'NO CHANGE')
    
    instruct = 'CHANGED'
    if (new_ante_strloc == -1) or (ante not in new_text):
        instruct = 'FALSE'
    return (new_text, new_prp_strloc, new_ante_strloc, URL_keyword, instruct)
  

def count_name_np(tree,loc_1,loc_2):
    '''
    Count the number of proper names and other NPs between the given locations
    '''
    name_count = 0
    np_count = 0
    start = min(loc_1,loc_2)
    end = max(loc_1,loc_2)
    for i in range(start+1,end):
        if isinstance(tree[i],nltk.Tree):
            label = tree[i].label()
            if label == 'Name':
                name_count += 1
            if label == 'NP':
                np_count += 1
            
    return(name_count,np_count)
def features(row,is_with_URL,row_idx):
	'''
	Features for classifying a pair of pronoun-antecedent
	'''
	#print(row_idx)
	text = row['Text']
	prp,prp_strloc = row['Pronoun']
	ante,ante_strloc = row['Antecedent']
	features = {}

	## Gender features
	PRONOUNS = {'he':'male',
	              'she':'female',
	              'his':'male',
	              'her':'female',
	              'him':'female',
	              'hers':'female'}
	prp_gender = PRONOUNS[prp.lower()]

	name_gender = {}
	for name in names.words('male.txt'):
	    name_gender[name] = 'male'
	for name in names.words('female.txt'):
	    name_gender[name] = 'female'   
	    
	name_token = ante.split()
	name_length = len(name_token)
	if name_token[0] in name_gender:
	    same_gender = prp_gender == name_gender[name_token[0]]
	else:
	    same_gender = -1
	    
	(token_list,sent_token,tree,  ante_token_loc, ante_sent_loc, ante_tree_loc) = create_np_tree(text,ante_strloc) 
	(_,_,_,  prp_token_loc, prp_sent_loc, prp_tree_loc) = create_np_tree(text,prp_strloc) 


	## Generate new text and extra information from wikipedia page
	instruct = 'NO CHANGE'
	in_title = -1
	if is_with_URL:
	    try:
	        (new_text, new_prp_strloc, new_ante_strloc, title, instruct) = new_text_from_wiki(row,sent_token,prp_sent_loc,ante_sent_loc)
	        if instruct == 'FALSE':
	            return features, instruct
	        if (new_text[prp_strloc : prp_strloc+len(prp)] == prp) and (new_text[ante_strloc : ante_strloc+len(ante)] == ante):
	            text = new_text
	            prp_strloc = new_prp_strloc
	            ante_strloc = new_ante_strloc
	            in_title = int(ante in title)
	            (token_list,sent_token,tree,  ante_token_loc, ante_sent_loc, ante_tree_loc) = create_np_tree(text,ante_strloc) 
	            (_,_,_,  prp_token_loc, prp_sent_loc, prp_tree_loc) = create_np_tree(text,prp_strloc) 
	    except:
	        pass
	        


	## Distance features
	is_before = (ante_strloc < prp_strloc)
	str_dist = abs(prp_strloc - ante_strloc)/len(text)
	token_dist = abs(prp_token_loc - ante_token_loc)/len(token_list)
	sent_dist = abs(prp_sent_loc - ante_sent_loc)
	tree_dist = abs(prp_tree_loc[0] - ante_tree_loc[0])/len(tree)

	str_dist = round(str_dist,1)
	token_dist = round(token_dist,1)
	tree_dist = round(tree_dist,1)

	## Quantity features
	ante_count = text.count(ante)
	prev_ante_count = text[:prp_strloc].count(ante)
	name_count, np_count = count_name_np(tree, ante_tree_loc[0], prp_tree_loc[0]) ## Number of name and NP between pronoun and antecedent

	prev_ante_count = min(prev_ante_count,3)
	ante_count = min(ante_count,3)

	## In-between entity features
	is_comma = False
	is_wdt = False
	is_wp = False
	start = min(ante_token_loc,prp_token_loc)
	end = max(ante_token_loc,prp_token_loc)

	if sent_dist == 0:
	    for (token,pos) in token_list[start:end]:
	        if pos == ',':
	            is_comma = True
	        elif pos == 'WDT':
	            is_wdt = True
	        elif pos == 'WP':
	            is_wp = True
	     

	features = {'prp':prp,
	            'prp_gender':prp_gender,
	            #'ante':ante,
	            'same_gender':same_gender,
	            'in_title': in_title,
	            'is_before':is_before,
	            'str_dist':str_dist,
	            'sent_dist':sent_dist,
	            'token_dist':token_dist,
	            'tree_dist':tree_dist,
	            'ante_count':ante_count,
	            'prev_ante_count':prev_ante_count,
	            'name_count':name_count,
	            'np_count':np_count,
	            'is_comma':is_comma,
	            'is_wdt':is_wdt,
	            'is_wp':is_wp}
	            #'left_pos_prp':left_pos_prp,
	            #'right_pos_prp':right_pos_prp,
	            #'left_pos_ante':left_pos_ante,
	            #'right_pos_ante':right_pos_ante}



	return features, instruct


def training(data,is_with_URL):
    featuresets = []
    for i,row in data.iterrows():
        (feature,instruct) = features(row,is_with_URL,i)
        if instruct != 'FALSE':
            featuresets.append((feature,row['Coref']))
    classifier = nltk.NaiveBayesClassifier.train(featuresets)
    return classifier
def predict(classifier,data,master_data,is_with_URL):
    
    prediction = master_data.copy()
    
    for i in range(len(master_data)):   
        ## Coref A
        row = data.iloc[2*i]
        feature,instruct = features(row,is_with_URL,i)
        truth = row['Coref']
        if instruct == 'FALSE':
            expected_A = False
            prob_A = 1
        else:
            dist = classifier.prob_classify(feature)
            expected_A = dist.max()
            prob_A = dist.prob(expected_A)
            
        ## Coref B
        row = data.iloc[2*i+1]
        feature,instruct = features(row,is_with_URL,i)
        truth = row['Coref']
        if instruct == 'FALSE':
            expected_B = False
            prob_B = 1
        else:
            dist = classifier.prob_classify(feature)
            expected_B = dist.max()
            prob_B = dist.prob(expected_B)
            
    
        if (expected_A == True) and (expected_B == True):
            if prob_A - prob_B > 0.3:
                expected_B = False
            elif prob_B - prob_A > 0.3:
                expacted_A = False
         
        prediction['A-coref'][i] = expected_A
        prediction['B-coref'][i] = expected_B
        
    return prediction


train = pd.read_csv(r'./gap-development.tsv', delimiter = '\t')
validation = pd.read_csv(r'./gap-validation.tsv', delimiter = '\t') 
test = pd.read_csv(r'./gap-test.tsv', delimiter = '\t')

new_train = format_data(train)
new_validation = format_data(validation)
new_test= format_data(test)


is_with_URL = True
classifier = training(new_train,is_with_URL)
prediction = predict(classifier,new_test,test,is_with_URL)
if is_with_URL:
    filename = 'CS372_HW5_page_output_20190749.tsv'
else:
    filename = 'CS372_HW5_snippet_output_20190749.tsv'
prediction[['ID','A-coref','B-coref']].to_csv(filename,sep='\t',index=False,header=False)
score = gap_scorer.run_scorer('gap-test.tsv', filename)
print(score)

is_with_URL = False
classifier = training(new_train,is_with_URL)
prediction = predict(classifier,new_test,test,is_with_URL)
if is_with_URL:
    filename = 'CS372_HW5_page_output_20190749.tsv'
else:
    filename = 'CS372_HW5_snippet_output_20190749.tsv'
prediction[['ID','A-coref','B-coref']].to_csv(filename,sep='\t',index=False,header=False)
score = gap_scorer.run_scorer('gap-test.tsv', filename)
print(score)
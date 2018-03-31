## import modules here
import pandas as pd
import numpy as np
import string
from sklearn.preprocessing import LabelEncoder, scale, normalize
from sklearn.feature_extraction import DictVectorizer
from collections import deque,Counter,OrderedDict
#from sklearn.metrics import f1_score,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import re
import nltk
import pickle

RANDOM_SEED = 9

################# helper data ##############

# Vowel Phonemes
vowels = ('AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH'
          , 'IY', 'OW', 'OY', 'UH', 'UW')

# Consonants Phonemes
consonants = ('P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N',
              'NG', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH')

strong_suffixes = set(('al','ance', 'ancy', 'ant','ard','ary','ate','auto','ence', 'ency', 'ent',
                  'ery','est','ial', 'ian' ,'iana','en','esce','ic','ify','ine','ion', 'tion',
                  'ity','ive','ory','ous','ual','ure' ,'wide','y','se','ade','e','ee','een',
                  'eer','ese','esque','ette','eur','ier','oon','que'))

strong_prefixes = set(('ad','co','con','counter','de','di','dis','e','en','ex','in','mid','ob','para','pre','re','sub',
                  'a','be','with','for')) 

neutral_prefixes=set(('down','fore','mis','over','out','un','under','up','anti','bi','non','pro','tri','contra','counta',
                   'de','dis','extra','inter','intro','multi','non','post','retro','super','trans','ultra'))
                  
neutral_suffixes= set(('able','age','al','ate','ed','en','er','est','ful','hood','ible','ing','ile','ish','ism',
                   'ist','ize','less','like','ly''man','ment','most','ness','old','s','ship','some','th','ward','wise','y'))

suffixes = (
'inal','ain', 'tion', 'sion', 'osis', 'oon', 'sce', 'que', 'ette', 'eer', 'ee', 'aire', 'able', 'ible', 'acy', 'cy', 'ade',
'age', 'al', 'al', 'ial', 'ical', 'an', 'ance', 'ence',
'ancy', 'ency', 'ant', 'ent', 'ant', 'ent', 'ient', 'ar', 'ary', 'ard', 'art', 'ate', 'ate', 'ate', 'ation', 'cade',
'drome', 'ed', 'ed', 'en', 'en', 'ence', 'ency', 'er', 'ier',
'er', 'or', 'er', 'or', 'ery', 'es', 'ese', 'ies', 'es', 'ies', 'ess', 'est', 'iest', 'fold', 'ful', 'ful', 'fy', 'ia',
'ian', 'iatry', 'ic', 'ic', 'ice', 'ify', 'ile',
'ing', 'ion', 'ish', 'ism', 'ist', 'ite', 'ity', 'ive', 'ive', 'ative', 'itive', 'ize', 'less', 'ly', 'ment', 'ness',
'or', 'ory', 'ous', 'eous', 'ose', 'ious', 'ship', 'ster',
'ure', 'ward', 'wise', 'ize', 'phy', 'ogy')

prefixes = (
'ac', 'ad', 'af', 'ag', 'al', 'an', 'ap', 'as', 'at', 'an', 'ab', 'abs', 'acer', 'acid', 'acri', 'act', 'ag', 'acu',
'aer', 'aero', 'ag', 'agi',
'ig', 'act', 'agri', 'agro', 'alb', 'albo', 'ali', 'allo', 'alter', 'alt', 'am', 'ami', 'amor', 'ambi', 'ambul', 'ana',
'ano', 'andr', 'andro', 'ang',
'anim', 'ann', 'annu', 'enni', 'ante', 'anthrop', 'anti', 'ant', 'anti', 'antico', 'apo', 'ap', 'aph', 'aqu', 'arch',
'aster', 'astr', 'auc', 'aug',
'aut', 'aud', 'audi', 'aur', 'aus', 'aug', 'auc', 'aut', 'auto', 'bar', 'be', 'belli', 'bene', 'bi', 'bine', 'bibl',
'bibli', 'biblio', 'bio', 'bi',
'brev', 'cad', 'cap', 'cas', 'ceiv', 'cept', 'capt', 'cid', 'cip', 'cad', 'cas', 'calor', 'capit', 'capt', 'carn',
'cat', 'cata', 'cath', 'caus', 'caut'
, 'cause', 'cuse', 'cus', 'ceas', 'ced', 'cede', 'ceed', 'cess', 'cent', 'centr', 'centri', 'chrom', 'chron', 'cide',
'cis', 'cise', 'circum', 'cit',
'civ', 'clam', 'claim', 'clin', 'clud', 'clus claus', 'co', 'cog', 'col', 'coll', 'con', 'com', 'cor', 'cogn', 'gnos',
'com', 'con', 'contr', 'contra',
'counter', 'cord', 'cor', 'cardi', 'corp', 'cort', 'cosm', 'cour', 'cur', 'curr', 'curs', 'crat', 'cracy', 'cre',
'cresc', 'cret', 'crease', 'crea',
'cred', 'cresc', 'cret', 'crease', 'cru', 'crit', 'cur', 'curs', 'cura', 'cycl', 'cyclo', 'de', 'dec', 'deca', 'dec',
'dign', 'dei', 'div', 'dem', 'demo',
'dent', 'dont', 'derm', 'di', 'dy', 'dia', 'dic', 'dict', 'dit', 'dis', 'dif', 'dit', 'doc', 'doct', 'domin', 'don',
'dorm', 'dox', 'duc', 'duct', 'dura',
'dynam', 'dys', 'ec', 'eco', 'ecto', 'en', 'em', 'end', 'epi', 'equi', 'erg', 'ev', 'et', 'ex', 'exter', 'extra',
'extro', 'fa', 'fess', 'fac', 'fact',
'fec', 'fect', 'fic', 'fas', 'fea', 'fall', 'fals', 'femto', 'fer', 'fic', 'feign', 'fain', 'fit', 'feat', 'fid', 'fid',
'fide', 'feder', 'fig', 'fila',
'fili', 'fin', 'fix', 'flex', 'flect', 'flict', 'flu', 'fluc', 'fluv', 'flux', 'for', 'fore', 'forc', 'fort', 'form',
'fract', 'frag',
'frai', 'fuge', 'fuse', 'gam', 'gastr', 'gastro', 'gen', 'gen', 'geo', 'germ', 'gest', 'giga', 'gin', 'gloss', 'glot',
'glu', 'glo', 'gor', 'grad', 'gress',
'gree', 'graph', 'gram', 'graf', 'grat', 'grav', 'greg', 'hale', 'heal', 'helio', 'hema', 'hemo', 'her', 'here', 'hes',
'hetero', 'hex', 'ses', 'sex', 'homo',
'hum', 'human', 'hydr', 'hydra', 'hydro', 'hyper', 'hypn', 'an', 'ics', 'ignis', 'in', 'im', 'in', 'im', 'il', 'ir',
'infra', 'inter', 'intra', 'intro', 'ty',
'jac', 'ject', 'join', 'junct', 'judice', 'jug', 'junct', 'just', 'juven', 'labor', 'lau', 'lav', 'lot', 'lut', 'lect',
'leg', 'lig', 'leg', 'levi', 'lex',
'leag', 'leg', 'liber', 'liver', 'lide', 'liter', 'loc', 'loco', 'log', 'logo', 'ology', 'loqu', 'locut', 'luc', 'lum',
'lun', 'lus', 'lust', 'lude', 'macr',
'macer', 'magn', 'main', 'mal', 'man', 'manu', 'mand', 'mania', 'mar', 'mari', 'mer', 'matri', 'medi', 'mega', 'mem',
'ment', 'meso', 'meta', 'meter', 'metr',
'micro', 'migra', 'mill', 'kilo', 'milli', 'min', 'mis', 'mit', 'miss', 'mob', 'mov', 'mot', 'mon', 'mono', 'mor',
'mort', 'morph', 'multi', 'nano', 'nasc',
'nat', 'gnant', 'nai', 'nat', 'nasc', 'neo', 'neur', 'nom', 'nom', 'nym', 'nomen', 'nomin', 'non', 'non', 'nov', 'nox',
'noc', 'numer', 'numisma', 'ob', 'oc',
'of', 'op', 'oct', 'oligo', 'omni', 'onym', 'oper', 'ortho', 'over', 'pac', 'pair', 'pare', 'paleo', 'pan', 'para',
'pat', 'pass', 'path', 'pater', 'patr',
'path', 'pathy', 'ped', 'pod', 'pedo', 'pel', 'puls', 'pend', 'pens', 'pond', 'per', 'peri', 'phage', 'phan', 'phas',
'phen', 'fan', 'phant', 'fant', 'phe',
'phil', 'phlegma', 'phobia', 'phobos', 'phon', 'phot', 'photo', 'pico', 'pict', 'plac', 'plais', 'pli', 'ply', 'plore',
'plu', 'plur', 'plus', 'pneuma',
'pneumon', 'pod', 'poli', 'poly', 'pon', 'pos', 'pound', 'pop', 'port', 'portion', 'post', 'pot', 'pre', 'pur',
'prehendere', 'prin', 'prim', 'prime',
'pro', 'proto', 'psych', 'punct', 'pute', 'quat', 'quad', 'quint', 'penta', 'quip', 'quir', 'quis', 'quest', 'quer',
're', 'reg', 'recti', 'retro', 'ri', 'ridi',
'risi', 'rog', 'roga', 'rupt', 'sacr', 'sanc', 'secr', 'salv', 'salu', 'sanct', 'sat', 'satis', 'sci', 'scio',
'scientia', 'scope', 'scrib', 'script', 'se',
'sect', 'sec', 'sed', 'sess', 'sid', 'semi', 'sen', 'scen', 'sent', 'sens', 'sept', 'sequ', 'secu', 'sue', 'serv',
'sign', 'signi', 'simil', 'simul', 'sist', 'sta',
'stit', 'soci', 'sol', 'solus', 'solv', 'solu', 'solut', 'somn', 'soph', 'spec', 'spect', 'spi', 'spic', 'sper',
'sphere', 'spir', 'stand', 'stant', 'stab',
'stat', 'stan', 'sti', 'sta', 'st', 'stead', 'strain', 'strict', 'string', 'stige', 'stru', 'struct', 'stroy', 'stry',
'sub', 'suc', 'suf', 'sup', 'sur', 'sus',
'sume', 'sump', 'super', 'supra', 'syn', 'sym', 'tact', 'tang', 'tag', 'tig', 'ting', 'tain', 'ten', 'tent', 'tin',
'tect', 'teg', 'tele', 'tem', 'tempo', 'ten',
'tin', 'tain', 'tend', 'tent', 'tens', 'tera', 'term', 'terr', 'terra', 'test', 'the', 'theo', 'therm', 'thesis',
'thet', 'tire', 'tom', 'tor', 'tors', 'tort'
, 'tox', 'tract', 'tra', 'trai', 'treat', 'trans', 'tri', 'trib', 'tribute', 'turbo', 'typ', 'ultima', 'umber',
'umbraticum', 'un', 'uni', 'vac', 'vade', 'vale',
'vali', 'valu', 'veh', 'vect', 'ven', 'vent', 'ver', 'veri', 'verb', 'verv', 'vert', 'vers', 'vi', 'vic', 'vicis',
'vict', 'vinc', 'vid', 'vis', 'viv', 'vita', 'vivi'
, 'voc', 'voke', 'vol', 'volcan', 'volv', 'volt', 'vol', 'vor', 'with', 'zo')

# Upper Convert set ot upper
def upper(iterable):
    return {x.upper() for x in iterable}

neutral_prefixes  = upper(neutral_prefixes)
neutral_suffixes  = upper(neutral_suffixes)
strong_prefixes   = upper(strong_prefixes)
strong_suffixes   = upper(strong_suffixes)
full_suffixes_set = upper(suffixes)
full_prefixes_set = upper(prefixes)

# Classification Map
classifications = { '10'  : 1,
                    '100' : 1,
                    '1000': 1,
                    '01'  : 2,
                    '001' : 3,
                    '0001': 4,
                    '010' : 2,
                    '0100': 2,
                    '0010': 3
                    }

vector_map = vowels + consonants


Noun_set=set(('NN','NNS','NNPS','NNPS'))
Verb_set=set(('VB','VBD','VBG','VBN','VBP','VBZ'))
Adj_set=set(('JJ','JJR','JJS'))

################# training #################

def train(data, classifier_file):  # do not change the heading of the function
    train, test = train_test_split(data, test_size = 0.2)
    words =word_data(data)
    train =word_data(train)
    test=word_data(test)
    #print(words)
    #classifier_type=DecisionTreeClassifier
    feature_list = ['str_pre','str_suf','neu_pre','neu_suf','prefix','suffix','phoneme_length','vowel_count','noun','verb','adj','open_close','consonant_count']
    #clf=DecisionTreeClassifier(criterion = "gini")
    clf = DecisionTreeClassifier(criterion = "gini", max_depth = 14, random_state = RANDOM_SEED)
    #clf = DecisionTreeClassifier(criterion = "entropy")
    #clf=classifier(classifier_type,criterion='entropy')
    x_train=np.array(train.df[feature_list])
    x_test=np.array(test.df[feature_list])
    y_train = train.df.primary_stress_index
    y_test = test.df.primary_stress_index
    #clf.train(x_train,y_train)
    
    dtree=clf.fit(x_train,y_train)
    #print(dtree.score(x_train,y_train))
    #print(dtree.score(x_test,y_test))
    
    save_Pickle((train,clf),classifier_file)
    return


################# testing #################

def test(data, classifier_file):  # do not change the heading of the function
    words, clf = get_Pickle(classifier_file)
    test_words = test_word_data(data)
    #features = classifier_cls.get_features()
    features = ['str_pre','str_suf','neu_pre','neu_suf','prefix','suffix','phoneme_length','vowel_count','noun','verb','adj','open_close','consonant_count']
    feature_array = np.array(test_words.df[features])
    #print(feature_array)
    predicted_Y =clf.predict(feature_array)

    test_words.set_predicted_classes(predicted_Y)
    pred = test_words.df.predicted_primary_index.tolist()
    return pred
    

################# classes ##################

'''

word_data       = Class to hold word data and perform all requisite pre-processing

    Attributes
lines           = List of word and stressed phonemes
df              = dataframe to hold and process word data
pn_list         = list of phonemes
vowel_map       = 2-4 bit string depicting location of primary stress
classifications = Group Index of stressed vowel, 0 is 1st, 3 is last irrespective of vowel count/word length
                  1 and 2 are then 2nd and 3rd respecively.
ngrams          = All possible ngrams of pn_list
ngrams_counts   = Dict object of ngrams 


'''
class word_data(object):
    def __init__(self,data,train_type_tags=[]):
        self.lines = [line_split(line) for line in data]
        self.df = pd.DataFrame(data=self.lines, columns=('word', 'pronunciation'))
        self.df['pn_list'] = self.df.pronunciation.apply(str.split)
        #print(self.df['pn_list'])
        self.df['phoneme_length'] = self.df.pn_list.str.len()
        self.df['destressed_pn_list'] = self.df.pronunciation.apply(filter_stress, args=('[012]',))
        self.df['open_close']=self.df.destressed_pn_list.apply(open_close_fuction,args=(vowels,))
        #print(self.df['open_close'])
        self.df['vowel_map'] = self.df.destressed_pn_list.apply(phoneme_map, args=(vowels,))
        self.df['consonant_map'] = self.df.destressed_pn_list.apply(phoneme_map, args=(consonants,))
        self.df['vowel_count'] = self.df.vowel_map.apply(np.sum)
        #print(self.df['vowel_count'])
        self.df['consonant_count'] = self.df.consonant_map.apply(np.sum)
        #print(self.df['consonant_count'])
        self.df['vowel_map_string'] = self.df.vowel_map.apply(to_string)
        self.df['stress_map'] = self.df.pn_list.apply(get_stress_map)
        #print(self.df['stress_map'])
        self.df['primary_stress_index'] = self.df.stress_map.apply(get_classification)
        #print(self.df['classification'])
        #self.df['primary_stress_index'] = self.df.apply(get_classsification_index,args=('classification',) ,axis=1)
        #print(self.df['primary_stress_index'])
        self.df['ngrams'] = self.df.pn_list.apply(get_all_ngrams)
        self.df['ngram_counts'] = self.df.ngrams.apply(Counter)
        self.df['destressed_ngrams'] = self.df.destressed_pn_list.apply(get_all_ngrams)
        self.df['destressed_ngram_counts'] = self.df.destressed_ngrams.apply(Counter)
        self.df['prefix'] = self.df.word.apply(check_prefix,args=(full_prefixes_set,))
        self.df['suffix'] = self.df.word.apply(check_suffix,args=(full_suffixes_set,))
        self.df['str_pre'] = self.df.word.apply(check_prefix,args=(strong_prefixes,))
        self.df['str_suf'] = self.df.word.apply(check_suffix,args=(strong_suffixes,))
        self.df['neu_pre'] = self.df.word.apply(check_prefix,args=(neutral_prefixes,))
        self.df['neu_suf'] = self.df.word.apply(check_suffix,args=(neutral_suffixes,))
        self.df['type_tag'] = self.df.word.apply(get_pos_tag)
        self.df['noun']=self.df.word.apply(check_tag,args=(Noun_set,))
        #print(self.df['noun'])
        self.df['verb']=self.df.word.apply(check_tag,args=(Verb_set,))
        self.df['adj']=self.df.word.apply(check_tag,args=(Adj_set,))
    def set_predicted_classes(self,classes):
        self.df['predicted_classes'] = classes
        self.df['predicted_primary_index'] = self.df.apply(get_classsification_index,args=('predicted_classes',), axis=1)
        
class test_word_data(object):
    def __init__(self,data,train_type_tags=[]):
        self.lines = [line_split(line) for line in data]
        self.df = pd.DataFrame(data=self.lines, columns=('word', 'pronunciation'))
        self.df['pn_list'] = self.df.pronunciation.apply(str.split)
        self.df['phoneme_length'] = self.df.pn_list.str.len()
        self.df['destressed_pn_list'] = self.df.pronunciation.apply(filter_stress, args=('[012]',))
        #print(self.df['destressed_pn_list'])
        self.df['vowel_map'] = self.df.destressed_pn_list.apply(phoneme_map, args=(vowels,))
        self.df['open_close']=self.df.destressed_pn_list.apply(open_close_fuction,args=(vowels,))
        #print(self.df['vowel_map'])
        self.df['consonant_map'] = self.df.destressed_pn_list.apply(phoneme_map, args=(consonants,))
        self.df['vowel_count'] = self.df.vowel_map.apply(np.sum)
        self.df['consonant_count'] = self.df.consonant_map.apply(np.sum)
        self.df['vowel_map_string'] = self.df.vowel_map.apply(to_string)
        #print(self.df['vowel_map_string'])
        self.df['stress_map'] = self.df.pn_list.apply(get_stress_map)
       # print(self.df['stress_map'] )
        self.df['classification'] = 0
        #self.df['primary_stress_index'] = self.df.apply(get_classsification_index,args=('classification',) ,axis=1)
        self.df['primary_stress_index']=1
        self.df['ngrams'] = self.df.pn_list.apply(get_all_ngrams)
        self.df['ngram_counts'] = self.df.ngrams.apply(Counter)
        self.df['destressed_ngrams'] = self.df.destressed_pn_list.apply(get_all_ngrams)
        self.df['destressed_ngram_counts'] = self.df.destressed_ngrams.apply(Counter)
        self.df['prefix'] = self.df.word.apply(check_prefix,args=(full_prefixes_set,))
        self.df['suffix'] = self.df.word.apply(check_suffix,args=(full_suffixes_set,))
        self.df['str_pre'] = self.df.word.apply(check_prefix,args=(strong_prefixes,))
        self.df['str_suf'] = self.df.word.apply(check_suffix,args=(strong_suffixes,))
        self.df['neu_pre'] = self.df.word.apply(check_prefix,args=(neutral_prefixes,))
        self.df['neu_suf'] = self.df.word.apply(check_suffix,args=(neutral_suffixes,))
        self.df['type_tag'] = self.df.word.apply(get_pos_tag)
        self.df['noun']=self.df.word.apply(check_tag,args=(Noun_set,))
        self.df['verb']=self.df.word.apply(check_tag,args=(Verb_set,))
        self.df['adj']=self.df.word.apply(check_tag,args=(Adj_set,))
    def set_predicted_classes(self,classes):
        #self.df['predicted_classes'] = classes
        self.df['predicted_primary_index'] = classes
        #self.df.apply(get_classsification_index,args=('predicted_classes',), axis=1)  


################# helper functions #########

# Pickler
def save_Pickle(obj, file):
    with open(file,'wb') as f:
        pickle.dump(obj,f)
    f.close()
    
def get_Pickle(file):
    with open(file,'rb') as f:
        obj = pickle.load(f)
    f.close()
    return (obj_i for obj_i in obj)

def open_close_fuction(pn_list,vowel):
    if pn_list[-1] in vowel:
        return 1
    return 0

# Return all ngrams of particular length
def get_ngram_possibilities(pronunciation_list, length):
    return tuple(zip(*(pronunciation_list[i:] for i in range(length))))


# Develop deque of all possible ngrams
def get_all_ngrams(pn_list,restrict_length = None):
    ngrams = set()
    if not restrict_length:
        restrict_length = len(pn_list)
    for i in range(2,restrict_length + 1):
        ngrams.update(get_ngram_possibilities(pn_list,i))
    return ngrams


# Convert list to tuple
def as_tuple(list_to_convert):
    return tuple(list_to_convert)


# Filter stress from string

def filter_stress(string_to_be_filtered, to_filter=None):
    if type(string_to_be_filtered) in [list, tuple]:
        string_to_be_filtered = ' '.join(string_to_be_filtered)
    return tuple(re.sub(to_filter,'',string_to_be_filtered).split())


# Filter non-important stresses
def filter_non_primary_stress(pronunciation):
    pronunciation = pronunciation.replace('0', '')
    return pronunciation.replace('2', '')


# Maps the location of the stress, 1 if stress at position
# 0 otherwise
def stress_map(pronunciation, stress='1'):
    return [1 if stress in num else 0 for num in pronunciation]


# Maps the the location of phenom, 1 in phenom_list
# 0 otherwise
def phoneme_map(pronunciation, phoneme_list):
    return [1 if phoneme in phoneme_list else 0 for phoneme in pronunciation]


# Map existence of one iterable in another
def iterable_map(list_to_map, iterable):
    return [1 if iter_item in list_to_map else 0 for iter_item in iterable]


# Get nltk pos_tag
def get_pos_tag(word):
    return nltk.pos_tag([word])[0][1]

# Returning string as a classification
def get_stress_position(stress_map_list, stress=1):
    return str(stress_map_list.index(stress) + 1)

def check_tag(word,nltk_set):
    if nltk.pos_tag([word])[0][1] in nltk_set:
        return 1
    return 0

# Check if prefix exists
def check_prefix(word,prefixes_set):
    for letter_idx in range(len(word) + 1):
        if word[:letter_idx] in prefixes_set:
            return 1
    return 0


# Check if suffix exists
def check_suffix(word,suffixes_set):
    word_length = len(word)
    for letter_idx in range(word_length + 1):
        if word[abs(letter_idx - word_length):] in suffixes_set:
            return 1
    return 0


# Get ascii index of first letter
def get_first_letter_idx(word):
    return string.ascii_lowercase.index(word[0].lower()) + 1


# Return the stressed vowel
def get_stressed_vowel(pn_list):
    for vowel in pn_list:
        if '1' in vowel:
            return filter_stress(vowel,to_filter='1')[0]


# Return all possible consecutive tuples length n from list
def sub_string(pronunciation_list, n):
    return tuple(zip(*(pronunciation_list[i:] for i in range(n))))


# Build a dict of all possible sequences of phonemes
def get_sequences(phoneme_series):
    sequences = {}
    max_length = max(phoneme_series.str.len())
    for i in range(2, max_length + 1):
        for pn_list in phoneme_series:
            # Next iteration if pn_list is shorter then the sequence length be built
            if len(pn_list) < i:
                continue
            word_sequences = sub_string(pn_list, i)
            for seq in word_sequences:
                sequences[seq] = sequences.get(seq, 0) + 1
    return sequences


def in_list(pn_list, sequence):
    if pn_list in sequence:
        return 1
    return 0


# Return 1 if sequence has a primary stress in it
def is_primary(sequence):
    for phoneme in sequence:
        if '1' in phoneme:
            return True
    return False


# Return classification for pn_list
def get_stress_map(pn_list):
    vowels = str()
    for pn in pn_list:
        if pn in consonants:
            continue
        elif '1' in pn:
            vowels += '1'
        elif '0' in pn or '2' in pn:
            vowels += '0'
    return vowels

def get_classification(vowel_map):
    return classifications[vowel_map]

# Return the index of the stressed vowel based on classification
def get_classsification_index(df,classification_column):
    vowel_idx = [idx.start() for idx in re.finditer('1',df.vowel_map_string)]
    if df[classification_column] > len(vowel_idx) - 1:
        #print('1')
        return vowel_idx[-1]
    if df[classification_column] < 3:
        #print('2')
        return vowel_idx[df[classification_column]]
    else:
        #print('3')
        return vowel_idx[-1]

def to_string(list_to_convert):
    return ''.join([str(x) for x in list_to_convert])


def line_split(line):
    line = line.split(':')
    return line[0], line[1]

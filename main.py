import nltk
from nltk.util import ngrams
nltk.download("punkt")
import string
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import requests

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

nlp = spacy.load("en_core_web_sm")
punc = list(string.punctuation)

def fetch_text_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return (response.text).split(",")

url = 'https://gist.githubusercontent.com/ZohebAbai/513218c3468130eacff6481f424e4e64/raw/b70776f341a148293ff277afa0d0302c8c38f7e2/gist_stopwords.txt'
stopwords = fetch_text_from_url(url)

add_stop = ['said', 'say', '...', 'like', 'cnn', 'ad', 'etc', 'aforementioned', 'accordance', 'according', 'do', 'did', 'does', 'done', 'possible',
           'consider', 'concern', 'concerning', 'conсerned', 'regard', 'regarding', 'regards', 'have', 'has', 'had', 'having', 'refer', 'referred', 'shall'] # away too much almost other actually other скачать с сайта по in hanced stop word list

stop_words = ENGLISH_STOP_WORDS.union(add_stop)
stop_words = stop_words.union(stopwords)
except_w=['across',"acute","dyspnea" 'act', 'amount', 'announce', 'arise', 'begin', 'beginning', 'beginnings', 'begins', 'believe', 'bill', 'bottom', 'brief', 'call', 'changes', 'course',
          'date', 'detail', 'down', 'effect', 'eight', 'eighty', 'eleven', 'empty', 'end', 'ending', 'fifteen', 'fifth', 'fill', 'fire', 'first', 'five', 'fix', 'forth',
          'forty', 'four', 'front', 'full', 'help', 'home', 'hundred', 'index', 'information', 'inner', 'interest', 'invention', 'left', 'line', 'little', 'made', 'make',
          'makes', 'mill', 'mug', 'name', 'new', 'nine', 'ninety', 'non', 'novel', 'off', 'old', 'on', 'one', 'outside', 'over', 'owing', 'own', 'page', 'pagecount', 'pages',
          'part', 'past', 'placed', 'plus', 'research', 'research-articl', 'results', 'right', 'second', 'section', 'self', 'sent', 'seven', 'side', 'six', 'sixty', 'stop',
          'system', 'thin', 'think', 'third', 'thousand', 'three', 'tip', 'twelve', 'twenty', 'twice', 'two', 'use', 'value', 'way', 'words', 'world', 'zero']

stop_words=stop_words-set(except_w)



# Modify tokenizer infix patterns
infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\\-\\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        # Commented out regex that splits on hyphens between letters:
        # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)

infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer

def tokinizer(text):
  sent_tokens = []
  index=0
  text_t=nlp(text)
  for sent in text_t.sents:
      list_tok=[]
      for i in sent:
        list_tok.append((i.text.lower(), i.pos_,index))
        index+=1
      sent_tokens.append(list_tok)
  return sent_tokens

def concatenate_ngrams(candidate):
  cand_temp= []
  temp=''
  if type(candidate) !=type(str()):
     for w in candidate:
        if (w not in punc) and (len(temp)>0) and ((temp[-1]=="'") or ((w[0] not in punc) and (temp[-1] not in punc))):
             temp=temp+" "+str(w)
        else:
             temp=temp+str(w)
  else:
    temp=candidate
  temp = temp.lower()
  return str(temp)

list_seq_2=  [[["PROPN","NOUN"],"*"],
              ["ADJ",'*', ["PROPN","NOUN"], '*'],
              ["ADJ","*"],
              ['VERB', 'ADJ', ["PROPN","NOUN"],'*'],
              [["PROPN","NOUN"],'*','ADJ','*',["PROPN","NOUN"],'*'],
              ['ADJ','VERB',["PROPN","NOUN"], '*'],
              ['VERB','*',["PROPN","NOUN"],'*'],
              [["PROPN","NOUN"], '*','ADJ','*',["PROPN","NOUN"], '*'],
              ["ADV","*","ADJ","*"],
              [["PROPN","NOUN"],'ADP',["PROPN","NOUN"]],
              [["PROPN","NOUN"],"*","PART",["PROPN","NOUN"],"*"],
              [['PROPN','NOUN','ADJ','VERB',"ADV","X"]],
              [['PROPN','NOUN','ADJ','VERB',"ADV","X"],["NOUN","PROPN"],"*"],
              [["PROPN","NOUN"],"*", "ADP",["PROPN","NOUN"],"*"]
              ]


def filter_propn_noun(mwe_list):
    filtred_ngrams=[]
    for i in mwe_list:
      checker2=True
      if (("NOUN" in i[1][-1]) or ("PROPN" in i[1][-1]) or ("NOUN" in i[1][-2]) or ("PROPN" in i[1][-2])) and (("ADJ" not in i[1][-1]) and ("ADJ" not in i[1][-2])) :
        temp_seq=str(concatenate_ngrams(i[0]))
        temp_token=nlp(temp_seq)
        if (temp_token[-1]).pos_ not in ["NOUN","PROPN","VERB"]:
           checker2=False

      if checker2==True:
        filtred_ngrams.append(i)
    return filtred_ngrams

def filter_stop_words(mwe_list):
    filtred_ngrams=[]
    for mwe in mwe_list:
      checker3=True
      temp=mwe
      for i, w in enumerate(mwe[0]):
        if mwe[2][i] not in ["PROPN","ADP"]:
               if w in stop_words:
                checker3=False

      if checker3==True:
        filtred_ngrams.append(mwe)
    return filtred_ngrams

def filter_ngrams_by_pos_tag(sentence, sequense):
    filtered_ngrams=[]
    t=0
    for seq in sequense:
        for i in range(len(sentence)):
            temp=[]
            temp_index=[]
            temp_pos=[]
            checker=True


            if ((sentence[i][1] in seq[0]) or (sentence[i][1] == seq[0])) and (sentence[i][0] not in punc) :
               seq_index=0
               sent_index=0

               while seq_index<len(seq) and i+sent_index<len(sentence) and checker==True:
                   if (sentence[i+sent_index][1] in seq[seq_index]) or (sentence[i+sent_index][0] in seq[seq_index]) :
                       temp.append(sentence[i+sent_index][0])
                       temp_pos.append(sentence[i+sent_index][1])
                       temp_index.append(sentence[i+sent_index][2])
                       seq_index += 1
                       sent_index += 1


                   elif seq[seq_index]=="*" and (sentence[i+sent_index][1] in seq[seq_index-1]):
                       if seq_index<len(seq)-1:
                              temp.append(sentence[i+sent_index][0])
                              temp_pos.append(sentence[i+sent_index][1])
                              temp_index.append(sentence[i+sent_index][2])
                              sent_index += 1

                       elif seq_index==len(seq)-1:
                             if (len(temp)>1  or "-" in "".join(temp)) and (len(set("".join(temp)).intersection(set(punc)-set("-'")))==0):
                                 temp_2=temp.copy()
                                 temp_pos2=temp_pos.copy()
                                 temp_index2=temp_index.copy()
                                 if [temp_2, seq, temp_pos2, temp_index2,len(temp_2)] not in filtered_ngrams and temp[-1] not in punc:
                                     filtered_ngrams.append([temp_2, seq, temp_pos2, temp_index2,len(temp_2)])


                             if (i+sent_index)<len(sentence):
                                temp.append(sentence[i+sent_index][0])
                                temp_pos.append(sentence[i+sent_index][1])
                                temp_index.append(sentence[i+sent_index][2])
                                sent_index += 1

                   elif seq[seq_index]=="*" and (sentence[i+sent_index][1] not in seq[seq_index-1]):
                      seq_index += 1

                   else:
                      checker=False
               #
               if seq_index==len(seq) and (len(temp)>1  or "-" in "".join(temp)) and len(set("".join(temp)).intersection(set(punc)-set("-'")))==0:
                    if [temp, seq, temp_pos, temp_index,len(temp)] not in filtered_ngrams and temp[-1] not in punc:
                         filtered_ngrams.append([temp, seq, temp_pos, temp_index,len(temp)])

    return  filtered_ngrams

import string

class PhraseExtractor:
    def __init__(self, text):
        self.text = text

    def extract_phrases(self):
        text_sent_tokens = tokinizer(self.text)
        mwe_list = []
        for sent in text_sent_tokens:
            temp_mwe_list = filter_ngrams_by_pos_tag(sent, list_seq_2)
            temp_mwe_list = filter_propn_noun(temp_mwe_list)
            temp_mwe_list = filter_stop_words(temp_mwe_list)
            mwe_list += temp_mwe_list

        mwe_list_n = [tuple(i[0]) for i in mwe_list]
        candidates = []
        for i in set(mwe_list_n):
            candidates.append(concatenate_ngrams(i))

        candidates = [i for i in candidates if ((i[-1] not in punc) and (i[-1] not in string.punctuation))]
        candidates = [i for i in candidates if len(set('1234567890').intersection(set(i))) == 0]

        return candidates

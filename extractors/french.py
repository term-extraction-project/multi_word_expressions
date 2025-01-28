import nltk
from nltk.util import ngrams
nltk.download("punkt")
import string
import spacy
import requests

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from operator import itemgetter
import pandas as pd

pos_tag_patterns=[[["NOUN","ADJ","PROPN"],"*"],
                    [["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*"],
                    [["NOUN","ADJ"],"*","ADP","DET",["NOUN","ADJ"],"*"],
                    ["NOUN","VERB"],
                    ["VERB","ADJ"],
                    ["ADJ","VERB"],
                    [["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*"],
                    [["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*","ADP",["NOUN","ADJ"],"*"]]


from spacy.lang.fr.examples import sentences
url = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-fr/master/stopwords-fr.txt'
stop_words = (requests.get(url).text).split("\n")


# setting up punctuation lists for checking, punc_without does not contain hyphens and apostrophes as they can be part of phrases. punc_all is needed to check if there is a hyphen at the beginning or at the end of a phrase
punc_without = list(string.punctuation)+["»","«"]
punc_without.remove('-')
punc_without.remove("'")
punc_all=list(string.punctuation)+["»","«"]

# Text tokenizer, input text with original case NOT in lowercase
# output a set of tokens marked by sentences, an element in the list is a sentence that contains tokens with information about them
# [ [("token1", pos, index),("token2", pos, index),("token3", pos, index)],
# [("token1", pos, index),("token2", pos, index),("token3", pos, index)]]
def tokinizer(text, nlp):
  sent_tokens = []
  index=0
  text_t=nlp(text)
  for sent in text_t.sents:
      list_tok=[]
      for i in sent:
        list_tok.append((i.text.lower(), i.pos_,index))  # creating a list of tokens with content, the actual unigram in lower case, its part of speech, position number in the text
        index+=1
      sent_tokens.append(list_tok)
  return sent_tokens



# function for combining phrase tokens into a single string
# input is a list of words ["word1","word2", "word3"]
# output is "word1 word2 word3"
# no space is put between the hyphen and the apostrophe
def concatenate_ngrams(candidate):
  cand_temp= []
  temp=''
  if type(candidate) !=type(str()):
     for w in candidate:
        if (w not in punc_without) and (len(temp)>0) and ((temp[-1]=="'") or ((w[0] not in punc_without) and (temp[-1] not in punc_without))):
             temp=temp+" "+str(w)
        else:
             temp=temp+str(w)
  else:
    temp=candidate
  temp = temp.lower()
  return str(temp)



# Cleaning phrases from stop words,
# if a word in a phrase is a stop word, the phrase is deleted. all words in the phrase are checked except prepositions and PROPN
# at the input is a list of phrases and a list of stop words
# at the output is a filtered list
def filter_stop_words(mwe_list, stop_words):
    filtred_ngrams=[]
    for mwe in mwe_list:
      checker3=True
      temp=mwe
      if mwe[2][0]=="ADP" or mwe[2][-1]=="ADP" :
               checker3=False

      for i, w in enumerate(mwe[0]):
        if mwe[2][i] not in ["PROPN","ADP"]:
               if w in stop_words:
                  checker3=False

      if checker3==True:
        filtred_ngrams.append(mwe)
    return filtred_ngrams



# extracting candidates based on part-of-speech templates
# input: list of tokens in one sentence [("token1", pos, index),("token2", pos, index),("token3", pos, index)] and part-of-speech template
# output: list of extracted candidates with information about them:
# [[list of words in the phrase], template by which the candidate was extracted, sequence of parts of speech of the candidate, indexes of word positions, number of words, number of characters in the candidate]
def filter_ngrams_by_pos_tag(sentence, sequense):
    filtered_ngrams=[]
    t=0
    for seq in sequense:
        for i in range(len(sentence)):
            temp=[]
            temp_index=[]
            temp_pos=[]
            checker=True


            if ((sentence[i][1] in seq[0]) or (sentence[i][1] == seq[0])) and (sentence[i][0] not in punc_without) :
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
                             if (len(temp)>1  or "-" in "".join(temp)) and (len(set("".join(temp)).intersection(set(punc_without)-set("-'")))==0):
                                 temp_2=temp.copy()
                                 temp_pos2=temp_pos.copy()
                                 temp_index2=temp_index.copy()
                                 if [temp_2, seq, temp_pos2, temp_index2,len(temp_2)] not in filtered_ngrams and temp[-1] not in punc_without:
                                     temp_2=[word.lower() for word in temp_2]
                                     filtered_ngrams.append([temp_2, seq, temp_pos2, temp_index2,len(temp_2),len(concatenate_ngrams(temp_2))])


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
               if seq_index==len(seq) and (len(temp)>1  or "-" in "".join(temp)) and len(set("".join(temp)).intersection(set(punc_without)-set("-'")))==0:
                    if [temp, seq, temp_pos, temp_index,len(temp)] not in filtered_ngrams and temp[-1] not in punc_without:
                         temp=[word.lower() for word in temp]
                         filtered_ngrams.append([temp, seq, temp_pos, temp_index,len(temp),len(concatenate_ngrams(temp))])
    return  filtered_ngrams

# calculation of the rectified frequency - the number of phrases in the text, not in phrases longer than
# input: list of words of the phrase (one candidate), all texts in the form of a single string, list f_raw_req_list which contains the frequencies of phrases longer than/or it is empty, since it is filled in during the function call
# output: rectified frequency of the phrase
# the function is called as many times as candidates. to calculate the rectified frequency, the calculation is made from the longest to the shortest phrase, since for its calculation it is necessary to know the frequency of phrases longer than the target
def f_req_calc(mwe, all_txt, f_raw_req_list):
  temp=all_txt
  mwe_c=concatenate_ngrams(mwe)
  for i in f_raw_req_list:
    i_c=concatenate_ngrams(i[0])
    if mwe_c in i_c and mwe_c!=i_c and len(mwe)!=len(i[0]):
      temp=temp.replace(i_c," ")
  f=temp.count(mwe_c)
  return f

# Grouping phrases into groups by word positions.
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] =  self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def group_items(lst):
    n = len(lst)
    uf = UnionFind(n)

    # Union sets with intersecting numbers
    for i in range(n):
        for j in range(i + 1, n):
            if set(lst[i][3]).intersection(lst[j][3]):
                uf.union(i, j)

    # Assign group numbers
    group_map = {}
    group_number = 1
    for i in range(n):
        root = uf.find(i)
        if root not in group_map:
            group_map[root] = group_number
            group_number += 1
        lst[i].append(group_map[root])

    return lst


# main body of phrase extraction
# input text and various parameters
# output list of phrases: ["phrase 1", "phrase 2", "phrase 3"]
class FrenchPhraseExtractor:
    def __init__(self, text, stop_words=stop_words, list_seq=pos_tag_patterns,  cohision_filter=True, additional_text="1", f_raw_sc=9, f_req_sc=3):
        self.text = text            # text in original case
        self.cohision_filter=cohision_filter     #  Enable or disable the cohesive filter
        self.additional_text=additional_text   # if there is additional text, it is used to calculate frequencies, terms are NOT extracted from it
        self.f_req_sc=f_req_sc     # rectified frequency threshold
        self.f_raw_sc=f_raw_sc       # raw frequency threshold
        self.stop_words=stop_words     # stop word list
        self.list_seq=list_seq         # list of part of speech patterns
        self.model_nlp = spacy.load("fr_core_news_sm")

     # Extracting phrases
    def extract_phrases(self):

        # Change the tokenizer so that it does not separate words with hyphens.
        # IT-developers create innovative solutions. --> ["IT-developers","create","innovative","solutions","."]
        # Instead of ["IT","-","developers"] tokenized as a whole token "IT-developers", this helps to avoid extracting unigrams that are part of the word, which reduces noise

        nlp = self.model_nlp

        infixes = (
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\\-\\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\\.(?=[{au}{q}])".format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                ),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ]
        )
        infix_re = compile_infix_regex(infixes)
        nlp.tokenizer.infix_finditer = infix_re.finditer


        # removing extra spaces
        text=self.text.replace(" -","-").replace("- ","-").replace(" '","'").replace("  "," ")

        #  text tokenization, parts of speech and position index
        text_sent_tokens = tokinizer(text, nlp)
        mwe_list = []

        # extracting candidates from each sentence separately
        for sent in text_sent_tokens:
            temp_mwe_list = filter_ngrams_by_pos_tag(sent, self.list_seq)  # Part-of-speech extraction
            temp_mwe_list = filter_stop_words(temp_mwe_list, self.stop_words)   # stop word filtering

            # temp_mwe_list contains lists of candidates with additional information about them:
            # [[list of words of the phrase], the template by which the candidate was extracted, the sequence of parts of speech of the candidate, the indices of the positions of words, the number of words, the number of characters in the candidate]
          
            sent_text=" ".join([s[0] for s in sent])
            temp_mwe_list=[mwe+[sent_text] for mwe in temp_mwe_list]
            mwe_list += temp_mwe_list

        # creating a list of candidates containing only the words of the candidates: [("phrase","one"),("mwe","next"),("phrase","other")]
        mwe_list_n = [tuple(i[0]) for i in mwe_list]
        candidates = []
        term_mws=[]
        words_scores=[]
        candid_word=[]

       # combination of words of a phrase: ["phrase one", "mwe next","phrase other"]
        for i in set(mwe_list_n):
            candidates.append(concatenate_ngrams(i))

       # cleaning phrases from punctuation and if it consists entirely of punctuation and numbers
        candidates = [i for i in candidates if ((i[-1] not in punc_without) and (i[-1] not in string.punctuation))]
        candidates = [i for i in candidates if len(set('1234567890').intersection(set(i))) == 0]

        # text for cohesive filter
        all_txt=''
        if len(self.additional_text)>10:  # if there is text (number 10 is random, the main thing is that it is not empty) then we combine it with the text from which we extracted candidates
             text_ref=self.additional_text.replace(" -","-").replace("- ","-").replace(" '","'").replace("  "," ").lower()
             all_txt=self.text+". "+text_ref
        else:
             all_txt=self.text
        all_txt=all_txt.lower()

        # if the cohesive filter is on
        if  self.cohision_filter==True:
            f_raw_req_list=[]
            all_cand_r=[tuple(i[0]) for i in mwe_list]
            possible_mwe=sorted(set(all_cand_r), key=len, reverse=True)    # sort phrases from longest to shortest

            for mwe in possible_mwe:
                f_raw=all_txt.count(concatenate_ngrams(mwe))    # raw frequency calculation
                f_req=f_req_calc(mwe,all_txt, f_raw_req_list)    # rectified frequency calculation
                f_raw_req_list.append([mwe,f_raw,f_req])          # adding information about a phrase and its frequency to the list

            mwe_f=[]
            f_raw_req_list_ind=[concatenate_ngrams(i[0]) for i in f_raw_req_list]

            # Merging Phrase Frequency and Phrase Info Lists
            for mwe in mwe_list:
                k=f_raw_req_list_ind.index(concatenate_ngrams(mwe[0]))
                mwe_f.append([mwe[0],f_raw_req_list[k][1], f_raw_req_list[k][2],mwe[3], mwe[-1]])

            # grouping phrases by common word position
            # input: [candidate, raw frequency, straightened frequency, word position indices, sentence in which it is located])
            # output: phrases grouped by position (at the end, the number of the group to which the phrase belongs is indicated):
            # [candidate, raw frequency, straightened frequency, word position indices, sentence in which it is located, group number])
            grouped_data = group_items(mwe_f)

            candidates=[]
            candid_q=[]
            remover=[]
            df=pd.DataFrame(grouped_data, columns=["mwe","f raw","f req","index","sent","group"])

             # selecting a candidate from the group with the highest rectified or raw frequency
            for i in range(1,len(set(df["group"]))+1):
                df_temp=df[df['group'] == i]
                while len(df_temp)>0:
                      max=df_temp["f req"].max()
                      cand=df_temp[df_temp["f req"]==max].values.tolist()[0]
                      if max>1:
                         candidates.append(cand)
                      elif df_temp["f raw"].max()>1:
                         candid_q.append(cand)

                      index=df_temp["index"][df_temp["f req"]==max].values.tolist()[0]
                      drop=df_temp.index[df_temp['index'].apply(lambda x: any(i in index for i in x))].tolist()
                      dd=df_temp.loc[drop].values.tolist()
                      df_temp=df_temp[~df_temp.index.isin(drop+index)]
                      remover+=dd

            # or phrases are accepted if they have a rectified or raw frequency above the specified threshold
            data1=df[df["f req"]>=self.f_req_sc].values.tolist()
            data2=df[df["f raw"]>=self.f_raw_sc].values.tolist()

            # combine words of a phrase and create a single list of extracted phrases: ["phrase 1","phrase 2","phrase3"]
            cand_mwe=[concatenate_ngrams(i[0]) for i in candidates+candid_q]
            cand_mwe1=[concatenate_ngrams(i[0]) for i in data1]
            cand_mwe2=[concatenate_ngrams(i[0]) for i in data2]

            cand=cand_mwe+cand_mwe1+cand_mwe2
            candidates = [i for i in cand if ((i[-1] not in punc_without) and (i[-1] not in string.punctuation))]

        return candidates

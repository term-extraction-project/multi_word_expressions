import nltk
from nltk.util import ngrams
nltk.download("punkt")
import string

from operator import itemgetter
import pandas as pd
import string


import stanza
stanza.download('kk')
nlp = stanza.Pipeline('kk')

punc_without = list(string.punctuation)+["»","«"]
punc_without.remove('-')
punc_without.remove("'")
punc_all=list(string.punctuation)+["»","«"]


list_seq_2=  [[["PROPN","NOUN"],"*"],
              ["ADJ",'*', ["PROPN","NOUN"], '*']]

def tokinizer(sent):
  index=0
  list_tok=[]
  for i in sent.words:
        list_tok.append((i.text.lower(), i.upos, index, i.lemma.lower()))
        index+=1
  return list_tok

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

def filter_ngrams_by_pos_tag(sentence, sequense):
    filtered_ngrams=[]
    t=0
    for seq in sequense:
        for i in range(len(sentence)):
            temp=[]
            temp_index=[]
            temp_pos=[]
            temp_lemma=[]
            checker=True

            if ((sentence[i][1] in seq[0]) or (sentence[i][1] == seq[0])) and (sentence[i][0] not in punc_without) :
               seq_index=0
               sent_index=0

               while seq_index<len(seq) and i+sent_index<len(sentence) and checker==True:
                   if (sentence[i+sent_index][1] in seq[seq_index]) or (sentence[i+sent_index][0] in seq[seq_index]) :
                       temp.append(sentence[i+sent_index][0])
                       temp_pos.append(sentence[i+sent_index][1])
                       temp_index.append(sentence[i+sent_index][2])
                       temp_lemma.append(sentence[i+sent_index][3])
                       seq_index += 1
                       sent_index += 1

                   elif seq[seq_index]=="*" and (sentence[i+sent_index][1] in seq[seq_index-1]):
                       if seq_index<len(seq)-1:
                              temp.append(sentence[i+sent_index][0])
                              temp_pos.append(sentence[i+sent_index][1])
                              temp_index.append(sentence[i+sent_index][2])
                              temp_lemma.append(sentence[i+sent_index][3])
                              sent_index += 1

                       elif seq_index==len(seq)-1:
                             if (len(temp)>1  or "-" in "".join(temp)) and (len(set("".join(temp)).intersection(set(punc_without)-set("-'")))==0):
                                 temp_2=temp.copy()
                                 temp_pos2=temp_pos.copy()
                                 temp_index2=temp_index.copy()
                                 temp_lemma2=temp_lemma.copy()
                                 if [temp_2, seq, temp_pos2, temp_index2,len(temp_2), len(concatenate_ngrams(temp_2)), temp_lemma2] not in filtered_ngrams and temp[-1] not in punc_without:
                                     filtered_ngrams.append([temp_2, seq, temp_pos2, temp_index2,len(temp_2),len(concatenate_ngrams(temp_2)),temp_lemma2])

                             if (i+sent_index)<len(sentence):
                                temp.append(sentence[i+sent_index][0])
                                temp_pos.append(sentence[i+sent_index][1])
                                temp_index.append(sentence[i+sent_index][2])
                                temp_lemma.append(sentence[i+sent_index][3])
                                sent_index += 1

                   elif seq[seq_index]=="*" and (sentence[i+sent_index][1] not in seq[seq_index-1]):
                      seq_index += 1

                   else:
                      checker=False
      
               if seq_index==len(seq) and (len(temp)>1  or "-" in "".join(temp)) and len(set("".join(temp)).intersection(set(punc_without)-set("-'")))==0:
                    if [temp, seq, temp_pos, temp_index,len(temp),len(concatenate_ngrams(temp)),temp_lemma] not in filtered_ngrams and temp[-1] not in punc_without:
                         filtered_ngrams.append([temp, seq, temp_pos, temp_index,len(temp),len(concatenate_ngrams(temp)),temp_lemma])

    return  filtered_ngrams


def f_req_calc(mwe, all_txt, f_raw_req_list):
  temp=all_txt
  mwe_c=concatenate_ngrams(mwe)
  for i in f_raw_req_list:
    i_c=concatenate_ngrams(i[0])
    if mwe_c in i_c and mwe_c!=i_c and len(mwe)!=len(i[0]):
      temp=temp.replace(i_c," ")
  f=temp.count(mwe_c)
  return f


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

class KazakhPhraseExtractor:
    def __init__(self, text,  list_seq=list_seq_2,  cohision_filter=True, additional_text="1", f_raw_sc=9, f_req_sc=3):
        self.text = text
        self.cohision_filter=cohision_filter
        self.additional_text=additional_text
        self.f_req_sc=f_req_sc
        self.f_raw_sc=f_raw_sc
        self.list_seq=list_seq

    def extract_phrases(self):  
        sent=tokinizer(self.text)
        mwe_list = filter_ngrams_by_pos_tag(sent, self.list_seq)

        mwe_list_n = [tuple(i[0]) for i in mwe_list]
        candidates = []
        for i in set(mwe_list_n):
            candidates.append(concatenate_ngrams(i))

        candidates = [i for i in candidates if ((i[-1] not in punc_without) and (i[-1] not in string.punctuation))]
        candidates = [i for i in candidates if len(set('1234567890').intersection(set(i))) == 0]

        #text for cohision filter and calculate frequency
        all_txt=(self.additional_text).lower()

        #Filter based on frequency
        if  self.cohision_filter==True:

            f_raw_req_list=[]
            all_cand_r=[tuple(i[6]) for i in mwe_list]
            possible_mwe=sorted(set(all_cand_r), key=len, reverse=True)

            for mwe in possible_mwe:
                f_raw=all_txt.count(concatenate_ngrams(mwe))
                f_req=f_req_calc(mwe,all_txt, f_raw_req_list)
                f_raw_req_list.append([mwe,f_raw,f_req])

            mwe_f=[]
            f_raw_req_list_ind=[concatenate_ngrams(i[0]) for i in f_raw_req_list]

            for mwe in mwe_list:
                k=f_raw_req_list_ind.index(concatenate_ngrams(mwe[6]))
                mwe_f.append([mwe[0],f_raw_req_list[k][1], f_raw_req_list[k][2],mwe[3]])


            grouped_data = group_items(mwe_f)

            candidates=[]
            remover=[]
            df=pd.DataFrame(grouped_data, columns=["mwe","f raw","f req","index","group"])

            for i in range(1,len(set(df["group"]))+1):
                df_temp=df[df['group'] == i]
                while len(df_temp)>0:
                      max=df_temp["f req"].max()
                      cand=df_temp[df_temp["f req"]==max].values.tolist()[0]
                      if df_temp["f raw"].max()>0:
                            candidates.append(cand)

                      index=df_temp["index"][df_temp["f req"]==max].values.tolist()[0]
                      drop=df_temp.index[df_temp['index'].apply(lambda x: any(i in index for i in x))].tolist()
                      dd=df_temp.loc[drop].values.tolist()
                      df_temp=df_temp[~df_temp.index.isin(drop+index)]
                      remover+=dd

            data1=df[df["f req"]>=self.f_req_sc].values.tolist()
            data2=df[df["f raw"]>=self.f_raw_sc].values.tolist()

            cand_mwe=[concatenate_ngrams(i[0]) for i in candidates]
            cand_mwe1=[concatenate_ngrams(i[0]) for i in data1]
            cand_mwe2=[concatenate_ngrams(i[0]) for i in data2]

            cand=cand_mwe+cand_mwe1+cand_mwe2
            candidates = [i for i in cand if ((i[-1] not in punc_without) and (i[-1] not in string.punctuation))]

        return candidates

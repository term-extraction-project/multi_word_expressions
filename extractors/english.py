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

# загруска стоп слов
url = 'https://raw.githubusercontent.com/term-extraction-project/stop_words/main/stop_words_en.txt'
stop_words = (requests.get(url).text).split(",")

# шаблоны частей речи
pos_tag_patterns=  [[["PROPN","NOUN"],"*"],
                     ["ADJ",'*', ["PROPN","NOUN"], '*'],
                     ["ADJ","*"],
                     ['VERB', 'ADJ', ["PROPN","NOUN"],'*'],
                     [["PROPN","NOUN"],'*','ADJ','*',["PROPN","NOUN"],'*'],
                     ['ADJ','VERB',["PROPN","NOUN"], '*'],
                     ['VERB','*',["PROPN","NOUN"],'*'],
                     [["PROPN","NOUN"], '*','ADJ','*',["PROPN","NOUN"], '*'],
                     ["ADV","*","ADJ","*"],
                     [["PROPN","NOUN"],'ADP',["PROPN","NOUN"]],
                     [["ADJ","PROPN","NOUN"],"*","PART",["PROPN","NOUN"],"*"],
                     [['VERB',"ADV","X"]],
                     [["ADJ","PROPN","NOUN"],"*", "ADP",["PROPN","NOUN"],"*"]
                      ]
# настройка списков пунктуаций для проверки, punc_without не содержит дефиса и апострофа ак как они могут быть в составе фраз.  punc_all необходим для проверки есть ли дефис в начале или в конце фразы
punc_without = list(string.punctuation)+["»","«"]
punc_without.remove('-')
punc_without.remove("'")
punc_all=list(string.punctuation)+["»","«"]

# Токенизатор текста, на вход текст с оригинальным регистром НЕ в нижнем регистре 
#  на выходе набор размеченных токенов по предложениям, элемент в списке - предложение, которое содержит токены с информацией о них
#  [ [("token1", pos, index),("token2", pos, index),("token3", pos, index)],    
#    [("token1", pos, index),("token2", pos, index),("token3", pos, index)]]
def tokinizer(text, nlp):
  sent_tokens = []
  index=0
  text_t=nlp(text)
  for sent in text_t.sents:
      list_tok=[]
      for i in sent:
        list_tok.append((i.text.lower(), i.pos_,index))  # создание списка токенов с содержанием, собственно униграммы в нижнем регистре, его часть речи, номер позиции в тексте
        index+=1
      sent_tokens.append(list_tok)
  return sent_tokens



# функция объединения токенов фразы в единый string
# на вход список слов ["word1","word2", "word3"]
# на выходе  "word1 word2 word3"
# пробел между дефисом и апострофом не ставится
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




# фильтр на основе изменяющейся части речи 
# на вход список кандидатов
# на выходе отфильтрованный список
# если при повторной разметке фразы которые заканчиваются на NOUN или PROPN поменяла часть речи на иную кроме "NOUN","PROPN","VERB", то такая фраза не полная, и она удаляется
# при разметке Spacy даже дефис может быть PROPN или NOUN если он является часть целого слова, например IT-developers
def filter_propn_noun(mwe_list,nlp):
    filtred_ngrams=[]
    for i in mwe_list:
      checker2=True
      if concatenate_ngrams(i[0])[-1] in punc_all and concatenate_ngrams(i[0])[0] in punc_all:
         checker2=False

      if len(i[2])>1:
       if (("NOUN" in i[1][-1]) or ("PROPN" in i[1][-1]) or ("NOUN" in i[1][-2]) or ("PROPN" in i[1][-2])) and (("ADJ" not in i[1][-1]) and ("ADJ" not in i[1][-2])) :
        temp_seq=str(concatenate_ngrams(i[0]))
        temp_token=nlp(temp_seq)
        if (temp_token[-1]).pos_ not in ["NOUN","PROPN","VERB"]:
           checker2=False

      if checker2==True:
        filtred_ngrams.append(i)
    return filtred_ngrams

# Очистка фраз от стоп-слов,
# если слово в фразе является стоп-словом, фраза удаляется. проверяются все слова в фразе кроме предлогов и PROPN
# на входе список фраз и список стоп-слов
# На выходе отфильтрированный список
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



# извлечение кандидатов на основе шаблонов частей речи
# на вход список токенов в одном предложении [("token1", pos, index),("token2", pos, index),("token3", pos, index)] и  шаблон частей речи
# на выходе список извлеченных кандидатов с информацией о них: 
# [[список слов фразы], шаблон по которому был извлечен кандидат, последовательность частей речи канддата, индексы позиций слов,количество слов, количество символов в кандидате ]
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

# расчет выпрямленной частоты - количество фразы в тексте, не в составе фраз длинее
# на вход: список слов фразы(одного кандидата), все тексты в виде единого string, список f_raw_req_list в котором содержатся частоты фраз длинее/или он пуст, так как заполняется в процессе вызова функции
# на выходе : выпрямленная частота фразы
# функция вызывается столько же раз сколько и кандидатов. для расчета выпрямленной частоты, расчет производится от самой длинной до самой короткой фразы, так как для ее расчет необходимо знать частоту фраз длинее целевой
def f_req_calc(mwe, all_txt, f_raw_req_list):
  temp=all_txt
  mwe_c=concatenate_ngrams(mwe)
  for i in f_raw_req_list:
    i_c=concatenate_ngrams(i[0])
    if mwe_c in i_c and mwe_c!=i_c and len(mwe)!=len(i[0]):
      temp=temp.replace(i_c," ")
  f=temp.count(mwe_c)
  return f

# 
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
  

# основное тело извлечения фраз
# на вход текст и разные параметры
# на выходе список фраз: ["phrase 1", "phrase 2", "phrase 3"]
class EnglishPhraseExtractor:
    def __init__(self, text, stop_words=stop_words, list_seq=pos_tag_patterns,  cohision_filter=True, additional_text="1", f_raw_sc=9, f_req_sc=3):
        self.text = text            # текст в оригинальном регистре
        self.cohision_filter=cohision_filter     #  Включить или отключить когезионный фильтр
        self.additional_text=additional_text   # если есть дополнительный текст, используется для вычисления частот, из него термины НЕ извлекаються
        self.f_req_sc=f_req_sc     # порог выпрямленной частоты
        self.f_raw_sc=f_raw_sc       # порог сырой частоты
        self.stop_words=stop_words     # список стоп-сло
        self.list_seq=list_seq         # список шаблонов частей речи
        self.model_nlp = spacy.load("en_core_web_sm")     # модель Spacy

     # Извлечение фраз 
    def extract_phrases(self):
           
        # Изменение токенизатора , чтобы не разделял слова с дефисом.
        # IT-developers create innovative solutions.  --> ["IT-developers","create","innovative","solutions","."]
        # Вместо ["IT","-","developers"] токенизируется как цельный токен "IT-developers", это помогает избежать извлечения униграм которые находятся в  составе слова, что уменьшает шум

         
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


        # удаление лишних пробелов
        text=self.text.replace(" -","-").replace("- ","-").replace(" '","'").replace("  "," ")

        #   токенизация текста, растановка частей речи и индекса позиций
        text_sent_tokens = tokinizer(text, nlp)
        mwe_list = []

        # извлечение кандидатов из каждого предложения по отдельности
        for sent in text_sent_tokens:
            temp_mwe_list = filter_ngrams_by_pos_tag(sent, self.list_seq)  # извлечение на основе частей речи
            temp_mwe_list = filter_propn_noun(temp_mwe_list,nlp)             #  фильтрация от изменяющихся частей речи
            temp_mwe_list = filter_stop_words(temp_mwe_list, self.stop_words)   # фильтрация от стоп-слов
           
            # temp_mwe_list  содержит списки кандидатов с дополнительной информацией о них:
            # [[список слов фразы], шаблон по которому был извлечен кандидат, последовательность частей речи канддата, индексы позиций слов,количество слов, количество символов в кандидате ]

            sent_text=" ".join([s[0] for s in sent])
            temp_mwe_list=[mwe+[sent_text] for mwe in temp_mwe_list]
            mwe_list += temp_mwe_list

        # создание списка кандидатов содержащий только слова кандидатов: [("phrase","one"),("mwe","next"),("phrase","other")]
        mwe_list_n = [tuple(i[0]) for i in mwe_list]
        candidates = []
        term_mws=[]
        words_scores=[]
        candid_word=[]

       # объединение слов фразы:  ["phrase one", "mwe next","phrase other"]
        for i in set(mwe_list_n):
            candidates.append(concatenate_ngrams(i))

       # очистка фраз от пунктуации и если полностью состоит из пунктуации и цифр
        candidates = [i for i in candidates if ((i[-1] not in punc_without) and (i[-1] not in string.punctuation))]
        candidates = [i for i in candidates if len(set('1234567890').intersection(set(i))) == 0]

        #текст для когезионного фильтра
        all_txt=''
        if len(self.additional_text)>10:  # если есть текст (цифра 10 рандомная, главное чтобы не пустой)  то объединяем с текстом из которого извлекаличь кандилаты
             text_ref=self.additional_text.replace(" -","-").replace("- ","-").replace(" '","'").replace("  "," ").lower()
             all_txt=self.text+". "+text_ref
        else:
             all_txt=self.text
        all_txt=all_txt.lower()

        # если когезионный фильтр включен
        if  self.cohision_filter==True:
            f_raw_req_list=[]
            all_cand_r=[tuple(i[0]) for i in mwe_list]
            possible_mwe=sorted(set(all_cand_r), key=len, reverse=True)    # сортировка фраз от самой длинной к самой короткой

            for mwe in possible_mwe:
                f_raw=all_txt.count(concatenate_ngrams(mwe))    # вычисление сырой частоты
                f_req=f_req_calc(mwe,all_txt, f_raw_req_list)    # расчет выпрямленной частоты
                f_raw_req_list.append([mwe,f_raw,f_req])          # добавление в список информации о фразе и ее частоте

            mwe_f=[]
            f_raw_req_list_ind=[concatenate_ngrams(i[0]) for i in f_raw_req_list]

            # Объединение списоков частоты фразы и информации о фразе
            for mwe in mwe_list:
                k=f_raw_req_list_ind.index(concatenate_ngrams(mwe[0]))
                mwe_f.append([mwe[0],f_raw_req_list[k][1], f_raw_req_list[k][2],mwe[3], mwe[-1]])

            # группирование фраз по общей позиции слов
           # на вход : [кандидат,сырая частота, выпремленная частота, индексы позиций слов, предложение в котором находится])
           # на выходе сгруппированные фразы по позициям(вконце указан номер группы, к которой принадлежит фраза): 
          # [кандидат,сырая частота, выпремленная частота, индексы позиций слов, предложение в котором находится, номер группы ])
           
            grouped_data = group_items(mwe_f)

            candidates=[]
            candid_q=[]
            remover=[]
            df=pd.DataFrame(grouped_data, columns=["mwe","f raw","f req","index","sent","group"])

             # выбор кандидата из группы с наибольшей выпрямленной или сырой частотой 
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

            # либо фраз принимаются если имеют выпрямленную или сырую частоту выше указанного порога
            data1=df[df["f req"]>=self.f_req_sc].values.tolist()
            data2=df[df["f raw"]>=self.f_raw_sc].values.tolist()

            # объединение слов фразы и создание единого списка извлеченных фраз: ["phrase 1","phrase 2","phrase3"] 
            cand_mwe=[concatenate_ngrams(i[0]) for i in candidates+candid_q]
            cand_mwe1=[concatenate_ngrams(i[0]) for i in data1]
            cand_mwe2=[concatenate_ngrams(i[0]) for i in data2]

            cand=cand_mwe+cand_mwe1+cand_mwe2
            candidates = [i for i in cand if ((i[-1] not in punc_without) and (i[-1] not in string.punctuation))]

        return candidates

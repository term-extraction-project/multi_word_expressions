from multi_word_expressions.english import EnglishPhraseExtractor
from multi_word_expressions.kazakh import KazakhPhraseExtractor

KK_POS_PATTERNS=  [[["PROPN","NOUN"],"*"],
              ["ADJ",'*', ["PROPN","NOUN"], '*']]

import requests

url = 'https://raw.githubusercontent.com/term-extraction-project/stop_words/main/stop_words_en.txt'
EN_STOP_WORDS = (requests.get(url).text).split(",")

EN_POS_PATTERNS=  [[["PROPN","NOUN"],"*"],
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


class PhraseExtractor:
    def __init__(self, text, lang="en", stop_words=None, list_seq=None, cohision_filter=True, additional_text="1", f_raw_sc=9, f_req_sc=3):

        self.text = text
        self.lang = lang
        self.cohision_filter = cohision_filter
        self.additional_text = additional_text
        self.f_raw_sc = f_raw_sc
        self.f_req_sc = f_req_sc
        self.stop_words=stop_words
        self.list_seq=list_seq

        if lang == "en":
            stop_words =  self.stop_words if  self.stop_words is not None else EN_STOP_WORDS
            list_seq = self.list_seq if self.list_seq is not None else EN_POS_PATTERNS
            self.extractor = EnglishPhraseExtractor(
                text=self.text,
                stop_words=stop_words,
                list_seq=list_seq,
                cohision_filter=self.cohision_filter,
                additional_text=self.additional_text,
                f_raw_sc=self.f_raw_sc,
                f_req_sc=self.f_req_sc
            )
        elif lang == "kk":
            list_seq = self.list_seq if self.list_seq is not None else KK_POS_PATTERNS
            self.extractor = KazakhPhraseExtractor(
                text=self.text,
                list_seq=list_seq,
                cohision_filter=self.cohision_filter,
                additional_text=self.additional_text,
                f_raw_sc=self.f_raw_sc,
                f_req_sc=self.f_req_sc
            )
        else:
            raise ValueError(f"Unsupported language: {lang}")

    def extract_phrases(self):

        return self.extractor.extract_phrases()

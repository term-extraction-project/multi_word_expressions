from multi_word_expressions.english import EnglishPhraseExtractor
from multi_word_expressions.kazakh import KazakhPhraseExtractor

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
            self.extractor = EnglishPhraseExtractor(
                text=self.text,
                stop_words=self.stop_words,
                list_seq=self.list_seq,
                cohision_filter=self.cohision_filter,
                additional_text=self.additional_text,
                f_raw_sc=self.f_raw_sc,
                f_req_sc=self.f_req_sc
            )
        elif lang == "kk":
            self.extractor = KazakhPhraseExtractor(
                text=self.text,
                stop_words=self.stop_words
                list_seq=self.list_seq,
                cohision_filter=self.cohision_filter,
                additional_text=self.additional_text,
                f_raw_sc=self.f_raw_sc,
                f_req_sc=self.f_req_sc
            )
        else:
            raise ValueError(f"Unsupported language: {lang}")

    def extract_phrases(self):

        return self.extractor.extract_phrases()

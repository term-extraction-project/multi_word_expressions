from multi_word_expressions.english import EnglishPhraseExtractor
from multi_word_expressions.kazakh import KazakhPhraseExtractor

class PhraseExtractor:
    def __init__(self, text, cohision_filter=True, lang="en", f_raw_sc=9, f_req_sc=3):
        self.text = text
        self.cohision_filter = cohision_filter
        self.lang = lang
        self.f_raw_sc = f_raw_sc
        self.f_req_sc = f_req_sc

        if lang == "en":
            self.extractor = EnglishPhraseExtractor(text, cohision_filter, f_raw_sc, f_req_sc)
        elif lang == "kk":
            self.extractor = KazakhPhraseExtractor(text, cohision_filter, f_raw_sc, f_req_sc)
        else:
            raise ValueError(f"Unsupported language: {lang}")

    def extract_phrases(self):
        return self.extractor.extract_phrases()

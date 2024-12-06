# Phrase Extraction Project

A Python package for extracting phrases from text.

## Installation

You can install the package directly from GitHub using the following command:

```bash
pip install git+https://github.com/term-extraction-project/multi_word_expressions.git
```

### Usage

Here is an example of how to use the Phrase Extractor for english from the package:

```bash
from multi_word_expressions.english import EnglishPhraseExtractor

# Your input text
text = "Your input text here."
additional_text="It is your additional text."
                   
extractor = EnglishPhraseExtractor(   text=text,
                                      stop_words=stop_words,                # стоп-слова, по умолчанию установлены
                                      cohision_filter=True,                 # Фильтрация по когезии
                                      additional_text=additional_text,      # Дополнительный текст (если имеется)
                                      list_seq=pos_tag_patterns,            # Пользовательские POS-шаблоны, по умолчанию установлены
                                      f_raw_sc=9,                           # Частотный фильтр для сырого текста
                                      f_req_sc=3)                           # Частотный фильтр для отобранных кандидатов
candidates = extractor.extract_phrases()
print(candidates)
```


Here is an example of how to use the Phrase Extractor for kazakh from the package:

```bash
from multi_word_expressions.kazakh import KazakhPhraseExtractor

text = "Сіздің мәтініңіз қазақ тіліндегі."
additional_text="Міне, сіздің қосымша мәтініңіз."
                   
extractor = KazakhPhraseExtractor(   text=text,
                                      stop_words=stop_words,                # стоп-слова, по умолчанию установлены
                                      cohision_filter=True,                 # Фильтрация по когезии
                                      additional_text=additional_text,      # Дополнительный текст (если имеется)
                                      list_seq=pos_tag_patterns,            # Пользовательские POS-шаблоны, по умолчанию установлены
                                      f_raw_sc=9,                           # Частотный фильтр для сырого текста
                                      f_req_sc=3)                           # Частотный фильтр для отобранных кандидатов
candidates = extractor.extract_phrases()
print(candidates)
```

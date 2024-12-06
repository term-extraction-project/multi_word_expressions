# Phrase Extraction Project

A Python package for extracting phrases from text.

## Installation

You can install the package directly from GitHub using the following command:

```bash
pip install git+https://github.com/term-extraction-project/multi_word_expressions.git

from multi_word_expressions.english import EnglishPhraseExtractor
from multi_word_expressions.kazakh import KazakhPhraseExtractor
```

### Usage

Here is an example of how to use the PhraseExtractor class from the package:

```bash
from main import PhraseExtractor
# Your input text
from multi_word_expressions import PhraseExtractor

text = "Your input text here."
additional_text="It is your additional text."
extractor_en = PhraseExtractor(
              text=text,
              lang="en",
              #stop_words=custom_stop_words_en,   # Пользовательские стоп-слова, по умолчанию установлены
              #list_seq=custom_pos_patterns_en,   # Пользовательские POS-шаблоны, по умолчанию установлены
              cohision_filter=True,               # Фильтрация по когезии
              additional_text=additional_text,    # Дополнительный текст (если требуется)
              f_raw_sc=9,                         # Частотный фильтр для сырого текста
              f_req_sc=3)                         # Частотный фильтр для отобранных кандидатов
          
candidates = extractor.extract_phrases()
print(candidates)
```

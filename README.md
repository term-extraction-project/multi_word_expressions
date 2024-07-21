# Term Extraction Project

A Python package for extracting phrases from text.

## Installation

You can install the package directly from GitHub using the following command:

```bash
!pip install git+https://github.com/term-extraction-project/multi_word_expressions.git
```

### Usage

Here is an example of how to use the PhraseExtractor class from the package:

```bash
from main import PhraseExtractor
# Your input text
text = "Your input text here."
extractor = PhraseExtractor(text)
candidates = extractor.extract_phrases()
print(candidates)
```

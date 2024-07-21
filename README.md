# Term Extraction Project

A Python package for extracting phrases from text.

## Installation

You can install the package directly from GitHub using the following command:

```bash
!pip install git+https://github.com/term-extraction-project/multi_word_expressions.git

### Usage
Here is an example of how to use the PhraseExtractor class from the package:

python
Копировать код
from main import PhraseExtractor

# Your input text
text = "Your input text here."

# Create an instance of PhraseExtractor
extractor = PhraseExtractor(text)

# Extract phrases
candidates = extractor.extract_phrases()

# Print the extracted phrases
print(candidates)
Example
Below is a complete example demonstrating the usage of the PhraseExtractor class:

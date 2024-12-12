from setuptools import setup, find_packages

setup(
    name="multi_word_expressions",
    version="0.1.0",
    description="A tool for extracting multi-word expressions for English and Kazakh.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Aliya Kalykulova",
    author_email="aliyakalykulova@mail.ru",
    url="https://github.com/term-extraction-project/multi_word_expressions",
    packages=find_packages(),
    install_requires=[],  # Укажите зависимости, если они есть
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

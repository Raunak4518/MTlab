import unicodedata
from importlib.metadata import version
from sacremoses import MosesTokenizer
from indicnlp.tokenize import indic_tokenize


_ET = MosesTokenizer(lang="en")
def normalize_text(text:str)-> str :
    text = unicodedata.normalize("NFC",text)
    text = " ".join(text.split())
    return text
def tokenize_english(text: str):
    text = normalize_text(text)
    if not text:
        return []
    tokens = _ET.tokenize(text,return_str=False, escape=False,)
    return tokens
def tokenize_hindi(text:str):
    text =normalize_text(text)
    if not text:
        return []
    tokens = indic_tokenize.trivial_tokenize(
        text , lang="hi"
    )
    return tokens

    
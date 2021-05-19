from typing import Optional
from thinc.api import Model
from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
from .punctuation import TOKENIZER_PREFIXES, TOKENIZER_INFIXES
from .punctuation import TOKENIZER_SUFFIXES
from .stop_words import STOP_WORDS
from .syntax_iterators import SYNTAX_ITERATORS
from spacy.language import Language
from spacy.pipeline import Lemmatizer


class NorwegianNynroskDefaults(Language.Defaults):
    tokenizer_exceptions = TOKENIZER_EXCEPTIONS
    prefixes = TOKENIZER_PREFIXES
    infixes = TOKENIZER_INFIXES
    suffixes = TOKENIZER_SUFFIXES
    syntax_iterators = SYNTAX_ITERATORS
    stop_words = STOP_WORDS


class NorwegianNynrosk(Language):
    lang = "nn"
    Defaults = NorwegianNynroskDefaults


@NorwegianNynrosk.factory(
    "lemmatizer",
    assigns=["token.lemma"],
    default_config={"model": None, "mode": "rule", "overwrite": False},
    default_score_weights={"lemma_acc": 1.0},
)
def make_lemmatizer(
    nlp: Language, model: Optional[Model], name: str, mode: str, overwrite: bool
):
    return Lemmatizer(nlp.vocab, model, name, mode=mode, overwrite=overwrite)


__all__ = ["NorwegianNynrosk"]

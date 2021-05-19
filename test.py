from spacy.lang.nn import NorwegianNynrosk

nlp = NorwegianNynrosk()

print(nlp.lang, [token.is_stop for token in nlp("nynorsk og bokmal")])
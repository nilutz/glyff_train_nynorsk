import spacy

nlp = spacy.load('nn_pipeline')


for text in [
    'Korleis gjer eg dette',
    'Språksituasjonen i Noreg',
    'Da ho hadde trilla ei lang, lang stund, motet ho ei gas']:

    print(text)
    print('\n')
    doc = nlp.make_doc(text)
    for name, proc in nlp.pipeline:

        print(name, proc)
        doc = proc(doc)
    print('-'*20)
    print('\n')
    for token in doc:
        print(token, token.pos_, token.dep_, token.tag_)

    print('='*20)
    print('\n\n')
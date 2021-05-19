# Train Nynorsk in Spacy v3

ideally create a project

we are basically retraining/updating the bøkmal model

## 0. Prepare

    python3 -m venv .venv

    pip install spacy


## 1. Create Nynrosk [Language Class](https://spacy.io/api/language#defaults)

[Creating a custom language subclass](https://spacy.io/usage/linguistic-features#language-data)

Create a nn class in ".venv/lib/python3.9/site-packages/spacy/lang/nn" or somewhere else

-> (possibly) also update, create, correct:
 * abbr.
 * stopwords
 * lemmatizer
 *  morphological analyzer
 *  etc.

here

now it can be e.g. referenced as:

```python
from spacy.lang.nn import NorwegianNynrosk

nlp = NorwegianNynrosk()

print(nlp.lang, [token.is_stop for token in nlp("nynorsk og bokmal")])
```

## 2. Create training data

download [UD treebank Nynrosk](https://github.com/UniversalDependencies/UD_Norwegian-Nynorsk) and put into ```/data```

    mkdir train_data

And for all the data

    spacy convert -n 10 data/UD_Norwegian-Nynorsk/no_nynorsk-ud-test.conllu train_data

    spacy convert -n 10 data/UD_Norwegian-Nynorsk/no_nynorsk-ud-train.conllu train_data

    spacy convert -n 10 data/UD_Norwegian-Nynorsk/no_nynorsk-ud-dev.conllu train_data

## 3. The Config
Create a base config [here](https://spacy.io/usage/training#spacy-train-cli)

    python -m spacy init fill-config base_config.cfg config.cfg

## 4. Pretrain

could use a [pretrain](https://spacy.io/api/cli#pretrain) approach on tok2vec with eg, fastext, word2vec, etc

## 5. And train !

    python -m spacy train config.cfg --output ./output --paths.train ./train_data/no_nynorsk-ud-train.spacy --paths.dev ./train_data/no_nynorsk-ud-dev.spacy

instead of using step 1 one can do ```--code, -c	```to include the same scripts

```bash
python -m spacy train config.cfg --output ./output --paths.train ./train_data/no_nynorsk-ud-train.spacy --paths.dev ./train_data/no_nynorsk-ud-dev.spacy
ℹ Using CPU

=========================== Initializing pipeline ===========================
[2021-05-18 20:29:52,229] [INFO] Set up nlp object from config
[2021-05-18 20:29:52,241] [INFO] Pipeline: ['tok2vec', 'tagger', 'parser', 'ner']
[2021-05-18 20:29:52,249] [INFO] Created vocabulary
[2021-05-18 20:29:52,250] [INFO] Finished initializing nlp object
[2021-05-18 20:30:15,046] [INFO] Initialized pipeline components: ['tok2vec', 'tagger', 'parser', 'ner']
✔ Initialized pipeline

============================= Training pipeline =============================
ℹ Pipeline: ['tok2vec', 'tagger', 'parser', 'ner']
ℹ Initial learn rate: 0.001
E    #       LOSS TOK2VEC  LOSS TAGGER  LOSS PARSER  LOSS NER  TAG_ACC  DEP_UAS  DEP_LAS  SENTS_F  ENTS_F  ENTS_P  ENTS_R  SCORE
---  ------  ------------  -----------  -----------  --------  -------  -------  -------  -------  ------  ------  ------  ------
  0       0          0.00       166.59       366.34     88.50    35.70    12.17     6.74     0.13    0.00    0.00    0.00    0.15
  0     200       3019.67     14583.43     30630.41    281.79    83.54    65.30    53.03    79.93    0.00    0.00    0.00    0.48
  0     400       4694.74      7717.34     22757.49      0.00    87.88    69.43    59.86    74.64    0.00    0.00    0.00    0.51
  0     600       5233.93      6197.61     20871.13      0.00    89.20    73.25    64.59    82.21    0.00    0.00    0.00    0.53
  0     800       5651.34      5648.10     20514.28      0.00    90.03    73.84    65.96    72.71    0.00    0.00    0.00    0.53
  0    1000       5670.19      5058.28     17700.58      0.00    90.88    76.91    69.35    88.31    0.00    0.00    0.00    0.55
  0    1200       6065.36      4856.02     17669.07      0.00    91.43    77.39    70.18    86.83    0.00    0.00    0.00    0.55
  0    1400       6132.10      4549.24     16925.64      0.00    91.43    76.12    69.07    86.12    0.00    0.00    0.00    0.55
  1    1600       6750.81      3915.63     16412.45      0.00    92.01    78.25    71.48    87.31    0.00    0.00    0.00    0.56
  1    1800       6900.66      3733.95     15573.94      0.00    92.23    78.63    72.33    89.63    0.00    0.00    0.00    0.56
  1    2000       8203.88      4211.78     17663.30      0.00    92.44    79.95    73.50    90.17    0.00    0.00    0.00    0.57
  1    2200      10485.59      5298.89     21927.45      0.00    92.61    80.26    74.10    88.37    0.00    0.00    0.00    0.57
  1    2400      13423.41      6708.82     27535.07      0.00    93.08    81.10    75.50    90.51    0.00    0.00    0.00    0.57
  2    2600      15760.75      6615.98     30854.50      0.00    93.35    81.19    76.14    90.02    0.00    0.00    0.00    0.58
  2    2800      20693.44      7978.55     37577.29      0.00    93.55    82.48    77.42    91.27    0.00    0.00    0.00    0.58
  3    3000      25613.86      9021.18     44319.46      0.00    93.70    82.84    77.95    90.85    0.00    0.00    0.00    0.58
  3    3200      33367.47     10023.25     52253.02      0.00    93.91    83.60    79.11    90.83    0.00    0.00    0.00    0.59
  4    3400      34719.11      9495.34     50427.25      0.00    94.04    84.09    79.58    90.95    0.00    0.00    0.00    0.59
  5    3600      36071.09      8761.74     48938.64      0.00    93.99    84.52    79.97    91.14    0.00    0.00    0.00    0.59
  6    3800      36378.90      7979.04     46696.28      0.00    94.08    84.16    79.61    91.17    0.00    0.00    0.00    0.59
  6    4000      38678.32      7716.23     45608.69      0.00    94.24    84.52    80.10    91.56    0.00    0.00    0.00    0.59
  7    4200      37688.96      6858.87     42784.53      0.00    94.31    84.33    79.83    90.97    0.00    0.00    0.00    0.59
  8    4400      38688.06      6681.43     41316.72      0.00    94.12    84.57    79.99    90.45    0.00    0.00    0.00    0.59
  9    4600      40457.42      6504.52     41010.86      0.00    94.09    84.20    79.69    90.71    0.00    0.00    0.00    0.59
  9    4800      40428.16      5958.62     38461.33      0.00    94.25    84.67    80.05    90.55    0.00    0.00    0.00    0.59
 10    5000      39888.29      5546.82     37225.23      0.00    94.18    84.32    80.03    90.63    0.00    0.00    0.00    0.59
 11    5200      41209.65      5433.85     36570.23      0.00    94.27    84.61    80.32    90.88    0.00    0.00    0.00    0.59
 12    5400      41468.53      5153.74     35525.53      0.00    94.31    84.72    80.48    90.89    0.00    0.00    0.00    0.59
 12    5600      43174.94      5101.18     35071.64      0.00    94.29    84.89    80.50    90.58    0.00    0.00    0.00    0.59
 13    5800      41884.75      4444.11     33254.45      0.00    94.37    85.04    80.92    91.00    0.00    0.00    0.00    0.59
 14    6000      44225.76      4808.72     34154.68      0.00    94.29    84.78    80.61    90.43    0.00    0.00    0.00    0.59
 15    6200      43767.43      4504.56     32706.56      0.00    94.40    85.08    80.76    90.08    0.00    0.00    0.00    0.59
 15    6400      44956.06      4319.07     31624.79      0.00    94.38    84.92    80.56    90.18    0.00    0.00    0.00    0.59
 16    6600      44728.34      4095.51     31102.16      0.00    94.41    84.72    80.54    91.15    0.00    0.00    0.00    0.59
 17    6800      45849.50      4036.50     30963.34      0.00    94.37    84.95    80.67    90.31    0.00    0.00    0.00    0.59
 18    7000      47736.66      4137.85     30653.05      0.00    94.38    84.92    80.68    90.48    0.00    0.00    0.00    0.59
 18    7200      46329.14      3750.24     29078.77      0.00    94.28    84.84    80.41    90.93    0.00    0.00    0.00    0.59
 19    7400      46343.98      3709.54     28788.30      0.00    94.36    85.23    80.91    91.03    0.00    0.00    0.00    0.59
 20    7600      47199.97      3774.36     28129.44      0.00    94.39    84.58    80.34    89.82    0.00    0.00    0.00    0.59
 21    7800      48616.68      3741.70     28226.25      0.00    94.36    85.08    80.79    89.94    0.00    0.00    0.00    0.59
 21    8000      48540.14      3364.63     27156.42      0.00    94.39    84.86    80.70    90.51    0.00    0.00    0.00    0.59
 22    8200      48132.33      3379.65     26776.90      0.00    94.44    85.07    80.79    91.00    0.00    0.00    0.00    0.59
 23    8400      49758.17      3325.92     27111.11      0.00    94.44    85.05    80.89    90.18    0.00    0.00    0.00    0.59
 23    8600      50841.37      3274.89     26195.97      0.00    94.42    85.01    80.81    91.44    0.00    0.00    0.00    0.59
 24    8800      48796.49      3126.64     25412.67      0.00    94.37    85.21    80.91    90.48    0.00    0.00    0.00    0.59
 25    9000      48732.32      3178.71     24717.17      0.00    94.44    85.12    80.81    90.34    0.00    0.00    0.00    0.59
✔ Saved pipeline to output directory
output/model-last
```

## 6. Evaluate

    python -m spacy evaluate model data_path --output --code --gold-preproc --gpu-id --displacy-path --displacy-limit

## 7. Package it

be aware if a package is named 'nn-pipeline' you can load it with 'nn_pipeline'

    python -m spacy package ./output/model-best /package

and install

    pip install package/nn_pipeline-0.0.0/dist/nn_pipeline-0.0.0.tar.gz

## 8. Future
Idea publish this and make it public a nynorsk spacy module!
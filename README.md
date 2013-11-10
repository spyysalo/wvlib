wvlib - word vector library
===========================

This README is TODO. See scripts for documentation.

Try the following:

Find 10 words closest to "protein" using word2vec vectors induced on
the text8 demo data

    echo protein | python distance.py text8.tar.gz -n 10

Evaluate the vectors on the binary classification task using words
from McIntosh and Curran "Reducing semantic drift with bagging and
distributional similarity" (ACL 2009)

    python evalclass.py text8.tar.gz word-classes/McIC-09/*.txt

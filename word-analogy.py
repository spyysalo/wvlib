#!/usr/bin/env python

"""Given words w1, w2 and w3, find nearest neighbors to
vec(w2)-vec(w1)+vec(w2) in given word representation.

This is a python + wvlib version of word-analogy.c from word2vec
(https://code.google.com/p/word2vec/). The primary differences to
word-analogy.c are support for additional word vector formats,
increased configurability, and reduced speed.
"""

import sys
import os

import wvlib

from distance import process_arguments, query_loop

def words_to_vector_analogy(wv, words, options):
    vectors = []
    for w in words:
        if w in wv:
            if not options.quiet:
                print '\nWord: %s  Position in vocabulary: %d' % (w, wv.rank(w))
            vectors.append(wv[w])
        else:
            print >> sys.stderr, 'Out of dictionary word: "%s"' % w
    if len(vectors) != len(words):        
        return None # fail on any OOV
    elif len(vectors) < 2:
        print >> sys.stderr, 'Enter at least three words'
        return None
    else:
        if len(vectors) > 3:
            print >> sys.stderr, 'Warning: ignoring words after the third'
        return wvlib.unit_vector(vectors[1] - vectors[0] + vectors[2])

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        wv, options = process_arguments(argv, prompt='Enter three words')
    except Exception, e:
        print >> sys.stderr, 'Error: %s' % str(e)
        return 1
    return query_loop(wv, options, words_to_vector_analogy)

if __name__ == '__main__':
    sys.exit(main(sys.argv))

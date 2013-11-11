#!/usr/bin/env python

"""Given phrases p1, p2 and p3, find nearest neighbors to
vec(p2)-vec(p1)+vec(p3) in given word representation.

This is a python + wvlib extension of word-analogy.c from word2vec
(https://code.google.com/p/word2vec/). The primary differences to
word-analogy.c are support for additional word vector formats,
increased configurability, and reduced speed.
"""

import sys
import os

import wvlib

from distance import process_options, get_query, output_nearest

def process_query(wv, query, options=None):
    try:        
        vectors = [wv.words_to_vector(q) for q in query]
    except KeyError, e:
        print >> sys.stderr, 'Out of dictionary word: %s' % str(e)
        return False

    words = [w for q in query for w in q]
    if not options.quiet:
        for w in words:
            print '\nWord: %s  Position in vocabulary: %d' % (w, wv.rank(w))

    assert len(vectors) == 3, 'internal error'
    vector = wvlib.unit_vector(vectors[1] - vectors[0] + vectors[2])
    nncount = options.number if options else 10
    nearest = wv.nearest(vector, n=nncount, exclude=words)
    output_nearest(nearest, options)

    return True

def query_loop(wv, options):
    while True:
        try:
            query = get_query(options.prompt, options.multiword, 
                              options.exit_word, 3)
        except EOFError:
            return 0
        if not query:
            continue
        if options.echo:
            print query
        if len(query) < 3:
            print >> sys.stderr, 'Enter three words/phrases'
            continue
        if len(query) > 3:
            print >> sys.stderr, 'Ignoring words/phrases after the third'
            query = query[:3]
        process_query(wv, query, options)

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        wv, options = process_options(argv[1:])
    except Exception, e:
        print >> sys.stderr, 'Error: %s' % str(e)
        return 1
    return query_loop(wv, options)

if __name__ == '__main__':
    sys.exit(main(sys.argv))

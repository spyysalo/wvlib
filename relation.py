#!/usr/bin/env python

"""Given phrases p1 and p2, find nearest neighbors to both and rank
pairs of neighbors by similarity to vec(p2)-vec(p1) in given word
representation.

The basic idea is a straightforward combination of nearest neighbors
and analogy as in word2vec (https://code.google.com/p/word2vec/).
"""

import sys
import os

import numpy

import wvlib

from distance import process_options, get_query

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

    nncount = 100 # TODO: add CLI parameter
    nearest = [wv.nearest(v, n=nncount, exclude=words) for v in vectors]
    nearest = [[(n[0], n[1], wv[n[0]]) for n in l] for l in nearest]
    assert len(nearest) == 2, 'internal error'
    pairs = [(n1, n2, 
              numpy.dot(wvlib.unit_vector(vectors[1]-vectors[0]+n1[2]), n2[2]))
             for n1 in nearest[0] for n2 in nearest[1] if n1[0] != n2[0]]
    pairs.sort(lambda a, b: cmp(b[2], a[2]))

    nncount = options.number if options else 10
    for p in pairs[:nncount]:
        print '%s\t---\t%s\t%f' % (p[0][0], p[1][0], p[2])

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
        if len(query) < 2:
            print >> sys.stderr, 'Enter two words/phrases'
            continue
        if len(query) > 2:
            print >> sys.stderr, 'Ignoring words/phrases after the second'
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

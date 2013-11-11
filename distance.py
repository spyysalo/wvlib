#!/usr/bin/env python

"""Find nearest neighbors to input word(s) in given word representation.

This is a python + wvlib version of distance.c from word2vec
(https://code.google.com/p/word2vec/). The primary differences to
distance.c are support for additional word vector formats, increased
configurability, and reduced speed.
"""

import sys
import os

import wvlib

# word2vec distance.c output header
output_header = "\n                                              Word       Cosine distance\n------------------------------------------------------------------------"

def argparser():
    try:
        import argparse
    except ImportError:
        import compat.argparse as argparse

    ap=argparse.ArgumentParser()
    ap.add_argument('vectors', nargs=1, metavar='FILE', help='word vectors')
    ap.add_argument('-e', '--echo', default=False, action='store_true',
                    help='echo query word(s)')
    ap.add_argument('-n', '--number', metavar='INT', default=40, type=int,
                    help='number of nearest words to retrieve')
    ap.add_argument('-q', '--quiet', default=False, action='store_true',
                    help='minimal output')
    ap.add_argument('-x', '--no-exit', default=False, action='store_true',
                    help='don\'t exit on input "EXIT"')
    return ap

def process_arguments(argv, prompt='Enter words'):
    options = argparser().parse_args(argv[1:])
    if options.quiet:
        options.prompt = ''
    elif options.no_exit:
        options.prompt = prompt + ' (CTRL-D to break):\n'
    else:
        options.prompt = prompt + ' (EXIT or CTRL-D to break):\n'    
    return wvlib.load(options.vectors[0]).normalize(), options

def words_to_vector_average(wv, words, options):
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
    else:
        return wvlib.unit_vector(sum(vectors))

def output_nearest(nearest, options, out=sys.stdout):
    if not options.quiet:
        print >> out, output_header
        fmt = '%50s\t\t%f'
    else:
        fmt = '%s\t%f'
    for w, s in nearest:
        print >> out, fmt % (w, s)
    print >> out

def process_query(wv, query, options, words_to_vector=None):
    if words_to_vector is None:
        words_to_vector = words_to_vector_average
    words = query.split()
    v = words_to_vector(wv, words, options)
    if v is not None:
        nearest = wv.nearest(v, n=options.number, exclude=words)
        output_nearest(nearest, options)
    else:
        pass

def query_loop(wv, options, words_to_vector=None):
    while True:
        try:
            s = raw_input(options.prompt)
        except EOFError:
            return 0
        if s.strip() == 'EXIT' and not options.no_exit:
            return 0
        if options.echo:
            print s
        process_query(wv, s, options, words_to_vector)

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        wv, options = process_arguments(argv)
    except Exception, e:
        print >> sys.stderr, 'Error: %s' % str(e)
        return 1
    return query_loop(wv, options)

if __name__ == '__main__':
    sys.exit(main(sys.argv))

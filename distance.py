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
    ap.add_argument('-m', '--multiword', default=False, action='store_true',
                    help='multiword input')
    ap.add_argument('-n', '--number', metavar='INT', default=40, type=int,
                    help='number of nearest words to retrieve')
    ap.add_argument('-r', '--max-rank', metavar='INT', default=None, 
                    type=int, help='only consider r most frequent words')
    ap.add_argument('-q', '--quiet', default=False, action='store_true',
                    help='minimal output')
    ap.add_argument('-x', '--exit-word', default='EXIT',
                    help='exit on word (default "EXIT")')
    return ap

def process_options(args, prompt='Enter words'):    
    options = argparser().parse_args(args)
    if options.quiet:
        options.prompt = ''
    elif not options.exit_word:
        options.prompt = prompt + ' (CTRL-D to break):\n'
    else:
        options.prompt = prompt + ' (%s or CTRL-D to break):\n' % \
            options.exit_word
    if options.max_rank is not None and options.max_rank < 1:
        raise ValueError('max-rank must be >= 1')
    wv = wvlib.load(options.vectors[0], max_rank=options.max_rank)
    return wv.normalize(), options

def output_nearest(nearest, options, out=sys.stdout):
    if not options.quiet:
        print >> out, output_header
        fmt = '%50s\t\t%f'
    else:
        fmt = '%s\t%f'
    for w, s in nearest:
        print >> out, fmt % (w, s)
    print >> out

def process_query(wv, query, options=None):
    words = [w for q in query for w in q]
    try:
        vector = wv.words_to_vector(words)
    except KeyError, e:
        print >> sys.stderr, 'Out of dictionary word: %s' % str(e)
        return False

    if options and not options.quiet:
        for w in words:
            print '\nWord: %s  Position in vocabulary: %d' % (w, wv.rank(w))    

    nncount = options.number if options else 10
    nearest = wv.nearest(vector, n=nncount, exclude=words)
    output_nearest(nearest, options)

    return True

def get_line(prompt, exit_word=None):
    s = raw_input(prompt)
    if s.strip() == exit_word:
        raise EOFError('exit word in input')
    return s

def get_query(prompt='', multiword=False, exit_word=None, max_phrases=None):
    """Return query from user input.

    Input is returned as one or more lists of words (phrases), for
    example

        [["paris"], ["france"], ["tokyo"]]

    or

        [["new", "york"], ["united", "states"], ["kuala lumpur"]

    If multiword evaluates to False, prompt for one phrase of single
    words, otherwise prompts for up to max_phrases of one or more
    words. Return None on end of input or if exit_word is given as
    input.
    """

    if not multiword:
        query = [[w] for w in get_line(prompt, exit_word).split()]
    else:
        query = [get_line(prompt, exit_word).split()]
        while True:
            line = get_line('', exit_word)
            if not line or line.isspace():
                print "blank break"
                break
            query.append(line.split())
            if max_phrases and len(query) >= max_phrases:
                print "len break"
                break
    return query

def empty_query(query):
    return not query or not any(p for p in query)

def query_loop(wv, options):
    while True:
        try:
            query = get_query(options.prompt, options.multiword, 
                              options.exit_word)
        except EOFError:
            return 0
        if empty_query(query):
            continue
        if options.echo:
            print query
        process_query(wv, query, options)

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        wv, options = process_options(argv[1:])
    except Exception, e:
        if str(e):
            print >> sys.stderr, 'Error: %s' % str(e)
        else:
            raise
    return query_loop(wv, options)

if __name__ == '__main__':
    sys.exit(main(sys.argv))

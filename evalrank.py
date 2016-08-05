#!/usr/bin/env python

"""Evaluate word representation by similarity ranking to reference."""

import sys
import codecs
import logging

import numpy
import wvlib

from os.path import basename, splitext
from collections import OrderedDict
from scipy.stats import spearmanr
from logging import info

DEFAULT_ENCODING = 'UTF-8'

class FormatError(Exception):
    pass

def argparser():
    try:
        import argparse
    except ImportError:
        import compat.argparse as argparse

    ap=argparse.ArgumentParser()
    ap.add_argument('-l', '--lowercase', default=None, action='store_true',
                    help='lowercase words in reference')
    ap.add_argument('-r', '--max-rank', metavar='INT', default=None, 
                    type=int, help='only consider r most frequent words')
    ap.add_argument('-q', '--quiet', default=False, action='store_true')
    ap.add_argument('vectors', help='word vectors')
    ap.add_argument('references', metavar='FILE', nargs='+',
                    help='reference similarities')
    return ap

def cosine(v1, v2):
    return numpy.dot(v1/numpy.linalg.norm(v1), v2/numpy.linalg.norm(v2))

def dot(v1, v2):
    return numpy.dot(v1, v2)

def evaluate(wv, reference):
    """Evaluate wv against reference, return (rho, count) where rwo is
    Spearman's rho and count is the number of reference word pairs
    that could be evaluated against.
    """
    gold, predicted, oov = [], [], OrderedDict()
    for words, sim in sorted(reference, key=lambda ws: ws[1]):
        w1, w2, v1, v2 = words[0], words[1], None, None
        try:
            v1 = wv[w1]
        except KeyError:
            oov[w1] = True
        try:
            v2 = wv[w2]
        except KeyError:
            oov[w2] = True
        if v1 is None or v2 is None:
            continue
        gold.append((words, sim))
        predicted.append((words, cosine(v1, v2)))
    if oov:
        info('OOV: ' + ', '.join(oov.keys()))
    simlist = lambda ws: [s for w,s in ws]
    rho, p = spearmanr(simlist(gold), simlist(predicted))
    return (rho, len(gold))
    
def read_reference(name, options=None, encoding=DEFAULT_ENCODING):
    """Return similarity ranking data as list of ((w1, w2), sim) tuples."""
    data = []
    with codecs.open(name, 'rU', encoding=encoding) as f:
        for line in f:
            # try tab-separated first, fall back to any space
            fields = line.strip().split('\t')
            if len(fields) != 3:
                fields = line.strip().split()
            if len(fields) != 3:
                raise FormatError(line)
            if options and options.lowercase:
                fields[0], fields[1] = fields[0].lower(), fields[1].lower()
            try:
                data.append(((fields[0], fields[1]), float(fields[2])))
            except ValueError:
                raise FormatError(line)
    return data

def baseroot(name):
    return splitext(basename(name))[0]

def main(argv=None):
    if argv is None:
        argv = sys.argv

    options = argparser().parse_args(argv[1:])

    if options.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    try:
        wv = wvlib.load(options.vectors, max_rank=options.max_rank)
        wv = wv.normalize()
    except Exception, e:
        print >> sys.stderr, 'Error: %s' % str(e)
        return 1    
    references = [(r, read_reference(r, options)) for r in options.references]

    print '%20s\trho\tmissed\ttotal\tratio' % 'dataset'
    for name, ref in references:
        rho, count = evaluate(wv, ref)
        total, miss = len(ref), len(ref) - count
        print '%20s\t%.4f\t%d\t%d\t(%.2f%%)' % \
            (baseroot(name), rho, miss, total, 100.*miss/total)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))

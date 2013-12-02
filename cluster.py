#!/usr/bin/env python

import sys
import math
import logging

import scipy.cluster
import wvlib

from itertools import izip

KMEANS = 'kmeans'
DEFAULT_METHOD = KMEANS
methods = [
    KMEANS,
]

def argparser():
    try:
        import argparse
    except ImportError:
        import compat.argparse as argparse

    ap=argparse.ArgumentParser()
    ap.add_argument('vectors', nargs=1, metavar='FILE', help='word vectors')
    ap.add_argument('-k', default=None, type=int,
                    help='number of clusters (default sqrt(words/2))')
    ap.add_argument('-m', '--method', default=DEFAULT_METHOD, choices=methods,
                    help='clustering method to apply')
    ap.add_argument('-n', '--normalize', default=False, action='store_true',
                    help='normalize vectors to unit length')
    ap.add_argument('-r', '--max-rank', metavar='INT', default=None, 
                    type=int, help='only consider r most frequent words')
    ap.add_argument('-w', '--whiten', default=False, action='store_true',
                    help='normalize features to unit variance ')
    return ap

def process_options(args):    
    options = argparser().parse_args(args)

    if options.max_rank is not None and options.max_rank < 1:
        raise ValueError('max-rank must be >= 1')
    if options.k is not None and options.k < 2:
        raise ValueError('cluster number must be >= 2')

    wv = wvlib.load(options.vectors[0], max_rank=options.max_rank)

    if options.k is None:
        options.k = int(math.ceil((len(wv.words())/2)**0.5))
        logging.info('set k=%d (%d words)' % (options.k, len(wv.words())))

    if options.normalize:
        logging.info('normalize vectors to unit length')
        wv.normalize()

    words, vectors = wv.words(), wv.vectors()

    if options.whiten:
        logging.info('normalize features to unit variance')
        vectors = scipy.cluster.vq.whiten(vectors)

    return words, vectors, options

def kmeans(vectors, k):
    codebook, distortion = scipy.cluster.vq.kmeans(vectors, k)
    cluster_ids, dist = scipy.cluster.vq.vq(vectors, codebook)
    return cluster_ids

def write_strict_clusters(words, cluster_ids, out=None):
    """Write given list of words and their corresponding cluster ids to out."""

    assert len(words) == len(cluster_ids), 'word/cluster ids number mismatch'

    if out is None:
        out = sys.stdout
    for word, cid in izip(words, cluster_ids):
        print >> out, '%s\t%d' % (word, cid)

def cluster(words, vectors, options):
    if options.method == KMEANS:
        cluster_ids = kmeans(vectors, options.k)
        write_strict_clusters(words, cluster_ids)
    else:
        raise NotImplementedError

def main(argv=None):
    if argv is None:
        argv = sys.argv

    try:
        words, vectors, options = process_options(argv[1:])
    except Exception, e:
        if str(e):
            print >> sys.stderr, 'Error: %s' % str(e)
            return 1
        else:
            raise
    try:
        cluster(words, vectors, options)
    except Exception, e:
        #print >> sys.stderr, 'Error: %s' % (str(e))
        raise

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))

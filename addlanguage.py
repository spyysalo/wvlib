#!/usr/bin/env python

"""Add language labels to output of nearest.py."""

from __future__ import print_function

import sys
import re
import codecs
import logging

from os import path
from logging import warn, info


logging.getLogger().setLevel(logging.INFO)


# Regex for lines to add language codes to
WORD_SIM_RE = re.compile(r'^\s*(\S+)\s+([0-9.]+)\s*$')


class FormatError(Exception):
    pass


def argparser():
    try:
        import argparse
    except ImportError:
        import compat.argparse as argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('-f', '--file', default=None,
                    help='file to add languages to (default STDIN)')
    ap.add_argument('-r', '--max-rank', metavar='INT', default=None, type=int,
                    help='only read r most frequent words')
    ap.add_argument('-v', '--verbose', default=False, action='store_true')
    ap.add_argument('vocabs', nargs='+', metavar='FILE',
                    help='vocabulary files')
    return ap


def language_label(fn):
    """Guess language label from vocabulary filename."""
    bn = path.basename(fn)
    n = path.splitext(bn)[0]
    n = n.replace('_vocab', '')
    return n


def load_vocab(fn, options):
    freq_by_word = {}
    with codecs.open(fn, encoding='utf-8') as f:
        for i, l in enumerate(f, start=1):
            if options.max_rank and i > options.max_rank:
                break
            l = l.rstrip()
            fields = l.split(None, 1)
            if len(fields) != 2:
                raise FormatError('line {} in {}: {}'.format(i, fn, l))
            try:
                freq = int(fields[0])
            except ValueError:
                raise FormatError('line {} in {}: {}'.format(i, fn, l))
            word = fields[1]
            if word in freq_by_word:
                warn('duplicate word in {}: {}'.format(fn, word))
            else:
                freq_by_word[word] = freq
    return freq_by_word


def load_vocabs(files, options):
    vocab_by_label = {}
    for fn in files:
        label = language_label(fn)
        if label in vocab_by_label:
            raise ValueError('duplicate language {}'.format(label))
        vocab = load_vocab(fn, options)
        info('read {} words for {} from {}'.format(len(vocab), label, fn))
        vocab_by_label[label] = vocab
    # Group into dict of dicts, outer keyed by word, inner by language
    # label, values are frequencies in language.
    combined = {}
    for label, vocab in vocab_by_label.items():
        for word, freq in vocab.items():
            if word not in combined:
                combined[word] = {}
            combined[word][label] = freq
    return combined


def format_labels(word, vocabs, options):
    if word not in vocabs:
        return '<NONE>'
    freq_by_label = vocabs[word]
    freq_and_label = [(f, l) for l, f in freq_by_label.items()]
    freq_and_label = list(reversed(sorted(freq_and_label)))
    if not options.verbose:
        return freq_and_label[0][1]    # Most frequent label only
    else:
        return '\t'.join('{} ({})'.format(l, f) for f, l in freq_and_label)


def add_languages(flo, vocabs, options, out=None):
    if out is None:
        out = codecs.getwriter('utf-8')(sys.stdout)
    for l in flo:
        l = l.rstrip()
        m = WORD_SIM_RE.match(l)
        if m:
            word, sim = m.groups()
            labels = format_labels(word, vocabs, options)
            l += '\t{}'.format(labels)
        print(l, file=out)


def main(argv):
    options = argparser().parse_args(argv[1:])
    vocabs = load_vocabs(options.vocabs, options)
    if options.file is None:
        with codecs.getreader('utf-8')(sys.stdin) as f:
            add_languages(f, vocabs, options)
    else:
        with codecs.open(options.file, encoding='utf-8') as f:
            add_languages(f, vocabs, options)


if __name__ == '__main__':
    sys.exit(main(sys.argv))

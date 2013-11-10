#!/usr/bin/env python

"""Word vector library.

Provides functionality for reading and writing word vectors in various
formats and support for basic operations using word vectors such as
cosine similarity.

The word2vec binary and text formats are supported for input, and tar,
tar.gz and directory-based variants of the wvlib format are supported
for input and output.

Variables:

formats -- list of formats recognized by load().

Functions:

load() -- load word vectors from a file in a supported input format.

Classes:

WVData -- represents word vectors and related data.
Word2VecData -- WVData for word2vec vectors.

Examples:

Load word2vec vectors from "vectors.bin", save in wvlib format in
"vectors.tar.gz":

>>> import wvlib
>>> wv = wvlib.load("vectors.bin")
>>> wv.save("vectors.tar.gz")

Load word vectors from name, compare similarity of two words:

>>> import wvlib
>>> wv = wvlib.load(name)
>>> wv.word_similarity("dog", "cat")

Load word vectors, normalize, and find words nearest to given word:

>>> import wvlib
>>> wv = wvlib.load(name).normalize()
>>> wv.nearest("dog")

normalize() irreversibly alters the word vectors, but considerably
speeds up calculations using word vector similarity.
"""

import sys
import os

import json
import codecs
import tarfile
import logging

import numpy
import heapq

from functools import partial
from itertools import izip
from collections import OrderedDict
from StringIO import StringIO
from time import time

logging.getLogger().setLevel(logging.DEBUG)

DEFAULT_ENCODING = "UTF-8"

WV_FORMAT_VERSION = 1

CONFIG_NAME = 'config.json'
VOCAB_NAME = 'vocab.tsv'
VECTOR_BASE = 'vectors'

# supported formats and likely filename extensions for each
WORD2VEC_FORMAT = 'w2v'
WORD2VEC_TEXT = 'w2vtxt'
WORD2VEC_BIN = 'w2vbin'
WVLIB_FORMAT = 'wvlib'

extension_format_map = {
    '.txt' : WORD2VEC_TEXT,
    '.bin' : WORD2VEC_BIN,
    '.tar' : WVLIB_FORMAT,
    '.tgz' : WVLIB_FORMAT,
    '.tar.gz' : WVLIB_FORMAT,
}

formats = sorted(list(set(extension_format_map.values())))

class FormatError(Exception):
    pass

class WVData(object):
    TAR, DIR = range(2)

    def __init__(self, config, vocab, vectors):
        # TODO: check config-vocab-vectors consistency
        self.config = config
        self.vocab = vocab
        self.vectors = vectors
        self._w2v_map = None
        self._normalized = False

    def words(self):
        """Return list of words in the vocabulary."""

        return self.vocab.words()

    def word_to_vector(self, w):
        """Return vector for given word.

        For large numbers of queries, consider word_to_vector_mapping().
        """
        
        return self.word_to_vector_mapping()[w]

    def word_to_vector_mapping(self):
        """Return dict mapping from words to vectors.

        The returned data is shared and should not be modified by the
        caller.
        """

        if self._w2v_map is None:
            self._w2v_map = dict(iter(self))
        return self._w2v_map

    def word_similarity(self, w1, w2):
        """Return cosine similarity of vectors for given words.

        In cases involving many invocations of this function, consider
        calling normalize() first to avoid repeatedly normalizing the
        same vectors.
        """

        w2v = self.word_to_vector_mapping()
        v1, v2 = w2v[w1], w2v[w2]
        if not self._normalized:
            v1, v2 = v1/numpy.linalg.norm(v1), v2/numpy.linalg.norm(v2)
        return numpy.dot(v1, v2)

    def nearest(self, w, n=10):
        w2v = self.word_to_vector_mapping()
        v = w2v[w]/numpy.linalg.norm(w2v[w])
        if not self._normalized:
            sim = partial(self._item_similarity, v=v)
        else:
            sim = partial(self._item_similarity_normalized, v=v)
        # +1 for the input itself
        nearest = heapq.nlargest(n+1, w2v.iteritems(), sim)
        return [(n[0], sim(n)) for n in nearest if n[0] != w]

    def normalize(self):
        """Normalize word vectors.

        Irreversible. Has potentially high invocation cost, but should
        reduce overall time when there are many invocations of
        word_similarity()."""

        if self._normalized:
            return
        self._invalidate()
        self.vectors.normalize()
        self._normalized = True
        return self

    def save(self, name, format=None):
        """Save in format to pathname name.

        If format is None, determine format heuristically.
        """

        if format is None:
            format = self.guess_format(name)
        if format == self.TAR:
            return self.save_tar(name)
        elif format == self.DIR:
            return self.save_dir(name)
        else:
            raise NotImplementedError

    def save_tar(self, name, mode=None):
        """Save in tar format to pathname name using mode.

        If mode is None, determine mode from filename extension.
        """

        if mode is None:
            if name.endswith('.tgz') or name.endswith('.gz'):
                mode = 'w:gz'
            elif name.endswith('.bz2'):
                mode = 'w:bz2'
            else:
                mode = 'w'

        f = tarfile.open(name, mode)
        try:
            vecformat = self.config.format
            vecfile_name = VECTOR_BASE + '.' + vecformat
            self._save_in_tar(f, CONFIG_NAME, self.config.savef)
            self._save_in_tar(f, VOCAB_NAME, self.vocab.savef)            
            self._save_in_tar(f, vecfile_name, partial(self.vectors.savef,
                                                       format = vecformat))
        finally:
            f.close()

    def save_dir(self, name):
        """Save to directory name."""

        vecformat = self.config.format
        vecfile_name = VECTOR_BASE + '.' + vecformat
        self.config.save(os.path.join(name, CONFIG_NAME))
        self.vocab.save(os.path.join(name, VOCAB_NAME))
        self.vectors.save(os.path.join(name, vecfile_name, vecformat))

    def _invalidate(self):
        """Invalidate cached values."""

        self._w2v_map = None

    def __getitem__(self, key):
        """Return vector for given word."""

        return self.word_to_vector_mapping()[key]

    def __getattr__(self, name):
        # delegate access to nonexistent attributes to config
        return getattr(self.config, name)

    def __iter__(self):
        """Iterate over (word, vector) pairs."""

        #return izip(self.vocab.words(), self.vectors)
        return izip(self.vocab.iterwords(), iter(self.vectors))

    @classmethod
    def load(cls, name):
        """Return WVData from pathname name."""

        format = cls.guess_format(name)
        if format == cls.TAR:
            return cls.load_tar(name)
        elif format == cls.DIR:
            return cls.load_dir(name)
        else:
            raise NotImplementedError

    @classmethod
    def load_tar(cls, name):
        """Return WVData from tar or tar.gz name."""

        f = tarfile.open(name, 'r')
        try:
            return cls._load_collection(f)
        finally:            
            f.close()

    @classmethod
    def load_dir(cls, name):
        """Return WVData from directory name."""

        d = _Directory.open(name, 'r')
        try:
            return cls._load_collection(d)
        finally:
            d.close()

    @classmethod
    def _load_collection(cls, coll):
        # abstracts over tar and directory
        confname, vocabname, vecname = None, None, None
        for i in coll:
            if i.isdir():
                continue
            elif not i.isfile():
                logging.warning('unexpected item: %s' % i.name)
                continue
            base = os.path.basename(i.name)
            if base == CONFIG_NAME:
                confname = i.name
            elif base == VOCAB_NAME:
                vocabname = i.name
            elif os.path.splitext(base)[0] == VECTOR_BASE:
                vecname = i.name
            else:
                logging.warning('unexpected file: %s' % i.name)
        for i, n in ((confname, CONFIG_NAME),
                     (vocabname, VOCAB_NAME),
                     (vecname, VECTOR_BASE)):
            if i is None:
                raise FormatError('missing %s' % n)
        config = Config.loadf(coll.extractfile(confname))
        vocab = Vocabulary.loadf(coll.extractfile(vocabname))
        vectors = Vectors.loadf(coll.extractfile(vecname), config.format)
        return cls(config, vocab, vectors)

    @staticmethod
    def _save_in_tar(tar, name, savef):
        # helper for save_tar
        i = tar.tarinfo(name)
        s = StringIO()
        savef(s)
        s.seek(0)
        i.size = s.len
        i.mtime = time()
        tar.addfile(i, s)

    @staticmethod
    def _item_similarity(i, v):
        """Similarity of (word, vector) pair with normalized vector."""

        return numpy.dot(i[1]/numpy.linalg.norm(i[1]), v)

    @staticmethod
    def _item_similarity_normalized(i, v):
        """Similarity of normalized (word vector) pair with vector."""

        return numpy.dot(i[1], v)

    @staticmethod
    def guess_format(name):
        if os.path.isdir(name):
            return WVData.DIR
        else:
            return WVData.TAR

class _FileInfo(object):
    """Implements minimal part of TarInfo interface for files."""

    def __init__(self, name):
        self.name = name

    def isdir(self):
        return os.path.isdir(self.name)

    def isfile(self):
        return os.path.isfile(self.name)

class _Directory(object):
    """Implements minimal part part of TarFile interface for directories."""

    def __init__(self, name):
        self.name = name
        self.listing = None
        self.open_files = []
        
    def __iter__(self):
        if self.listing is None:
            self.listing = os.listdir(self.name)
        return (_FileInfo(os.path.join(self.name, i)) for i in self.listing)

    def extractfile(self, name):
        f = open(name)
        self.open_files.append(f)
        return f

    def close(self):
        for f in self.open_files:
            f.close()
        
    @classmethod
    def open(cls, name, mode=None):
        return cls(name)

class Vectors(object):

    default_format = 'npy'
    _text_format = set(['tsv'])
    
    def __init__(self, vectors):
        self.vectors = vectors
        self._normalized = False

    def normalize(self):
        if self._normalized:
            return
        for i, v in enumerate(self.vectors):
            self.vectors[i] = v/numpy.linalg.norm(v)
        self._normalized = True
        return self

    def to_rows(self):
        return self.vectors

    def save_tsv(self, f):
        """Save as TSV to file-like object f."""

        f.write(self.__str__())

    def savef(self, f, format):
        """Save in format to file-like object f."""

        if format == 'tsv':
            return self.save_tsv(f)
        elif format == 'npy':
            return numpy.save(f, self.vectors)
        else:
            raise NotImplementedError(format)

    def save(self, name, format=None, encoding=DEFAULT_ENCODING):
        """Save in format to pathname name.

        If format is None, determine format from filename extension.
        If format is None and the extension is empty, use default
        format and append the appropriate extension.
        """

        if format is None:
            format = os.path.splitext(name)[1].replace('.', '')
        if not format:
            format = self.default_format
            name = name + '.' + format

        logging.debug('save vectors in %s to %s' % (format, name))

        if format in Vectors._text_format:
            with codecs.open(name, 'wt', encoding=encoding) as f:
                return self.savef(f, format)
        else:
            with codecs.open(name, 'wb') as f:
                return self.savef(f, format)
        
    def to_tsv(self):
        return self.__str__()

    def __str__(self):
        return '\n'.join('\t'.join(str(i) for i in r) for r in self.to_rows())

    def __iter__(self):
        return iter(self.vectors)

    @classmethod
    def from_rows(cls, rows):
        return cls(rows)

    @classmethod
    def load_tsv(cls, f):
        """Return Vectors from file-like object f in TSV."""

        rows = []
        for l in f:
            #row = [float(i) for i in l.rstrip('\n').split()]
            row = numpy.fromstring(l, sep='\t')
            rows.append(row)
        return cls.from_rows(rows)

    @classmethod
    def load_numpy(cls, f):
        """Return Vectors from file-like object f in NumPy format."""

        return cls(numpy.load(f))

    @classmethod
    def loadf(cls, f, format):
        """Return Vectors from file-like object f in format."""

        if format == 'tsv':
            return cls.load_tsv(f)
        elif format == 'npy':
            return cls.load_numpy(f)
        else:
            raise NotImplementedError(format)

    @classmethod
    def load(cls, name, format=None, encoding=DEFAULT_ENCODING):
        """Return Vectors from pathname name in format.

        If format is None, determine format from filename extension.
        """

        if format is None:
            format = os.path.splitext(name)[1].replace('.', '')

        if format in Vectors._text_format:
            with codecs.open(name, 'rt', encoding=encoding) as f:
                return cls.loadf(f, format)
        else:
            with codecs.open(name, 'rb') as f:
                return cls.loadf(f, format)
    
class Vocabulary(object):
    def __init__(self, word_freq):
        self.word_freq = OrderedDict(word_freq)

    def to_rows(self):
        return self.word_freq.iteritems()

    def savef(self, f):
        """Save as TSV to file-like object f."""

        f.write(self.__str__())

    def save(self, name, encoding=DEFAULT_ENCODING):
        """Save as TSV to pathname name."""

        with codecs.open(name, 'wt', encoding=encoding) as f:
                return self.savef(f)

    def words(self):
        return self.word_freq.keys()

    def iterwords(self):
        return self.word_freq.iterkeys()

    def __str__(self):
        return '\n'.join('\t'.join(str(i) for i in r) for r in self.to_rows())

    def __getitem__(self, key):
        return self.word_freq[key]

    def __setitem__(self, key, value):
        self.word_freq[key] = value

    def __delitem__(self, key):
        del self.word_freq[key]

    def __iter__(self):
        return iter(self.word_freq)

    def __contains__(self, item):
        return item in self.word_freq

    @classmethod
    def from_rows(cls, rows):
        return cls(rows)

    @classmethod
    def loadf(cls, f):
        """Return Vocabulary from file-like object f in TSV format."""

        rows = []
        for l in f:
            l = l.rstrip()
            row = l.split('\t')
            if len(row) != 2:
                raise ValueError('expected 2 fields, got %d: %s' % \
                                     (len(row), l))
            try:
                row[1] = int(row[1])
            except ValueError, e:
                raise TypeError('expected int, got %s' % row[1])
            rows.append(row)
        return cls.from_rows(rows)

    @classmethod
    def load(cls, name, encoding=DEFAULT_ENCODING):
        """Return Vocabulary from pathname name in TSV format."""

        with codecs.open(name, 'rU', encoding=encoding) as f:
            return cls.loadf(f)

class ConfigError(Exception):
    pass

class Config(object):
    def __init__(self, version, word_count, vector_dim, format):
        self.version = version
        self.word_count = word_count
        self.vector_dim = vector_dim
        self.format = format

    def to_dict(self):
        return { 'version' : self.version,
                 'word_count' : self.word_count,
                 'vector_dim' : self.vector_dim,
                 'format' : self.format,
               }

    def savef(self, f):
        """Save as JSON to file-like object f."""

        return json.dump(self.to_dict(), f, sort_keys=True,
                         indent=4, separators=(',', ': '))

    def save(self, name, encoding=DEFAULT_ENCODING):
        """Save as JSON to pathname name."""

        with codecs.open(name, 'wt', encoding=encoding) as f:
            return self.savef(f)

    def __str__(self):
        return str(self.to_dict())

    @classmethod
    def from_dict(cls, d):        
        try:
            return cls(int(d['version']),
                       int(d['word_count']), 
                       int(d['vector_dim']),
                       d['format'])
        except KeyError, e:
            raise ConfigError('missing %s' % str(e))

    @classmethod
    def loadf(cls, f):
        """Return Config from file-like object f in JSON format."""

        try:
            config = json.load(f)
        except ValueError, e:
            raise ConfigError(e)
        return cls.from_dict(config)

    @classmethod
    def load(cls, name, encoding=DEFAULT_ENCODING):
        """Return Config from pathname name in JSON format."""

        with codecs.open(name, 'rU', encoding=encoding) as f:
            return cls.loadf(f)

    @classmethod
    def default(cls, word_count, vector_dim):
        """Return default Config for given settings."""
        
        return cls(WV_FORMAT_VERSION, word_count, vector_dim, 
                   Vectors.default_format)

class Word2VecData(WVData):

    def __init__(self, data):
        config = Config.default(len(data), len(data[0][1]))
        logging.warning('word2vec load: filling in 0s for word counts')
        vocab = Vocabulary([(row[0], 0) for row in data])
        vectors = Vectors([row[1] for row in data])
        super(Word2VecData, self).__init__(config, vocab, vectors)
        self.w2vdata = data

    @classmethod
    def load_textf(cls, f):
        """Return Word2VecData from file-like object f in the word2vec
        text format."""

        return cls(cls.read(f, cls.read_text_line))

    @classmethod
    def load_binaryf(cls, f):
        """Return Word2VecData from file-like object f in the word2vec
        binary format."""

        return cls(cls.read(f, cls.read_binary_line))

    @classmethod
    def load_binary(cls, name):
        """Return Word2VecData from pathname name in the word2vec
        binary format."""

        with open(name, 'rb') as f:
            return cls.load_binaryf(f)

    @classmethod
    def load_text(cls, name, encoding=DEFAULT_ENCODING):
        """Return Word2VecData from pathname name in the word2vec text
        format."""

        with codecs.open(name, 'rU', encoding=encoding) as f:
            return cls.load_textf(f)
    
    @classmethod
    def load(cls, name, binary=None, encoding=DEFAULT_ENCODING):
        """Return Word2VecData from pathname name in the word2vec
        binary or text format.

        If binary is None, determine format heuristically.
        """

        if binary is None:
            binary = not cls.is_w2v_text(name, encoding)
        if binary:
            return cls.load_binary(name)
        else:
            return cls.load_text(name, encoding)

    @staticmethod
    def read_size_line(f):
        """Read line from file-like object f as word2vec format
        header, return (word count, vector size)."""
        
        try:
            wcount, vsize = f.readline().rstrip('\n').split()
            return int(wcount), int(vsize)
        except ValueError:
            raise FormatError('expected two ints, got "%s"' % l)

    @staticmethod
    def read_text_line(f, vsize):
        """Read line from file-like object f as word2vec text format
        and word vector, return (word, vector)."""

        l = f.readline().rstrip(' \n')
        fields = l.split(' ')
        try:
            return fields[0], numpy.array([float(f) for f in fields[1:]])
        except ValueError:
            raise FormatError('expected word and floats, got "%s"' % l)
        if len(vec) != vsize:
            raise FormatError('expected vector of size %d, got %d for "%s"' %
                              (vsize, len(v), word))

    @staticmethod
    def read_word(f):
        """Read and return word from file-like object f."""

        # too terse (http://docs.python.org/2/library/functions.html#iter)
#         return = ''.join(iter(lambda: f.read(1), ' '))

        wchars = []
        while True:
            c = f.read(1)
            if c == ' ':
                break
            if not c:
                raise FormatError("preliminary end of file")
            wchars.append(c)
        return ''.join(wchars)

    @staticmethod
    def read_binary_line(f, vsize):
        """Read line from file-like object f as word2vec binary format
        word and vector, return (word, vector)."""
        
        word = Word2VecData.read_word(f)
        vector = numpy.fromfile(f, numpy.float32, vsize)
        # discard terminal newline
        f.read(1)
        return word, vector

    @staticmethod
    def read(f, read_line=read_binary_line):
        """Read word2vec data from file-like object f using function
        read_line to parse individual lines. 

        Return list of (word, vector) pairs.
        """
        
        wcount, vsize = Word2VecData.read_size_line(f)
        rows = []
        for i in range(wcount):
            rows.append(read_line(f, vsize))
        if len(rows) != wcount:
            raise FormatError('expected %d words, got %d' % (wcount, len(rows)))
        return rows

    @staticmethod
    def is_w2v_textf(f):
        """Return True if file-like object f is in the word2vec text
        format, False otherwise."""

        try:
            wcount, vsize = Word2VecData.read_size_line(f)
            w, v = Word2VecData.read_text_line(f, vsize)
            return True
        except FormatError:
            return False            

    @staticmethod
    def is_w2v_text(name, encoding=DEFAULT_ENCODING):
        """Return True if pathname name is in the word2vec text
        format, False otherwise."""

        with codecs.open(name, 'rU', encoding=encoding) as f:
            return Word2VecData.is_w2v_textf(f)

def _guess_format(name):
    for ext, format in extension_format_map.items():
        if name.endswith(ext):
            return format
    if os.path.isdir(name):
        return WVLIB_FORMAT
    return None

def load(name, format=None):
    """Load word vectors from pathname name in format.

    If format is None, determine format heuristically.
    """

    if format is None:
        format = _guess_format(name)
    if format is None:
        # TODO more appropriate exception
        raise FormatError('failed to guess format')

    logging.info('reading %s as %s' % (name, format))

    if format == WVLIB_FORMAT:
        return WVData.load(name)
    if format == WORD2VEC_FORMAT: # binary vs. text unspecified
        return Word2VecData.load(name)
    elif format == WORD2VEC_TEXT:
        return Word2VecData.load_text(name)
    elif format == WORD2VEC_BIN:
        return Word2VecData.load_binary(name)
    else:
        raise NotImplementedError        

def argparser():
    import argparse

    ap=argparse.ArgumentParser()
    ap.add_argument('vectors', metavar='FILE', help='vectors to load')
    ap.add_argument('-f', '--format', default=None, choices=formats)
    ap.add_argument('-o', '--output', metavar='FILE/DIR', default=None,
                    help='save vectors to file or directory')
    ap.add_argument('-n', '--nearest', metavar='WORD', default=None,
                    help='output words nearest to given word')
    return ap

def main(argv=None):
    if argv is None:
        argv = sys.argv

    options = argparser().parse_args(argv[1:])

    wv = load(options.vectors, options.format)

    if options.nearest:
        print '\n'.join(str(n) for n in wv.nearest(options.nearest))

    if options.output:
        wv.save(options.output)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))


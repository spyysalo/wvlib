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

vector_formats -- list of vector formats recognized by WVData.save()

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

Load word vectors and compare similarity of two words:

>>> import wvlib
>>> wv = wvlib.load("text8.tar.gz")
>>> wv.similarity("dog", "cat")

Load word vectors, normalize, and find words nearest to given word:

>>> import wvlib
>>> wv = wvlib.load("text8.tar.gz").normalize()
>>> wv.nearest("dog")

(normalize() irreversibly alters the word vectors, but considerably
speeds up calculations using word vector similarity.)

Load word vectors, normalize, and find word that has the same
relationship to "japan" as "paris" has to "france" (see https://code.google.com/p/word2vec/#Interesting_properties_of_the_word_vectors).

>>> import wvlib
>>> wv = wvlib.load("text8.tar.gz").normalize()
>>> v = wv["paris"] - wv["france"] + wv["japan"]
>>> wv.nearest(v)[0]

Load word vectors and save with vectors in TSV format:

>>> import wvlib
>>> wv = wvlib.load("text8.tar.gz")
>>> wv.save("text8-tsv.tar.gz", vector_format="tsv")
"""

import sys
import os

import math
import json
import codecs
import tarfile
import logging

import numpy
import heapq

from functools import partial
from itertools import tee, izip, islice
from StringIO import StringIO
from types import StringTypes
from time import time
from collections import defaultdict

try:
    from collections import OrderedDict
except ImportError:
    from compat.ordereddict import OrderedDict

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
    '.tar.bz2' : WVLIB_FORMAT,
}

# supported vector formats and filename extensions
NUMPY_FORMAT = 'npy'
TSV_FORMAT = 'tsv'

formats = sorted(list(set(extension_format_map.values())))

vector_formats = sorted([NUMPY_FORMAT, TSV_FORMAT])

class FormatError(Exception):
    pass

class WVData(object):
    TAR = 'tar'
    DIR = 'dir'

    def __init__(self, config, vocab, vectors):
        # TODO: check config-vocab-vectors consistency
        self.config = config
        self.vocab = vocab
        self.vectors = vectors
        self._w2v_map = None
        self._normalized = False
        self._lsh = None

    def words(self):
        """Return list of words in the vocabulary."""

        return self.vocab.words()

    def rank(self, w):
        """Return rank (ordinal, 0-based) of word w in the vocabulary."""

        return self.vocab.rank(w)

    def word_to_vector(self, w):
        """Return vector for given word.

        For large numbers of queries, consider word_to_vector_mapping().
        """
        
        return self.word_to_vector_mapping()[w]

    def words_to_vector(self, words):
        """Return average vector for given words."""

        w2v = self.word_to_vector_mapping()
        return sum(w2v[w] for w in words)/len(words)

    def word_to_unit_vector(self, w):
        """Return unit (normalized) vector for given word.

        For large numbers of queries, consider normalize() and
        word_to_vector_mapping().
        """

        v = self.word_to_vector_mapping()[w]
        if self._normalized:
            return v
        else:
            return v/numpy.linalg.norm(v)

    def word_to_vector_mapping(self):
        """Return dict mapping from words to vectors.

        The returned data is shared and should not be modified by the
        caller.
        """

        if self._w2v_map is None:
            self._w2v_map = dict(iter(self))
        return self._w2v_map

    def similarity(self, v1, v2):
        """Return cosine similarity of given words or vectors.

        If v1/v2 is a string, look up the corresponding word vector.
        This is not particularly efficient function. Instead of many
        invocations, consider word_similarity() or direct computation.
        """
        
        vs = [v1, v2]
        for i, v in enumerate(vs):
            if isinstance(v, StringTypes):
                v = self.word_to_unit_vector(v)
            else:
                v = v/numpy.linalg.norm(v) # costly but safe
            vs[i] = v
        return numpy.dot(vs[0], vs[1])

    def word_similarity(self, w1, w2):
        """Return cosine similarity of vectors for given words.

        For many invocations of this function, consider calling
        normalize() first to avoid repeatedly normalizing the same
        vectors.
        """

        w2v = self.word_to_vector_mapping()
        v1, v2 = w2v[w1], w2v[w2]
        if not self._normalized:
            v1, v2 = v1/numpy.linalg.norm(v1), v2/numpy.linalg.norm(v2)
        return numpy.dot(v1, v2)

    def nearest(self, v, n=10, exclude=None, candidates=None):
        """Return nearest n words and similarities for given word or vector,
        excluding given words.

        If v is a string, look up the corresponding word vector.
        If exclude is None and v is a string, exclude v.
        If candidates is not None, only consider (word, vector)
        values from iterable candidates.
        Return value is a list of (word, similarity) pairs.
        """

        if isinstance(v, StringTypes):
            v, w = self.word_to_unit_vector(v), v
        else:
            v, w = v/numpy.linalg.norm(v), None
        if exclude is None:
            exclude = [] if w is None else set([w])
        if not self._normalized:
            sim = partial(self._item_similarity, v=v)
        else:
            sim = partial(self._item_similarity_normalized, v=v)
        if candidates is None:
            candidates = self.word_to_vector_mapping().iteritems()
        nearest = heapq.nlargest(n+len(exclude), candidates, sim)
        wordsim = [(p[0], sim(p)) for p in nearest if p[0] not in exclude]
        return wordsim[:n]

    def approximate_nearest(self, v, n=10, exclude=None, 
                            evalnum=1000, bits=None,
                            search_hash_neighborhood=True):
        """Return approximate nearest n words and similarities for
        given word or vector, excluding given words.

        Uses random hyperplane-based locality sensitive hashing (LSH)
        with given number of bits, evaluating evalnum approximate
        neighbors exactly.

        LSH is initialized on the first invocation, which may take
        long for large numbers of word vectors. For a small number of
        NN queries, nearest() may be more efficient.

        If search_hash_neighborhood is True, find approximate neigbors
        by searching the Hamming neighborhood. This is more efficient
        for hashes with comparatively high load factors (~1) but
        inefficient for ones with low load factors.

        If v is a string, look up the corresponding word vector.
        If exclude is None and v is a string, exclude v.
        If bits is None, estimate number of bits to use.
        Return value is a list of (word, similarity) pairs.
        """

        if self._lsh is None or (bits is not None and bits != self._lsh.bits):
            self._initialize_lsh(bits)

        if isinstance(v, StringTypes):
            v, w = self.word_to_unit_vector(v), v
        else:
            v, w = v/numpy.linalg.norm(v), v

        if search_hash_neighborhood:
            candidates = islice(self._lsh.neighbors(v), evalnum)
            return self.nearest(w, n, exclude, candidates)
        else:
            # evalnum nearest by hash similarity
            sim = partial(self._lsh.item_similarity, h=self._lsh.hash(v), 
                          bits=self._lsh.bits)
            anearest = heapq.nlargest(evalnum, self._lsh.iteritems(), sim)
            # nearest vectors for up to evalnum of nearest hashes
            candidates = islice((wv for _, wvs in anearest for wv in wvs), 
                                evalnum)
            return self.nearest(w, n, exclude, candidates)

    def _initialize_lsh(self, bits):
        if bits is None:
            w = self.config.word_count
            bits = max(4, int(math.ceil(math.log(w, 2))))
            logging.debug('init lsh: %d vectors, %d bits' % (w, bits))

        self._lsh = RandomHyperplaneLSH(self.config.vector_dim, bits)
        for w, v in self:
            self._lsh.add(v, (w, v))

        lf = self._lsh.load_factor()
        logging.debug('init lsh: load factor %.2f' % lf)
        if lf < 0.1:
            logging.warning('low lsh load factor (%f), neighbors searches '
                            'may be slow' % lf)

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

    def filter_by_rank(self, r):
        """Discard vectors for words other than the r most frequent."""
        
        if r < self.config.word_count:
            self._invalidate()
            self.config.word_count = r
            self.vocab.shrink(r)
            self.vectors.shrink(r)

    def save(self, name, format=None, vector_format=None):
        """Save in format to pathname name.

        If format is None, determine format heuristically.
        If vector_format is not None, save vectors in vector_format
        instead of currently set format (config.format).
        """

        vf = self.config.format
        try:
            if vector_format is not None:
                self.config.format = vector_format
            if format is None:
                format = self.guess_format(name)
            if format is None:
                raise ArgumentError('failed to guess format for %s' % name)

            logging.info('saving %s as %s with %s vectors' % 
                         (name, format, self.config.format))

            if format == self.TAR:
                return self.save_tar(name)
            elif format == self.DIR:
                return self.save_dir(name)
            else:
                raise NotImplementedError
        finally:
            self.config.format = vf

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
        self._lsh = None

    def __getitem__(self, word):
        """Return vector for given word."""

        return self.word_to_vector_mapping()[word]

    def __contains__(self, word):
        """Return True iff given word is in vocabulary (has vector)."""

        return word in self.word_to_vector_mapping()

    def __getattr__(self, name):
        # delegate access to nonexistent attributes to config
        return getattr(self.config, name)

    def __iter__(self):
        """Iterate over (word, vector) pairs."""

        #return izip(self.vocab.words(), self.vectors)
        return izip(self.vocab.iterwords(), iter(self.vectors))

    @classmethod
    def load(cls, name, max_rank=None):
        """Return WVData from pathname name.

        If max_rank is not None, only load max_rank most frequent words.
        """

        format = cls.guess_format(name)
        if format == cls.TAR:
            wv = cls.load_tar(name, max_rank=max_rank)
        elif format == cls.DIR:
            wv = cls.load_dir(name, max_rank=max_rank)
        else:
            raise NotImplementedError
        if max_rank is not None:
            wv.filter_by_rank(max_rank)
        return wv
            
    @classmethod
    def load_tar(cls, name, max_rank=None):
        """Return WVData from tar or tar.gz name.

        If max_rank is not None, only load max_rank most frequent words."""

        f = tarfile.open(name, 'r')
        try:
            return cls._load_collection(f, max_rank=max_rank)
        finally:            
            f.close()

    @classmethod
    def load_dir(cls, name, max_rank=None):
        """Return WVData from directory name.

        If max_rank is not None, only load max_rank most frequent words."""

        d = _Directory.open(name, 'r')
        try:
            return cls._load_collection(d, max_rank=max_rank)
        finally:
            d.close()

    @classmethod
    def _load_collection(cls, coll, max_rank=None):
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
        vocab = Vocabulary.loadf(coll.extractfile(vocabname), 
                                 max_rank=max_rank)
        vectors = Vectors.loadf(coll.extractfile(vecname), config.format,
                                max_rank=max_rank)
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

    default_format = NUMPY_FORMAT
    
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

    def shrink(self, s):
        """Discard vectors other than the first s."""

        self.vectors = self.vectors[:s]

    def to_rows(self):
        return self.vectors

    def save_tsv(self, f):
        """Save as TSV to file-like object f."""

        f.write(self.__str__())

    def savef(self, f, format):
        """Save in format to file-like object f."""

        if format == TSV_FORMAT:
            return self.save_tsv(f)
        elif format == NUMPY_FORMAT:
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
    def load_tsv(cls, f, max_rank=None):
        """Return Vectors from file-like object f in TSV.

        If max_rank is not None, only load max_rank first vectors."""

        rows = []
        for i, l in enumerate(f):
            if max_rank is not None and i >= max_rank:
                break
            #row = [float(i) for i in l.rstrip('\n').split()]
            row = numpy.fromstring(l, sep='\t')
            rows.append(row)
            if (i+1) % 10000 == 0:
                logging.debug('read %d TSV rows' % (i+1))
        return cls.from_rows(rows)

    @classmethod
    def load_numpy(cls, f, max_rank=None):
        """Return Vectors from file-like object f in NumPy format.

        If max_rank is not None, only load max_rank first vectors."""

        # NOTE: the bit below would be better but doesn't work, as mmap 
        # cannot use existing file handles.
        # if max_rank is not None:
        #     return cls(numpy.array(numpy.load(f, mmap_mode='r')[:max_rank]))

        v = cls(numpy.load(f))
        if max_rank is not None:
            # see comment above
            logging.debug('note: full numpy.load() despite max_rank')
            v.shrink(max_rank)
        return v

    @classmethod
    def loadf(cls, f, format, max_rank=None):
        """Return Vectors from file-like object f in format.

        If max_rank is not None, only load max_rank first vectors."""

        if format == TSV_FORMAT:
            return cls.load_tsv(f, max_rank=max_rank)
        elif format == NUMPY_FORMAT:
            return cls.load_numpy(f, max_rank=max_rank)
        else:
            raise NotImplementedError(format)

    @classmethod
    def load(cls, name, format=None, encoding=DEFAULT_ENCODING, max_rank=None):
        """Return Vectors from pathname name in format.

        If format is None, determine format from filename extension.
        If max_rank is not None, only load max_rank first vectors."""

        if format is None:
            format = os.path.splitext(name)[1].replace('.', '')

        if Vectors.is_text_format(format):
            with codecs.open(name, 'rt', encoding=encoding) as f:
                return cls.loadf(f, format, max_rank=max_rank)
        else:
            with codecs.open(name, 'rb') as f:
                return cls.loadf(f, format, max_rank=max_rank)

    @staticmethod
    def is_text_format(format):
        if format == TSV_FORMAT:
            return True
        elif format == NUMPY_FORMAT:
            return False
        else:
            raise ArgumentError('Unknown format %s' % format)

class Vocabulary(object):
    def __init__(self, word_freq):
        assert not any((j for i, j in pairwise(word_freq) if i[1] < j[1])), \
            'words not ordered by descending frequency'
        self.word_freq = OrderedDict(word_freq)
        self._rank = None

    def words(self):
        return self.word_freq.keys()

    def rank(self, w):
        if self._rank is None:
            self._rank = dict(((j, i) for i, j in enumerate(self.words())))
        return self._rank[w]

    def shrink(self, s):
        """Discard words other than the first s."""

        self._invalidate()
        self.word_freq = OrderedDict(islice(self.word_freq.iteritems(), s))

    def iterwords(self):
        return self.word_freq.iterkeys()

    def to_rows(self):
        return self.word_freq.iteritems()

    def savef(self, f):
        """Save as TSV to file-like object f."""

        f.write(self.__str__())

    def save(self, name, encoding=DEFAULT_ENCODING):
        """Save as TSV to pathname name."""

        with codecs.open(name, 'wt', encoding=encoding) as f:
                return self.savef(f)

    def _invalidate(self):
        """Invalidate cached values."""

        self._rank = None

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
    def loadf(cls, f, max_rank=None):
        """Return Vocabulary from file-like object f in TSV format.

        If max_rank is not None, only load max_rank most frequent words."""

        rows = []
        for i, l in enumerate(f):
            if max_rank is not None and i >= max_rank:
                break
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
    def load(cls, name, encoding=DEFAULT_ENCODING, max_rank=None):
        """Return Vocabulary from pathname name in TSV format.

        If max_rank is not None, only load max_rank most frequent words."""

        with codecs.open(name, 'rU', encoding=encoding) as f:
            return cls.loadf(f, max_rank=max_rank)

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
    def load_textf(cls, f, max_rank=None):
        """Return Word2VecData from file-like object f in the word2vec
        text format.

        If max_rank is not None, only load max_rank most frequent words.
        """

        return cls(cls.read(f, cls.read_text_line, max_rank=max_rank))

    @classmethod
    def load_binaryf(cls, f, max_rank=None):
        """Return Word2VecData from file-like object f in the word2vec
        binary format.

        If max_rank is not None, only load max_rank most frequent words.
        """

        return cls(cls.read(f, cls.read_binary_line, max_rank=max_rank))

    @classmethod
    def load_binary(cls, name, max_rank=None):
        """Return Word2VecData from pathname name in the word2vec
        binary format.

        If max_rank is not None, only load max_rank most frequent words.
        """

        with open(name, 'rb') as f:
            return cls.load_binaryf(f, max_rank)

    @classmethod
    def load_text(cls, name, encoding=DEFAULT_ENCODING, max_rank=None):
        """Return Word2VecData from pathname name in the word2vec text
        format.

        If max_rank is not None, only load max_rank most frequent words.
        """

        with codecs.open(name, 'rU', encoding=encoding) as f:
            return cls.load_textf(f, max_rank)
    
    @classmethod
    def load(cls, name, binary=None, encoding=DEFAULT_ENCODING, max_rank=None):
        """Return Word2VecData from pathname name in the word2vec
        binary or text format.

        If binary is None, determine format heuristically.
        If max_rank is not None, only load max_rank most frequent words.
        """

        if binary is None:
            binary = not cls.is_w2v_text(name, encoding)
        if binary:
            return cls.load_binary(name, max_rank=max_rank)
        else:
            return cls.load_text(name, encoding, max_rank=max_rank)

    @staticmethod
    def read_size_line(f):
        """Read line from file-like object f as word2vec format
        header, return (word count, vector size)."""
        
        l = f.readline().rstrip('\n')
        try:
            wcount, vsize = l.split()
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
    def read(f, read_line=read_binary_line, max_rank=None):
        """Read word2vec data from file-like object f using function
        read_line to parse individual lines. 

        Return list of (word, vector) pairs.
        If max_rank is not None, only load max_rank most frequent words.
        """
        
        wcount, vsize = Word2VecData.read_size_line(f)
        rows = []
        if max_rank is not None and wcount > max_rank:
            wcount = max_rank
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

class RandomHyperplaneLSH(object):
    """Random hyperplane-based locality sensitive hash following
    Charikar (2002)."""

    def __init__(self, dim, bits):
        """Initialize for dim-dimensional vectors hashed to bits-bit
        signatures."""
        
        self.dim = dim
        self.bits = bits
        self.vectors = numpy.random.randn(bits, dim)
        # note: normalization not strictly required
        self.vectors = [v/numpy.linalg.norm(v) for v in self.vectors]
        self._values = {}
        self._entries = 0

    def hash(self, v):
        """Return hash for given vector."""

        # note: normalization not strictly required
        v = v/numpy.linalg.norm(v)
        h = 0
        for u in self.vectors:
            h <<= 1
            if numpy.dot(u, v) > 0:
                h |= 1
        return h

    def similarity(self, v1, v2):
        """Return approximate cosine similarity of given vectors or
        hashes.

        If v1/v2 is a vector, hash before calculating similarity."""

        if not isinstance(v1, (int, long)):
            v1 = self.hash(v1)
        if not isinstance(v2, (int, long)):
            v2 = self.hash(v2)
        return self.hash_similarity(v1, v2, self.bits)

    def neighbors(self, h, min_dist=0, number=None):
        """Yield neighbors of given hash or vector ordered by
        increasing Hamming distance.

        If h is a vector, hash before finding neighbors.
        If number is not None, yield at most given number of neighbors.

        Note: is the load factor is low, this function may be very slow.
        """

        # TODO: revert to linear search on low load factor

        if min_dist < 0:
            raise ValueError('min_dist must be >= 0')
        if not isinstance(h, (int, long)):
            h = self.hash(h)        
        i = 0
        for distance in range(min_dist, self.bits):
            for n in hamming_neighbors(h, self.bits, distance):
                for v in self[n]:
                    if number is not None and i >= number:
                        raise StopIteration
                    yield v
                    i += 1        

    def load_factor(self):
        return float(self._entries)/2**self.bits

    def iteritems(self, max_items=None):
        return self._values.iteritems()

    def add(self, key, value):
        if not isinstance(key, (int, long)):
            key = self.hash(key)
        if key not in self._values:
            self._values[key] = []
        self._values[key].append(value)
        self._entries += 1

    def __getitem__(self, key):
        if not isinstance(key, (int, long)):
            key = self.hash(key)
        return self._values.get(key, [])

    @staticmethod
    def item_similarity(i, h, bits):
        return hash_similarity(i[0], h, bits)

def _guess_format(name):
    for ext, format in extension_format_map.items():
        if name.endswith(ext):
            return format
    if os.path.isdir(name):
        return WVLIB_FORMAT
    return None

def load(name, format=None, max_rank=None):
    """Load word vectors from pathname name in format.

    If format is None, determine format heuristically.
    If max_rank is not None, only load max_rank most frequent words.
    """

    if not os.path.exists(name):
        raise IOError('no such file or directory: %s' % name)
    if format is None:
        format = _guess_format(name)
    if format is None:
        raise FormatError('failed to guess format: %s' % name)

    logging.info('loading %s as %s' % (name, format))

    if format == WVLIB_FORMAT:
        load_func = WVData.load
    elif format == WORD2VEC_FORMAT: # binary vs. text unspecified
        load_func = Word2VecData.load
    elif format == WORD2VEC_TEXT:
        load_func = Word2VecData.load_text
    elif format == WORD2VEC_BIN:
        load_func = Word2VecData.load_binary
    else:
        raise NotImplementedError        
    return load_func(name, max_rank=max_rank)

### misc. helper functions

# from http://docs.python.org/2/library/itertools.html#recipes
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

# transated from http://www.hackersdelight.org/hdcodetxt/snoob.c.txt
def lex_next_bits(i):
    """Return next number with same number of set bits as i."""
    
                                 # i = xxx0 1111 0000
    smallest = i & -i;           #     0000 0001 0000
    ripple = i + smallest;       #     xxx1 0000 0000
    ones = i ^ ripple;           #     0001 1111 0000
    ones = (ones >> 2)/smallest; #     0000 0000 0111
    return ripple | ones;        #     xxx1 0000 0111

def hamming_neighbors(i, bits, distance):
    """Yield numbers with given Hamming distance to i."""

    if distance == 0:
        yield i
    else:
        mask = int(distance * '1', 2)
        while mask < 1 << bits:
            yield i ^ mask
            mask = lex_next_bits(mask)

def hash_similarity(h1, h2, bits):
    # 1 - (Hamming distance/max Hamming distance).
    # set bit count per http://stackoverflow.com/a/9831671
    return 1 - bin(h1^h2).count('1')/float(bits)
    
def unit_vector(v):
    return v/numpy.linalg.norm(v)

### CLI stuff

def argparser():
    try:
        import argparse
    except ImportError:
        import compat.argparse as argparse

    ap=argparse.ArgumentParser()
    ap.add_argument('vectors', metavar='FILE', help='vectors to load')
    ap.add_argument('-f', '--format', default=None, choices=formats,
                    help='input FILE format')
    ap.add_argument('-v', '--vector_format', default=None, 
                    choices=vector_formats,
                    help='output vector format (with -o)')
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
        wv.save(options.output, vector_format=options.vector_format)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))


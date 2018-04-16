__version__ = '1.3.3'

import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools
import functools
import h5py
import numpy as np
import os

from libscrappy import ffi, lib

ftype = np.float32
size_ftype = np.dtype(ftype).itemsize
vsize = 4 #SSE vector length

def _none_if_null(p):
    # convert cffi NULL to None
    if p == ffi.NULL:
        p = None
    return p


def _gsp():
    # allowing up to 9mer, we can have up to base7 and be unique
    alpha_len = range(4,8)
    kmer_len = range(1,10)
    pairs = [(a, k) for a, k in itertools.product(alpha_len, kmer_len)]
    state_lookup = {a**k: (a, k) for a, k in pairs}
    assert len(state_lookup) == len(pairs) # check all are unique

    def guess_state_properties(nstate):
        """Find likely kmer length and alphabet size from transducer state-space size.
        
        :param nstate: number of states in transducer.
    
        :returns: alphabet size, kmer length.
        """
        nkmers = nstate - 1 # for stay
        return state_lookup[nkmers]
    return guess_state_properties

guess_state_properties = _gsp()


class RawTable(object):
    def __init__(self, data, start=0, end=None):
        """Representation of a scrappie `raw_table`.

        :param data: `nd.array` containing raw data.

        ..note:: The class stores a reference to a contiguous numpy array of
            the correct type to be passed to the extension library. The class
            provides safety against the original data being garbage collected.
            To obtain an up-to-date (possibly trimmed and scaled) copy of the
            data use `raw_table.data(as_numpy=True)`.
        """
        if end is None:
            end = len(data)

        self._data = np.ascontiguousarray(data.astype(ftype, order='C', copy=True))
        rt = ffi.new('raw_table *')
        rt.n = len(self._data)
        rt.start = start
        rt.end = end
        rt.raw = ffi.cast("float *", ffi.from_buffer(self._data))
        self._rt = rt[0]

    def data(self, as_numpy=False):
        """Current data as either C object or realised numpy copy.

        :param as_numpy: If True, return a numpy copy of the current data. If
            currently selected range empty, a (0,) shape array is returned.
        """
        if as_numpy:
            return np.copy(self._data[self.start:self.end])
        else:
            return self._rt

    @property
    def start(self):
        """Currently set start sample."""
        return self._rt.start

    @property
    def end(self):
        """Currently set end sample."""
        return self._rt.end

    def trim(self, start=200, end=10, varseg_chunk=100, varseg_thresh=0.0):
        """Trim data.

        :param start: lower bound on number of samples to trim from start
        :param end: lower bound on number of samples to trim from end
        :param varseg_chunk: chunk size for analysing variance of signal
        :param varseq_thresh: quantile to be calculated to use for thresholding

        """
        self._rt = _trim_raw(
            self._rt, start=start, end=end,
            varseg_chunk=varseg_chunk, varseg_thresh=varseg_thresh
        )
        return self

    def scale(self):
        """Normalize data using med/mad scaling."""
        self._rt = _scale_raw(self._rt)
        return self



def _trim_raw(rt, start=200, end=10, varseg_chunk=100, varseg_thresh=0.0):
    """Trim a `raw_table`.

    :param rt: a `raw_table`.
    :param start: lower bound on number of samples to trim from start
    :param end: lower bound on number of samples to trim from end
    :param varseg_chunk: chunk size for analysing variance of signal
    :param varseg_thresh: quantile to be calculated to use for thresholding

    :returns: new scrappie raw data structure.
    """
    trimmed_rt = _none_if_null(lib.trim_raw_by_mad(rt, varseg_chunk, varseg_thresh))
    pstart, pend = 0, 0
    if trimmed_rt is not None:
        pstart, pend = trimmed_rt.start, trimmed_rt.end
        pstart = min(trimmed_rt.n, pstart + start)
        pend = max(0, pend - end)
        if pend < pstart:
            pstart, pend = 0, 0
    rt.start = pstart
    rt.end = pend
    return rt


def _scale_raw(rt):
    """Scale a `raw_table` in place.

    :param rt: `raw_table` to scale`.

    :returns: input (this is solely to be explicit, `rt` is modified in place).
    """
    lib.medmad_normalise_array(rt.raw + rt.start, rt.end - rt.start)
    return rt


class ScrappyMatrix(object):
    def __init__(self, scrappy_matrix):
        """Container to manage lifetime of a bare scrappie_matrix."""
        self._data = scrappy_matrix

    def __del__(self):
        _free_matrix(self._data)

    @property
    def shape(self):
        """Tuple (columns, rows)

        .. note:: The rows value is of the realised array. The actual data is
            stored padded for ease of use with SSE vectors. 
        """
        return self._data.nc, self._data.nr
    
    def data(self, as_numpy=False, sloika=True):
        """Current data as either C object or realised numpy copy. In the
        latter case, padding due to SSE vector use is removed.

        :param as_numpy: If True, return a numpy copy of the current data.
        :param sloika: return sloika compatible posterior matrix (stay is
            first state). Only valid when `as_numpy` is `True`.

        :returns: a contiguous `np.ndarray` of shape (blocks, states).

        """
        if as_numpy:
            return _scrappie_to_numpy(self._data, sloika=sloika)
        else:
            return self._data


def _free_matrix(matrix):
    """Free a `scrappie_matrix`.

    :param matrix: a scrappie matrix

    :returns: `None`
    """
    lib.free_scrappie_matrix(matrix)


def _scrappie_to_numpy(matrix, sloika=True):
    """Convert a `scrappie_matrix` to a numpy array. Removes padding due to
    SSE vectors and optionally reorders states.

    :param matrix: a `scrappie_matrix` to use as source data.
    :param sloika: return sloika compatible matrix (stay is first state).

    :returns: a contiguous `np.ndarray` of shape (blocks, states).

    ..note:: a copy of the data is made, so the input matrix should be freed
        at the earliest convenience with `_free_matrix`.
    """
    np_matrix = np.frombuffer(ffi.buffer(
        matrix.data.f, size_ftype * vsize * matrix.nrq * matrix.nc),
        dtype=ftype
    ).reshape(matrix.nc, vsize * matrix.nrq)

    nblock, nstate = matrix.nc, matrix.nr
    np_matrix = np_matrix[:, :nstate]

    # sloika requires stay state first
    if sloika:
        p1 = np_matrix[:, nstate - 1:nstate]
        p2 = np_matrix[:, 0:nstate - 1]
        np_matrix = np.hstack((p1, p2))
    np_matrix = np.ascontiguousarray(np_matrix)
    return np_matrix


def calc_post(rt, model='rgrgr_r94', min_prob=1e-6, log=True):
    """Run a network network to obtain class probabilities.

    :param rt: a `RawTable`.
    :param min_prob: minimum bound of probabilites.
    :param log: return log-probabilities.

    :returns: a `ScrappyMatrix`
    """
    if not log and model == 'rnnrf_r94':
        raise ValueError("Returning non-log transformed matrix not supported for model type 'rnnrf_r94'.")
    if not isinstance(rt, RawTable):
        raise TypeError('`rt` should be a RawTable.')

    try:
        network = _models_[model]
    except KeyError:
        raise KeyError("Model type '{}' not recognised.".format(model))
    else:
        matrix = _none_if_null(network(rt.data(), min_prob, log))
        if matrix is None:
            raise RuntimeError('An unknown error occurred during posterior calculation.')
        else:
            return ScrappyMatrix(matrix)


def decode_post(post, model='rgrgr_r94', **kwargs):
    """Decode a posterior to retrieve basecall, score, and states. This
    function merely dispatches to a relevant function governed by the model.

    :param post: a `ScrappyMatrix` containing matrix to be decoded.
    :param model: model type.
    :param **kwargs: See the functions `_decode_...` for kwargs
        relevant to each model.

    :returns: tuple containing (call, score, call positions per raw block).
    """
    if not isinstance(post, ScrappyMatrix):
        raise TypeError('`post` should be a ScrappyMatrix.')
    try:
        decoder = _decoders_[model]
    except KeyError:
        raise KeyError("Model type '{}' not recognised.".format(model))
    else:
        return decoder(post, **kwargs)


def _decode_post(post, stay_pen=0.0, skip_pen=0.0, local_pen=2.0, use_slip=False):
    """Decode a posterior using Viterbi algorithm for transducer.

    :param post: a `ScrappyMatrix` containing transducer posteriors.
    :param stay_pen: penalty for staying.
    :param skip_pen: penalty for skipping a base.
    :param local_pen: penalty for local basecalling.
    :param use_slip: allow slipping (movement more than 2 bases).

    :returns: tuple containing (call, score, call positions per raw block).
    """
    nblock, nstate = post.shape

    path = ffi.new("int[{}]".format(nblock + 1))
    score = lib.decode_transducer(
        post.data(), stay_pen, skip_pen, local_pen,
        path, use_slip
    )

    pos = np.zeros(nblock + 1, dtype=np.int32)
    p_pos = ffi.cast("int *", pos.ctypes.data)
    basecall = lib.overlapper(path, nblock + 1, nstate - 1, p_pos)

    return ffi.string(basecall).decode(), score, pos


def _decode_post_crf(post):
    """Decode a posterior using Viterbi algorithm for conditional random field.

    :param post: a `ScrappyMatrix` containing CRF transitions.

    :returns: tuple containing (basecall, score, call positions per raw data block).
    """
    nblock, nstate = post.shape

    path = ffi.new("int[{}]".format(nblock + 1))
    score = lib.decode_crf(post.data(), path)

    pos = np.ascontiguousarray(np.zeros(nblock + 1, dtype=np.int32))
    p_pos = ffi.cast("int *", ffi.from_buffer(pos))
    basecall = lib.crfpath_to_basecall(path, nblock, p_pos)

    return ffi.string(basecall).decode(), score, pos


# Network and decoder functions used above
_models_ = {
    'rgrgr_r94': lib.nanonet_rgrgr_r94_posterior,
    'rgrgr_r95': lib.nanonet_rgrgr_r95_posterior,
    'rnnrf_r94': lib.nanonet_rnnrf_r94_transitions,
}

_decoders_ = {
    'rgrgr_r94': _decode_post,
    'rgrgr_r95': _decode_post,
    'rnnrf_r94': _decode_post_crf,
}


def get_model_stride(model):
    """Obtain the stride length of a model from its name.

    :param model: model name:

    :returns: the model stride.
    """
    stride = lib.get_raw_model_stride_from_string(model.encode())
    if stride == -1:
        raise ValueError("Invalid scrappie model '{}'.")
    return stride


def basecall_raw(data, model='rgrgr_r94', with_base_probs=False, **kwargs):
    """Basecall from raw data in a numpy array to demonstrate API.

    :param data: `ndarray` containing raw signal data.
    :param model: model to use in calculating basecall.
    :param with_base_probs: calculate per-block base (ACGT-) probabilities.
    :param kwargs: kwargs passed to `decode_post`.

    :returns: tuple containing: (basecall, score, per-block call positions
        data start index, data end index, base probs). The last item will
        be `None` for `with_base_probs == False`.
    """
    if with_base_probs and model != 'rnnrf_r94':
        ValueError("Base probabilities can only be returned for model 'rnnrf_r94'.")

    raw = RawTable(data)
    raw.trim().scale()

    post = calc_post(raw, model, log=True)
    seq, score, pos = decode_post(post, model, **kwargs)

    base_probs = None
    if with_base_probs:
        base_post = lib.posterior_crf(post.data())
        base_probs = _scrappie_to_numpy(base_post, sloika=False)
        _free_matrix(base_post)

    return seq, score, pos, raw.start, raw.end, base_probs


def sequence_to_squiggle(sequence, rescale=False):
    """Simulate a squiggle from a base sequence.

    :param sequence: base sequence to model.
    :param rescale:
    :param as_numpy: return a numpy array rather than a `scrappie_matrix`.

    :returns: a simulated squiggle as either a `ScrappyMatrix` or an ndarray.

    """

    seq_len = len(sequence)
    seq = _none_if_null(lib.encode_bases_to_integers(sequence.encode(), seq_len, 1))
    if seq is None:
        raise RuntimeError('An unknown error occurred whilst encoding sequence.')

    squiggle = _none_if_null(lib.dna_squiggle(seq, seq_len, rescale))
    if squiggle is None:
        raise RuntimeError('An unknown error occurred whilst generating squiggle.')

    return ScrappyMatrix(squiggle)


def map_signal_to_squiggle(data, sequence, back_prob=0.0, local_pen=2.0, min_score=5.0):
    """Align a squiggle to a sequence using a simulated squiggle.

    :param data: `ndarray` containing raw signal data.
    :param sequence: base sequence to which to align data.
    :param back_prob: probability of backward movement.
    :param local_pen: penalty for local alignment.
    :param min_score: floor on match score.

    :returns: tuple containing (alignment score, alignment path)
    """
    raw = RawTable(data)
    raw.trim().scale()

    squiggle = sequence_to_squiggle(sequence)

    path = np.ascontiguousarray(np.zeros(raw._rt.n, dtype=np.int32))
    p_path = ffi.cast("int32_t *", ffi.from_buffer(path))

    score = lib.squiggle_match_viterbi(raw.data(), squiggle.data(), back_prob, local_pen, min_score, p_path)

    return score, path


def map_post_to_sequence(post, sequence, stay_pen=0, skip_pen=0, local_pen=4.0,
                         viterbi=False, path=False, bands=None):
    """Block-based local-global alignment of a squiggle to a sequence using
    either Forward or Viterbi algorithm. For the latter the Viterbi path can
    optionally be calculated.

    :param post: a `ScrappyMatrix` containing log-probabilities (as from
       `calc_post`).
    :param sequence: a base sequence which to map.
    :param stay_pen: penalty for zero-state movement from one block to next.
    :param skip_pen: penalty for two-state movement from one block to next.
    :param local_pen: penalty for local alignment through blocks
    :param viterbi: use Viterbi algorithm rather than forward.
    :param path: calculate alignment path (only valid for `viterbi==True`
        and `bands==None`).
    :param bands: two sequences containing lower and upper extremal allowed
        positions for each block. Should be length corresponding to number
        of blocks of `post`. If a single number is given, a diagonal band with
        width 2 * `bands` * #states / #blocks will be used. If `None` is given
        banding is not used (a full DP matrix is evaluated).

    :returns: (score, path), (or (None, *) in the case of failure).

    ..note:: if `viterbi`==False or `path`==False, the returned path will
        be `None`.
    """

    if path and not viterbi:
        raise ValueError('Cannot calulate path with `viterbi==False`.')
    if not isinstance(post, ScrappyMatrix):
        raise TypeError('`post` should be a ScrappyMatrix.')

    nblock, nstate = post.shape 
    alpha_len, kmer_len = guess_state_properties(nstate)

    seq_len = len(sequence) - kmer_len + 1
    p_seq = _none_if_null(lib.encode_bases_to_integers(sequence.encode(), len(sequence), kmer_len))
    if p_seq is None:
        raise RuntimeError('An unknown error occurred whilst encoding sequence.')

    if viterbi and path:
        path_data = np.zeros(nblock, dtype=np.int32)
        p_path = ffi.cast("int *", ffi.from_buffer(path_data))
    else:
        path_data = None
        p_path = ffi.NULL

    if bands is None:
        if viterbi:
            score = lib.map_to_sequence_viterbi(
                post.data(), stay_pen, skip_pen, local_pen, p_seq, seq_len, p_path)
        else:
            score = lib.map_to_sequence_forward(
                post.data(), stay_pen, skip_pen, local_pen, p_seq, seq_len)
    else:
        if isinstance(bands, int):
            # create a monotonic diagonal band
            gradient = seq_len / nblock
            bands = 2 * bands * gradient
            hband = bands / 2
            bands = [np.ascontiguousarray(np.array(x, dtype=np.int32)) for x in (
                [(max(0,       x * gradient - hband)) for x in range(nblock)],
                [(min(seq_len, x * gradient + hband)) for x in range(nblock)]
            )]
        elif len(bands) == 2:
            bands = [
                np.ascontiguousarray(x, dtype=np.int32) for x in bands]
        else:
            raise ValueError('`bands` should be `None`, an integer, or length 2.')

        p_poslow, p_poshigh = (
            ffi.cast("int *", ffi.from_buffer(x)) for x in bands)
        if not lib.are_bounds_sane(p_poslow, p_poshigh, nblock, seq_len):
            raise ValueError('Supplied banding structure is not valid.')

        if viterbi:
            func = lib.map_to_sequence_viterbi_banded
        else:
            func = lib.map_to_sequence_forward_banded
        score = func(
            post.data(), stay_pen, skip_pen, local_pen, p_seq, seq_len, p_poslow, p_poshigh)

    score = _none_if_null(score)
    if score is None:
        raise RuntimeError('An unknown error occurred during alignment.')

    return score, path_data


def _raw_gen(filelist):
    for fname in filelist:
        with h5py.File(fname, 'r') as h:
            pass
        try:
            data = None
            with h5py.File(fname, 'r') as h:
                base = 'Raw/Reads'
                read_name = list(h[base].keys())[0]
                data = h['{}/{}/Signal'.format(base, read_name)][()]
        except:
            raise RuntimeError('Failed to read signal data from {}.'.format(fname))
        else:
            yield os.path.basename(fname), data


def _basecall():
    # Entry point for testing/demonstration.
    parser = argparse.ArgumentParser(
        description="Basecall a single .fast5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('fast5', nargs='+',
        help='path to .fast5s to basecall.')
    parser.add_argument('model', choices=list(_models_.keys()),
        help='Choice of model.')
    parser.add_argument('--threads', default=None, type=int,
        help='Number of threads to use.')
    parser.add_argument('--process', action='store_true',
        help='Use ProcesPool rather than ThreadPool.')

    args = parser.parse_args()

    worker = functools.partial(basecall_raw, model=args.model)
    if args.threads is None:
        for fname, data in _raw_gen(args.fast5):
            seq, score, _, start, end, _ = worker(data)
            print(">{} {} {}-{}\n{}".format(fname, score, start, end, seq))
    else:
        iter0, iter1 = itertools.tee(_raw_gen(args.fast5))
        Executor = ProcessPoolExecutor if args.process else ThreadPoolExecutor
        with Executor(max_workers=args.threads) as executor:
            datas = (x[1] for x in iter0)
            fnames = (x[0] for x in iter1)
            results = executor.map(worker, datas)
            for fname, (seq, score, _, start, end, _) in zip(fnames, results):
                print(">{} {} {}-{}\n{}".format(fname, score, start, end, seq))


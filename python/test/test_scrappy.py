from concurrent.futures import ThreadPoolExecutor
import functools
from io import StringIO
import os
import shutil
import subprocess
from timeit import default_timer as now
import unittest

import h5py
import numpy as np

import scrappy

_path_ = os.path.join(os.path.dirname(__file__), '..', '..', 'reads')
_test_reads_ = tuple(os.path.abspath(os.path.join(_path_, x)) for x in (
    'MINICOL228_20161012_FNFAB42578_MN17976_mux_scan_HG_52221_ch174_read172_strand.fast5',
    'MINICOL228_20161012_FNFAB42578_MN17976_mux_scan_HG_52221_ch271_read66_strand.fast5',
    'read_ch228_file118.fast5',
))
_test_fasta_ = {
    x:'{}.fa'.format(os.path.splitext(x)[0]) for x in _test_reads_
}

class TestScrappy(unittest.TestCase):

    def _parse_fasta(fh):
        # return name, seq pairs from a fh
        #   NB: assumes sequences are on a single line
        name, seq = None, None
        for i, line in enumerate(fh.readlines()):
            if i % 2 == 0:
                # remove ">" and take up to first whitespace
                name = os.path.basename(line[1:].split()[0])
            else:
                seq = line.rstrip()
                yield name, seq


    @classmethod
    def setUpClass(self):
        print("* TestScrappy")
        self.scrappie_exe = shutil.which('scrappie')
        if self.scrappie_exe is None:
            raise RuntimeError("Ensure scrappie is on PATH before running scrappy tests.")
        self.model = 'rgrgr_r94'
        self.expected_states = 1025
        self.expected_stride = 5

        # get scrappie basecalls for two test files
        cmd = ['scrappie', 'raw', '--model', self.model]
        cmd.extend(_test_reads_)
        scrappie_fasta = subprocess.check_output(cmd).decode()
        self.scrappie_seqs = dict()
        name, seq = None, None
        for name, seq in self._parse_fasta(StringIO(scrappie_fasta)):
            self.scrappie_seqs[name] = seq
            if len(self.scrappie_seqs) == len(_test_reads_):
                break

        # get signal data
        self.signals = dict(scrappy._raw_gen(_test_reads_))

        # just one signal and ref
        chosen_one = _test_reads_[0]
        self.one_signal = self.signals[os.path.basename(chosen_one)]
        with open(_test_fasta_[chosen_one], 'r') as fh:
            self.one_ref = next(self._parse_fasta(fh))[1]
        print("  finished setUp.")


    def test_000_same_as_scrappie(self):
        for fname, data in self.signals.items():
            seq, score, pos, _, _, _ = scrappy.basecall_raw(data, model=self.model)
            self.assertEqual(seq, self.scrappie_seqs[os.path.basename(fname)])


    def test_010_raw_table_type(self):
        # type should be correct
        for t in (np.float32, np.float64):
            rt = scrappy.RawTable(self.one_signal.astype(t))
            self.assertEqual(rt._data.dtype, scrappy.ftype, 'Raw table internal type.')


    def test_011_raw_table_methods(self):
        rt = scrappy.RawTable(self.one_signal)
        # check trim
        rt.trim(start=200)
        self.assertEqual(rt.start, 200, 'Trimming applied.')
        # check scale
        rt.scale()
        self.assertEqual(np.median(rt._data[rt.start:rt.end]), 0.0, 'Scaling shifts median to 0.0.')
        # .data(as_numpy=True) should own its data
        new_data = rt.data(as_numpy=True)
        self.assertIsInstance(new_data, np.ndarray)
        self.assertTrue(new_data.flags.owndata)


    def test_012_bad_trimming(self):
        rt = scrappy.RawTable(self.one_signal)
        rt.trim(start=200, end=len(rt._data) - 5)
        self.assertEqual(rt.start, 0, 'Empty gives start=0.')
        self.assertEqual(rt.end, 0, 'Empty gives end=0.')
        self.assertEqual(len(rt.data(as_numpy=True)), 0, 'Empty give len=0 array')


    def test_020_intermediates(self):
        rt = scrappy.RawTable(self.one_signal)
        self.assertIsInstance(rt._rt, scrappy.ffi.CData)
        rt.trim().scale()
        self.assertIsInstance(rt, scrappy.RawTable)
        self.assertIsInstance(rt._rt, scrappy.ffi.CData)
        post = scrappy.calc_post(rt, self.model, log=True)
        self.assertIsInstance(post, scrappy.ScrappyMatrix)

        # Check matrix is formed sanely
        sloika_post = scrappy._scrappie_to_numpy(post._data, sloika=True)
        self.assertIsInstance(sloika_post, np.ndarray)
        self.assertEqual(sloika_post.shape[1], self.expected_states)

        # check types, shouldn't leak cffi abstraction
        seq, score, pos = scrappy.decode_post(post, self.model)
        self.assertIsInstance(seq, str, 'sequence is str.')
        self.assertIsInstance(score, float, 'score is float.')
        self.assertIsInstance(pos, np.ndarray, 'pos is ndarray.')


    def test_030_threaded_call(self):
        # Just run this to check we don't die
        worker = functools.partial(scrappy.basecall_raw, model=self.model)
        with ThreadPoolExecutor(max_workers=1) as executor:
            results = executor.map(worker, self.signals.values())
            for res in results:
                pass


    def test_040_squiggle_map_r94(self):
        # Just check mapping runs without fail
        score, path = scrappy.map_signal_to_squiggle(self.one_signal, self.one_ref, model='squiggle_r94')
        self.assertIsInstance(score, float, 'score is float.')
        self.assertIsInstance(path, np.ndarray, 'path is ndarray.')
        self.assertEqual(len(self.one_signal), len(path), 'Length of path is length of signal.')

    def test_041_squiggle_map_r10(self):
        # Just check mapping runs without fail
        score, path = scrappy.map_signal_to_squiggle(self.one_signal, self.one_ref, model='squiggle_r10')
        self.assertIsInstance(score, float, 'score is float.')
        self.assertIsInstance(path, np.ndarray, 'path is ndarray.')

    def test_045_post_forward_mapping(self):
        rt = scrappy.RawTable(self.one_signal)
        rt.trim().scale()
        post = scrappy.calc_post(rt, self.model, log=True)

        t0 = now()
        score_band, _ = scrappy.map_post_to_sequence(
            post, self.one_ref, stay_pen=0, skip_pen=0, local_pen=4.0,
            viterbi=False, path=False, bands=100)
        t1 = now()
        score_no_band, _ = scrappy.map_post_to_sequence(
            post, self.one_ref, stay_pen=0, skip_pen=0, local_pen=4.0,
            viterbi=False, path=False, bands=None)
        t2 = now()
        self.assertIsInstance(score_no_band, float, 'score is float.')
        self.assertLess(t1 - t0, t2 - t0, 'banded mapping is faster.')

        with self.assertRaises(ValueError):
            # can't calculate path with Forward
            score_no_band = scrappy.map_post_to_sequence(
                post, self.one_ref, stay_pen=0, skip_pen=0, local_pen=4.0,
                viterbi=False, path=True, bands=None)


    def test_046_post_viterbi_mapping(self):
        rt = scrappy.RawTable(self.one_signal)
        rt.trim().scale()
        post = scrappy.calc_post(rt, self.model, log=True)

        t0 = now()
        score_band, _ = scrappy.map_post_to_sequence(
            post, self.one_ref, stay_pen=0, skip_pen=0, local_pen=4.0,
            viterbi=True, path=False, bands=100)
        t1 = now()
        score_no_band, _ = scrappy.map_post_to_sequence(
            post, self.one_ref, stay_pen=0, skip_pen=0, local_pen=4.0,
            viterbi=True, path=False, bands=None)
        t2 = now()
        self.assertIsInstance(score_no_band, float, 'score is float.')
        self.assertLess(t1 - t0, t2 - t0, 'banded mapping is faster.')

        score_band, path = scrappy.map_post_to_sequence(
            post, self.one_ref, stay_pen=0, skip_pen=0, local_pen=4.0,
            viterbi=True, path=True, bands=100)
        self.assertIsInstance(path, np.ndarray, 'path is ndarray.')


    def test_050_model_stride(self):
        stride = scrappy.get_model_stride(self.model)
        self.assertIsInstance(stride, int, 'stride is int.')
        self.assertEqual(stride, self.expected_stride, 'stride is as expected.')


    def test_051_model_stride_unknown(self):
        with self.assertRaises(ValueError):
            stride = scrappy.get_model_stride('garbage_model')


    def test_060_matrix_conversions(self):
        rt = scrappy.RawTable(self.one_signal)
        rt.trim().scale()
        original = scrappy.calc_post(rt, self.model, log=True)
        from_np = scrappy.ScrappyMatrix(
            original.data(as_numpy=True, sloika=False))

        # decoding both as a check on the C structure
        scores = [scrappy.decode_post(x)[1] for x in (original, from_np)]
        np.testing.assert_almost_equal(scores[0], scores[1], err_msg='Scores equal after round trip.')


    def test_065_matrix_view(self):
        rt = scrappy.RawTable(self.one_signal)
        rt.trim().scale()
        original = scrappy.calc_post(rt, self.model, log=True)

        all_view = original[:]
        self.assertSequenceEqual(original.shape, all_view.shape, 'All view is same shape.')
        small_view = original[100:200]
        self.assertSequenceEqual((100, original.shape[1]), small_view.shape, 'Slice has correct shape.')
        smaller_view = small_view[10:50]
        self.assertSequenceEqual((40, original.shape[1]), smaller_view.shape, 'Slice of slice has correct shape.')

        np_original = original.data(as_numpy=True, sloika=False)
        np_smaller = smaller_view.data(as_numpy=True, sloika=False)
        np.testing.assert_allclose(np_original[110:150,], np_smaller, err_msg='Slice contains correct data.')




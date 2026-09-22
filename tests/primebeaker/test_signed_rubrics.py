import importlib.util
import random
import unittest
from pathlib import Path

path=Path(__file__).resolve().parents[2]/'src/primebeaker/environments/signed_rubrics.py'
spec=importlib.util.spec_from_file_location('signed_rubrics',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SignedRubricTests(unittest.TestCase):
    def test_penalty_presence_reduces_reward(self):
        r=[{'weight':5},{'weight':-5}]
        denominator=m.weight_normalizer(r)
        self.assertEqual(denominator,5)
        self.assertEqual((5-5)/denominator,0)
        self.assertEqual(-5/denominator,-1)
        self.assertEqual(5/denominator,1)
    def test_all_negative_and_positive_only(self):
        with self.assertRaises(ValueError):
            m.weight_normalizer([{'weight':-3}])
        self.assertEqual(m.weight_normalizer([{'weight':3},{'weight':-7}]),3)
        self.assertEqual(m.weight_normalizer([{'weight':3},{'weight':7}]),10)
    def test_invalid_weights(self):
        for w in [True,0,float('nan'),float('inf'),'3']:
            with self.assertRaises((TypeError,ValueError)):m.nonzero_weight(w)
    def test_sample_is_uniform_and_reproducible(self):
        r=[{'weight':5}]*30+[{'weight':-3}]*8
        a=m.signed_sample_indexes(r,8,random.Random(17))
        self.assertEqual(a,m.signed_sample_indexes(r,8,random.Random(17)))
        self.assertEqual(len(set(a)),8)
        self.assertEqual(a,sorted(random.Random(17).sample(range(len(r)),8)))
        flipped=[{'weight':-v['weight']} for v in r]
        self.assertEqual(a,m.signed_sample_indexes(flipped,8,random.Random(17)))
        self.assertEqual(len(m.signed_sample_indexes(r,1,random.Random(17))),1)
    def test_positive_sampling_unchanged(self):
        r=[{'weight':5}]*20
        self.assertEqual(m.signed_sample_indexes(r,8,random.Random(17)),sorted(random.Random(17).sample(range(20),8)))
    def test_judge_polarity_explicit(self):
        self.assertIn('pass ONLY',m.judge_criterion({'weight':-3,'text':'The answer invents a constant.'}))
        self.assertEqual(m.judge_criterion({'weight':3,'text':'Correct constant.'}),'Correct constant.')
if __name__=='__main__':unittest.main()

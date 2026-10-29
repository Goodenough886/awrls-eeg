import unittest
import numpy as np
from awrls3 import Config,clean
from awrls3.estimate import fit,spectral_system
from awrls3.dictionary import ShapeDictionary


class NumericalTests(unittest.TestCase):
    def test_flat_gls_is_ls(self):
        rng=np.random.default_rng(4);R=rng.normal(size=(151,5));d=rng.normal(size=151)
        a,b=spectral_system(d,R,200,12)
        c,e=spectral_system(d,R,200,12,(np.array([0,100]),np.ones(2)))
        np.testing.assert_allclose(a,c,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(b,e,rtol=1e-12,atol=1e-12)

    def test_real_frequency_objective_matches_time_domain(self):
        from scipy.signal.windows import tukey
        rng=np.random.default_rng(5);R=rng.normal(size=(128,4));d=rng.normal(size=128)
        G,b=spectral_system(d,R,200,100)
        A=R*tukey(128,.12)[:,None];y=d*tukey(128,.12)
        np.testing.assert_allclose(G,A.T@A,atol=1e-11)
        np.testing.assert_allclose(b,A.T@y,atol=1e-11)
        beta=np.linalg.solve(G,b)
        np.testing.assert_allclose(beta,np.linalg.lstsq(A,y,rcond=None)[0],atol=1e-11)

    def test_column_scale_invariance(self):
        rng=np.random.default_rng(6);A=rng.normal(size=(160,3));B=np.column_stack([np.ones(160),np.linspace(-1,1,160)])
        d=rng.normal(size=160);cfg=Config(dictionary=False)
        a,_=fit(d,A,B,np.zeros(2),200,cfg)
        b,_=fit(d,A*np.array([100,.002,7]),B,np.zeros(2),200,cfg)
        np.testing.assert_allclose(a,b,atol=1e-10)

    def test_invalid_and_clean_constant(self):
        cfg=Config(dictionary=False)
        y,diag=clean(np.ones(400),200,cfg)
        np.testing.assert_array_equal(y,np.ones(400))
        for x in [np.ones((2,400)),np.array([np.nan]*400),np.ones(3)]:
            with self.assertRaises(ValueError):clean(x,200,cfg)
        with self.assertRaises(ValueError):clean(np.ones(400),200,Config())

    def test_dictionary_round_trip(self):
        import tempfile
        from pathlib import Path
        t=np.linspace(-.6,.6,241);d=ShapeDictionary(np.array([np.exp(-(t/.1)**2)]),200,{'train':[1,2]})
        with tempfile.TemporaryDirectory() as p:
            path=Path(p)/'d.npz';d.save(path);other=ShapeDictionary.load(path)
            np.testing.assert_array_equal(d.shapes,other.shapes)
            self.assertEqual(d.metadata,other.metadata)
            match,_,_=other.match(d.shapes[0],120,200)
            self.assertGreater(match,.999)

if __name__=='__main__':unittest.main()

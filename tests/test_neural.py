import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from awrls3.neural import Denoiser,NeuralCleaner


class NeuralTests(unittest.TestCase):
    def test_zero_head_and_required_proposal(self):
        m=Denoiser(False).eval();x=torch.randn(2,1,512)
        with torch.inference_mode():y,_=m(x,torch.zeros_like(x))
        torch.testing.assert_close(y,x,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.pt';model=Denoiser(True)
            torch.save(dict(state_dict=model.state_dict(),hybrid=True,channels=32),p)
            c=NeuralCleaner(p)
            with self.assertRaises(ValueError):c.clean(np.ones(512),256)

    def test_checkpoint_inference_scaling_and_resampling(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.pt';model=Denoiser(False)
            # Nonzero learned-looking residual makes this test exercise resampling.
            torch.nn.init.normal_(model.head.weight,std=.01)
            torch.save(dict(state_dict=model.state_dict(),hybrid=False,channels=32),p)
            c=NeuralCleaner(p);x=np.random.default_rng(7).normal(size=2000)
            y=c.clean(x,200);z=c.clean(3*x,200)
            self.assertEqual(y.shape,x.shape);self.assertTrue(np.isfinite(y).all())
            np.testing.assert_allclose(z,3*y,atol=2e-6)

if __name__=='__main__':unittest.main()

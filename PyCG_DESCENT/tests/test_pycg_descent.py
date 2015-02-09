import unittest
import numpy as np
import os

from pele.potentials._pythonpotential import CppPotentialWrapper
from pele.potentials import BasePotential, _lj_cpp
from PyCG_DESCENT import CGDescent

ndof = 4
_xrand = np.random.uniform(-1, 1, [ndof])
_xmin = np.zeros(ndof)
_emin = 0.


class _E(BasePotential):
    def getEnergy(self, x):
        return np.dot(x, x)

class _EG(object):
    def getEnergy(self, x):
        return np.dot(x, x)

    def getEnergyGradient(self, x):
        return self.getEnergy(x), 2. * x


class _Raise(BasePotential):
    def getEnergy(self, x):
        raise NotImplementedError

    def getEnergyGradient(self, x):
        raise NotImplementedError


class TestCGDescent_PP(unittest.TestCase):
    def test_raises(self):
        with self.assertRaises(NotImplementedError):
            cgd = CGDescent(_xrand, _Raise())
            cgd.run()


class TestCGDescent(unittest.TestCase):
    def do_check(self, pot, **kwargs):
        cgd = CGDescent(_xrand, pot, **kwargs)
        res = cgd.run()
        self.assertAlmostEqual(res.energy, _emin, 4)
        self.assertTrue(res.success)
        self.assertLess(np.max(np.abs(res.coords - _xmin)), 1e-2)
        self.assertGreater(res.nfev, 0)

    def test_E(self):
        self.do_check(_E())

    def test_EG(self):
        self.do_check(_EG())

    def assert_same(self, res1, res2):
        self.assertEqual(res1.energy, res2.energy)
        self.assertEqual(res1.rms, res2.rms)
        self.assertEqual(res1.nfev, res2.nfev)

    def test_run_niter(self):
        cgd1 = CGDescent(_xrand, _EG())
        res1 = cgd1.run()
        cgd2 = CGDescent(_xrand, _EG())
        res2 = cgd2.run(res1.nsteps)
        self.assert_same(res1, res2)

    def test_result(self):
        cgd = CGDescent(_xrand, _EG())
        res = cgd.run()
        self.assertIn("gnorm", res)
        self.assertIn("itersub", res)
        self.assertIn("numsub", res)
        self.assertIn("nfunc", res)
        self.assertIn("ngrad", res)
        self.assertIn("energy", res)
        self.assertIn("grad", res)
        self.assertIn("success", res)
        self.assertIn("coords", res)
        self.assertIn("rms", res)
        self.assertIn("nsteps", res)
        self.assertIn("nfev", res)


class TestCGDescent_Raises(unittest.TestCase):
    def test_raises(self):
        pot = _lj_cpp._ErrorPotential()
        with self.assertRaises(RuntimeError):
            cgd = CGDescent(_xrand, pot)
            cgd.run()


class TestCGDescent_LJ(unittest.TestCase):
    def setUp(self):
        from pele.potentials import LJ

        self.x0 = np.zeros(6)
        self.x0[0] = 2.
        self.pot = LJ()

    def test_reset(self):
        cgd1 = CGDescent(self.x0, self.pot)
        cgd1.run()
        res1 = cgd1.get_result()

        x2 = self.x0.copy()
        x2[1] = 2.
        cgd2 = CGDescent(x2, self.pot)
        cgd2.reset(self.x0)
        cgd2.run()
        res2 = cgd2.get_result()

        self.assertEqual(res1.rms, res2.rms)
        self.assertEqual(res1.nfev, res2.nfev)
        self.assertEqual(res1.nsteps, res2.nsteps)
        self.assertTrue(np.all(res1.coords == res2.coords))


if __name__ == "__main__":
    unittest.main()

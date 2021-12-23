import numpy as np

from nuspacesim import ResultsTable
from nuspacesim.utils import decorators


def test_nss_result_store():
    @decorators.nss_result_store("columnA", "columnB")
    def test_base_f(input1, input2):
        """this is the docstring"""
        return input1 + input2, input1 * input2

    iA, iB = np.random.randn(2, 128)
    cA, cB = test_base_f(iA, iB)

    assert np.array_equal(cA, iA + iB)
    assert np.array_equal(cB, iA * iB)

    sim = ResultsTable()
    cA, cB = test_base_f(iA, iB, store=sim)

    assert np.array_equal(cA, iA + iB)
    assert np.array_equal(cB, iA * iB)
    assert np.array_equal(sim["columnA"], cA)
    assert np.array_equal(sim["columnB"], cB)

    assert test_base_f.__doc__ == "this is the docstring"


def test_nss_result_store_scalar():
    @decorators.nss_result_store_scalar(
        ["valueA", "valueB"],
        ["terse comment", "Verbose, pompous, and overly long commentB"],
    )
    def test_base_f(input1, input2):
        """this is the docstring"""
        return float(input1 + input2), int(input1 * input2)

    iA, iB = np.random.randn(2, 1)
    vA, vB = test_base_f(iA, iB)

    assert vA == iA + iB
    assert vB == int(iA * iB)

    sim = ResultsTable()
    vA, vB = test_base_f(iA, iB, store=sim)

    assert vA == iA + iB
    assert vB == int(iA * iB)
    assert sim.meta["valueA"][0] == vA
    assert sim.meta["valueB"][0] == vB
    assert sim.meta["valueA"][1] == "terse comment"
    assert sim.meta["valueB"][1] == "Verbose, pompous, and overly long commentB"

    assert test_base_f.__doc__ == "this is the docstring"


def test_nss_result_plot():

    plot_written = False
    iA, iB = np.random.randn(2, 128)

    def plotter(inputs, results, *args, **kwargs):
        nonlocal plot_written
        plot_written = True
        assert plot_written
        assert len(inputs) == 2
        assert len(results) == 2
        assert len(args) == 0
        assert len(kwargs) == 0
        assert np.array_equal(inputs[0], iA)
        assert np.array_equal(inputs[1], iB)
        assert np.all(np.equal(results[0], 0.0))
        assert np.all(np.equal(results[1], 1.0))

    @decorators.nss_result_plot(plotter)
    def test_base_f(input1, input2):
        """this is the docstring"""
        return np.zeros_like(input1), np.ones_like(input2)

    from nuspacesim.utils.plot_function_registry import registry

    assert plotter.__name__ in registry

    # test plotter is not called without a plot argument
    assert not plot_written
    cA, cB = test_base_f(iA, iB)
    assert not plot_written
    assert np.all(np.equal(cA, 0.0))
    assert np.all(np.equal(cB, 1.0))

    # test plotter is called with a callable plot argument
    plot_written = False
    cA, cB = test_base_f(iA, iB, plot=plotter)
    assert plot_written
    assert np.all(np.equal(cA, 0.0))
    assert np.all(np.equal(cB, 1.0))

    # test plotter is called with a string plot argument
    plot_written = False
    cA, cB = test_base_f(iA, iB, plot=plotter.__name__)
    assert plot_written
    assert np.all(np.equal(cA, 0.0))
    assert np.all(np.equal(cB, 1.0))

    # test plotter is called with a list of callable plot arguments
    plot_written = False
    cA, cB = test_base_f(iA, iB, plot=list([plotter]))
    assert plot_written
    assert np.all(np.equal(cA, 0.0))
    assert np.all(np.equal(cB, 1.0))

    # test plotter is called with a list of string plot arguments
    plot_written = False
    cA, cB = test_base_f(iA, iB, plot=list([plotter.__name__]))
    assert plot_written
    assert np.all(np.equal(cA, 0.0))
    assert np.all(np.equal(cB, 1.0))

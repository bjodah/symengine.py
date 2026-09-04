"""Binding-private contract of the pickled LLVM artifact.

``LLVMDouble.__reduce__`` ships a versioned *native* artifact together with the
callable shape (input/output element counts) it was compiled for.  ``_load``
hands both to ``LLVMVisitor::loads``, which refuses an artifact whose recorded
shape disagrees -- otherwise a mismatched pickle would be JIT-loaded and then
called with the wrong buffer extents.

These assertions reach into ``symengine.lib.symengine_wrapper``'s private
``llvm_loading_func`` and into the exact layout of the ``__reduce__`` state
tuple, both of which are implementation details of *this* binding.  They are
deliberately kept out of ``test_pickling.py``, whose contracts are portable
across bindings and are mechanically transcribed for other wrappers.
"""
import unittest

from symengine import symbols, sin, sinh, have_llvm, have_numpy

if have_llvm:
    from symengine.lib.symengine_wrapper import llvm_loading_func


@unittest.skipUnless(have_llvm, "No LLVM support")
@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_llvm_double_load_rejects_wrong_shape():
    from symengine import Lambdify
    args = x, y, z = symbols('x y z')
    expr = sin(sinh(x + y) + z)
    l = Lambdify(args, expr, cse=True, backend='llvm')

    loader, state = l.__reduce__()
    assert loader is llvm_loading_func
    # state[0] is args_size, state[1] is tot_out_size.
    for index in (0, 1):
        wrong_shape = list(state)
        wrong_shape[index] += 1
        try:
            loader(*wrong_shape)
        except ValueError as exc:
            assert "A5-E019 entry-abi" in str(exc)
        else:
            raise AssertionError("LLVM artifact with a wrong shape was accepted")

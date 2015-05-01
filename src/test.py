#!/usr/bin/env python
try:
    import imp
    import sys
except ImportError:
    print "Problems importing either `imp` or `sys`! No other tests will be run."
    sys.exit(1)

def try_import(module):
    try:
        imp.find_module(module)
    except ImportError, e:
        return False
    return True

print "Testing if needed modules are possible to import:"

imports = ["time", "argparse", "numpy", "theano", "collections", "math", "os", "cPickle"]

for module in imports:
    print "Module", module, ".....",
    if try_import(module):
        print "present"
    else:
        print "NOT present"

if try_import("theano"):
    print "Running theano tests, this will take a while:"
    import theano
    theano.test()
else:
    print "Cannot run theano tests, since we were unable to import it"

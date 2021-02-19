#!/usr/bin/env python3

# =================================================================
# hier_object and related tests by https://github.com/SeeForTwo are
# licensed under a Creative Commons Attribution-NonCommercial 4.0
# International License.
# http://creativecommons.org/licenses/by-nc/4.0/
# =================================================================

"""
Print one pseudo random example created by test_hyp_hier_object.py
(for testing hier_object.py). Typically, each time this is run,
a different example is generated.
"""
from test_hyp_hier_object import hier_object_data
from object_create import HIERARCHY_TEST_PARAM

if __name__ == '__main__':
    example = hier_object_data(HIERARCHY_TEST_PARAM).example()
    if not example:
        print(example)
    else:
        for e in example:
            print(e)

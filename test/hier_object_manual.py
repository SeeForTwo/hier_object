#!/usr/bin/env python3

# =================================================================
# hier_object and related tests by https://github.com/SeeForTwo are
# licensed under a Creative Commons Attribution-NonCommercial 4.0
# International License.
# http://creativecommons.org/licenses/by-nc/4.0/
# =================================================================

"""
Example of manual testing of hier_object.py
"""

import test_hyp_hier_object

def test_h1f1():
    """
    One head, one face
    """
    info = [
        {'ImageID': '1', 'LabelName': 'head',
         'TestInfo': {
             "children_as_idx": {
                 "face": [
                     1
                 ]
             },
             "duplicates_as_idx": [],
             "idx": 0,
             "label": "head",
             "parents_as_idx": {},
             "xc": 0.45,
             "yc": 0.45
         },
         'XMax': '0.855', 'XMin': '0.045', 'YMax': '0.855', 'YMin': '0.045'},
        {'ImageID': '1', 'LabelName': 'face',
         'TestInfo': {
             "children_as_idx": {},
             "duplicates_as_idx": [],
             "idx": 1,
             "label": "face",
             "parents_as_idx": {
                 "head": [
                     0
                 ]
             },
             "xc": 0.45,
             "yc": 0.45
         },
         'XMax': '0.53019', 'XMin': '0.36981',
         'YMax': '0.53019', 'YMin': '0.36981'}]

    test_hyp_hier_object.do_hier_object_test(info, verbose=True)

if __name__ == "__main__":
    test_h1f1()

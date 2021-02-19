#!/usr/bin/env python3

# =================================================================
# hier_object and related tests by https://github.com/SeeForTwo are
# licensed under a Creative Commons Attribution-NonCommercial 4.0
# International License.
# http://creativecommons.org/licenses/by-nc/4.0/
# =================================================================

"""
Property-based, pseudo random unit tests for hier_object.py using
"hypothesis" library.
See test_unit_hier_object.py for basic unit tests.

The "hypothesis" library is similar to Haskell's QuickCheck. See
  http://hypothesis.works/
  https://hypothesis.readthedocs.io/en/latest/data.html

Tested with hypothesis 3.44.1 (e.g. Ubuntu 18) and 4.36.2
(e.g. Ubuntu 20). Versions older than 3.5.0 are known be incompatible.
"""

import json
from operator import attrgetter
import os.path
import sys
import unittest

try:
    from hypothesis import given, seed, settings, Verbosity, HealthCheck
except ImportError as ex:
    print('\nImportError')
    print('Install package "python3-hypothesis" for "hypothesis"?')
    print('Or see http://hypothesis.works/ for how to get this library')
    print('')
    raise ex

from allocate_space import DEFAULT_IMAGE_ID
from cls_test_info import TEST_INFO, TestInfo
from object_create import hier_object_data

sys.path.append(
    os.path.join(os.path.dirname(__file__), os.path.pardir))

import hier_object

def print_info(info, subset=None):
    """
    Print INFO from hier_object_data().

    If SUBSET is not None, only rows with TEST_INFO indexes in SUBSET
    are printed.
    """
    for row in info:
        ti = row.get(TEST_INFO)
        if subset and ti and (ti.get('idx') not in subset):
            continue
        print({(k, v) for (k, v) in row.items() if k != TEST_INFO})
        print(TEST_INFO, row.get(TEST_INFO))
        print('')

def print_object_data(ho_data):
    """
    Print datastructure of HierObject objects.
    """
    for cls, obj_lst in ho_data[DEFAULT_IMAGE_ID].items():
        print('All %d objects for %s' % (len(obj_lst), cls))
        for obj in obj_lst:
            print(json.dumps(
                obj, indent=1, sort_keys=True,
                cls=hier_object.HierObjectJSONEncoder))
            print('')

def print_object_list(ho_list, subset=None):
    """
    Print a list HierObject objects.
    If SUBSET is not None, it is a list/etc. of indexes
    of TestInfo objects to print.
    """
    for idx, obj in enumerate(ho_list):
        if subset and (idx not in subset):
            continue
        print(idx)
        print(json.dumps(
            obj, indent=1, sort_keys=True,
            cls=hier_object.HierObjectJSONEncoder))
        print('')

def info_to_objects_and_test_info(info, modify=False):
    """
    Create HierObject objects and separate test information from
    from INFO from hier_object_data.

    Returns a copy of info sorted in TestInfo order, the datastructure
    with HierObject objects, a flat list of all HierObjects in
    TestInfo order respectively and a corresponding list of TestInfo
    objects ordered by their IDX members.

    if MODIFY is True, TEST_INFO values in info that are dictionaries
    are modified to be TestInfo objects.
    """
    hier_object.clear_default_all()
    ti_list = []
    hobj1 = None
    for row_dict in info:
        ti = row_dict[TEST_INFO]
        if not isinstance(ti, TestInfo):
            ti = TestInfo(previous=ti)
            if modify:
                row_dict[TEST_INFO] = ti
        hobj1 = hier_object.HierObject(row_dict)
        ti.obj = hobj1
        ti_list.append(ti)
    ti_list.sort(key=attrgetter('idx'))
    if not info:
        info_ti_order = []
    elif isinstance(info[0][TEST_INFO], TestInfo):
        info_ti_order = sorted(info, key=lambda i: i[TEST_INFO].idx)
    else:
        info_ti_order = sorted(info, key=lambda i: i[TEST_INFO]['idx'])
    if hobj1 is None:
        ho_data = hier_object.create_all_dict()
        ho_list_ti_order = []
    else:
        ho_data = hobj1.all
        ho_list_ti_order = [i.obj for i in ti_list]
    # find class labels of parents and children
    child_parent_set = set()
    for ti in ti_list:
        child_parent_set.update([
            (c_label, ti.label)
            for c_label in ti.children_as_idx.keys()])
    hier_object.get_hierarchy(ho_data, child_parent_set)
    return info_ti_order, ho_data, ho_list_ti_order, ti_list

def not_duplicate(ti_list, idx_list):
    """
    Return the test information object indexes in a list of
    indexes IDX_LIST whose corresponding HierObject is not a
    duplicate.
    """
    if not idx_list:
        return []
    return [
        idx for idx in idx_list if not ti_list[idx].obj.duplicate]

def check_duplicates(ti_list, ti):
    """
    Check the information about duplicates in HierObject in TestInfo
    TI. TI_LIST is the list of TestInfo.
    """
    obj = ti.obj
    if ti.duplicates_as_idx:
        num_other = len(not_duplicate(ti_list, ti.duplicates_as_idx))
        if obj.duplicate:
            assert num_other == 1, 'num_other == %d, not 1 for ti[%d]' % (
                num_other, ti.idx)
        else:
            assert num_other == 0, 'num_other == %d, not 0 for ti[%d]' % (
                num_other, ti.idx)
    else:
        assert not obj.duplicate
    if obj.duplicate:
        assert not obj.parent
        assert not obj.children
        return

def check_parents(ti_list, ti):
    """
    Check the information about parents (objects at higher hierarchy
    levels ) in HierObject in TestInfo TI. TI_LIST is the list of
    TestInfo.
    """
    obj = ti.obj
    all_image = obj.all[DEFAULT_IMAGE_ID]
    parent_labels = []
    if ti.parents_as_idx and not ti.obj.duplicate:
        for p_label, pi_list in ti.parents_as_idx.items():
            parents_idx = not_duplicate(ti_list, pi_list)
            if len(parents_idx) == 1:
                parent_labels.append(p_label)
                assert obj.parent, 'assert obj.parent for ti[%d].obj ' % ti.idx
                p_ti = ti_list[parents_idx[0]]
                p_obj = all_image[p_ti.label][ti.obj.parent[p_ti.label]]
                assert p_ti.obj == p_obj
            else:
                if ti.obj.parent:
                    assert not ti.obj.parent.get(p_label)
    if obj.parent:
        for k, v in obj.parent.items():
            if k in parent_labels:
                continue
            assert not v, (
                'assert obj.parent[%s] is %s not empty for ti[%d].obj '
                % (k, str(v), ti.idx))

def check_children(ti_list, ti):
    """
    Check the information about children (objects in lower hierarchy
    levels) in HierObject in TestInfo TI. TI_LIST is the list of
    TestInfo.
    """
    obj = ti.obj
    all_image = obj.all[DEFAULT_IMAGE_ID]
    children_labels = []
    if ti.children_as_idx and not ti.obj.duplicate:
        for c_label, ci_list in ti.children_as_idx.items():
            children_idx = not_duplicate(ti_list, ci_list)
            if children_idx:
                c_ti_list = [ti_list[c_idx] for c_idx in children_idx]
                # remove children with two or move non-duplicate parents
                c_ti_list = [
                    i for i in c_ti_list
                    if len(not_duplicate(
                        ti_list, i.parents_as_idx.get(ti.label, []))) == 1]
                if c_ti_list:
                    children_labels.append(c_label)
                    assert obj.children
                    c_obj_list = [all_image[c_label][i] for i in
                                  obj.children.get(c_label, [])]
                    ti_obj_set = set([c_ti.obj for c_ti in c_ti_list])
                    assert ti_obj_set == set(c_obj_list), (
                        'ti_obj_set == set(c_obj_list) ti[%d]\n%s\n==\n%s'
                        % (ti.idx,
                           '\n'.join(map(str, ti_obj_set)),
                           '\n'.join(map(str, c_obj_list))))
    if obj.children:
        for k, v in obj.children.items():
            if k in children_labels:
                continue
            assert not v, (
                'assert obj.children[%s] is %s not empty for ti[%d].obj '
                % (k, str(v), ti.idx))

def check_object(ti_list, ti):
    """
    Check the HierObject in TestInfo TI. TI_LIST is the list of TestInfo.
    """
    obj = ti.obj
    # check center coordinates
    assert abs(obj.xc - ti.xc) < 0.0001
    assert abs(obj.yc - ti.yc) < 0.0001
    check_duplicates(ti_list, ti)
    check_parents(ti_list, ti)
    check_children(ti_list, ti)

def do_hier_object_test(info, verbose=False, subset=None):
    """
    Actual test for HierObject methods.
    INFO is from hypothesis library executing hier_object_data().
    if VERSBOSE is True, informational messages are printed.
    if SUBSET is not None, it is a list/etc. of indexes
    of TestInfo objects to check.
    """
    if verbose:
        print('START')
        print_info(info)
    (info_ti_order, ho_data, ho_list_ti_order,
     ti_list) = info_to_objects_and_test_info(info)
    if not ti_list:
        return
    if verbose:
        if subset:
            print_object_list(ho_list_ti_order, subset)
        else:
            print_object_data(ho_data)
    for ti in ti_list:
        if subset and (ti.idx not in subset):
            continue
        check_object(ti_list, ti)

@given(hier_object_data())
# @seed(245871566436678543376580270324379892737)
# @settings(verbosity=Verbosity.verbose, deadline=2000,
#           suppress_health_check=[HealthCheck.too_slow])
@settings(deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def hier_object_test(info):
    """
    Wrapper for test for HierObject methods.
    The hypothesis library runs this test by executing hier_object_data()
    to generate INFO.
    Just calls do_hier_object_test with INFO and VERSBOSE=False.
    """
    do_hier_object_test(info, verbose=False)

class HierObjectTestCaseHypothesis(unittest.TestCase):
    """
    A unittest class that wraps the Hypothesis library test.
    """
    def test_hier_object(self):
        """
        A unittest that calls hier_object_test.
        """
        hier_object_test()

if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3

# =================================================================
# hier_object and related tests by https://github.com/SeeForTwo are
# licensed under a Creative Commons Attribution-NonCommercial 4.0
# International License.
# http://creativecommons.org/licenses/by-nc/4.0/
# =================================================================

"""
Basic unit tests for hier_object.py.
See also test_hyp_hier_object.py for more, pseudo-random unit tests.
"""

import filecmp
import json
import os.path
import shutil
import sys
import unittest

sys.path.append(
    os.path.join(os.path.dirname(__file__), os.path.pardir))

import hier_object

class HierObjectTestCase(unittest.TestCase):
    """
    Methods starting with "test" are unit tests for HierObject.
    """

    RESULT_DIR_SETUP = False

    def setUp(self):
        """
        This is called before the start of each test.
        Clears class static storage.
        """
        hier_object.clear_default_all()

    def init_1(self):
        """
        Return one HierObject object.
        """
        return hier_object.HierObject(dict(
            ImageID='1', LabelName='Object',
            XMin='0.2', XMax='0.4', YMin='0.3', YMax='0.5'))

    def init_3(self):
        """
        Return three HierObject objects.
        """
        hobj1 = self.init_1()
        hobj2 = hier_object.HierObject(dict(
            ImageID='1', LabelName='Object',
            XMin='0.1', XMax='0.3', YMin='0.4', YMax='0.8'))
        hobj3 = hier_object.HierObject(dict(
            ImageID='1', LabelName='Group',
            XMin='0.1', XMax='0.5', YMin='0.1', YMax='0.5'))
        return (hobj1, hobj2, hobj3)

    def overlap_5(self):
        """
        Return five Overlap namedtuples.
        """
        return [
            hier_object.Overlap(
                index=0, self_inside_other=True, center_of_self=True,
                other_inside_self=None, center_of_other=None,
                intersection=0.59,
                iou=0.5, iom=0.5, obj=None),
            hier_object.Overlap(
                index=1, self_inside_other=False, center_of_self=True,
                other_inside_self=None, center_of_other=None,
                intersection=0.58,
                iou=0.6, iom=0.8, obj=None),
            hier_object.Overlap(
                index=2, self_inside_other=True, center_of_self=True,
                other_inside_self=None, center_of_other=None,
                intersection=0.57,
                iou=0.9, iom=0.3, obj=None),
            hier_object.Overlap(
                index=3, self_inside_other=False, center_of_self=True,
                other_inside_self=None, center_of_other=None,
                intersection=0.56,
                iou=0.4, iom=0.9, obj=None),
            hier_object.Overlap(
                index=5, self_inside_other=False, center_of_self=False,
                other_inside_self=None, center_of_other=None,
                intersection=0.55,
                iou=0.9, iom=0.7, obj=None)]

    def setup_directories(self):
        """
        Return paths to result and reference directories.
        Creates result directory.
        Deletes old results directory if it exists.
        """
        this_dir = os.path.realpath(os.path.dirname(__file__))
        result = os.path.join(this_dir, 'Result')
        reference = os.path.join(this_dir, 'Reference')
        if not os.path.isdir(reference):
            raise Exception('Reference directory %s not found.' % reference)
        if self.RESULT_DIR_SETUP:
            if not os.path.isdir(result):
                raise Exception('Result directory %s not found.' % result)
        else:
            if os.path.isdir(result):
                shutil.rmtree(result)
            os.mkdir(result)
            self.RESULT_DIR_SETUP = True
        return result, reference

    def test10Create(self):
        """
        Test creating a HierObject object.
        """
        hobj1 = self.init_1()
        self.assertLess(abs(hobj1.area - 0.04), 0.0000001)

    def test50GetHierarchy(self):
        """
        Test get_hierarchy().
        """
        hobj1, hobj2, hobj3 = self.init_3()
        hier_object.get_hierarchy(hobj1.all, [('Object', 'Group')])
        self.assertIsNone(hobj1.children)
        self.assertEqual(hobj1.parent.get('Group'), 0)
        self.assertIsNone(hobj2.children)
        self.assertEqual(len(hobj2.parent), 0)
        self.assertEqual(hobj3.children['Object'][0], 0)
        self.assertIsNone(hobj3.parent)

    def test50Json(self):
        """
        Test creating JSON using HierObjectJSONEncoder.
        """
        hobj1 = self.init_1()
        s = json.dumps(
            hobj1, indent=1, sort_keys=True,
            cls=hier_object.HierObjectJSONEncoder)
        result_dir, reference_dir = self.setup_directories()
        result_path = os.path.join(result_dir, 'hobj1.json')
        reference_path = os.path.join(reference_dir, 'hobj1.json')
        with open(result_path, 'w') as f:
            f.write(s + '\n')
        self.assertTrue(filecmp.cmp(result_path, reference_path),
                        msg='Files are not the same:\n%s\n%s' % (
                            result_path, reference_path))

    def test50Bbox(self):
        """
        Test creating JSON using HierObjectJSONEncoder.
        """
        hobj1 = self.init_1()
        self.assertEqual(hobj1.bbox(), (0.2, 0.3, 0.4, 0.5))

    def test50Rect(self):
        """
        Test creating JSON using HierObjectJSONEncoder.
        """
        hobj1 = self.init_1()
        self.assertEqual(hobj1.rect(), (0.2, 0.3, 0.2, 0.2))

    def test50Str(self):
        """
        Test __str__ for HierObject.
        """
        hobj1 = self.init_1()
        self.assertEqual(
            str(hobj1),
            'x0=0.200 x1=0.400 y0=0.300 y1=0.500 lbl=Object 0 img=1')

    def test60SortIou(self):
        """
        Test sort_iou.
        """
        overlap_list = self.overlap_5()
        result = hier_object.sort_iou(overlap_list)
        self.assertEqual(tuple([i.index for i in result]), (2, 5, 1, 0, 3))
        result = hier_object.sort_iou(reversed(overlap_list))
        self.assertEqual(tuple([i.index for i in result]), (5, 2, 1, 0, 3))

    def test60SortForParent(self):
        """
        Test sort_for_parent.
        """
        overlap_list = self.overlap_5()
        result = hier_object.sort_for_parent(overlap_list)
        self.assertEqual(tuple([i.index for i in result]), (0, 2, 3, 1, 5))
        result = hier_object.sort_for_parent(reversed(overlap_list))
        self.assertEqual(tuple([i.index for i in result]), (0, 2, 3, 1, 5))

    def test70GetHierarchyJSON(self):
        """
        Test get_hierarchy().
        """
        hobj1, hobj2, hobj3 = self.init_3()
        hier_object.get_hierarchy(hobj1.all, [('Object', 'Group')])
        s = json.dumps(
            [hobj1, hobj2, hobj3],
            indent=1, sort_keys=True,
            cls=hier_object.HierObjectJSONEncoder)
        result_dir, reference_dir = self.setup_directories()
        result_path = os.path.join(result_dir, 'three_obj.json')
        reference_path = os.path.join(reference_dir, 'three_obj.json')
        with open(result_path, 'w') as f:
            f.write(s + '\n')
        self.assertTrue(filecmp.cmp(result_path, reference_path),
                        msg='Files are not the same:\n%s\n%s' % (
                            result_path, reference_path))

if __name__ == '__main__':
    unittest.main()

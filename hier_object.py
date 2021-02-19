#!/usr/bin/env python3

# =================================================================
# hier_object and related tests by https://github.com/SeeForTwo are
# licensed under a Creative Commons Attribution-NonCommercial 4.0
# International License.
# http://creativecommons.org/licenses/by-nc/4.0/
# =================================================================

"""
Given (computer vision) objects that were independently annotated,
compute relationships between objects so they can be organized
into a hierarchy.

Initially for "Open Images" style annotations.
Support skipping objects that have been annotated twice.
Assigns children (inside objects) to parents (outer objects) if there
an unambiguous choice for a child's parent (e.g. in a hierarchy like
an "eye" is the child of a "face" and a "face is the parent of an
"eye").

See test/Reference/hobj1.json and test/Reference/three_obj.json
for examples of this representation.
See more description in README.md.
"""

from collections import defaultdict, namedtuple
import json
from operator import attrgetter

# Describe overlap between an object and an another object.
Overlap = namedtuple('Overlap', [
    'index',                       # index of other object
    'self_inside_other',           # boolean flag
    'center_of_self',              # boolean flag
    'other_inside_self',           # boolean flag
    'center_of_other',             # boolean flag
    'intersection', 'iou', 'iom',  # area metrics
    'obj'])                        # other object

def create_all_dict():
    """
    Create a datastructure to hold all objects of all classes in all
    images (i.e a dict with key image identifier of dicts with key
    object label of lists of objects).
    """
    return defaultdict(lambda: defaultdict(list))

class HierObject(object):
    """
    Represent a computer vision object (with a bounding box and class label).

    Static member ALL collects all the objects. It is a dict with key
    image identifier of dicts with key object label of lists of objects.
    """

    default_all = None # for default datastucture to hold all objects

    def __init__(self, obj, all_obj=None, **extra):
        """
        Initialize object OBJ using the parse() method.
        ALL_OBJ can be created with create_all_dict() or if
        None a default datastructure will be used.
        EXTRA is passed to PARSE which can use or ignore it.

        Members set by parse():
          label is the class label for the object
          imageID is the identifier for the image
          x0, y0, x1, y1 are float bounding box coordinates
          w is x1 - x0
          h is y1 - y0

        Member set by __init__():
          obj is OBJ
          xc, yc are center X and Y coordinates
          area is w * h
          index is the index of the object in list
          ALL_OBJ[image ID][class label]

        Members set by get_hierarchy():
          overlap is dict with key class label of lists of Overlap namedtuples

        Members set by find_duplicates()
          duplicate is True if there is another object of the same
            class with high overlap, False otherwise. Priority
            goes to object with larger index.

        Members set by find_parent()
          parent is a dict with key other class label and value
            other class index for the parent of the object if
            exactly one parent found.
          children is dict with key other class label and value
            list of the class index for the children of the object.
        """
        self.obj = obj
        if all_obj is None:
            if HierObject.default_all is None:
                HierObject.default_all = create_all_dict()
            self.all = HierObject.default_all
        else:
            self.all = all_obj
        self.parse(obj, **extra)
        rn = lambda x: round(x, 5)
        self.xc = rn((self.x0 + self.x1) / 2.0)
        self.yc = rn((self.y0 + self.y1) / 2.0)
        self.area = self.w * self.h # do not round area
        assert self.area > 0.0 # avoid future divide-by-zero
        self.all[self.imageID][self.label].append(self)
        self.index = len(self.all[self.imageID][self.label]) - 1
        self.overlap = None
        self.duplicate = None
        self.parent = None
        self.children = None

    def parse(self, obj, **extra):
        """
        Parse an "openimages" annotation row for an object.
        EXTRA is not used for "openimages" annotations.

        A derived class can override this to parse other
        objects. Use EXTRA in derived classes if more
        information is needed (e.g. to pass in an image
        identifier if it is not in OBJ).
        """
        self.label = obj['LabelName']
        self.imageID = obj['ImageID']
        rn = lambda x: round(x, 5)
        self.x0 = rn(float(obj['XMin']))
        self.x1 = rn(float(obj['XMax']))
        self.y0 = rn(float(obj['YMin']))
        self.y1 = rn(float(obj['YMax']))
        self.w = rn(self.x1 - self.x0)
        assert self.w > 0.0
        self.h = rn(self.y1 - self.y0)
        assert self.h > 0.0

    def __str__(self):
        """
        Return a one line string representation of the object.
        """
        s = 'x0=%5.3f x1=%5.3f y0=%5.3f y1=%5.3f ' % (
            self.x0, self.x1, self.y0, self.y1)
        s += 'lbl=%s %d img=%s' % (
            str(self.label), self.index, str(self.imageID))
        return s

    def has_intersection(self, other):
        """
        Return True if this object and another intersect.
        """
        w = min(self.x1, other.x1) - max(self.x0, other.x0)
        h = min(self.y1, other.y1) - max(self.y0, other.y0)
        return (w > 0) and (h > 0)

    def is_inside(self, other):
        """
        Return True if this object is inside (or the same as) OTHER.
        """
        return ((self.x0 >= other.x0)
                and (self.x1 <= other.x1)
                and (self.y0 >= other.y0)
                and (self.y1 <= other.y1))

    def center_is_inside(self, other):
        """
        Return True if the center of this object is (strictly) inside OTHER.
        """
        return ((self.xc > other.x0)
                and (self.xc < other.x1)
                and (self.yc > other.y0)
                and (self.yc < other.y1))

    def intersection_area(self, other):
        """
        Compute the intersection area with another object.
        """
        w = min(self.x1, other.x1) - max(self.x0, other.x0)
        h = min(self.y1, other.y1) - max(self.y0, other.y0)
        if (w <= 0.0) or (h <= 0.0): # sanity check
            return 0.0
        return w * h

    def redundant_and_not_last(self, threshold=0.5):
        """
        Return True is there is another object with this label with
        IoU > THRESHOLD (default 0.5) (so they are the same, so they
        are redundant) and if the other object has a larger index. For
        keeping only one annotation for objects that are duplicates.
        """
        over = self.overlap.get(self.label)
        if over:
            for other in over:
                if (other.iou > threshold) and (other.index > self.index):
                    return True
        return False

    def bbox(self):
        """
        Return bounding box x0, y0, x1, y1
        """
        return self.x0, self.y0, self.x1, self.y1

    def rect(self):
        """
        Return rectangle x0, y0, w, h
        """
        return self.x0, self.y0, self.w, self.h

    def dict_for_json(self, overlap_info=True):
        """
        Create a dictionary with members (for creating JSON)
        If OVERLAP_INFO is True, include information in OVERLAP
        member.
        """
        d = self.__dict__.copy()
        del d['all']
        if not overlap_info:
            del d['overlap']
        if d.get('overlap') is not None:
            d['overlap'] = {
                label: [{k: v
                         for (k, v) in over._asdict().items()
                         if k != 'obj'}
                        for over in overlap_list]
                for label, overlap_list in d['overlap'].items()
                }
        return d

def clear_default_all():
    """
    Clear the class static variable use to hold all objects by default
    """
    HierObject.default_all = None

class HierObjectJSONEncoder(json.JSONEncoder):
    """
    A class for JSON encoding that handles HierObject objects.
    """
    def default(self, obj):
        """
        Use dict_for_json() method of HierObject objects to
        convert them to JSON.
        """
        if isinstance(obj, HierObject):
            return obj.dict_for_json()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

def get_hierarchy(all_obj, child_parent_list=None):
    """
    Compute hierarchy information on all objects in ALL_OBJ
    (i.e. self.all from any object, HierObject.default_all
    if default datastructure used).
    Calls get_overlap and find_parent.

    CHILD_PARENT_LIST is a list of tuples of (child class label,
    parent class label).
    """
    find_overlapping_objects(all_obj)
    find_duplicates(all_obj)
    if child_parent_list:
        for child_label, parent_label in child_parent_list:
            find_parent(all_obj, child_label, parent_label)

def find_overlapping_objects(all_obj):
    """
    Compute hierarchy information on all objects in ALL_OBJ
    (i.e. self.all from any object, HierObject.default_all
    if default datastructure used).
    """
    for object_dict in all_obj.values(): # image
        for object_list in object_dict.values(): # class label
            for obj in object_list:
                obj.overlap = defaultdict(list)
        flat = [
            obj
            for object_list in object_dict.values()
            for obj in object_list]
        for idx1 in range(len(flat)-1):
            for idx2 in range(idx1+1, len(flat)):
                compare(flat[idx1], flat[idx2])

def compare(obj1, obj2):
    """
    Save information comparing overlapping objects in member OVERLAP.
    """
    intersection_flag = obj1.has_intersection(obj2)
    if intersection_flag:
        center1 = obj1.center_is_inside(obj2)
        center2 = obj2.center_is_inside(obj1)
        inside1 = obj1.is_inside(obj2)
        inside2 = obj2.is_inside(obj1)
        intersection = obj1.intersection_area(obj2)
        # Intersection over Union, Jaccard similarity coefficient
        rn = lambda x: round(x, 5)
        iou = rn(intersection / (obj1.area + obj2.area - intersection))
        # Intersection over minimum area
        iom = rn(intersection / min(obj1.area, obj2.area))
        obj1.overlap[obj2.label].append(Overlap(
            index=obj2.index,
            self_inside_other=inside1, center_of_self=center1,
            other_inside_self=inside2, center_of_other=center2,
            intersection=rn(intersection),
            iou=iou, iom=iom, obj=obj2))
        obj2.overlap[obj1.label].append(Overlap(
            index=obj1.index,
            self_inside_other=inside2, center_of_self=center2,
            other_inside_self=inside1, center_of_other=center1,
            intersection=rn(intersection),
            iou=iou, iom=iom, obj=obj1))

def find_duplicates(all_obj):
    """
    Find duplicate objects and set DUPLICATE member of all
    but one object. Priority goes to larges index.
    Uses 0.5 as the IoU threshold.
    """
    for object_dict in all_obj.values(): # image
        for object_list in object_dict.values(): # class label
            for obj in object_list:
                obj.duplicate = obj.redundant_and_not_last()

def find_parent(all_obj, child_label, parent_label):
    """
    Find the parent of each child object, given class labels for each.
    Set the PARENT member if a child and append to the
    CHILDREN member of parent if exactly one parent found for the child.
    """
    for object_dict in all_obj.values(): # image
        child_list = object_dict.get(child_label)
        if not child_list:
            continue
        all_parent_class = object_dict.get(parent_label)
        if not all_parent_class:
            continue
        for parent in all_parent_class:
            if parent.duplicate:
                pass
            if parent.children is None:
                parent.children = defaultdict(list)
        for child in child_list:
            if child.parent is None:
                child.parent = {}
            if child.duplicate:
                continue
            parent_overlap_list = [ # get the overlap for each child object
                over
                for over in child.overlap.get(parent_label, [])
                if over.center_of_self and not over.obj.duplicate]
            if not parent_overlap_list:
                continue
            if len(parent_overlap_list) > 1:
                parent_overlap_list = sort_for_parent(parent_overlap_list)
                if parent_overlap_list[0].self_inside_other:
                    # if one child is completely inside, ignore any
                    # others where just the center of the child is inside
                    parent_overlap_list = [
                        i for i in parent_overlap_list if i.self_inside_other]
            if len(parent_overlap_list) == 1:
                parent = parent_overlap_list[0].obj
                child.parent[parent_label] = parent.index
                parent.children[child_label].append(child.index)

def sort_iou(overlap_list):
    """
    Return a copy of OVERLAP_LIST sorted by IoU (for object similarity,
    e.g. detecting duplicate annotations).
    Largest IoU first.
    """
    return sorted(overlap_list, reverse=True, key=attrgetter('iou'))

def sort_for_parent(overlap_list):
    """
    Return a copy of OVERLAP_LIST sorted by "self_inside_other",
    "center_of_self", and IoM (for determining if object is a child of
    another object).  Inside/high overlap first, minimal overlap last.
    """
    return sorted(overlap_list, reverse=True,
                  key=lambda x: (
                      int(x.self_inside_other), int(x.center_of_self),
                      x.iom))

if __name__ == '__main__':
    print('Example for hier_object library...')
    hobj1 = HierObject(dict(
        ImageID='i1', LabelName='Object',
        XMin='0.2', XMax='0.4', YMin='0.3', YMax='0.5'))
    hobj2 = HierObject(dict(
        ImageID='i1', LabelName='Object',
        XMin='0.1', XMax='0.3', YMin='0.4', YMax='0.8'))
    hobj3 = HierObject(dict(
        ImageID='i1', LabelName='Group',
        XMin='0.1', XMax='0.5', YMin='0.1', YMax='0.5'))
    assert hobj1 == hobj3.all['i1']['Object'][0]
    assert hobj2 == hobj3.all['i1']['Object'][1]
    assert hobj3 == hobj1.all['i1']['Group'][0]
    get_hierarchy(hobj1.all, [('Object', 'Group')])
    assert hobj3.children.get('Object') == [0]
    assert hobj1.parent.get('Group') == 0
    assert hobj2.parent.get('Group') is None
    print(json.dumps(
        hobj1, indent=1, sort_keys=True, cls=HierObjectJSONEncoder))
    print('Done')


# =================================================================
# hier_object and related tests by https://github.com/SeeForTwo are
# licensed under a Creative Commons Attribution-NonCommercial 4.0
# International License.
# http://creativecommons.org/licenses/by-nc/4.0/
# =================================================================

"""
Create a Hypothesis library strategy described by
ObjectStrategyParam namedtuples.

See README.md for more information.
"""

from collections import defaultdict, namedtuple
from itertools import product

try:
    import hypothesis.strategies as st
except ImportError as ex:
    print('\nImportError')
    print('Install package "python3-hypothesis" for "hypothesis"?')
    print('Or see http://hypothesis.works/ for how to get this library')
    print('')
    raise ex

from bbox_utils import get_bbox, center_is_inside, object_is_inside
from bbox_utils import calc_intersection, calc_iou
from allocate_space import create_object_inside, create_overlapping_object
from allocate_space import create_duplicate, split_bbox
from cls_test_info import TEST_INFO

ObjectStrategyParam = namedtuple(
    'ObjectStrategyParam',
    ['label',    # class label string or None for an overlapping group
     'obj_min',  # minimum number of objects per parent
     'obj_max',  # maximum number of objects per parent
     'dup_min',  # maximum number of duplicate annotations per object
     'dup_max',  # minimum number of duplicate annotations per object
     'min_size', # minimum width, height (float, >0.0, <1.0, or None)
                 # used to terminate splitting when allocating space
     'suggested_min_size', # minimum width, height (float, >0.0, <1.0, or None)
                           # used to prioritize splitting when allocating space
     'ratio'     # maximum width/height is parent width/height * ratio
    ])

GROUP_PARAM = ObjectStrategyParam(
    label=None, obj_min=1, obj_max=20, dup_min=None, dup_max=None,
    min_size=0.02, suggested_min_size=0.1, ratio=0.99)
HEAD_PARAM = ObjectStrategyParam(
    label='head', obj_min=1, obj_max=5, dup_min=0, dup_max=1,
    min_size=0.02, suggested_min_size=0.1, ratio=0.99)
FACE_ROOT_PARAM = ObjectStrategyParam(
    label='face', obj_min=1, obj_max=5, dup_min=0, dup_max=1,
    min_size=0.02, suggested_min_size=0.1, ratio=1.0)
FACE_INSIDE_PARAM = ObjectStrategyParam(
    label='face', obj_min=1, obj_max=1, dup_min=0, dup_max=1,
    min_size=0.02, suggested_min_size=0.1, ratio=0.99)
EYE_PARAM = ObjectStrategyParam(
    label='eye', obj_min=0, obj_max=3, dup_min=0, dup_max=1,
    min_size=0.01, suggested_min_size=0.05, ratio=0.5)
NOSE_PARAM = ObjectStrategyParam(
    label='nose', obj_min=0, obj_max=2, dup_min=0, dup_max=1,
    min_size=0.01, suggested_min_size=0.05, ratio=0.5)
LEAF_PARAM = (EYE_PARAM, NOSE_PARAM)

GROUP_TEST_PARAM = ObjectStrategyParam(
    label=None, obj_min=3, obj_max=3, dup_min=None, dup_max=None,
    min_size=0.02, suggested_min_size=0.1, ratio=0.99)
HEAD_TEST_PARAM = ObjectStrategyParam(
    label='head', obj_min=2, obj_max=2, dup_min=0, dup_max=0,
    min_size=0.02, suggested_min_size=0.1, ratio=0.99)
EYE_TEST_PARAM = ObjectStrategyParam(
    label='eye', obj_min=2, obj_max=2, dup_min=0, dup_max=0,
    min_size=0.01, suggested_min_size=0.05, ratio=0.5)

# HIERARCHY_PARAM is described in README.md
HIERARCHY_PARAM = [
    [GROUP_PARAM, HEAD_PARAM, LEAF_PARAM],
    [GROUP_PARAM, FACE_ROOT_PARAM, LEAF_PARAM],
    [GROUP_PARAM, HEAD_PARAM, FACE_INSIDE_PARAM, LEAF_PARAM],
    [LEAF_PARAM]]

HIERARCHY_TEST_PARAM = [[GROUP_TEST_PARAM, HEAD_TEST_PARAM, EYE_TEST_PARAM]]

# An ObjectParam namedtuple contains the parameters for generating
# annotations for one object (in contrast to ObjectStrategyParam which
# contains parameters for generating annotations for a number of objects).
# The object_strategy() function converts an ObjectStrateyParm
# into a list of ObjectParam namedtuple.
ObjectParam = namedtuple(
    'ObjectParam',
    ['label',    # class label
     'dup',  # duplicate annotations per object
     'min_size', # minimum width, height (float, >0.0, <1.0_, or None)
                 # used to terminate splitting when allocating space
     'suggested_min_size', # minimum width, height (float, >0.0, <1.0, or None)
                           # used to prioritize splitting when allocating space
     'ratio'    # maximum width/height is parent width/height * ratio
    ])

def object_strategy(osp):
    """
    Return a hypothesis library strategy described by
    a ObjectStrategyParam namedtuple.
    This corresponds to a sequence of ObjectParam namedtuple.
    """
    return st.lists(
        st.tuples( # This tuple is in the same order as ObjectParam
            st.just(osp.label),
            st.just(None) if osp.dup_min is None
            else st.integers(min_value=osp.dup_min, max_value=osp.dup_max),
            st.just(osp.min_size),
            st.just(osp.suggested_min_size),
            st.just(osp.ratio)),
        min_size=osp.obj_min,
        max_size=osp.obj_max)

def split_helper(draw, op_list, total_bbox):
    """
    Split the area in bounding box TOTAL_BBOX into one part
    for each item in OP_LIST, which is list of ObjectParam named tuples.

    DRAW is provided by the hypothesis library and is used
    to choose parameters.
    """
    min_size = None
    suggested_min_size = None
    for op in op_list:
        if min_size is None:
            min_size = op.min_size
        else:
            min_size = max(min_size, op.min_size)
        if suggested_min_size is None:
            suggested_min_size = op.suggested_min_size
        else:
            suggested_min_size = max(
                suggested_min_size, op.suggested_min_size)
    bbox_list = [total_bbox]
    if len(op_list) > 1:
        bbox_list = split_bbox(
            draw, total_bbox, len(op_list), min_size, suggested_min_size)
    return bbox_list

def create_object_and_duplicates(draw, objects, overlap_flag, op, o_bbox,
                                 p_bbox, pc_list):
    """
    Create an object and any duplicates of that object.

    DRAW is provided by the hypothesis library and is used
    to choose parameters.

    Objects (row dictionaries) are appended to list OBJECTS.

    Overlapping objects using bounding box O_BBOX are created if
    OVERLAP_FLAG is True, otherwise non-overlapping using bounding box
    P_BBOX are used. OP is an ObjectParam namedtuple describing how to
    create the object(s).

    PC_LIST is a list of indices of objects in an overlapping group,
    i.e. objects that could be parents, children or duplicates of another
    object in the list.

    Returns the intersection of the bounding boxes of the objects.

    """
    cur_indexes = [len(objects)]
    if not overlap_flag:
        p = create_object_inside(
            draw, o_bbox, op.label, ratio=op.ratio, objects=objects,
            pc_list=pc_list)
        max_bbox = o_bbox
    else:
        p = create_overlapping_object(
            draw, o_bbox, p_bbox, op.label,
            objects=objects, pc_list=pc_list)
        max_bbox = p_bbox
    prev_bbox = get_bbox(p)
    intersection_bbox = prev_bbox
    if op.dup is not None:
        for dup_idx_ignored in range(op.dup):
            cur_indexes.append(len(objects))
            p = create_duplicate(
                draw, prev_bbox, op.label, objects, max_bbox=max_bbox,
                pc_list=pc_list)
            intersection_bbox = calc_intersection(
                intersection_bbox, get_bbox(p))
        for i in cur_indexes:
            objects[i][TEST_INFO].duplicates_as_idx = [
                j for j in cur_indexes if j != i]
    return intersection_bbox

def create_parents_and_children(draw, p_list, p_bbox, objects,
                                parents_children,
                                overlap_cur=False, group_idx=None):
    """
    Create objects (row dictionaries) for parents and then recursively
    call this function to create objects for all their children.

    P_LIST describes the objects using a nested lists of ObjectParam.

    P_BBOX is the bounding box for all parents. If OVERLAP_CUR is
    True, created objects overlap.

    Objects are appended to list OBJECTS.

    PARENTS_CHILDREN is a dict with key arbitrary index and value that
    is a list of objects that potentially could be parents or
    children of each other. The keys separate objects that are known
    to not overlap based on how they are generated.

    DRAW is provided by the hypothesis library and is used to choose
    parameters.
    """
    level = p_list[0]
    # The level of the hierarchy:
    #   'head' is one level (e.g. p_list[0])
    #   'eye' and 'nose' are another level (e.g. p_list[1])

    # Create a list ObjectParam named tuples for the randomly
    # generated (by DRAW) objects at this level
    op_list = [
        ObjectParam(*op_tuple)
        for obj_st_param in level
        for op_tuple in draw(object_strategy(obj_st_param))]
    o_bbox_list = split_helper(draw, op_list, p_bbox)
    next_group_idx = group_idx
    if group_idx is None:
        next_group_idx = 0
    for op, o_bbox in zip(op_list, o_bbox_list):
        # If o_bbox_list is shorter that op_list, zip will
        # ignore extra elements of op_list, which will happen
        # if there is not enough space for all the requested objects
        # in op_list
        if op.label is None:
            # group of objects, not an actual object
            intersection_bbox = o_bbox
        else:
            intersection_bbox = create_object_and_duplicates(
                draw, objects, overlap_cur, op, o_bbox, p_bbox,
                pc_list=parents_children[group_idx])
        if len(p_list) > 1:
            overlap_flag = (op.label is None)
            create_parents_and_children(
                draw, p_list[1:], intersection_bbox, objects,
                parents_children,
                overlap_cur=overlap_flag,
                group_idx=next_group_idx)
        if (group_idx is None) and not overlap_cur:
            next_group_idx += 1

def find_parents_children(objects, parents_children, params):
    """
    Find parents and children of objects.

    OBJECTS is a list of objects (row dictionaries).

    PARENTS_CHILDREN is a dict with key arbitrary index and value that
    is a list of objects that potentially could be parents or
    children of each other. The keys separate objects that are known
    to not overlap based on how they are generated.

    PARAMS describes the objects using a nested lists of ObjectParam.
    """
    # Find class labels for possible child-parent pairs based on PARAMS
    child_parent_set = set()
    for idx1 in range(len(params)-1):
        level1 = [i for i in params[idx1] if i.label is not None]
        if not level1:
            continue
        for idx2 in range(idx1+1, len(params)):
            level2 = [i for i in params[idx2] if i.label is not None]
            if not level2:
                continue
            for parent, child in product(level1, level2):
                child_parent_set.add((child.label, parent.label))
    for pc_list in parents_children.values():
        for child_label, parent_label in child_parent_set:
            children = [i for i in pc_list
                        if objects[i]['LabelName'] == child_label]
            if not children:
                continue
            parents = [i for i in pc_list
                       if objects[i]['LabelName'] == parent_label]
            if not parents:
                continue
            for ci in children:
                pi_lst = [pi for pi in parents
                          if object_is_inside(objects[ci], objects[pi])]
                if not pi_lst:
                    pi_lst = [pi for pi in parents
                              if center_is_inside(objects[ci], objects[pi])]
                if pi_lst:
                    c_ti = objects[ci][TEST_INFO]
                    c_ti.parents_as_idx[parent_label].extend(pi_lst)
                    for pi in pi_lst:
                        p_ti = objects[pi][TEST_INFO]
                        p_ti.children_as_idx[child_label].append(ci)

def find_additional_duplicates(objects, parents_children):
    """
    Find parents and children of objects.

    OBJECTS is a list of objects (row dictionaries).

    PARENTS_CHILDREN is a dict with key arbitrary index and value that
    is a list of objects that potentially chould be parents or
    children of each other. The keys separate objects that are known
    to not overlap based on how they are generated.
    """
    for pc_list in parents_children.values():
        label_set = set([objects[i]['LabelName'] for i in pc_list])
        for label in label_set:
            indexes = [i for i in pc_list
                       if objects[i]['LabelName'] == label]
            n = len(indexes)
            for idx1 in range(n-1):
                obj_idx1 = indexes[idx1]
                for idx2 in range(idx1+1, n):
                    obj_idx2 = indexes[idx2]
                    if obj_idx2 in objects[obj_idx1][TEST_INFO].duplicates_as_idx:
                        continue
                    if calc_iou(objects[obj_idx1], objects[obj_idx2]) > 0.5:
                        objects[obj_idx1][TEST_INFO].duplicates_as_idx.append(obj_idx2)
                        objects[obj_idx2][TEST_INFO].duplicates_as_idx.append(obj_idx1)

@st.composite
def hier_object_data(draw, param_list=None, total_bbox=None):
    """
    Pseudo randomly generate data than models annotations
    of objects.

    DRAW is provided by the hypothesis library and is used
    to choose parameters.

    PARAM_LIST is a list of lists of ObjectStrategyParam namedtuples or a list
    of lists of tuples of ObjectStrategyParam namedtuples.
    The default for PARAM_LIST is HIERARCHY_PARAM.

    TOTAL_BBOX is a bounding box of all objects and defaults to
    (0.0, 0.0, 1.0, 1.0).

    See README.md for more explanation.
    """
    if param_list is None:
        param_list = HIERARCHY_PARAM
    # for consistency, make always param_list be a list of lists of tuples
    param_list = [
        [tuple([j]) if isinstance(j, ObjectStrategyParam) else j
         for j in i]
        for i in param_list]
    params = draw(st.one_of([st.just(i) for i in param_list]))
    if total_bbox is None:
        total_bbox = (0.0, 0.0, 1.0, 1.0)
    objects = []
    parents_children = defaultdict(list)
    create_parents_and_children(
        draw, params, total_bbox, objects, parents_children)
    find_parents_children(objects, parents_children, params)
    find_additional_duplicates(objects, parents_children)
    objects = draw(st.permutations(objects)) # shuffle
    return objects

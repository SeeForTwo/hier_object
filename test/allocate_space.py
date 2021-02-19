
# =================================================================
# hier_object and related tests by https://github.com/SeeForTwo are
# licensed under a Creative Commons Attribution-NonCommercial 4.0
# International License.
# http://creativecommons.org/licenses/by-nc/4.0/
# =================================================================

"""
Allocate space for objects.

split_bbox() is used to divide space prior to creation of objects at the
same hierarchy level.

create_object_inside() and create_overlapping_object() create a child
object in the space of a parent object, i.e. create an object at a
lower hierarchy level in the space of an object at a higher hierarchy
level.

create_duplicate() creates an object in the same space as another
object at the same hierarchy level.
"""

import heapq

try:
    import hypothesis.strategies as st
except ImportError as ex:
    print('\nImportError')
    print('Install package "python3-hypothesis" for "hypothesis"?')
    print('Or see http://hypothesis.works/ for how to get this library')
    print('')
    raise ex

from cls_test_info import TEST_INFO, TestInfo
from bbox_utils import calc_intersection

DEFAULT_IMAGE_ID = '1'

def draw_plus_minus_one(draw):
    """
    Return a random floating point value between -1.0 and 1.0
    (inclusive) using DRAW provided by the hypothesis library.

    While the hypothesis library can generate pseudo random floats
    these "have a complicated and hard to explain shrinking behavior."
    Fractions shrink to towards smaller denominators and then towards
    zero, the middle of the -1.0 to 1.0 interval.
    """
    return float(draw(st.fractions(
        min_value=-1, max_value=1, max_denominator=32)))

def draw_zero_one(draw):
    """
    Return a random floating point value between 0.0 and 1.0 (inclusive)
    using DRAW provided by the hypothesis library. The value shrinks
    to 0.0.
    """
    return float(draw(st.fractions(
        min_value=0, max_value=1, max_denominator=32)))

def split_bbox(draw, start_bbox, max_splits, min_size=None,
               suggested_min_size=None):
    """
    Splits bounding box (rectangle) START_BBOX, which is a bounding
    box tuple of (x0, y0, x1, y1) describing an axis aligned rectangle
    into at most MAX_SPLITS (axis aligned) rectangles that have width
    >= MIN_SIZE and HEIGHT >= MIN_SIZE. Splits are pseudo random based
    on DRAW provided by the hypothesis library.  The resulting
    rectangles do not overlap and their union is start_bbox.

    The return value is a list of bounding box tuples of
    (x0, y0, x1, y0) floats. The list of bounding boxes is sorted so
    largest area is first.

    Uses recursive guillotine cuts (that split a rectangle
    horizontally or vertically into two rectangles).

    Consider the example where START_BBOX is (0.0, 0.0, 4.0, 4.0),
    MAX_SPLITS is 100 and MIN_SIZE = 1.0. It is possible that the
    pseudo random DRAW will choose 1.0, 2.0 and 3.0 for cuts in both
    dimensions and 4*4 = 16 rectangle are returned.  If other cuts are
    chosen, then as few as 3*3 = 9 rectangles are returned. So in this
    case, between 9 and 16 rectangles inclusive would be returned.
    """
    very_small = 0.0000001
    min_size2 = None
    if min_size is not None:
        min_size2 = 2.0 * min_size
        very_small = min_size / 100.0
    else:
        min_size = very_small
    suggested_min_size2 = min_size2
    if suggested_min_size is not None:
        suggested_min_size2 = 2.0 * suggested_min_size
        if min_size2 is not None:
            suggested_min_size2 = max(min_size2, suggested_min_size2)
    sr = tuple(map(float, start_bbox))
    start_area = (sr[2] - sr[0]) * (sr[3] - sr[1])
    # rec_heap is a tuple of (-area, bounding_box). -area is used
    # for sorting. Python's heap starts with the smallest element,
    # which in this case is the one with the largest area.
    rec_heap = [(-start_area, sr)]
    while len(rec_heap) < max_splits:
        ignore_area, bbox = heapq.heappop(rec_heap)
        # decide whether to split the height/horizontally
        # or the width/vertically
        p_x0, p_y0, p_x1, p_y1 = bbox
        p_w = p_x1 - p_x0
        p_h = p_y1 - p_y0
        assert p_w > 0.00001
        assert p_h > 0.00001
        if (min_size2 is not None) and (max(p_w, p_h) < min_size2):
            break # cannot cut largest area rectangle, stop splitting
        min_hw = min(p_w, p_h)
        if ((suggested_min_size2 is not None)
                or (min_hw < suggested_min_size2)):
            # cannot cut so each is larger than suggested_min_size,
            # so decrease suggested_min_size
            suggested_min_size2 = 0.75 * min_hw
            if min_size2 is not None:
                suggested_min_size2 = max(min_size2, suggested_min_size2)
        if ((suggested_min_size2 is None)
                or (min_hw >= suggested_min_size2)):
            # can split in either dimension, so choose pseudo randomly
            hor_cut = draw(st.booleans())
        else:
            hor_cut = p_h > p_w # can only cut larger dimension
        if hor_cut:
            center = (p_y0 + p_y1) / 2.0
            middle = p_h / 2.0 - min_size # cut can be +/- middle from center
            if middle < very_small:
                cut = center
            else:
                assert middle > 0
                cut = center + middle * draw_plus_minus_one(draw)
            assert cut - p_y0 > 0.00001
            assert p_y1 - cut > 0.00001
            new1 = (- p_w * (cut - p_y0), (p_x0, p_y0, p_x1, cut))
            new2 = (- p_w * (p_y1 - cut), (p_x0, cut, p_x1, p_y1))
        else:
            center = (p_x0 + p_x1) / 2.0
            middle = p_w / 2.0 - min_size # cut can be +/- middle from center
            if middle < very_small:
                cut = center
            else:
                assert middle > 0
                cut = center + middle * draw_plus_minus_one(draw)
            assert cut - p_x0 > 0.00001
            assert p_x1 - cut > 0.00001
            new1 = (- (cut - p_x0) * p_h, (p_x0, p_y0, cut, p_y1))
            new2 = (- (p_x1 - cut) * p_h, (cut, p_y0, p_x1, p_y1))
        heapq.heappush(rec_heap, new1)
        heapq.heappush(rec_heap, new2)
    return [i[1] for i in sorted(rec_heap)]

def create_object_with_bbox(label, o_x0, o_y0, o_x1, o_y1, objects, pc_list):
    """
    Create an object with class label LABEL, bounding box
    (o_x0, o_y0), (o_x1, o_y1).
    The object is appended to list OBJECTS.
    The index of the object is appended to PC_LIST.
    """
    r5 = lambda x: round(x, 5)
    assert o_x0 < o_x1
    assert o_y0 < o_y1
    r_x0 = r5(o_x0)
    r_y0 = r5(o_y0)
    r_x1 = r5(o_x1)
    r_y1 = r5(o_y1)
    assert r_x0 < r_x1
    assert r_y0 < r_y1
    obj = dict(
        ImageID=DEFAULT_IMAGE_ID,
        LabelName=label,
        XMin=str(r_x0),
        XMax=str(r_x1),
        YMin=str(r_y0),
        YMax=str(r_y1))
    idx = len(objects)
    objects.append(obj)
    pc_list.append(idx)
    ti = TestInfo(
        bbox=(o_x0, o_y0, o_x1, o_y1), label=label, idx=idx)
    obj[TEST_INFO] = ti
    return obj

def create_object_inside(draw, p_bbox, label, ratio, objects,
                         pc_list=None):
    """
    Create a class LABEL object that is inside bounding box P_BBOX.

    DRAW is provided by the hypothesis library and is used
    to choose parameters.

    The maximum width and height of the created object is RATIO *
    width and RATIO * height of P_BBOX respectively.
    RATIO must be less than or equal to 1.0 and greater than 0.0.

    The minimum width and height of the created object
    is 1/5 of the maximum width and height respectively.

    The object is appended to list OBJECTS.

    PC_LIST is passed to create_object_with_bbox.

    """
    assert ratio > 0.0
    assert ratio <= 1.0
    min_ratio = 5.0

    p_x0, p_y0, p_x1, p_y1 = p_bbox
    p_w = p_x1 - p_x0
    p_h = p_y1 - p_y0
    assert p_w > 0.00001
    assert p_h > 0.00001

    max_w = ratio * p_w
    min_w = max_w / min_ratio
    max_value = max_w - min_w
    x_offset = (p_w - (min_w + max_value * draw_zero_one(draw))) / 2.0

    max_h = ratio * p_h
    min_h = max_h / min_ratio
    max_value = max_h - min_h
    y_offset = (p_h - (min_h + max_value * draw_zero_one(draw))) / 2.0

    o_x0 = p_x0 + x_offset
    o_x1 = p_x1 - x_offset
    o_y0 = p_y0 + y_offset
    o_y1 = p_y1 - y_offset
    return create_object_with_bbox(
        label, o_x0, o_y0, o_x1, o_y1, objects, pc_list)

def create_overlapping_object(draw, p_bbox, group_bbox, label, objects,
                              pc_list=None):
    """
    Create a class LABEL object in group_bbox with center in P_BBOX
    and some portion in GROUP_BBOX but not in P_BBOX. P_BBOX must
    be inside and smaller than GROUP_BBOX.

    DRAW is provided by the hypothesis library and is used
    to choose parameters.

    The object is appended to list OBJECTS.

    PC_LIST is passed to create_object_with_bbox.
    """
    p_x0, p_y0, p_x1, p_y1 = p_bbox
    p_w = p_x1 - p_x0
    p_h = p_y1 - p_y0
    g_x0, g_y0, g_x1, g_y1 = group_bbox
    g_w = g_x1 - g_x0
    g_h = g_y1 - g_y0

    assert g_x0 <= p_x0
    assert g_y0 <= p_y0
    assert g_x1 >= p_x1
    assert g_y1 >= p_y1

    overlap_param = 1.4 # how much bigger object is than P_BBOX
    margin_param = 0.9 # how much smaller object is than GROUP_BBOX
    w = min(overlap_param * p_w, margin_param * g_w)
    x_margin = w / 20.0
    w2 = w / 2.0 - x_margin
    w8 = w2 / 4.0
    h = min(overlap_param * p_h, margin_param * g_h)
    y_margin = h / 20.0
    h2 = h / 2.0 - y_margin
    h8 = h2 / 4.0
    min_xc = max(g_x0 + w2, p_x0 + w8) + x_margin
    min_yc = max(g_y0 + h2, p_y0 + h8) + y_margin
    max_xc = min(g_x1 - w2, p_x1 - w8) - x_margin
    max_yc = min(g_y1 - h2, p_y1 - h8) - y_margin

    # see split_bbox() for discussion of st.fractions
    if min_xc < max_xc:
        xc = min_xc + draw_zero_one(draw) * (max_xc - min_xc)
    else: # handle min_xc == max_xc
        xc = (min_xc + max_xc) / 2.0
    if min_yc < max_yc:
        yc = min_yc + draw_zero_one(draw) * (max_yc - min_yc)
    else: # handle min_xc == max_xc
        yc = (min_yc + max_yc) / 2.0
    o_x0 = xc - w2
    o_x1 = xc + w2
    o_y0 = yc - h2
    o_y1 = yc + h2

    # check that object is mostly in p_bbox
    i_x0, i_y0, i_x1, i_y1 = calc_intersection(
        (o_x0, o_y0, o_x1, o_y1), p_bbox)
    area_i = (i_x1 - i_x0) * (i_y1 - i_y0)
    area_p = (p_x1 - i_x0) * (p_y1 - p_y0)
    assert area_i > (0.6 * area_p)

    return create_object_with_bbox(
        label, o_x0, o_y0, o_x1, o_y1, objects, pc_list)

def create_duplicate(draw, bbox, label, objects,
                     max_bbox=None, min_iou=0.8, pc_list=None):
    """
    Create a class LABEL object that is a duplicate of the object with
    BBOX and LABEL, with IoU (intersect / union, Jaccard similarity
    coefficient) at least MIN_IOU. The object is appended to list OBJECTS.

    If max_bbox is not None, the duplicate is contained in bbox.
    PC_LIST is passed to create_object_with_bbox.
    """
    p_x0, p_y0, p_x1, p_y1 = bbox
    assert p_x0 < p_x1
    assert p_y0 < p_y1
    p_w = p_x1 - p_x0
    p_h = p_y1 - p_y0
    p_area = p_w * p_h
    p_xc = (p_x1 + p_x0) / 2.0
    p_yc = (p_y1 + p_y0) / 2.0

    # see split_bbox() for discussion of st.fractions
    dx0 = p_w * 0.1
    dy0 = p_h * 0.1
    dx1 = dx0
    dy1 = dy0
    assert min_iou < 0.9999
    assert min_iou > 0.4
    f = 1.0 - min_iou / 2.0
    if max_bbox is not None:
        m_x0, m_y0, m_x1, m_y1 = bbox
        dx0 = min(dx0, abs(m_x0 - p_x0) * f, abs(p_xc - p_x0) * f)
        dy0 = min(dy0, abs(m_y0 - p_y0) * f, abs(p_yc - p_y0) * f)
        dx1 = min(dx1, abs(m_x1 - p_x1) * f, abs(p_xc - p_x1) * f)
        dy1 = min(dy1, abs(m_y1 - p_y1) * f, abs(p_yc - p_y1) * f)

    dx0 = dx0 * draw_plus_minus_one(draw)
    dy0 = dy0 * draw_plus_minus_one(draw)
    dx1 = dx1 * draw_plus_minus_one(draw)
    dy1 = dy1 * draw_plus_minus_one(draw)

    while 1:
        # check IoU
        o_x0 = p_x0 + dx0
        o_y0 = p_y0 + dy0
        o_x1 = p_x1 + dx1
        o_y1 = p_y1 + dy1
        o_area = max(0.0, o_x1 - o_x0) * max(0.0, o_y1 - o_y0)

        i_x0, i_y0, i_x1, i_y1 = calc_intersection(
            (o_x0, o_y0, o_x1, o_y1), bbox)
        i_area = (i_x1 - i_x0) * (i_y1 - i_y0)
        iou = i_area / (p_area + o_area - i_area)
        if iou >= min_iou:
            break

        # if IoU too small, reduce largest of dx0, dy0, dx1, dy1
        max_diff = max(map(abs, [dx0, dy0, dx0, dy0]))
        if abs(dx0) == max_diff:
            dx0 *= 0.7
        if abs(dy0) == max_diff:
            dy0 *= 0.7
        if abs(dx1) == max_diff:
            dx1 *= 0.7
        if abs(dy1) == max_diff:
            dy1 *= 0.7
    return create_object_with_bbox(
        label, o_x0, o_y0, o_x1, o_y1, objects, pc_list)

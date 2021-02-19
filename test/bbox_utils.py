
# =================================================================
# hier_object and related tests by https://github.com/SeeForTwo are
# licensed under a Creative Commons Attribution-NonCommercial 4.0
# International License.
# http://creativecommons.org/licenses/by-nc/4.0/
# =================================================================

"""
Utilities related to the bounding boxes for objects.
"""

def get_bbox(row):
    """
    Given a row dictionary describing an object, return its bounding
    box (x0, y0, x1, y1) as a tuple of floats.
    """
    return tuple(
        map(float,
            (row['XMin'], row['YMin'], row['XMax'], row['YMax'])))

def get_center(row):
    """
    Given a row dictionary describing an object, return the center of
    its bounding box (xc, yc) as a tuple of floats.
    """
    x0, y0, x1, y1 = get_bbox(row)
    xc = (x0 + x1) / 2.0
    yc = (y0 + y1) / 2.0
    return (xc, yc)

def center_is_inside(row1, row2):
    """
    Return True if center of bounding ROW1 is inside ROW2.
    """
    xc, yc = get_center(row1)
    x0, y0, x1, y1 = get_bbox(row2)
    return (x0 <= xc) and (xc <= x1) and (y0 <= yc) and (yc <= y1)

def object_is_inside(row1, row2):
    """
    Return True if bounding box of ROW1 is inside bounding box of ROW2.
    """
    x0a, y0a, x1a, y1a = get_bbox(row1)
    x0b, y0b, x1b, y1b = get_bbox(row2)
    return (x0b <= x0a) and (x1a <= x1b) and (y0b <= y0a) and (y1a <= y1b)

def calc_intersection(bbox1, bbox2):
    """
    Compute the intersection of two bounding boxes (x0, y0, x1, y1).
    Either bounding box may be None.
    """
    if not bbox1:
        return bbox2
    if not bbox2:
        return bbox1
    x0a, y0a, x1a, y1a = bbox1
    x0b, y0b, x1b, y1b = bbox2
    x0 = max(x0a, x0b)
    y0 = max(y0a, y0b)
    x1 = min(x1a, x1b)
    y1 = min(y1a, y1b)
    if x1 < x0:
        x1 = x0
    if y1 < y0:
        y1 = y0
    return (x0, y0, x1, y1)

def area(bbox):
    """
    Return the area of a bounding box (x0, y0, x1, y1).
    """
    x0, y0, x1, y1 = bbox
    if x1 <= x0:
        return 0.0
    if y1 <= y0:
        return 0.0
    return (x1-x0) * (y1-y0)

def calc_iou(row1, row2):
    """
    Compute the Intersection over Union, Jaccard similarity
    coefficient of two row dictionaries describing objects.
    The value is rounding to 5 decimal places.
    """
    bbox1 = get_bbox(row1)
    bbox2 = get_bbox(row2)
    rn = lambda x: round(x, 5)
    area1 = area(bbox1)
    area2 = area(bbox2)
    intersection = area(calc_intersection(bbox1, bbox2))
    iou = rn(intersection / (area1 + area2 - intersection))
    return iou

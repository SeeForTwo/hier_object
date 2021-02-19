
# =================================================================
# hier_object and related tests by https://github.com/SeeForTwo are
# licensed under a Creative Commons Attribution-NonCommercial 4.0
# International License.
# http://creativecommons.org/licenses/by-nc/4.0/
# =================================================================

"""
Definition for TestInfo class used as structure to collect information
used in testing.
"""

from collections import defaultdict
import json

# Constant strings for dict indexing
TEST_INFO = 'TestInfo'

class TestInfo(object):
    """
    Class used as structure to collect information used in testing.
    """
    def __init__(self, bbox=None, label=None, idx=None, previous=None):
        """
        Create a TestInfo object initialized with optional bounding box
        (BBOX) coordinates (x0, y0, x1, y1) and/or option class label
        string LABEL and/or index IDX or optional dict of PREVIOUS
        members.
        """
        r5 = lambda x: round(x, 5)

        # center coordinates
        self.xc = None
        self.yc = None
        if bbox:
            x0, y0, x1, y1 = bbox
            self.xc = r5((x0 + x1)/2.0)
            self.yc = r5((y0 + y1)/2.0)

        self.label = label

        # Index for this object in various roles
        self.idx = idx

        # Dictionaries with key class names and value List indexes of
        # other related objects.
        # These are indexes, not objects, so __repr__ of these can be
        # used later as Python examples or serialized to JSON.
        self.parents_as_idx = defaultdict(list)
        self.children_as_idx = defaultdict(list)
        # List of indexes of duplicate objects.
        self.duplicates_as_idx = []

        self.obj = None # class HierObject object

        if previous:
            self.load(previous)

    def load(self, d):
        """
        Load members from a dict D.
        D may be a dict or a JSON string of a dict.
        """
        if isinstance(d, str):
            d = json.loads(d)
        for k, v in d.items():
            if k in self.__dict__:
                self.__dict__[k] = v

    def __repr__(self):
        """
        Return a string with a JSON compatible dict of members.
        Ignores OBJ member.
        """
        return json.dumps(
            {k:v for (k, v) in self.__dict__.items() if k != 'obj'},
            indent=1, sort_keys=True)

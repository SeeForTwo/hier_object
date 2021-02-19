# hier_object

*hier_object* is a Python library for putting computer vision ground
truth object bounding box annotations into a hierarchy: it determines what
objects are inside of or part of other objects.
It considers containing objects that can overlap, objects that have
duplicate annotations and bounding boxes of objects inside another
that extend beyond the containing object.
This `README` focuses on details.
For the motivation for this library, see
[Putting Annotations in a Hierarchy](https://SeeForTwo.github.io/hierarchical_annotations)
and
[Detecting Left and Right Eyes](https://SeeForTwo.github.io/left_right_eyes).

Here is an example of using it, where `hobj1` is inside `hobj3`:

```python
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
```

First, a `HierObject` class object is created for each computer vision object.

```
class HierObject(object)
    def __init__(self, obj, all_obj=None, **extra):
        # ...
        self.parse(obj, **extra)
        # ...
```

The provided `parse` method uses an `obj` parameter that is a `dict` with:
* 0.0 to 1.0 `float` (or strings that can be converted to `float`)
  bounding box coordinate values corresponding to keys `XMin`, `XMax`,
  `YMin` and `YMax`,
* a computer vision class label value corresponding to key `LabelName`,
* a image identifier value corresponding to key `imageID`.

This supports annotations from the
[Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html).
A derived class can override this to parse other computer vision
object representations. A derived class can optionally use the `extra`
parameter.
By default, a static member collects all the objects, but this can be
changed by specifying the `all_obj` parameter.

Second, the `get_hierarchy` function is called with the `all` member
of an object and a `list` of `tuple` of pairs of computer vision class
names. The first name is for the class of the `children`, the inside
or contained objects. The second name is for the class of the
`parent`, the outside or containing objects. Any number of pairs may
be specified to handle arbitrary hierarchies.

Each `HierObject` object has the following members:
* `all` is a `dict` with key image identifier of `dict` with key object
  class label of `list` of `HierObject` objects.
* `parent` is a `dict` with key that is a parent class name and value
  that is an integer index of another object if this object is
  contained by that other object with this parent class and not other
  objects of this parent class.
* `children` is a `dict` with key that is a child
  class name and value that is a `list` of integer indexes of all objects
  of this child class that are contained by this object and no other object
  with the same class as this object.  
* `duplicate` is a `bool` that is `True` if there is another object
  with the same class (which has `duplicate` `False`) with high
  overlap (IoU > 0.5).
* `overlap` is a `dict` with key object class name and values that are
  a `list` of `Overlap` `namedtuple` with information about all
  objects of this class with a bounding box that intersects this
  object's bounding box.
* Various `float` geometry members are: area `area`, height `h`, width
  `w`, bounding box `x0`, `y0`, `x1` and `y1`, and bounding box center
  `xc`, `yc`.
  
* `label` and `imageID` are from creation.
* `index` is this object's index in the `list` in `all` for this object's class
   for this object's image.

An `Overlap` `namedtuple` describes the intersection of an object with another:
* `self_inside_other` is `True` if this object's bounding box is
  inside the other object.
* `center_of_self` is `True` if this object's center is inside the
  other object.
* `other_inside_self` is `True` if the other object's bounding box is
  inside this object.
* `center_of_other` is `True` if the other object's center is inside this object.
* `intersection` is the area of intersection (`float`).
* `iou` is the Intersection over Union / Jaccard similarity
  coefficient (`float`).
* `iom` is the Intersection over Minimum Area (`float`).
* `index` is the integer index of the other object.
* `obj` is the other `HierObject` itself.

When setting the `parent` and `children` members, the `get_hierarchy`
function first considers child objects that are inside a parent object
and if there are none, considers child objects with centers inside a
parent object. The additional `Overlap` `namedtuple` information can
be used to implement alternative rules for determining
parent-children. This information can also be used for determining if
an object has no parent or children assigned because there are no
possible choices versus if there are ambiguous choices.

`HierObject` are JSON serializable using the `HierObjectJSONEncoder` encoder.
Here is the JSON printed by the example above:

```javascript
{
 "area": 0.04000000000000001,
 "children": null,
 "duplicate": false,
 "h": 0.2,
 "imageID": "i1",
 "index": 0,
 "label": "Object",
 "obj": {
  "ImageID": "i1",
  "LabelName": "Object",
  "XMax": "0.4",
  "XMin": "0.2",
  "YMax": "0.5",
  "YMin": "0.3"
 },
 "overlap": {
  "Group": [
   {
    "center_of_other": false,
    "center_of_self": true,
    "index": 0,
    "intersection": 0.04,
    "iom": 1.0,
    "iou": 0.25,
    "other_inside_self": false,
    "self_inside_other": true
   }
  ],
  "Object": [
   {
    "center_of_other": false,
    "center_of_self": false,
    "index": 1,
    "intersection": 0.01,
    "iom": 0.25,
    "iou": 0.09091,
    "other_inside_self": false,
    "self_inside_other": false
   }
  ]
 },
 "parent": {
  "Group": 0
 },
 "w": 0.2,
 "x0": 0.2,
 "x1": 0.4,
 "xc": 0.3,
 "y0": 0.3,
 "y1": 0.5,
 "yc": 0.4
}
```

See [test/README.md](test/README.md) for the manually created unit tests and property
based (pseudo-random) tests for this library.

Please use "Discussions" or create an issue for questions or comments.

## License

hier_object and related tests by <https://github.com/SeeForTwo> are
licensed under a Creative Commons Attribution-NonCommercial 4.0
International License.
<http://creativecommons.org/licenses/by-nc/4.0/>

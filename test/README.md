# hier_object/test

Tests for *hier_object*, which is a Python library for [putting
computer vision ground truth object bounding box annotations into a
hierarchy](https://SeeForTwo.github.io/hierarchical_annotations),
are in this directory. 

The tests can be run using the following python programs:

File | Description
-----|------------
test_all.py | Run all tests.
test_unit_hier_object.py | Run a small number of manually created unit tests.
test_hyp_hier_object.py | Run a large number of tests created randomly be Hypothesis property-pased testing.

## Details about property based (pseudo random) testing using Hypothesis

The remainder of this `README` covers details of using the the
Hypothesis library for property based testing (in the style of
Haskell's QuickCheck) to create pseudo-random examples for
testing. The Hypothesis library drives the creation of examples, and
if it finds an example that makes a test fail, attempts to find a
simpler example that causes the same failure. See
<https://hypothesis.works/> and
<https://hypothesis.readthedocs.io/en/latest/> for general information
about the Hypothesis library.
For discussion about the purpose and effectiveness of these tests, see
<https://SeeForTwo.github.io/random_testing>


The following python programs use examples outside of the automated tests.

File | Description
-----|------------
hier_object_examples.py | Print one pseudo random example created in a similar manner to what test_hyp_hier_object.py creates. Typically, each time this is run, a different example is generated.
hier_object_manual.py | Example of manual testing of hier_object.py using a previously generated example, i.e. an an example of how to re-run an example found by running test_hyp_hier_object.py.

Internally, test_hyp_hier_object.py uses the following.

File | Description
-----|------------
test/allocate_space.py | Allocate space for objects.
test/bbox_utils.py | Utilities related to the bounding boxes for objects.
test/cls_test_info.py | Definition for TestInfo class used as structure to collect information used in testing.
test/object_create.py | Create a hypothesis library strategy described by ObjectStrategyParam namedtuples.

The `ObjectStrategyParam` `namedtuple` in `object_create.py` describes random computer vision objects or groups of objects to create using the following fields:
* `label`, is the class label string or `None` for an overlapping group.
* `obj_min` is the minimum number of objects per parent.
* `obj_max` is the maximum number of objects per parent.
* `dup_min` is the minimum number of duplicate annotations per object.
* `dup_max` is the maximum number of duplicate annotations per object.
* `min_size` is the minimum width, height (`float`, >0.0, <1.0, or None)
   used to terminate splitting when allocating space.
* `suggested_min_size` is the minimum width, height (`float`, >0.0, <1.0, or None)
  used to prioritize splitting when allocating space.
* `ratio` is used to determined a maximum width/height equal to parent width/height * ratio.

An important property guaranteed by the generation of bounding boxes
is that objects that are not in the same group can never be inside one
another or duplicates.

A nested structure of `ObjectStrategyParam` is used to define a
hierarchy.  This structure is a `list` of alternative hierarchies that
are `list` of either `ObjectStrategyParam` or `tuple` of
`ObjectStrategyParam` for each hierarchy level, in order of parent
(containing object, outside object) to children (object contained,
inside object).  The example below shows the default parameters,
`HIERARCHY_PARAM`.

```
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

HIERARCHY_PARAM = [
    [GROUP_PARAM, HEAD_PARAM, LEAF_PARAM],
    [GROUP_PARAM, FACE_ROOT_PARAM, LEAF_PARAM],
    [GROUP_PARAM, HEAD_PARAM, FACE_INSIDE_PARAM, LEAF_PARAM],
    [LEAF_PARAM]]
```

Generated test data is a `list` of `dict` describing each computer
vision object annotation plus `TestInfo` with the expected result.
Below is an example (in JSON format) of five generated objects (or
strictly speaking a case with five annotations for four objects where
the first and last annotations are duplicate annotations of the same
object) that can be used as input for testing `../heir_object.py`. The
`TestInfo` has been replaced `null` for all of these so
`../heir_object.py` gets no extra information.
`test_hyp_hier_object.py` saves the `TestInfo` information and uses it
to determine if the result from `../heir_object.py` is correct. The
`TestInfo` for the example below would indicate that no objects have
parents or children, and the first and last objects are duplicates of
each other. (See [hier_object_manual.py](hier_object_manual.py) for an example with
`TestInfo` or generate more examples with `TestInfo` using
[hier_object_examples.py](hier_object_examples.py).)

```javascript
[
 {
  "ImageID": "1",
  "LabelName": "nose",
  "TestInfo": null,
  "XMax": "0.275",
  "XMin": "0.225",
  "YMax": "0.45018",
  "YMin": "0.36435"
 },
 {
  "ImageID": "1",
  "LabelName": "eye",
  "TestInfo": null,
  "XMax": "0.86731",
  "XMin": "0.63269",
  "YMax": "0.76607",
  "YMin": "0.48893"
 },
 {
  "ImageID": "1",
  "LabelName": "eye",
  "TestInfo": null,
  "XMax": "0.75",
  "XMin": "0.25",
  "YMax": "0.17556",
  "YMin": "0.07944"
 },
 {
  "ImageID": "1",
  "LabelName": "eye",
  "TestInfo": null,
  "XMax": "0.2875",
  "XMin": "0.2125",
  "YMax": "0.86786",
  "YMin": "0.69167"
 },
 {
  "ImageID": "1",
  "LabelName": "nose",
  "TestInfo": null,
  "XMax": "0.275",
  "XMin": "0.225",
  "YMax": "0.45018",
  "YMin": "0.36435"
 }
]
```

Using `HIERARCHY_PARAM`, generating and testing 100 examples takes 6
to 10 seconds (which is what `test_hyp_hier_object.py` does) if no
errors are found. Generating and testing 10,000 examples takes 10 to
20 minutes.  This is reported as "extremely slow" by the Hypothesis
library, but quite reasonable in the context of a project with
computer vision Deep Neural Network (DNN) training.  Decreasing the
size of examples would increase speed.

Please use "Discussions" or create an issue for questions or comments.
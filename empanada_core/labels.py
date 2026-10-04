r"""Pure dense-label-array algorithms extracted from the GUI merge/split widgets.

Everything here operates on a dense numpy label array. All viewer/layer
plumbing (the viewer object, the Labels/Points/Shapes layers, the viewer dims
reads, the layer rename refresh hook and the widget decorator closures) has
been removed and replaced by explicit array and scalar arguments. The
numerical behaviour is unchanged from the widget bodies.
"""

from empanada.array_utils import crop_and_binarize, take, put
from skimage.measure import regionprops
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage import morphology as morph
from skimage.feature import peak_local_max
from skimage import draw
import numpy as np


def map_points(world_points):
    r"""Map points to indices in the (local) data coordinates of the labels array.

    The widget version took the labels layer and called
    ``labels_layer.world_to_data``. Labels layers are required to have a scale
    of all ones (the widget asserted this), so world coordinates and data
    coordinates are identical and the layer is no longer needed.

    Args:
        world_points: Array of (n, labels.ndim) world coordinates.

    Returns:
        local_points: List of tuples of int data coordinates.

    """
    # labels layer must have scale of all ones, i.e. world == data
    local_points = []
    for pt in world_points:
        local_points.append(tuple([int(c) for c in pt]))

    return local_points


def get_local_points(labels, label_ids):
    r"""Get the (truncated) centroids of the given label ids.

    Args:
        labels: Dense array of (h, w) or (d, h, w) label values.
        label_ids: Iterable of label values to get centroids for.

    Returns:
        local_points: List of tuples of int centroid coordinates.

    """
    local_points = []
    for rp in regionprops(labels):
        if rp.label in label_ids:
            world_points = rp.centroid
            local_points.append(tuple([int(c) for c in world_points]))

    return local_points


def _box_to_slice(shed_box):
    n = len(shed_box)
    n_dim = n // 2

    slices = []
    for i in range(n_dim):
        s = shed_box[i]
        e = shed_box[i + n_dim]
        slices.append(slice(s, e))

    return tuple(slices)


def _pad_box(shed_box, shape, radius=0):
    n = len(shed_box)
    n_dim = n // 2

    padded = [0] * len(shed_box)
    for i in range(n_dim):
        s = max(0, shed_box[i] - radius)
        e = min(shape[i], shed_box[i + n_dim] + radius)
        padded[i] = s
        padded[i + n_dim] = e

    return tuple(padded)


def morph_labels(
    labels,
    operation,
    points=None,
    radius=1,
    hole_size=64,
    apply3d=False,
    axis=0,
    plane=None,
    plane1=None,
    plane2=None,
):
    r"""Morphologically operate on the labels under the given points.

    Mutates ``labels`` in place and returns None (the widget assigned the
    array back to ``labels_layer.data``).

    Args:
        labels: Dense array of (h, w), (d, h, w) or (t, z, h, w) label values.
        operation: Str, one of 'Dilate', 'Erode', 'Close', 'Open', 'Fill holes'.
        points: Array of (n, labels.ndim) coordinates or None. If None, the
            operation is applied to every label in ``labels``.
        radius: Int, radius of the structuring element (ignored for 'Fill holes').
        hole_size: Max hole size to fill if operation is 'Fill holes'.
        apply3d: Bool, apply the operation in 3D.
        axis: Int, replaced ``viewer.dims.order[0]``, the displayed non-planar
            axis used to take 2D planes out of 3D labels.
        plane: Int, replaced ``viewer.dims.current_step[0]`` for 3D labels when
            no points are given.
        plane1: Int, replaced ``viewer.dims.current_step[0]`` for 4D labels when
            points are given.
        plane2: Int, replaced ``viewer.dims.current_step[1]`` for 4D labels when
            points are given.

    """
    ops = {
        "Dilate": morph.binary_dilation,
        "Erode": morph.binary_erosion,
        "Close": morph.binary_closing,
        "Open": morph.binary_opening,
        "Fill holes": morph.remove_small_holes,
    }

    hole_size = int(hole_size)

    if operation == "Fill holes":
        op_arg = hole_size
    elif labels.ndim == 3 and apply3d:
        op_arg = morph.ball(radius)
    else:
        op_arg = morph.disk(radius)

    if apply3d and labels.ndim != 3:
        print("Apply 3D checked, but labels are not 3D. Ignoring.")

    if points is None:
        label_ids = np.unique(labels)[1:].tolist()
        local_points = get_local_points(labels, label_ids)
    else:
        local_points = map_points(points)

        label_ids = [labels[pt].item() for pt in local_points]

    # drop any label_ids equal to 0 in case point
    # was placed on the background
    label_ids = list(filter(lambda x: x > 0, label_ids))

    if len(label_ids) == 0:
        print("No labels selected!")
        return

    for label_id in label_ids:
        if labels.ndim == 2 or (labels.ndim == 3 and apply3d):
            shed_box = [rp.bbox for rp in regionprops(labels) if rp.label == label_id][
                0
            ]
            shed_box = _pad_box(shed_box, labels.shape, radius)
            slices = _box_to_slice(shed_box)

            # apply op
            binary = crop_and_binarize(labels, shed_box, label_id)

            labels[slices][binary] = 0
            binary = ops[operation](binary, op_arg)
            labels[slices][binary] = label_id

        elif labels.ndim == 3:
            if points is None:
                labels2d = labels[plane]

                shed_box = [
                    rp.bbox for rp in regionprops(labels2d) if rp.label == label_ids
                ]
                shed_box = _pad_box(shed_box, labels.shape, radius)
                slices = _box_to_slice(shed_box)

                binary = crop_and_binarize(labels2d, shed_box, label_id)
                labels2d[slices][binary] = 0
                binary = ops[operation](binary, op_arg)
                labels2d[slices][binary] = label_id

                put(labels, plane, labels2d)
            else:
                plane = local_points[0][axis]
                labels2d = take(labels, plane, axis)
                assert all(local_pt[axis] == plane for local_pt in local_points)

                shed_box = [
                    rp.bbox for rp in regionprops(labels2d) if rp.label == label_id
                ][0]
                shed_box = _pad_box(shed_box, labels.shape, radius)
                slices = _box_to_slice(shed_box)

                binary = crop_and_binarize(labels2d, shed_box, label_id)
                labels2d[slices][binary] = 0
                binary = ops[operation](binary, op_arg)
                labels2d[slices][binary] = label_id

                put(labels, plane, labels2d, axis)

        elif labels.ndim == 4:
            if points is not None:
                labels2d = labels[plane1, plane2]

                shed_box = [
                    rp.bbox for rp in regionprops(labels2d) if rp.label == label_ids
                ]
            else:
                plane1 = local_points[0][0]
                plane2 = local_points[0][1]
                assert all(local_pt[0] == plane1 for local_pt in local_points)
                assert all(local_pt[1] == plane2 for local_pt in local_points)

                labels2d = labels[plane1, plane2]

                shed_box = [
                    rp.bbox for rp in regionprops(labels2d) if rp.label == label_id
                ][0]

            shed_box = _pad_box(shed_box, labels.shape, radius)
            slices = _box_to_slice(shed_box)

            binary = crop_and_binarize(labels2d, shed_box, label_id)
            labels2d[slices][binary] = 0
            binary = ops[operation](binary, op_arg)
            labels2d[slices][binary] = label_id

            labels[plane1, plane2] = labels2d


def delete_labels(labels, points, apply3d=False, axis=0):
    r"""Delete the labels under the given points.

    Mutates ``labels`` in place and returns None.

    Args:
        labels: Dense array of (h, w), (d, h, w) or (t, z, h, w) label values.
        points: Array of (n, labels.ndim) coordinates.
        apply3d: Bool, delete the labels in 3D.
        axis: Int, replaced ``viewer.dims.order[0]``.

    """
    if points is None:
        print("Add points!")
        return

    if apply3d and labels.ndim != 3:
        print("Apply 3D checked, but labels are not 3D. Ignoring.")

    # get points as indices in local coordinates
    local_points = map_points(points)

    label_ids = [labels[pt].item() for pt in local_points]

    # drop any label_ids equal to 0 in case point
    # was placed on the background
    label_ids = list(filter(lambda x: x > 0, label_ids))

    if labels.ndim == 2 or (labels.ndim == 3 and apply3d):
        for l in label_ids:
            labels[labels == l] = 0
    elif labels.ndim == 3:
        # take labels along axis
        for local_pt in local_points:
            labels2d = take(labels, local_pt[axis], axis)
            for l in label_ids:
                labels2d[labels2d == l] = 0

            put(labels, local_pt[axis], labels2d, axis)
    elif labels.ndim == 4:
        # take labels along axis
        for local_pt in local_points:
            labels2d = labels[local_pt[0], local_pt[1]]
            for l in label_ids:
                labels2d[labels2d == l] = 0

            labels[local_pt[0], local_pt[1]] = labels2d

    print(f"Removed labels {label_ids}")


def _line_to_indices(line, axis):
    if len(line[0]) == 2:
        line = line.ravel().astype("int").tolist()
        indices = np.stack(draw.line(*line), axis=1)
    elif len(line[0]) == 3:
        plane = line[0][axis]
        keep_axes = [i for i in range(3) if i != axis]
        line = line[:, keep_axes]
        line = line.ravel().astype("int").tolist()
        y, x = draw.line(*line)
        # add plane to indices
        z = np.full_like(x, plane)
        indices = [y, x]
        indices.insert(axis, z)
        indices = np.stack(indices, axis=1)
    elif len(line[0]) == 4:
        assert axis == 0
        planes = line[0][:2]
        line = line[:, [2, 3]]
        line = line.ravel().astype("int").tolist()
        y, x = draw.line(*line)
        # add plane to indices
        t = np.full_like(x, planes[0])
        z = np.full_like(x, planes[1])
        indices = np.stack([t, z, y, x], axis=1)
    else:
        raise Exception("Only lines in 2d, 3d, and 4d are supported!")

    return indices


def merge_labels(
    labels,
    points=None,
    shapes=None,
    shape_types=None,
    apply3d=False,
    axis=0,
    selected_label=None,
):
    r"""Merge the labels under the given points/lines into a single label.

    Mutates ``labels`` in place and returns None (the widget assigned the
    array back to ``labels_layer.data``).

    Args:
        labels: Dense array of (h, w), (d, h, w) or (t, z, h, w) label values.
        points: Array of (n, labels.ndim) coordinates or None.
        shapes: Sequence of (m, labels.ndim) vertex arrays or None, as in
            ``shapes_layer.data``.
        shape_types: Sequence of shape type strings ('line', 'path', ...)
            parallel to ``shapes``, as in ``shapes_layer.shape_type``.
        apply3d: Bool, merge the labels in 3D.
        axis: Int, replaced ``viewer.dims.order[0]``.
        selected_label: Int or None, replaced ``labels_layer.selected_label``.
            Preferred as the merged label value when it is one of the selected
            labels; if None (or not selected) the minimum selected label is used.

    """
    if points is None and shapes is None:
        print("Add points!")
        return

    world_points = []
    if points is not None:
        world_points.append(points)

    if shapes is not None:
        for stype, shape in zip(shape_types, shapes):
            if stype == "line":
                world_points.append(_line_to_indices(shape, axis))
            elif stype == "path":
                n = len(shape)  # number of vertices
                for i in range(n):
                    world_points.append(_line_to_indices(shape[i : i + 2], axis))
                    if i == n - 2:
                        break

    world_points = np.concatenate(world_points, axis=0)

    if apply3d and labels.ndim != 3:
        print("Apply 3D checked, but labels are not 3D. Ignoring.")

    # get points as indices in local coordinates
    local_points = map_points(world_points)

    # clip local points outside of labels shape
    for idx, pt in enumerate(local_points):
        clipped_point = ()
        for i, size in enumerate(labels.shape):
            clipped_point += (min(size - 1, max(0, pt[i])),)

        local_points[idx] = clipped_point

    label_ids = [labels[pt].item() for pt in local_points]

    # drop any label_ids equal to 0 in case point
    # was placed on the background
    label_ids = list(filter(lambda x: x > 0, label_ids))
    label_ids = np.unique(label_ids)

    # get merged label value
    # prefer the currently selected label
    if selected_label is not None and selected_label in label_ids:
        new_label_id = selected_label
    else:
        new_label_id = min(label_ids)

    if labels.ndim == 2 or (labels.ndim == 3 and apply3d):
        # replace labels with minimum of the selected labels
        for l in label_ids:
            if l != new_label_id:
                labels[labels == l] = new_label_id
    elif labels.ndim == 3:
        # take labels along axis
        for local_pt in local_points:
            labels2d = take(labels, local_pt[axis], axis)
            # replace labels with minimum of the selected labels
            for l in label_ids:
                if l != new_label_id:
                    labels2d[labels2d == l] = new_label_id

            put(labels, local_pt[axis], labels2d, axis)
    elif labels.ndim == 4:
        # take labels along axis
        for local_pt in local_points:
            labels2d = labels[local_pt[0], local_pt[1]]
            for l in label_ids:
                if l != new_label_id:
                    labels2d[labels2d == l] = new_label_id

            labels[local_pt[0], local_pt[1]] = labels2d

    print(f"Merged labels {label_ids} to {new_label_id}")


def _translate_point_in_box(point, shed_box):
    n_dim = len(shed_box) // 2
    return tuple([int(point[i] - shed_box[i]) for i in range(n_dim)])


def _distance_markers(binary, min_distance):
    distance = ndi.distance_transform_edt(binary)
    energy = -distance

    # handle irritating quirk of peak_local_max
    # when any dimension is 1
    if any([s == 1 for s in distance.shape]):
        coords = peak_local_max(np.squeeze(distance), min_distance=min_distance)
        markers = np.zeros(np.squeeze(distance).shape, dtype=bool)
        markers[tuple(coords.T)] = True

        expand_axis = [s == 1 for s in distance.shape].index(True)
        markers = np.expand_dims(markers, axis=expand_axis)
    else:
        coords = peak_local_max(distance, min_distance=min_distance)
        markers = np.zeros(distance.shape, dtype=bool)
        markers[tuple(coords.T)] = True

    markers, _ = ndi.label(markers)
    return energy, markers


def _point_markers(binary, local_points, shed_box):
    markers = np.zeros(binary.shape, dtype=bool)
    for local_pt in local_points:
        markers[_translate_point_in_box(local_pt, shed_box)] = True

    markers, _ = ndi.label(markers)
    energy = binary
    return energy, markers


def split_labels(
    labels,
    points,
    min_distance=10,
    points_as_markers=False,
    apply3d=False,
    new_label=False,
    start_label=None,
    axis=0,
):
    r"""Split the labels under the given points with a watershed.

    Mutates ``labels`` in place and returns None (the widget assigned the
    array back to ``labels_layer.data``).

    Args:
        labels: Dense array of (h, w), (d, h, w) or (t, z, h, w) label values.
        points: Array of (n, labels.ndim) coordinates. Points on the background
            (label 0) are dropped.
        min_distance: Int, min distance between markers for the distance markers.
        points_as_markers: Bool, use the placed points as the watershed markers.
            If True, ``min_distance`` is ignored.
        apply3d: Bool, split the labels in 3D.
        new_label: Bool, specify the new label IDs instead of appending to
            ``labels.max()``.
        start_label: The label ID to start the new label IDs from. Required when
            ``new_label`` is True.
        axis: Int, replaced ``viewer.dims.order[0]``, the displayed non-planar
            axis used to take 2D planes out of 3D labels.

    """
    if points is None:
        return

    if new_label:
        assert start_label is not None, "new_label=True requires start_label"

    if apply3d and labels.ndim != 3:
        print("Apply 3D checked, but labels are not 3D. Ignoring.")

    # get points as indices in local coordinates
    local_points = map_points(points)

    label_ids = np.array([labels[pt].item() for pt in local_points])
    local_points = np.stack(local_points, axis=0)

    # drop any label_ids equal to 0; in case point
    # was placed on the background
    background_pts = label_ids == 0
    local_points = local_points[~background_pts]
    label_ids = label_ids[~background_pts]

    if len(label_ids) == 0:
        print("No labels selected!")
        return

    # group local_points by label_ids
    labels_points = {
        label_id: local_points[label_ids == label_id]
        for label_id in np.unique(label_ids)
    }

    for label_id, local_points in labels_points.items():
        if labels.ndim == 2 or (labels.ndim == 3 and apply3d):
            shed_box = [rp.bbox for rp in regionprops(labels) if rp.label == label_id][
                0
            ]
            binary = crop_and_binarize(labels, shed_box, label_id)

            if points_as_markers:
                energy, markers = _point_markers(binary, local_points, shed_box)
            else:
                energy, markers = _distance_markers(binary, min_distance)

            marker_ids = np.unique(markers)[1:]

            if len(marker_ids) > 1:
                new_labels = watershed(energy, markers, mask=binary)
                slices = _box_to_slice(shed_box)

                if new_label:
                    new_label_id = int(start_label) - 1
                    max_label = new_label_id
                else:
                    max_label = labels.max()

                # Check if any of the new label IDs are already in use
                new_labels_exist = any(labels.max() >= (marker_ids + max_label))
                if new_labels_exist:
                    print(
                        f"Label ID {start_label} is already in use. Please specify new label IDs."
                    )
                else:
                    labels[slices][binary] = new_labels[binary] + max_label
                    print(f"Split label {label_id} to {marker_ids + max_label}")
            else:
                print("Nothing to split.")

        elif labels.ndim == 3:
            # get the current viewer axis
            plane = local_points[0][axis]
            labels2d = take(labels, plane, axis)
            assert all(local_pt[axis] == plane for local_pt in local_points)

            shed_box = [
                rp.bbox for rp in regionprops(labels2d) if rp.label == label_id
            ][0]
            binary = crop_and_binarize(labels2d, shed_box, label_id)

            if points_as_markers:
                local_points2d = []
                for lp in local_points:
                    local_points2d.append([p for i, p in enumerate(lp) if i != axis])
                energy, markers = _point_markers(binary, local_points2d, shed_box)
            else:
                energy, markers = _distance_markers(binary, min_distance)

            marker_ids = np.unique(markers)[1:]

            if len(marker_ids) > 1:
                new_labels = watershed(energy, markers, mask=binary)
                slices = _box_to_slice(shed_box)

                if new_label:
                    new_label_id = int(start_label) - 1
                    max_label = new_label_id
                else:
                    max_label = labels2d.max()
                    # Check if any of the new label IDs are already in use
                new_labels_exist = any(labels2d.max() >= (marker_ids + max_label))
                if new_labels_exist:
                    print(
                        f"Label ID {start_label} is already in use. Please specify new label IDs."
                    )
                else:
                    labels2d[slices][binary] = new_labels[binary] + max_label
                    print(f"Split label {label_id} to {marker_ids + max_label}")
            else:
                print("Nothing to split.")

            put(labels, local_points[0][axis], labels2d, axis)

        elif labels.ndim == 4:
            # get the current viewer axes
            plane1 = local_points[0][0]
            plane2 = local_points[0][1]
            assert all(local_pt[0] == plane1 for local_pt in local_points)
            assert all(local_pt[1] == plane2 for local_pt in local_points)

            labels2d = labels[plane1, plane2]

            shed_box = [
                rp.bbox for rp in regionprops(labels2d) if rp.label == label_id
            ][0]
            binary = crop_and_binarize(labels2d, shed_box, label_id)

            if points_as_markers:
                energy, markers = _point_markers(binary, local_points, shed_box)
            else:
                energy, markers = _distance_markers(binary, min_distance)

            marker_ids = np.unique(markers)[1:]

            if len(marker_ids) > 1:
                new_labels = watershed(energy, markers, mask=binary)
                slices = _box_to_slice(shed_box)

                if new_label:
                    new_label_id = int(start_label) - 1
                    max_label = new_label_id
                else:
                    max_label = labels2d.max()
                # Check if any of the new label IDs are already in use
                new_labels_exist = any(labels2d.max() >= (marker_ids + max_label))
                if new_labels_exist:
                    print(
                        f"Label ID {start_label} is already in use. Please specify new label IDs."
                    )
                else:
                    labels2d[slices][binary] = new_labels[binary] + max_label
                    print(f"Split label {label_id} to {marker_ids + max_label}")
            else:
                print("Nothing to split.")

            labels[plane1, plane2] = labels2d

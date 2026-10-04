"""Non-GUI inference orchestration extracted from the GUI inference widgets.

This module is the GUI-free core of the former ``_slice_inference.py``
(``SliceInferenceWidget``) and ``_volume_inference.py``
(``VolumeInferenceWidget``).  Only the inference-driving logic is kept:
engine construction / parameter updates, the 2D ROI + batch helpers, and the
3D stack / ortho-plane drivers.  All GUI toolkit layer and viewer access (layer
creation, ``viewer.dims``, cursor position, viewport corners), the widget
decorators, the threaded worker wrappers and the widget classes themselves are
not reproduced here.  This module imports no GUI library at all.

Both source files had a method named ``get_engine``, so they are named
``create_engine_2d`` and ``create_engine_3d`` here.  The two
``config_and_run_inference`` entry points are not reproduced as functions;
their reusable pieces are ``get_store_url``, ``select_multiscale_level`` and
``squeeze_channel_dim``.

The caller owns every piece of state that used to live on the widgets:
``self.engine``, ``self.last_config``, ``self.store_url``, ``self.dims``,
``image_layer.data`` and ``image_layer.multiscale`` must be read and updated by
the caller.
"""

import os
import numpy as np
import dask.array as da
from time import time
from tqdm import tqdm
from skimage.draw import polygon

from empanada_core.inference import Engine2d, Engine3d
from empanada_core.multigpu import MultiGPUEngine3d
from empanada.array_utils import take

from skimage import measure
from scipy.ndimage import binary_fill_holes


# ---------------- Engine management ----------------
def create_engine_2d(
    model_config,
    model_config_name,
    engine=None,
    last_config=None,
    reload_engine=None,
    downsampling=1,
    confidence_thr=0.5,
    center_confidence_thr=0.1,
    min_distance_object_centers=3,
    fine_boundaries=False,
    semantic_only=False,
    maximum_objects_per_class=10000,
    tile_size=0,
    use_gpu=False,
    use_quantized=False,
):
    """Build an ``Engine2d`` or update the parameters of an existing one.

    ``SliceInferenceWidget.get_engine`` with ``self`` unpacked into explicit
    parameters:

    ============================  ==========================
    widget attribute              parameter
    ============================  ==========================
    ``self.model_config``         ``model_config``
    ``self.model_config_name``    ``model_config_name``
    ``self.engine``               ``engine``
    ``self.last_config``          ``last_config``
    ``self.downsampling``         ``downsampling``
    ``self.confidence_thr``       ``confidence_thr``
    ``self.center_confidence_thr``  ``center_confidence_thr``
    ``self.min_distance_object_centers``  ``min_distance_object_centers``
    ``self.fine_boundaries``      ``fine_boundaries``
    ``self.semantic_only``        ``semantic_only``
    ``self.maximum_objects_per_class``  ``maximum_objects_per_class``
    ``self.tile_size``            ``tile_size``
    ``self.using_gpu``            ``use_gpu``
    ``self.using_quantized``      ``use_quantized``
    ============================  ==========================

    ``reload_engine`` was a local of the widget's ``get_engine`` and is now an
    explicit parameter: it decides whether a new ``Engine2d`` is built or
    ``engine.update_params`` is called in place.  When left as ``None`` it is
    derived exactly like the original, from ``engine is None`` and
    ``last_config != model_config_name``.

    Returns the engine; the caller updates its own ``last_config`` (the widget
    assigned ``self.last_config = self.model_config_name`` after both branches)
    and passes the returned engine back in on the next call.
    """
    if reload_engine is None:
        reload_engine = engine is None or last_config != model_config_name

    if reload_engine:
        engine = Engine2d(
            model_config,
            inference_scale=downsampling,
            nms_kernel=min_distance_object_centers,
            nms_threshold=center_confidence_thr,
            confidence_thr=confidence_thr,
            label_divisor=maximum_objects_per_class,
            semantic_only=semantic_only,
            fine_boundaries=fine_boundaries,
            tile_size=tile_size,
            use_gpu=use_gpu,
            use_quantized=use_quantized,
        )
    else:
        # update the parameters of the engine
        # without reloading the model
        assert engine is not None, (
            "reload_engine=False requires an existing engine to update"
        )
        engine.update_params(
            inference_scale=downsampling,
            label_divisor=maximum_objects_per_class,
            nms_threshold=center_confidence_thr,
            nms_kernel=min_distance_object_centers,
            confidence_thr=confidence_thr,
            semantic_only=semantic_only,
            fine_boundaries=fine_boundaries,
            tile_size=tile_size,
        )

    return engine


# ---------------- Helper methods ----------------
def _fill_holes_in_segmentation(mask):
    unique_indices = np.unique(mask)
    rprops = measure.regionprops(mask)

    # crop labels and then apply fill holes
    for rp in tqdm(rprops, desc="filling holes in labels:"):
        if rp.label in unique_indices and rp.label > 0:
            minr, minc, maxr, maxc = rp.bbox

            tmp = mask[minr:maxr, minc:maxc]
            tmp = binary_fill_holes(tmp.astype(bool))
            mask[minr:maxr, minc:maxc] = tmp.astype(mask.dtype) * rp.label
    return mask


def _get_mask_from_shapes_roi(image_shape, shapes):
    """Rasterise ROI polygons into a boolean mask.

    ``shapes`` is the vertices array itself (a shape ``(N, 2)`` array per
    shape); the widget passed ``shapes_layer.data`` here.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=bool)
    for shape in shapes:
        rr, cc = polygon(shape[:, 0], shape[:, 1], (h, w))
        mask[rr, cc] = True
    return mask


def _get_bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("ROI is empty: no pixels selected in the ROI layer.")
    min_y, max_y = int(ys.min()), int(ys.max()) + 1
    min_x, max_x = int(xs.min()), int(xs.max()) + 1
    return min_y, min_x, max_y, max_x


def _get_roi_slice(image, roi_labels=None, roi_shapes=None):
    """Turn a label ROI or a vertex bbox into a cropped slice of ``image``.

    Takes arrays instead of layers: ``image`` is the 2D image array,
    ``roi_labels`` a 2D label array (``Labels.data``) and ``roi_shapes`` the
    vertices (``Shapes.data``).  Exactly one of the two ROI arguments is used,
    mirroring the ``isinstance(roi_layer, Labels)`` / ``Shapes`` branches of
    ``SliceInferenceWidget._get_roi_slice``.

    The label/shape validation that used to live in ``_get_labels_2d`` and in
    ``_get_image_2d`` (dask compute, ndim and shape checks) is the caller's
    responsibility, as is checking ``roi_labels.shape == image.shape``.
    """
    if roi_labels is not None:
        mask = roi_labels > 0
        min_y, min_x, max_y, max_x = _get_bbox_from_mask(mask)
    elif roi_shapes is not None:
        if len(roi_shapes) == 0:
            raise ValueError("ROI Shapes layer has no shapes.")
        # Keep vertex-based bbox for shapes (matches previous behavior / tests)
        shapes = np.array(roi_shapes)
        min_y, min_x = np.inf, np.inf
        max_y, max_x = -np.inf, -np.inf
        for shape in shapes:
            min_y = min(min_y, shape[:, 0].min())
            min_x = min(min_x, shape[:, 1].min())
            max_y = max(max_y, shape[:, 0].max())
            max_x = max(max_x, shape[:, 1].max())
        min_y, min_x, max_y, max_x = map(int, (min_y, min_x, max_y, max_x))
        mask = _get_mask_from_shapes_roi(image.shape, roi_shapes)
    else:
        raise TypeError(
            "ROI must be given as roi_labels (a label array) or roi_shapes "
            "(a list of vertex arrays)."
        )

    roi = image[min_y:max_y, min_x:max_x].copy()
    return roi, min_y, min_x, max_y, max_x, mask[min_y:max_y, min_x:max_x]


def select_multiscale_level(image, multiscale):
    """Use the highest resolution level of a multiscale image.

    Extracted from ``VolumeInferenceWidget.config_and_run_inference``
    L133-136.  As written there the block reads two layer attributes
    (``image_layer.data`` and ``image_layer.multiscale``), so it is not pure on
    its own: the layer access is dropped here and the flag becomes an explicit
    parameter, keeping the array-level selection and the print identical.
    """
    if multiscale:
        print(f"Multiscale image selected, using highest resolution level!")
        image = image[0]
    return image


def squeeze_channel_dim(image):
    """Drop an extraneous channel dimension from a 4D image volume.

    Extracted from ``VolumeInferenceWidget.config_and_run_inference``
    L140-151.  Pure array logic (``shape`` and slicing only).  The
    ``image.ndim in [3, 4]`` assert that guarded this block (L139) is left to
    the caller; a 3D image is returned unchanged.
    """
    if image.ndim == 4:
        # Channel dimensions are commonly 1, 3 and 4
        # Check for dimensions on zeroth and last axes
        shape = image.shape
        if shape[0] in [1, 3, 4]:
            image = image[0]
        elif shape[-1] in [1, 3, 4]:
            image = image[..., 0]
        else:
            raise Exception(f"Image volume must be 3D, got image of shape {shape}")

        print(
            f"Got 4D image of shape {shape}, extracted single channel of size {image.shape}"
        )
    return image


def get_store_url(store_dir, layer_name, model_config_name):
    """Create the zarr storage url from the layer name and the model config.

    Extracted from ``VolumeInferenceWidget.config_and_run_inference`` L124-128:
    ``os.path.join(store_dir, f'{layer_name}_{model_config_name}.zarr')``, or
    ``None`` for the ``'no zarr storage'`` default.  ``layer_name`` is
    ``image_layer.name`` and ``model_config_name`` the selected config, both
    read from the widget in the original code.  The ``str(store_dir)``
    normalisation that the widget's ``__init__`` performed is kept here.
    """
    store_dir = str(store_dir)
    # Create storage url from layer name and model config
    if store_dir == "no zarr storage":  # This is a default -
        store_url = None
        print(f"Running without zarr storage directory, this may use a lot of memory!")
    else:
        store_url = os.path.join(store_dir, f"{layer_name}_{model_config_name}.zarr")
    return store_url


# ---------------- Inference runners ----------------
def _run_model(engine, image, axis, plane, y, x, fill_holes):
    # create the inference engine
    start = time()
    seg = engine.infer(image)
    if fill_holes:
        seg = _fill_holes_in_segmentation(seg)
    print(f"Inference time:", time() - start)
    return seg, axis, plane, y, x


def _run_model_batch(engine, image, fill_holes, axis=0):
    # create the inference engine
    if image.ndim == 3:
        # Slice along whichever axis is currently being viewed (xy, xz, or yz),
        # instead of always assuming the array's first axis is xy.
        # The widget derived this axis from viewer.dims.order[0]; it is now the
        # explicit `axis` parameter (default 0, the widget's viewer-is-None
        # fallback).
        n_slices = image.shape[axis]
        print(f"Running batch mode inference on {n_slices} images along axis {axis}.")
        segmentations = []
        for plane in tqdm(range(n_slices)):
            img_slice = take(image, plane, axis)
            if type(img_slice) == da.Array:
                img_slice = img_slice.compute()
            img_slice = np.asarray(img_slice)

            seg = engine.infer(img_slice)
            if fill_holes:
                seg = _fill_holes_in_segmentation(seg)
            segmentations.append(seg)

        # stack segmentations with padding
        max_h = max(seg.shape[0] for seg in segmentations)
        max_w = max(seg.shape[1] for seg in segmentations)
        padded = []
        for seg in segmentations:
            h, w = seg.shape
            padh, padw = max_h - h, max_w - w
            padded.append(np.pad(seg, ((0, padh), (0, padw))))

        # stack along a new leading axis, then move it back to the axis
        # that was actually sliced so the output matches the input
        # volume's original orientation/shape.
        stacked = np.stack(padded, axis=0)
        if axis != 0:
            stacked = np.moveaxis(stacked, 0, axis)

        return stacked

    elif image.ndim == 2:
        start = time()
        if type(image) == da.Array:
            image = image.compute()

        plane = 0
        seg = engine.infer(image)
        if fill_holes:
            seg = _fill_holes_in_segmentation(seg)
        print(f"Inference time:", time() - start)
        return seg, None, None, None, None

    else:
        raise Exception(f"Batch mode supports 2d and 3d, got {image.ndim}d.")


# ---------------- Engine management (3D) ----------------
def create_engine_3d(
    model_config,
    model_config_name,
    engine=None,
    last_config=None,
    reload_engine=None,
    store_url=None,
    multigpu=False,
    use_gpu=False,
    use_quantized=False,
    downsampling=1,
    confidence_thr=0.5,
    center_confidence_thr=0.1,
    min_distance_object_centers=3,
    fine_boundaries=False,
    semantic_only=False,
    median_slices=3,
    min_size=500,
    min_extent=5,
    maximum_objects_per_class=10000,
    return_panoptic=False,
    label_erosion=0,
    label_dilation=0,
    fill_holes_in_segmentation=False,
    chunk_size=(256, 256, 256),
):
    """Build an ``Engine3d`` / ``MultiGPUEngine3d`` or update an existing one.

    ``VolumeInferenceWidget.get_engine`` with ``self`` unpacked into explicit
    parameters:

    ============================  ==========================
    widget attribute              parameter
    ============================  ==========================
    ``self.model_config``         ``model_config``
    ``self.model_config_name``    ``model_config_name``
    ``self.engine``               ``engine``
    ``self.last_config``          ``last_config``
    ``self.store_url``            ``store_url``
    ``self.multigpu``             ``multigpu``
    ``self.use_gpu``              ``use_gpu``
    ``self.use_quantized``        ``use_quantized``
    ``self.downsampling``         ``downsampling``
    ``self.confidence_thr``       ``confidence_thr``
    ``self.center_confidence_thr``  ``center_confidence_thr``
    ``self.min_distance_object_centers``  ``min_distance_object_centers``
    ``self.fine_boundaries``      ``fine_boundaries``
    ``self.semantic_only``        ``semantic_only``
    ``self.median_slices``        ``median_slices``
    ``self.min_size``             ``min_size``
    ``self.min_extent``           ``min_extent``
    ``self.maximum_objects_per_class``  ``maximum_objects_per_class``
    ``self.return_panoptic``      ``return_panoptic``
    ``self.label_erosion``        ``label_erosion``
    ``self.label_dilation``       ``label_dilation``
    ``self.fill_holes``           ``fill_holes_in_segmentation``
    ``self.chunk_size``           ``chunk_size``
    ============================  ==========================

    Control flow is replicated exactly, including its original quirk: when
    ``reload_engine`` is true (the default derivation, i.e. no engine yet or
    the config changed) the ``Engine3d`` branch wins, so the ``elif multigpu``
    branch is unreachable on that call.  A ``MultiGPUEngine3d`` is therefore
    only ever built by a *later* call whose ``last_config`` already equals
    ``model_config_name``.  This is intentionally not "fixed".

    ``reload_engine`` is an explicit parameter, derived like the original when
    left as ``None`` (``engine is None`` or
    ``last_config != model_config_name``).

    Returns the engine; the caller updates its own ``last_config`` (the widget
    assigned ``self.last_config = self.model_config_name`` inside the two
    rebuild branches).  The widget also assigned ``self.using_gpu =
    self.use_gpu`` in the reload branch, which was never read anywhere else and
    is therefore dropped.
    """
    if reload_engine is None:
        reload_engine = engine is None or last_config != model_config_name

    if reload_engine:
        engine = Engine3d(
            model_config,
            inference_scale=downsampling,
            median_kernel_size=median_slices,
            nms_kernel=min_distance_object_centers,
            nms_threshold=center_confidence_thr,
            confidence_thr=confidence_thr,
            min_size=min_size,
            min_extent=min_extent,
            fine_boundaries=fine_boundaries,
            label_divisor=maximum_objects_per_class,
            use_gpu=use_gpu,
            use_quantized=use_quantized,
            semantic_only=semantic_only,
            save_panoptic=return_panoptic,
            store_url=store_url,
            chunk_size=chunk_size,
            label_erosion=label_erosion,
            label_dilation=label_dilation,
            fill_holes_in_segmentation=fill_holes_in_segmentation,
        )

    elif multigpu:
        engine = MultiGPUEngine3d(
            model_config,
            inference_scale=downsampling,
            median_kernel_size=median_slices,
            nms_kernel=min_distance_object_centers,
            nms_threshold=center_confidence_thr,
            confidence_thr=confidence_thr,
            min_size=min_size,
            min_extent=min_extent,
            fine_boundaries=fine_boundaries,
            label_divisor=maximum_objects_per_class,
            semantic_only=semantic_only,
            save_panoptic=return_panoptic,
            store_url=store_url,
            chunk_size=chunk_size,
        )

    else:
        # update the parameters
        assert engine is not None, (
            "reload_engine=False requires an existing engine to update"
        )
        engine.update_params(
            inference_scale=downsampling,
            median_kernel_size=median_slices,
            nms_kernel=min_distance_object_centers,
            nms_threshold=center_confidence_thr,
            confidence_thr=confidence_thr,
            min_size=min_size,
            min_extent=min_extent,
            fine_boundaries=fine_boundaries,
            label_divisor=maximum_objects_per_class,
            semantic_only=semantic_only,
            save_panoptic=return_panoptic,
            store_url=store_url,
            chunk_size=chunk_size,
            label_erosion=label_erosion,
            label_dilation=label_dilation,
            fill_holes_in_segmentation=fill_holes_in_segmentation,
        )

    return engine


# ---------------- Inference runners (3D) ----------------
def _stack_inference(engine, volume, axis_name):
    stack, trackers = engine.infer_on_axis(volume, axis_name)
    trackers_dict = {axis_name: trackers}
    return stack, axis_name, trackers_dict


def _orthoplane_inference(engine, volume):
    trackers_dict = {}
    axes_dict = {}
    for axis_name in ["xy", "xz", "yz"]:
        stack, trackers = engine.infer_on_axis(volume, axis_name)
        trackers_dict[axis_name] = trackers

        # report instances per class
        for tracker in trackers:
            class_id = tracker.class_id
            print(
                f"Class {class_id}, axis {axis_name}, has {len(tracker.instances.keys())} instances"
            )
        axes_dict[axis_name] = stack
    return trackers_dict, axes_dict

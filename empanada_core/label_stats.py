"""GUI-free label statistics helpers.

This module holds the computational core of two former dock widgets of the
viewer GUI plugin, whose (unmodified) sources stay in this package:

* ``empanada_core._filter_small_labels`` - removal of small-area or
  boundary-touching labels.
* ``empanada_core._label_counter_widget`` - per-class counting of label IDs
  and optional export of the label lists to .xlsx files.

Everything here takes plain numpy/dask arrays plus explicit scalars and flags;
nothing in this module knows about a viewer or a layer object. The widget-only
parts were left behind in the source files: the GUI decorator declarations, the
``viewer`` / ``labels_layer`` arguments, ``viewer.dims.current_step``,
``labels_layer.name``, the ``_get_current_image`` helper (which used
``viewer.cursor.position``, ``viewer.dims.order`` and
``layer.world_to_data``), ``viewer.add_labels`` and
``enable_layer_rename_refresh``.

Consequences a caller must know about:

* the current plane index and the layer name are explicit parameters
  (``plane``, ``layer_name``);
* the plane extraction done by ``_get_current_image`` cannot be reproduced
  without a viewer, so for ``label_type == 'Current image'`` on a >2D array the
  caller must pass the array it wants counted (e.g. ``labels[plane]``); the
  original selected that plane from the cursor position;
* where the widget silently ``return``ed on invalid input, the extracted
  function returns ``None`` (``parse_class_names``) or no removals
  (``apply_label_filter``) after printing the same message.
"""

import itertools
import os

import dask.array as da
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.segmentation import clear_border
from tqdm import tqdm

# openpyxl 只在 xlsx 导出路径上需要, 因此放到函数内部按需导入,
# 这样没有安装 openpyxl 时本模块依然可以正常 import。


def remove_label_from_image(image_array, label):
    image_array[image_array == label] = 0
    return image_array


def filter_out_small_label_areas(img, minimum_area_allowed):
    rp = regionprops_table(img, properties=("label", "area"))
    rp = pd.DataFrame(rp)
    rp = rp.sort_values(by="area")

    smallest_label = rp["label"].iloc[0]
    smallest_area = int(rp["area"].iloc[0])
    # only keep area values less than 1000
    rp = rp[rp["area"] <= minimum_area_allowed]

    # remove label from image
    labels_removed = []
    for label in rp["label"]:
        img = remove_label_from_image(img, label)
        labels_removed.append(label)

    if len(labels_removed) == 0:
        print("No labels were removed.")
        print(
            f"The label ID corresponding to the smallest area is {smallest_label} with an area of {smallest_area} pixels/voxels."
        )
    else:
        num_removed = len(labels_removed)
        print(f"The following label IDs were removed: {labels_removed}")
        print(f"Total number of label IDs removed: {num_removed} ")

    return img, len(labels_removed)


def remove_boundary_labels(labels):
    labels_removed = []
    labels_kept = clear_border(labels)

    for label in np.unique(labels):
        if label not in np.unique(labels_kept):
            labels_removed.append(label)

    if len(labels_removed) == 0:
        print("No label IDs were removed.")
    else:
        num_removed = len(labels_removed)
        print(f"The following label Ids were removed: {labels_removed}")
        print(f"Total number of label IDs removed: {num_removed} ")

    labels = labels_kept

    return labels, len(labels_removed)


def apply_label_filter(labels, remove_opt, min_area, apply_to, plane=None):
    """Remove small-area or boundary labels from a label array.

    Extracted from the ``widget`` inner function of
    ``_filter_small_labels.filter_small_labels`` (source L98-137); the
    ``viewer``/``labels_layer`` annotations, the ``labels_layer is None`` check
    and the ``viewer.add_labels`` display call were dropped.

    Parameters
    ----------
    labels : array-like
        Label array, i.e. what the widget read as ``labels_layer.data``.
        ``np.asarray(labels).copy()`` is applied here, so the caller's array is
        never modified; work happens on the returned array.
    remove_opt : str
        The widget's ``remove_opt`` radio button: 'Small labels' or
        'Boundary labels (slow)'.
    min_area : int
        The widget's ``min_area`` spin box: minimum pixel/voxel area to keep
        (only used for 'Small labels').
    apply_to : str
        The widget's ``apply_to`` radio button: 'Current image',
        '2D patches' or '3D image or z-stack'. This is the explicit
        "current plane only" vs "whole array" flag.
    plane : int or None
        Index of the current plane; replaces ``viewer.dims.current_step[0]``.
        Only used when ``apply_to == 'Current image'`` and ``labels.ndim > 2``.

    Returns
    -------
    labels : numpy.ndarray
        The filtered array. The widget squeezed it before handing it to
        ``viewer.add_labels``; that ``np.squeeze`` is kept, applied only when
        something was removed.
    labels_removed : int or list
        Number of removed label IDs, or the empty list the widget kept when
        ``remove_opt`` matched neither option. An invalid
        ``apply_to == '3D image or z-stack'`` on non-3D labels prints the
        original message and returns ``(labels, 0)`` (the widget returned
        without displaying anything).
    """
    labels = np.asarray(labels).copy()
    if apply_to == "3D image or z-stack" and labels.ndim != 3:
        print("Apply 3D checked, but labels are not 3D. Ignoring.")
        return labels, 0
    labels_removed = []
    # print(labels.shape)
    if apply_to == "Current image":
        if remove_opt == "Small labels":
            labels_, labels_removed = filter_out_small_label_areas(
                labels[plane] if labels.ndim > 2 else labels, min_area
            )
            if labels.ndim > 2:
                labels[plane] = labels_

            else:
                labels = labels_
        elif remove_opt == "Boundary labels (slow)":
            labels_, labels_removed = remove_boundary_labels(
                labels[plane] if labels.ndim > 2 else labels
            )
            if labels.ndim > 2:
                labels[plane] = labels_
            else:
                labels = labels_

    elif labels.ndim == 3 and apply_to == "2D patches":
        for label in tqdm(range(labels.shape[0])):
            if remove_opt == "Small labels":
                labels[label], labels_removed = filter_out_small_label_areas(
                    labels[label], min_area
                )
            elif remove_opt == "Boundary labels (slow)":
                labels[label], labels_removed = remove_boundary_labels(labels[label])

    else:
        if remove_opt == "Small labels":
            labels, labels_removed = filter_out_small_label_areas(labels, min_area)
        elif remove_opt == "Boundary labels (slow)":
            labels, labels_removed = remove_boundary_labels(labels)

    if labels_removed != 0:
        labels = np.squeeze(labels)

    return labels, labels_removed


def save_label_lists(
    label_type, class_names, label_queue, save_dir, layer_name, plane=None
):
    """Save the label IDs of each class to .xlsx files (openpyxl only).

    Extracted verbatim from ``_label_counter_widget.save_label_lists``. The
    only change: the ``labels_layer`` argument was replaced by the explicit
    ``layer_name`` string, because the source only read ``labels_layer.name``
    (L20 and L48) to use it as the worksheet name.
    """
    from openpyxl import Workbook

    if label_type == "Current image":
        for class_name in class_names.values():
            if plane == "null":
                filename = f"{class_name}_label_ids.xlsx"
                sheet_name = layer_name
            else:
                filename = f"{class_name}_image_{plane}_label_ids.xlsx"
                current_image = plane
                sheet_name = f"Image {current_image}"
            file_path = os.path.join(save_dir, filename)
            if os.path.exists(file_path):
                new_filename = f"{class_name}_image_{plane}_label_ids_updated.xlsx"
                file_path = os.path.join(save_dir, new_filename)
            workbook = Workbook()
            sheet = workbook.create_sheet(title=sheet_name)
            sheet["A1"] = "Label ID"
            for class_id, label_ids in label_queue.items():
                curr_class_name = class_names[class_id]
                if curr_class_name == class_name:
                    for row_num, label_id in enumerate(label_ids, start=2):
                        sheet.cell(row=row_num, column=1, value=label_id)

                workbook.save(file_path)
            try:
                default_sheet = workbook["Sheet"]
                workbook.remove(default_sheet)
                workbook.save(file_path)
            except:
                pass
                # print(f'Saved Excel file for class {class_id} ({class_name}) to {file_path}')

    elif label_type == "3D volume or z-stack":
        sheet_name = layer_name
        workbook = Workbook()
        sheet = workbook.create_sheet(title=sheet_name)
        sheet["A1"] = "Label ID"
        for class_id, label_ids in label_queue.items():
            class_name = class_names[class_id]
            for row_num, label_id in enumerate(label_ids, start=2):
                sheet.cell(row=row_num, column=1, value=label_id)
            filename = f"{class_name}_volume_label_ids.xlsx"
            file_path = os.path.join(save_dir, filename)
            if os.path.exists(file_path):
                new_filename = f"{class_name}_volume_label_ids_updated.xlsx"
                file_path = os.path.join(save_dir, new_filename)
            workbook.save(file_path)

        try:
            default_sheet = workbook["Sheet"]
            workbook.remove(default_sheet)
            workbook.save(file_path)
        except:
            pass


def create_xlsx_from_label_queue_list(class_names, label_queues_list, save_dir, labels):
    from openpyxl import Workbook

    for class_name in class_names.values():
        filename = f"{class_name}_patch_label_ids.xlsx"
        file_path = os.path.join(save_dir, filename)
        if os.path.exists(file_path):
            new_filename = f"{class_name}_patch_label_ids_updated.xlsx"
            file_path = os.path.join(save_dir, new_filename)
        workbook = Workbook()

        for slice_num in range(labels.shape[0]):
            label_queue = label_queues_list[slice_num]
            for class_id, label_ids in label_queue.items():
                if class_names[class_id] == class_name:
                    sheet_name = f"Image {slice_num}"

                    if sheet_name in workbook.sheetnames:
                        sheet = workbook[sheet_name]
                        # Clear the existing contents of the sheet
                        sheet.delete_rows(2, sheet.max_row)
                    else:
                        # Create a new sheet
                        sheet = workbook.create_sheet(title=sheet_name)
                    sheet["A1"] = "Label ID"

                    for row_num, label_id in enumerate(label_ids, start=1):
                        sheet.cell(row=row_num + 1, column=1, value=label_id)
        try:
            default_sheet = workbook["Sheet"]
            workbook.remove(default_sheet)
            workbook.save(file_path)
        except:
            pass
        workbook.save(file_path)


def count_labels(label_values, label_divisor):
    label_queue = {}
    if label_divisor == 0:
        label_queue[1] = label_values.tolist()
        return label_queue, [1]

    class_ids = np.unique(np.floor_divide(label_values, label_divisor)).tolist()
    for ci in class_ids:
        min_id = ci * label_divisor
        max_id = (ci + 1) * label_divisor
        label_ids = label_values[(label_values >= min_id) & (label_values < max_id)]
        label_queue[ci] = label_ids.tolist()

    return label_queue, class_ids


def parse_label_divisor(label_divisor):
    """Parse the widget's ``label_divisor`` value (source L171-175).

    ``label_divisor`` is the raw GUI LineEdit value: the string 'None' or
    a string holding an integer. Returns the integer divisor (0 for 'None').
    """
    if label_divisor == "None":
        label_divisor = 0
    else:
        label_divisor = int(label_divisor)
    assert label_divisor > -1, "Label divisor must be a non-negative integer!"

    return label_divisor


def parse_class_names(label_text):
    """Parse the widget's ``label_text`` value (source L225-235).

    ``label_text`` is the raw TextEdit value, one 'class_number,class_name'
    entry per line. Returns the ``{class_id: class_name}`` dict, or ``None``
    when an entered class number is not an integer (the widget printed the same
    message and returned without counting anything).
    """
    class_names = {}
    for seg_class in label_text.split():
        class_id, class_name = seg_class.split(",")
        class_num = class_id.strip()
        class_name = class_name.strip()

        if not class_num.isdigit():
            print(
                f"The class number you entered is invalid. Please provide an integer value!"
            )
            return None
        class_names[int(class_num)] = class_name

    print(f"Class names: {class_names}")

    return class_names


def get_label_values(labels):
    """The unique non-zero label IDs of a label array (source L237-244).

    ``labels`` is a numpy or dask array. For a dask array the unique non-zero
    labels are collected chunk by chunk with ``np.unique(chunk)[1:]`` and
    concatenated, exactly as the widget did.
    """
    if isinstance(labels, da.Array):
        label_values = []
        for inds in itertools.product(*map(range, labels.blocks.shape)):
            chunk = labels.blocks[inds].compute()
            label_values.append(np.unique(chunk)[1:])
        label_values = np.concatenate(label_values)
    else:
        label_values = np.unique(labels)[1:]

    return label_values


def count_labels_current_image(
    labels,
    label_divisor,
    class_names,
    layer_name,
    plane="null",
    export_xlsx=False,
    save_dir=None,
):
    """Count label IDs per class for a single image, optionally exporting.

    Extracted from the ``label_type == 'Current image'`` branch of the widget
    (source L246-266). ``labels`` is the array to count: the widget counted
    ``_get_current_image``'s plane when ``plane`` was not 'null' (that cursor
    based plane selection is dropped - pass ``labels[plane]`` to reproduce it)
    and the whole array otherwise.

    ``class_names`` replaces the parsed ``label_text``, ``layer_name`` replaces
    ``labels_layer.name`` (used only for the worksheet name), ``plane`` replaces
    both ``viewer.dims.current_step[0]`` and the 'null' sentinel
    (``'null'`` = whole current image, any other value is used in the
    '{class_name}_image_{plane}_label_ids.xlsx' file name), and
    ``export_xlsx``/``save_dir`` replace the corresponding GUI widgets.

    Returns ``(label_queue, class_ids)``; the only output is printed.
    """
    label_values = get_label_values(labels)

    label_queue, class_ids = count_labels(label_values, label_divisor)
    if label_queue and class_ids:
        class_ids = np.array(class_ids)
        has_labels = np.isin(class_ids, list(label_queue.keys()))
        valid_class_ids = class_ids[has_labels]
        valid_label_lists = [
            np.unique(label_queue[class_id]) for class_id in valid_class_ids
        ]
        valid_label_counts = [len(label_list) for label_list in valid_label_lists]

        for class_id, label_list, label_count in zip(
            valid_class_ids, valid_label_lists, valid_label_counts
        ):
            if label_count > 0:
                print(
                    f"Label IDs in class {class_id} ({class_names[class_id]}): {label_list}"
                )
                print(
                    f"Total number of label IDs in class {class_id} ({class_names[class_id]}): {label_count}"
                )

            else:
                print(
                    f"No label IDs in class {class_id} ({class_names[class_id]}) found in current slice!"
                )

    if export_xlsx:
        assert save_dir is not None, "export_xlsx=True requires save_dir"
        os.makedirs(save_dir, exist_ok=True)
        save_label_lists(
            "Current image", class_names, label_queue, save_dir, layer_name, plane
        )
        print(f"Saved Excel file to {save_dir}")

    return label_queue, class_ids


def count_labels_2d_patches(
    labels, label_divisor, class_names, export_xlsx=False, save_dir=None
):
    """Count label IDs per class for every 2D slice of a stack.

    Extracted from the ``label_type == '2D patches'`` branch of the widget
    (source L268-300); the inlined per-slice dask/numpy dispatch was replaced
    by ``get_label_values``, which is the same code. ``class_names`` replaces
    the parsed ``label_text``; ``export_xlsx``/``save_dir`` replace the
    corresponding GUI widgets. No layer name is needed here (the source's
    export helper only uses ``labels.shape[0]``).

    Returns ``(label_queues_list, class_ids_list)``; the only output is
    printed.
    """
    class_ids_list = []
    label_queues_list = []

    for slice_num in range(labels.shape[0]):
        label_slice = labels[slice_num]

        slice_labels = get_label_values(label_slice)

        label_queue, class_ids = count_labels(slice_labels, label_divisor)
        label_queues_list.append(label_queue)
        class_ids_list.append(class_ids)

        if label_queue and class_ids:
            for class_id in class_ids:
                if class_id in label_queue:
                    label_list = np.unique((label_queue[class_id]))
                    if len(label_list) > 0:
                        print(
                            f"Total number of label IDs in class {class_id} ({class_names[class_id]}) in image {slice_num}:",
                            len(label_list),
                        )
                    else:
                        print(
                            f"No label IDs in class {class_id} ({class_names[class_id]}) in image {slice_num}!"
                        )

    if export_xlsx:
        assert save_dir is not None, "export_xlsx=True requires save_dir"
        os.makedirs(save_dir, exist_ok=True)
        create_xlsx_from_label_queue_list(
            class_names, label_queues_list, save_dir, labels
        )
        print(f"Saved Excel file to {save_dir}")

    return label_queues_list, class_ids_list


def count_labels_volume(
    labels, label_divisor, class_names, layer_name, export_xlsx=False, save_dir=None
):
    """Count label IDs per class for a 3D volume, optionally exporting.

    Extracted from the ``label_type == '3D volume or z-stack'`` branch of the
    widget (source L302-318). ``class_names`` replaces the parsed
    ``label_text``, ``layer_name`` replaces ``labels_layer.name`` (worksheet
    name), and ``export_xlsx``/``save_dir`` replace the corresponding GUI
    widgets.

    Returns ``(label_queue, class_ids)``; the only output is printed.
    """
    label_values = get_label_values(labels)

    label_queue, class_ids = count_labels(label_values, label_divisor)

    if label_queue and class_ids:
        for class_id in class_ids:
            if class_id in label_queue:
                label_list = np.unique((label_queue[class_id]))
                if len(label_list) > 0:
                    print(
                        f"Total number of label IDs in class {class_id} ({class_names[class_id]}) in volume:",
                        len(label_list),
                    )

                else:
                    print(
                        f"No label IDs in class {class_id} ({class_names[class_id]}) in volume!"
                    )

    if export_xlsx:
        assert save_dir is not None, "export_xlsx=True requires save_dir"
        os.makedirs(save_dir, exist_ok=True)
        save_label_lists(
            "3D volume or z-stack", class_names, label_queue, save_dir, layer_name
        )
        print(f"Saved Excel file to {save_dir}")

    return label_queue, class_ids

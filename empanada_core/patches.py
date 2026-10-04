"""Core (non-GUI) patch selection and dataset saving for the patch widgets.

Extracted from the original dock-widget module: every function here operates on
plain numpy/dask arrays, scalars and plain Python containers instead of GUI
layer objects, and no threading worker wiring is included.
"""

import os
import string
import random
import numpy as np
import dask.array as da
from empanada.array_utils import take


def _pad_patch(patch, size):
    h, w = patch.shape
    assert size[0] >= h and size[1] >= w

    ph, pw = size[0] - h, size[1] - w

    patch = np.pad(patch, ((0, ph), (0, pw)))
    assert patch.shape[0] == size[0]
    assert patch.shape[1] == size[1]

    return patch


def _pad_label_patch(label_patch, size):
    h, w = label_patch.shape
    assert size[0] >= h and size[1] >= w

    ph, pw = size[0] - h, size[1] - w

    label_patch = np.pad(label_patch, ((0, ph), (0, pw)))
    assert label_patch.shape[0] == size[0]
    assert label_patch.shape[1] == size[1]

    return label_patch


def _pad_flipbook(flipbook, size):
    assert flipbook.ndim == 3

    h, w = flipbook.shape[1:]
    assert size[0] >= h and size[1] >= w

    ph, pw = size[0] - h, size[1] - w

    flipbook = np.pad(flipbook, ((0, 0), (0, ph), (0, pw)))
    assert flipbook.shape[1] == size[0]
    assert flipbook.shape[2] == size[1]

    return flipbook


def _pad_label_flipbook(label_flipbook, size):
    assert label_flipbook.ndim == 3

    h, w = label_flipbook.shape[1:]
    assert size[0] >= h and size[1] >= w

    ph, pw = size[0] - h, size[1] - w

    label_flipbook = np.pad(label_flipbook, ((0, 0), (0, ph), (0, pw)))
    assert label_flipbook.shape[1] == size[0]
    assert label_flipbook.shape[2] == size[1]

    return label_flipbook


def _pick_patches(image, patch_size, num_patches, points):
    patches = []
    locs = []
    for _ in range(num_patches):
        plane = None
        if points and points is not None:
            patch_ctr = points.pop(0)
            if len(patch_ctr) == 2:
                ys = int(patch_ctr[0] - patch_size / 2)
                ys = min(ys, image.shape[0] - patch_size)
                ys = max(ys, 0)
                xs = int(patch_ctr[1] - patch_size / 2)
                xs = min(xs, image.shape[1] - patch_size)
                xs = max(xs, 0)

                ye = min(ys + patch_size, image.shape[0])
                xe = min(xs + patch_size, image.shape[1])
                patch = image[ys:ye, xs:xe]
            else:
                plane = patch_ctr[0]
                ys = int(patch_ctr[1] - patch_size / 2)
                ys = min(ys, image.shape[1] - patch_size)
                ys = max(ys, 0)
                xs = int(patch_ctr[2] - patch_size / 2)
                xs = min(xs, image.shape[2] - patch_size)
                xs = max(xs, 0)

                ye = min(ys + patch_size, image.shape[1])
                xe = min(xs + patch_size, image.shape[2])
                patch = image[plane, ys:ye, xs:xe]
        else:
            if image.ndim == 2:
                ys = np.random.choice(
                    np.arange(0, max(1, image.shape[0] - patch_size), patch_size)
                )
                xs = np.random.choice(
                    np.arange(0, max(1, image.shape[1] - patch_size), patch_size)
                )
                ye = min(ys + patch_size, image.shape[0])
                xe = min(xs + patch_size, image.shape[1])
                patch = image[ys:ye, xs:xe]
            else:
                plane = np.random.randint(0, image.shape[0])
                ys = np.random.choice(
                    np.arange(0, max(1, image.shape[1] - patch_size), patch_size)
                )
                xs = np.random.choice(
                    np.arange(0, max(1, image.shape[2] - patch_size), patch_size)
                )
                ye = min(ys + patch_size, image.shape[1])
                xe = min(xs + patch_size, image.shape[2])
                patch = image[plane, ys:ye, xs:xe]

        if type(patch) == da.Array:
            patch = patch.compute()

        patch = _pad_patch(patch, (patch_size, patch_size))

        patches.append(patch)
        if plane is None:
            locs.append((ys, ye, xs, xe))
        else:
            locs.append((plane, ys, ye, xs, xe))

    return np.stack(patches, axis=0), locs


def _pick_paired_patches(image, label, patch_size, num_patches, points):
    patches = []
    label_patches = []
    locs = []
    for _ in range(num_patches):
        plane = None
        if points and points is not None:
            patch_ctr = points.pop(0)
            if len(patch_ctr) == 2:
                ys = int(patch_ctr[0] - patch_size / 2)
                ys = min(ys, image.shape[0] - patch_size)
                ys = max(ys, 0)
                xs = int(patch_ctr[1] - patch_size / 2)
                xs = min(xs, image.shape[1] - patch_size)
                xs = max(xs, 0)

                ye = min(ys + patch_size, image.shape[0])
                xe = min(xs + patch_size, image.shape[1])
                patch = image[ys:ye, xs:xe]

                label_patch = label[ys:ye, xs:xe]
            else:
                plane = patch_ctr[0]
                ys = int(patch_ctr[1] - patch_size / 2)
                ys = min(ys, image.shape[1] - patch_size)
                ys = max(ys, 0)
                xs = int(patch_ctr[2] - patch_size / 2)
                xs = min(xs, image.shape[2] - patch_size)
                xs = max(xs, 0)

                ye = min(ys + patch_size, image.shape[1])
                xe = min(xs + patch_size, image.shape[2])
                patch = image[plane, ys:ye, xs:xe]

                label_patch = label[plane, ys:ye, xs:xe]
        else:
            if image.ndim == 2:
                ys = np.random.choice(
                    np.arange(0, max(1, image.shape[0] - patch_size), patch_size)
                )
                xs = np.random.choice(
                    np.arange(0, max(1, image.shape[1] - patch_size), patch_size)
                )
                ye = min(ys + patch_size, image.shape[0])
                xe = min(xs + patch_size, image.shape[1])
                patch = image[ys:ye, xs:xe]

                label_patch = label[ys:ye, xs:xe]
            else:
                plane = np.random.randint(0, image.shape[0])
                ys = np.random.choice(
                    np.arange(0, max(1, image.shape[1] - patch_size), patch_size)
                )
                xs = np.random.choice(
                    np.arange(0, max(1, image.shape[2] - patch_size), patch_size)
                )
                ye = min(ys + patch_size, image.shape[1])
                xe = min(xs + patch_size, image.shape[2])
                patch = image[plane, ys:ye, xs:xe]

                label_patch = label[plane, ys:ye, xs:xe]

        if type(patch) and type(label_patch) == da.Array:
            patch = patch.compute()
            label_patch = label_patch.compute()

        patch = _pad_patch(patch, (patch_size, patch_size))
        label_patch = _pad_label_patch(label_patch, (patch_size, patch_size))

        patches.append(patch)
        label_patches.append(label_patch)
        if plane is None:
            locs.append((ys, ye, xs, xe))
        else:
            locs.append((plane, ys, ye, xs, xe))

    return np.stack(patches, axis=0), np.stack(label_patches, axis=0), locs


def _pick_flipbooks(image, patch_size, num_patches, points, isotropic):
    flipbooks = []
    locs = []
    for _ in range(num_patches):
        if isotropic:
            axes = [0, 1, 2]
            axis = random.choice(axes)

            # set height and width axes
            del axes[axes.index(axis)]
            ha, wa = axes
        else:
            axis = 0
            ha, wa = 1, 2

        if points and points is not None:
            patch_ctr = points.pop(0)
            plane = patch_ctr[axis]
            plane = max(2, plane)
            plane = min(image.shape[axis] - 3, plane)
            fb_slice = slice(plane - 2, plane + 3)

            ys = int(patch_ctr[ha] - patch_size / 2)
            ys = min(ys, image.shape[ha] - patch_size)
            ys = max(ys, 0)
            xs = int(patch_ctr[wa] - patch_size / 2)
            xs = min(xs, image.shape[wa] - patch_size)
            xs = max(xs, 0)

            ye = min(ys + patch_size, image.shape[ha])
            xe = min(xs + patch_size, image.shape[wa])
        else:
            # pick a plane from sample of every 3
            plane = np.random.randint(2, image.shape[axis] // 3) * 3
            fb_slice = slice(plane - 2, plane + 3)

            ys = np.random.choice(
                np.arange(0, max(1, image.shape[ha] - patch_size), patch_size)
            )
            xs = np.random.choice(
                np.arange(0, max(1, image.shape[wa] - patch_size), patch_size)
            )
            ye = min(ys + patch_size, image.shape[ha])
            xe = min(xs + patch_size, image.shape[wa])

        flipbook = take(image, fb_slice, axis)
        flipbook = take(flipbook, slice(ys, ye), ha)
        flipbook = take(flipbook, slice(xs, xe), wa)

        if type(flipbook) == da.Array:
            flipbook = flipbook.compute()

        if axis == 1:
            flipbook = flipbook.transpose(1, 0, 2)
        elif axis == 2:
            flipbook = flipbook.transpose(2, 0, 1)

        flipbook = _pad_flipbook(flipbook, (patch_size, patch_size))

        flipbooks.append(flipbook)
        locs.append((axis, fb_slice.start, fb_slice.stop, ys, ye, xs, xe))

    return np.stack(flipbooks, axis=0), locs


def _pick_paired_flipbooks(image, label, patch_size, num_patches, points, isotropic):
    flipbooks = []
    label_flipbooks = []
    locs = []
    for _ in range(num_patches):
        if isotropic:
            axes = [0, 1, 2]
            axis = random.choice(axes)

            # set height and width axes
            del axes[axes.index(axis)]
            ha, wa = axes
        else:
            axis = 0
            ha, wa = 1, 2

        if points and points is not None:
            patch_ctr = points.pop(0)
            plane = patch_ctr[axis]
            plane = max(2, plane)
            img_plane = min(image.shape[axis] - 3, plane)
            label_plane = min(label.shape[axis] - 3, plane)
            img_fb_slice = slice(img_plane - 2, img_plane + 3)
            label_fb_slice = slice(label_plane - 2, label_plane + 3)

            ys = int(patch_ctr[ha] - patch_size / 2)
            ys = min(ys, image.shape[ha] - patch_size)
            ys = max(ys, 0)
            xs = int(patch_ctr[wa] - patch_size / 2)
            xs = min(xs, image.shape[wa] - patch_size)
            xs = max(xs, 0)

            ye = min(ys + patch_size, image.shape[ha])
            xe = min(xs + patch_size, image.shape[wa])
        else:
            # pick a plane from sample of every 3
            plane = np.random.randint(2, image.shape[axis] // 3) * 3
            img_fb_slice = slice(plane - 2, plane + 3)
            label_fb_slice = slice(plane - 2, plane + 3)

            ys = np.random.choice(
                np.arange(0, max(1, image.shape[ha] - patch_size), patch_size)
            )
            xs = np.random.choice(
                np.arange(0, max(1, image.shape[wa] - patch_size), patch_size)
            )
            ye = min(ys + patch_size, image.shape[ha])
            xe = min(xs + patch_size, image.shape[wa])

        flipbook = take(image, img_fb_slice, axis)
        flipbook = take(flipbook, slice(ys, ye), ha)
        flipbook = take(flipbook, slice(xs, xe), wa)

        label_flipbook = take(label, label_fb_slice, axis)
        label_flipbook = take(label_flipbook, slice(ys, ye), ha)
        label_flipbook = take(label_flipbook, slice(xs, xe), wa)

        if type(flipbook) and type(label_flipbook) == da.Array:
            flipbook = flipbook.compute()
            label_flipbook = label_flipbook.compute()

        if axis == 1:
            flipbook = flipbook.transpose(1, 0, 2)
            label_flipbook = label_flipbook.transpose(1, 0, 2)
        elif axis == 2:
            flipbook = flipbook.transpose(2, 0, 1)
            label_flipbook = label_flipbook.transpose(2, 0, 1)

        flipbook = _pad_flipbook(flipbook, (patch_size, patch_size))
        label_flipbook = _pad_label_flipbook(label_flipbook, (patch_size, patch_size))

        flipbooks.append(flipbook)
        label_flipbooks.append(label_flipbook)
        locs.append((axis, img_fb_slice.start, img_fb_slice.stop, ys, ye, xs, xe))

    return np.stack(flipbooks, axis=0), np.stack(label_flipbooks, axis=0), locs


def pick_patches(
    image,
    patch_size,
    num_patches,
    points,
    isotropic=False,
    is_2d_stack=False,
    pick_points_only=False,
    label=None,
):
    """Dispatch to the 2D patch or 3D flipbook picker.

    ``image`` (and ``label``, when picking paired data) are the arrays to crop
    from, already resolved to the wanted pyramid level, and ``points`` are the
    patch centres in the coordinate space of those arrays (an empty sequence or
    None picks random locations). ``patch_size``, ``isotropic``,
    ``is_2d_stack`` and ``pick_points_only`` hold the values the widget read
    from its gui parameters.
    """
    if label is not None:
        assert label.shape == image.shape

    ndim = image.ndim
    assert ndim in [2, 3], "Must be 2D or 3D data!"

    if pick_points_only:
        num_patches = len(points)

    if ndim == 2 or is_2d_stack:
        if label is not None:
            return _pick_paired_patches(image, label, patch_size, num_patches, points)
        else:
            return _pick_patches(image, patch_size, num_patches, points)
    else:
        if label is not None:
            return _pick_paired_flipbooks(
                image, label, patch_size, num_patches, points, isotropic
            )
        else:
            return _pick_flipbooks(image, patch_size, num_patches, points, isotropic)


def patch_suffices(locs, pyramid_level):
    """Suffixes used to name picked 2D patches and to crop them when saving."""
    if len(locs[0]) == 5:
        return [
            f"s{pyramid_level}-LOC-2d-{l[0]}_{l[1]}-{l[2]}_{l[3]}-{l[4]}" for l in locs
        ]
    else:
        return [f"s{pyramid_level}-LOC-2d_{l[0]}-{l[1]}_{l[2]}-{l[3]}" for l in locs]


def flipbook_suffices(locs, pyramid_level):
    """Suffixes used to name picked flipbooks and to crop them when saving."""
    return [
        f"s{pyramid_level}-LOC-{l[0]}_{l[1]}-{l[2]}_{l[3]}-{l[4]}_{l[5]}-{l[6]}"
        for l in locs
    ]


def _random_suffix(size=10):
    # printing letters
    letters = string.ascii_letters
    digits = string.digits

    pstr = []
    for _ in range(size):
        if random.random() < 0.5:
            pstr.append(random.choice(letters))
        else:
            pstr.append(random.choice(digits))

    return "".join(pstr)


def store_dataset(patches, patch_labels, save_dir, dataset_name, metadata=None):
    """Write picked patches and masks as tiffs under ``save_dir/dataset_name``.

    ``patches`` and ``patch_labels`` are the arrays the widget read from the
    image and labels layers, ``metadata`` the image layer's metadata dict
    (``{'prefix': ..., 'suffices': [...]}``), which supplies the file prefix and
    the suffixes used to crop padding back off; when it is empty or None a
    random suffix is used and nothing is cropped.
    """
    from skimage import io

    assert dataset_name, "Must provide a dataset name!"

    assert patches.shape == patch_labels.shape, "Patch and label shapes must match!"

    if metadata:
        has_metadata = True
        prefix = metadata["prefix"]
        suffices = metadata["suffices"]
    else:
        has_metadata = False
        prefix = "unknown"
        suffices = ["-" + _random_suffix() for _ in range(len(patches))]

    if patches.ndim == 4:
        # get middle images of flipbooks
        images = patches[:, 2]
        masks = patch_labels[:, 2]
    else:
        images = patches
        masks = patch_labels

    outdir = os.path.join(save_dir, dataset_name)
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
        print("Created directory", outdir)
    else:
        print("Adding images to existing directory", outdir)

    os.makedirs(os.path.join(outdir, f"{prefix}/images"), exist_ok=True)
    os.makedirs(os.path.join(outdir, f"{prefix}/masks"), exist_ok=True)

    for sfx, img, msk in zip(suffices, images, masks):
        fname = f"{prefix}{sfx}.tiff"

        # if we have metadata, use it to crop image and mask
        # and remove excess padding
        if has_metadata:
            hrange, wrange = sfx.split("_")[-2:]
            hmin, hmax = hrange.split("-")
            wmin, wmax = wrange.split("-")
            h, w = int(hmax) - int(hmin), int(wmax) - int(wmin)

            img = img[:h, :w]
            msk = msk[:h, :w]

        io.imsave(
            os.path.join(outdir, f"{prefix}/images/{fname}"), img, check_contrast=False
        )
        io.imsave(
            os.path.join(outdir, f"{prefix}/masks/{fname}"),
            msk.astype(np.int32),
            check_contrast=False,
        )

    print("Finished saving.")

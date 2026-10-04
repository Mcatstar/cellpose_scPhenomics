import math
import os

import dask.array as da
import numpy as np
from skimage import io


def export_batch_segs(
    image, mask, image_name, export_type, dataset_name, save_dir, grayscale
):
    """
    Save labels as individual 2D images or a single 3D image.

    image: grayscale image array, or None if only masks are written.
    mask: labels array.
    image_name: name used as the filename prefix.
    export_type: key of save_ops, '2D images' or '3D image'.
    dataset_name: folder name created inside save_dir.
    save_dir: directory in which to save segmentations.
    grayscale: also write the image array next to the masks.
    """
    save_ops = {
        "2D images": "2D images",
        "3D image": "3D image",
    }

    def _get_impaths_from_dask(dask_array):
        # delayed keys
        keys = [l for l in dask_array.dask.layers if "imread" in l]
        # absolute image paths
        return [dask_array.dask[k][1] for k in keys]

    assert dataset_name, "Must provide a dataset name!"

    outdir = os.path.join(save_dir, dataset_name)
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
        print("Created directory", outdir)
    else:
        print("Adding images to existing directory", outdir)

    if grayscale:
        os.makedirs(os.path.join(outdir, "images"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "masks"), exist_ok=True)

    export_option = save_ops[export_type]
    # the image is optional, shapes and names come from the labels array if it is None
    shape_source = mask if image is None else image

    assert shape_source.shape[0] == mask.shape[0], (
        f"Image and labels layer must have the same number of images, got {shape_source.shape} and {mask.shape}"
    )
    assert image is not None or not grayscale, (
        "grayscale=True requires an image array, got image=None"
    )

    if shape_source.ndim == 3:
        if isinstance(image, da.Array):
            imnames = [
                ".".join(os.path.basename(imp).split(".")[:-1]) + ".tiff"
                for imp in _get_impaths_from_dask(image)
            ]
        else:
            zpad = math.ceil(math.log(shape_source.shape[0], 10))
            imnames = [
                image_name + "_" + str(n).zfill(zpad) + ".tiff"
                for n in range(shape_source.shape[0])
            ]
            # imnames = [str(n).zfill(zpad) + '.tiff' for n in range(len(image))]

        if export_option == "3D image":
            # Creates a 3D image or 2D stack of images from the label image layer
            i, h, w = shape_source.shape
            seg_stack = np.squeeze(mask[:i, :h, :w]).astype(np.int32)
            imname = image_name + ".tiff"

            if grayscale:
                # grayscale implies image is not None (asserted above), so
                # shape_source is the image here.
                img_stack = np.squeeze(shape_source[:i, :h, :w])
                io.imsave(
                    os.path.join(outdir, f"images/{imname}"),
                    img_stack,
                    check_contrast=False,
                )
            io.imsave(
                os.path.join(outdir, f"masks/{imname}"), seg_stack, check_contrast=False
            )

        else:
            for i in range(shape_source.shape[0]):
                imname = imnames[i]
                if isinstance(image, da.Array):
                    h, w = image[i].compute().shape
                else:
                    h, w = shape_source[i].shape

                seg = np.squeeze(mask[i, :h, :w]).astype(np.int32)

                if grayscale:
                    # grayscale implies image is not None (asserted above).
                    img = np.squeeze(shape_source[i, :h, :w])
                    io.imsave(
                        os.path.join(outdir, f"images/{imname}"),
                        img,
                        check_contrast=False,
                    )
                io.imsave(
                    os.path.join(outdir, f"masks/{imname}"), seg, check_contrast=False
                )

    else:
        imname = image_name + ".tiff"
        if grayscale:
            io.imsave(
                os.path.join(outdir, f"images/{imname}"),
                shape_source,
                check_contrast=False,
            )
        io.imsave(
            os.path.join(outdir, f"masks/{imname}"),
            mask.astype(np.int32),
            check_contrast=False,
        )

    print("Segmentations exported!")

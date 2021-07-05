# credit: https://theaisummer.com/medical-image-processing/?fbclid=IwAR17i5PBnE_ihFGjQZ2kUr4g2w3x02QK9Z7asnUT-4YDYKYmUZ8hvstLOD4
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


def show_mid_slice(img_numpy, title='img'):
    """
    Accepts an 3D numpy array and shows median slices in all three planes
    """
    assert img_numpy.ndim == 3
    n_i, n_j, n_k = img_numpy.shape

    # sagittal (left image)
    center_i1 = int((n_i - 1) / 2)
    # coronal (center image)
    center_j1 = int((n_j - 1) / 2)
    # axial slice (right image)
    center_k1 = int((n_k - 1) / 2)

    show_slices([img_numpy[center_i1, :, :],
                 img_numpy[:, center_j1, :],
                 img_numpy[:, :, center_k1]])
    plt.suptitle(title)


def show_slices(slices):
    """
    Function to display a row of image slices
    Input is a list of numpy 2D image slices
    """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


def resize_data_volume_by_scale(data, scale):
    """
    Resize the data based on the provided scale
    """
    scale_list = [scale, scale, scale]
    return scipy.ndimage.interpolation.zoom(data, scale_list, order=0)


# result = resize_data_volume_by_scale(epi_img_numpy, 0.5)
# result2 = resize_data_volume_by_scale(epi_img_numpy, 2)

def random_zoom(matrix, min_percentage=0.7, max_percentage=1.2):
    z = np.random.sample() * (max_percentage - min_percentage) + min_percentage
    zoom_matrix = np.array([[z, 0, 0, 0],
                            [0, z, 0, 0],
                            [0, 0, z, 0],
                            [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(matrix, zoom_matrix)


def random_rotate3D(img_numpy, min_angle, max_angle):
    """
    Returns a random rotated array in the same shape
    :param img_numpy: 3D numpy array
    :param min_angle: in degrees
    :param max_angle: in degrees
    """
    assert img_numpy.ndim == 3, "provide a 3d numpy array"
    assert min_angle < max_angle, "min should be less than max val"
    assert min_angle > -360 or max_angle < 360
    all_axes = [(1, 0), (1, 2), (0, 2)]
    angle = np.random.randint(low=min_angle, high=max_angle + 1)
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]
    return scipy.ndimage.rotate(img_numpy, angle, axes=axes)


def random_flip(img, label=None):
    axes = [0, 1, 2]
    rand = np.random.randint(0, 3)
    img = flip_axis(img, axes[rand])
    img = np.squeeze(img)

    if label is None:
        return img
    else:
        label = flip_axis(label, axes[rand])
        label = np.squeeze(label)
    return img, label


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


# Medical image shifting (displacement)
def transform_matrix_offset_center_3d(matrix, x, y, z):
    offset_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(matrix, offset_matrix)


def random_shift(img_numpy, max_percentage=0.4):
    dim1, dim2, dim3 = img_numpy.shape
    m1, m2, m3 = int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2)
    d1 = np.random.randint(-m1, m1)
    d2 = np.random.randint(-m2, m2)
    d3 = np.random.randint(-m3, m3)
    return transform_matrix_offset_center_3d(img_numpy, d1, d2, d3)


# Random 3D crop
def crop_3d_volume(img_tensor, crop_dim, crop_size):
    assert img_tensor.ndim == 3, '3d tensor must be provided'
    full_dim1, full_dim2, full_dim3 = img_tensor.shape
    slices_crop, w_crop, h_crop = crop_dim
    dim1, dim2, dim3 = crop_size

    # check if crop size matches image dimensions
    if full_dim1 == dim1:
        img_tensor = img_tensor[:, w_crop:w_crop + dim2, h_crop:h_crop + dim3]
    elif full_dim2 == dim2:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, :, h_crop:h_crop + dim3]
    elif full_dim3 == dim3:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2, :]
    # standard crop
    else:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2, h_crop:h_crop + dim3]
    return img_tensor


def find_random_crop_dim(full_vol_dim, crop_size):
    assert full_vol_dim[0] >= crop_size[0], "crop size is too big"
    assert full_vol_dim[1] >= crop_size[1], "crop size is too big"
    assert full_vol_dim[2] >= crop_size[2], "crop size is too big"

    if full_vol_dim[0] == crop_size[0]:
        slices = crop_size[0]
    else:
        slices = np.random.randint(full_vol_dim[0] - crop_size[0])

    if full_vol_dim[1] == crop_size[1]:
        w_crop = crop_size[1]
    else:
        w_crop = np.random.randint(full_vol_dim[1] - crop_size[1])

    if full_vol_dim[2] == crop_size[2]:
        h_crop = crop_size[2]
    else:
        h_crop = np.random.randint(full_vol_dim[2] - crop_size[2])

    return slices, w_crop, h_crop


# Clip intensity values (outliers)
def percentile_clip(img_numpy, min_val=5, max_val=95):
    """
    Intensity normalization based on percentile
    Clips the range based on the quartile values.
    :param img_numpy: original
    :param min_val: should be in the range [0,100]
    :param max_val: should be in the range [0,100]
    :return: intensity normalized image
    """
    low = np.percentile(img_numpy, min_val)
    high = np.percentile(img_numpy, max_val)

    img_numpy[img_numpy < low] = low
    img_numpy[img_numpy > high] = high
    return img_numpy


def clip_range(img_numpy, min_intensity=10, max_intensity=240):
    return np.clip(img_numpy, min_intensity, max_intensity)


# Intensity normalization in medical images
def normalize_intensity(img_tensor, normalization="mean"):
    """
    Accepts an image tensor and normalizes it
    :param img_tensor: original
    :param normalization: choices = "max", "mean" , type=str
    For mean normalization we use the non zero voxels only.
    """
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val
    elif normalization == "max":
        max_val, min_val = img_tensor.max(), img_tensor.min()
        img_tensor = (img_tensor - min_val) / (max_val - min_val)
    return img_tensor


# Elastic deformation
def elastic_transform_3d(image, labels=None, alpha=4, sigma=35, bg_val=0.1):
    """
    Elastic deformation of images as described in
    Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual
    Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.

    Modified from:
    https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62

    Modified to take 3D inputs
    Deforms both the image and corresponding label file
    image linear/trilinear interpolated

    Label volumes nearest neighbour interpolated
    """
    assert image.ndim == 3
    shape = image.shape
    dtype = image.dtype

    # Define coordinate system
    coords = np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])

    # Initialize interpolators
    im_intrps = RegularGridInterpolator(coords, image,
                                        method="linear",
                                        bounds_error=False,
                                        fill_value=bg_val)

    # Get random elastic deformations
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha

    # Define sample points
    x, y, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))

    # Interpolate 3D image image
    image = np.empty(shape=image.shape, dtype=dtype)
    image = im_intrps(indices).reshape(shape)

    # Interpolate labels
    if labels is not None:
        lab_intrp = RegularGridInterpolator(coords, labels,
                                            method="nearest",
                                            bounds_error=False,
                                            fill_value=0)

        labels = lab_intrp(indices).reshape(shape).astype(labels.dtype)
        return image, labels

    return image

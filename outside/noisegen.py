"""
Credit: http://www.xiaoliangbai.com/2016/09/09/more-on-image-noise-generation
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc


# Ensure plots embeded in notebook
# %matplotlib inline


# Source of the code is based on an excelent piece code from stackoverflow
# http://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv

def noise_generator(noise_type, image):
    """
    Generate noise to a given Image based on required noise type

    Input parameters:
        image: ndarray (input image data. It will be converted to float)

        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    """
    row, col, ch = image.shape
    if noise_type == "gauss":
        mean = 0.0
        var = 0.01
        sigma = var ** 0.5
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy.astype('uint8')
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy
    else:
        return image


# Open Image
plt.figure(0)
im = misc.face()
plt.imshow(im)
plt.axis('off')
plt.show()
plt.close(0)
# print im

plt.figure(1)
sp_im = noise_generator('s&p', im)
plt.imshow(sp_im)
plt.axis('off')
plt.show()
plt.close(1)
# print sp_im

plt.figure(2)
gauss_im = noise_generator('gauss', im)
plt.subplot(1, 2, 1)
plt.title('Salt & Pepper Noise')
plt.imshow(sp_im)
plt.subplot(1, 2, 2)
plt.imshow(gauss_im)
plt.title('Gaussian Noise')
plt.show()
plt.close(2)

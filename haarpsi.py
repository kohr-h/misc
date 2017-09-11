import numpy as np
import pywt
import scipy.misc
import scipy.signal


# Global filter parameters
wave = pywt.Wavelet('haar')
dec_lo_lvl1 = np.array(wave.dec_lo)
dec_lo_lvl2 = np.repeat(dec_lo_lvl1, 2)
dec_lo_lvl3 = np.repeat(dec_lo_lvl2, 2)
dec_hi_lvl1 = np.array(wave.dec_hi)
dec_hi_lvl2 = np.repeat(dec_hi_lvl1, 2)
dec_hi_lvl3 = np.repeat(dec_hi_lvl2, 2)


def filter_image(image, fh, fv):
    image = scipy.signal.convolve(image, fh[:, None], mode='same',
                                  method='direct')
    return scipy.signal.convolve(image, fv[None, :], mode='same',
                                 method='direct')


def similarity(x, y, c):
    return (2 * x * y + c) / (x ** 2 + y ** 2 + c)


def logistic(x, alpha):
    return 1 / (1 + np.exp(-alpha * x))


def inv_logistic(x, alpha):
    return (np.log(x) - np.log(1 - x)) / alpha


def local_sim_horiz(img1, img2, c, alpha):
    img1_h1 = filter_image(img1, fh=dec_hi_lvl1, fv=dec_lo_lvl1)
    img1_h2 = filter_image(img1, fh=dec_hi_lvl2, fv=dec_lo_lvl2)

    img2_h1 = filter_image(img2, fh=dec_hi_lvl1, fv=dec_lo_lvl1)
    img2_h2 = filter_image(img2, fh=dec_hi_lvl2, fv=dec_lo_lvl2)

    sim_lvl1 = similarity(np.abs(img1_h1), np.abs(img2_h1), c)
    sim_lvl2 = similarity(np.abs(img1_h2), np.abs(img2_h2), c)

    sim = (sim_lvl1 + sim_lvl2) / 2
    return logistic(sim, alpha)


def local_sim_vert(img1, img2, c, alpha):
    img1_v1 = filter_image(img1, fh=dec_lo_lvl1, fv=dec_hi_lvl1)
    img1_v2 = filter_image(img1, fh=dec_lo_lvl2, fv=dec_hi_lvl2)

    img2_v1 = filter_image(img2, fh=dec_lo_lvl1, fv=dec_hi_lvl1)
    img2_v2 = filter_image(img2, fh=dec_lo_lvl2, fv=dec_hi_lvl2)

    sim_lvl1 = similarity(np.abs(img1_v1), np.abs(img2_v1), c)
    sim_lvl2 = similarity(np.abs(img1_v2), np.abs(img2_v2), c)

    sim = (sim_lvl1 + sim_lvl2) / 2
    return logistic(sim, alpha)


def weight_map_horiz(img1, img2):
    img1_h3 = filter_image(img1, fh=dec_hi_lvl3, fv=dec_lo_lvl3)
    img2_h3 = filter_image(img2, fh=dec_hi_lvl3, fv=dec_lo_lvl3)

    return np.maximum(np.abs(img1_h3), np.abs(img2_h3))


def weight_map_vert(img1, img2):
    img1_v3 = filter_image(img1, fh=dec_lo_lvl3, fv=dec_hi_lvl3)
    img2_v3 = filter_image(img2, fh=dec_lo_lvl3, fv=dec_hi_lvl3)

    return np.maximum(np.abs(img1_v3), np.abs(img2_v3))


def haarpsi(img1, img2, c, alpha):
    lsim_horiz = local_sim_horiz(img1, img2, c, alpha)
    lsim_vert = local_sim_vert(img1, img2, c, alpha)

    wmap_horiz = weight_map_horiz(img1, img2)
    wmap_vert = weight_map_vert(img1, img2)

    numer = np.sum(lsim_horiz * wmap_horiz + lsim_vert * wmap_vert)
    denom = np.sum(wmap_horiz + wmap_vert)

    return inv_logistic(numer / denom, alpha) ** 2


# %% Testing ground

img1 = np.asarray(scipy.misc.ascent(), dtype=float)
img1 *= 100 / img1.max()
img2 = np.sum(scipy.misc.face(), axis=-1)[:512, :512].astype(float)
img2 *= 100 / img2.max()

c = 30
alpha = 4.2
print(haarpsi(img1, img1, c, alpha))
print(haarpsi(img1, img2, c, alpha))
print(haarpsi(img1, img1 + 5 * np.random.rand(*img1.shape), c, alpha))

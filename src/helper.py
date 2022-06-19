import numpy as np


def to_freq(img):
    """
    Returns the Fourier transform of a 2D array
    """
    H, W = img.shape[0], img.shape[1]
    aug_img = np.zeros((2*H, 2*W))
    aug_img[:H, :W] = img
    return np.fft.fftshift(np.fft.fft2(aug_img))


def from_freq(img):
    """
    Returns the inverse Fourier transform of a 2D array
    """
    H, W = img.shape[0]//2, img.shape[1]//2
    aug_img = np.fft.ifft2(np.fft.ifftshift(img))
    img = np.abs(aug_img[:H, :W])
    return img


def to_freq_3d(img):
    """
    Returns the Fourier transform of a 3D array
    """
    T, H, W = img.shape
    aug_img = np.zeros((2*T, 2*H, 2*W))
    aug_img[:T, :H, :W] = img
    return np.fft.fftshift(np.fft.fftn(aug_img))


def from_freq_3d(img):
    """
    Returns the inverse Fourier transform of a 3D array
    """
    T, H, W = (np.array(img.shape)/2).astype('int')
    aug_img = np.fft.ifftn(np.fft.ifftshift(img))
    img = np.abs(aug_img[:T, :H, :W])
    return img


def triple_int(f, ranges, steps):
    """
    Calculates triple integration using Reimann sum
    """
    da, db, dc = map(lambda x: x, steps)
    [am, aM], [bm, bM], [cm, cM] = ranges
    vol = da*db*dc
    al = np.linspace(am, aM, int((aM-am)/da))
    bl = np.linspace(bm, bM, int((bM-bm)/db))
    cl = np.linspace(cm, cM, int((cM-cm)/dc))
    if len(al) == 0:
        vol = vol/da
        al = [am]
    if len(bl) == 0:
        vol = vol/db
        bl = [bm]
    if len(cl) == 0:
        vol = vol/dc
        cl = [cm]
    ans = 0*f(am, bm, cm)
    for a in al:
        for b in bl:
            for c in cl:
                ans += vol*f(a, b, c)
    return ans

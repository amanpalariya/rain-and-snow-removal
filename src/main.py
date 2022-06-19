import cv2 as cv
from helper import from_freq, from_freq_3d, to_freq, to_freq_3d, triple_int
import numpy as np


def b(a, z, f):
    """
    Returns the imaged length by a camera of focal length `f` if the actual length is `a` at depth `z`
    """
    return a*f/z


def s(a):
    """
    Returns terminal velocity (in mm/s) of raindrops of size `a` mm 
    """
    return 1000*np.dot(np.array([a**3, a**2, a**1, 1]), np.array([0.056584, -0.90441, 4.9625, -0.19274]))


def l(a, z, f, e):
    """
    Returns the imaged length of rain/snow streak
    """
    return b(a + s(a)*e, z, f)


def g(W, H, a, z, theta, mu_x, mu_y, f, e):
    """
    Return an image with single raindrop rendered according to given parameters
    """
    dgamma = 100
    length = l(a, z, f, e)
    drop_dia = b(a, z, f)
    w = 1 + drop_dia
    h = 1 + length + drop_dia
    _h, _w = int(h)+1, int(w)+1
    img = np.zeros((H+2*_h, W+2*_w))
    mu_x += _w
    mu_y += _h
    tl = np.floor([mu_x, mu_y])
    br = np.ceil([mu_x+w, mu_y+h])
    tlx, tly = tl.astype('int')
    brx, bry = br.astype('int')
    x, y, gamma = np.meshgrid(np.linspace(tlx, brx, (brx-tlx+1)), np.linspace(tly, bry, (bry-tly+1)), np.linspace(0, l(a, z, f, e), dgamma), sparse=False)
    img[tly:bry+1, tlx:brx+1] += np.trapz(np.exp(-(np.power(x - np.cos(np.pi/2)*gamma - mu_x, 2) + np.power(y-np.sin(np.pi/2)*gamma - mu_y, 2))/(np.power(b(a, z, f), 2))), gamma)  # Integration
    return cv.warpAffine(img, cv.getRotationMatrix2D((mu_x, mu_y), (np.pi/2-theta)*180/np.pi, 1), dsize=(_w+W, _h+H))[_h:, _w:]


def G(W, H, a, z, theta, mu_x, mu_y, f, e):
    """
    Return the Fourier transform of an image with single raindrop rendered according to given parameters
    """
    img = g(W, H, a, z, theta, mu_x, mu_y, f, e)
    aug = np.zeros(2*np.array(img.shape))
    aug[:H, :W] = img
    return np.fft.fftshift(np.fft.fft2(aug))


def R(W, H, L, theta_max, theta_min, f, e, z_max=350, z_min=150, a_max=3, a_min=0.1):
    """
    Returns the magnitude spectrum of the rain/snow model
    """
    def integrand3(theta, a, z):
        return ((z*z)/(z_max*z_max))*(G(W, H, a, z, theta, np.floor(np.random.uniform(0, W)), np.floor(np.random.uniform(0, H)), f, e))
    d_theta = 0.1
    d_a = 0.2
    d_z = 100
    return L*np.abs(triple_int(integrand3, [[theta_min, theta_max], [a_min, a_max], [z_min, z_max]], [d_theta, d_a, d_z]))


def generate_rain(W, H, theta, N=500):
    """
    Renders `N` streaks on a `W`x`H` image 
    """
    mu_x_list = np.random.uniform(0, W-1, N)
    mu_y_list = np.random.uniform(0, H-1, N)
    a_list = np.random.uniform(0.1, 3, N)
    z_list = np.clip(np.random.normal(800, 200, N), 10, np.inf)
    theta_list = np.random.normal(theta, 0.01, N)
    f, e = 30, 1/30
    img = np.zeros((H, W), float)
    for a, z, theta, mu_x, mu_y in zip(a_list, z_list, theta_list, mu_x_list, mu_y_list):
        img += g(W, H, a, z, theta, mu_x, mu_y, f, e)
    return img


def get_full_temporal_median(m, window_half_size=1):
    """
    Calculates temporal median in a window of size (2*window_half_size + 1)
    The size of output is same as that of input `m`
    """
    T, H, W = m.shape
    med = np.zeros((T, H, W))
    for i in range(T):
        med[i] = np.median(m[max(0, i-window_half_size):min(T-1, i+window_half_size)], axis=0)
    return med


def detect(m, theta_min, theta_max=None):
    """
    Returns the detected rain in a movie `m`
    The rain/snow is in range [theta_min, theta_max] where angle is radians and measured in clockwise direction from positive x-axis 
    NOTE: For large `m`, the program may consume a lot of resources and eventually halt
    """
    T, H, W = m.shape
    f, e = 30, 1/30

    f3d_m = to_freq_3d(m)

    L = np.median(np.abs(f3d_m))/np.median(R(W, H, 1, np.pi/2, np.pi/2, f, e))
    if theta_max is None:
        theta_max = theta_min

    r_star = R(W, H, L, theta_max, theta_min, f, e)

    p2 = np.zeros((T, H, W))
    for i in range(T):
        M_i = to_freq(m[i])
        p2[i] = from_freq(np.clip(np.abs(r_star)/(1+np.abs(M_i)), 0, 1)*np.exp(1j*np.angle(M_i)))

    P2 = to_freq_3d(p2)
    del p2

    r_new_freq = np.abs(r_star)/(1+np.abs(P2))*(np.exp(1j*np.angle(f3d_m)))
    del P2
    del r_star
    del f3d_m

    r_new = from_freq_3d(r_new_freq)
    del r_new_freq
    return r_new/r_new.max()


def remove(m, rain_mask, removal_rate, median_window_half_size=3):
    """
    Removes the rain from a movie based on the `rain_mask` and `removal_rate`
    """
    alpha_p = np.clip(rain_mask*removal_rate, 0, 1)
    return (1-alpha_p)*m + alpha_p*get_full_temporal_median(m, median_window_half_size)


def process_file(filename, output_filename, theta_min, theta_max=None, iterations=1):
    """
    Detects and removes rain from a video file
    """
    chunks = 5
    cap = cv.VideoCapture(filename)
    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"{W}×{H} @ {fps} fps")
    out = cv.VideoWriter(output_filename+'.avi', cv.VideoWriter_fourcc(*'XVID'), fps, (W, H))
    out2 = cv.VideoWriter(output_filename+'_detected.avi', cv.VideoWriter_fourcc(*'XVID'), fps, (W, H))
    ret, frame = cap.read()
    i = 0
    mb, mg, mr = np.zeros((chunks, H, W)), np.zeros((chunks, H, W)), np.zeros((chunks, H, W))
    try:
        while ret:
            b, g, r = cv.split(frame)
            mb[i % chunks] = b
            mg[i % chunks] = g
            mr[i % chunks] = r
            i += 1
            if i % chunks == 0:
                for j in range(iterations):
                    mb_detected = detect(mb, theta_min, theta_max)
                    mg_detected = detect(mg, theta_min, theta_max)
                    mr_detected = detect(mr, theta_min, theta_max)
                    mb = remove(mb, mb_detected, 3).astype('uint8')
                    mg = remove(mg, mg_detected, 3).astype('uint8')
                    mr = remove(mr, mr_detected, 3).astype('uint8')
                for j in range(chunks):
                    out.write(cv.merge((mb[j], mg[j], mr[j])))
                    out2.write(cv.merge(
                        (
                            np.interp(mb_detected[j], [mb_detected[j].min(), mb_detected[j].max()], [0, 255]).astype('uint8'),
                            np.interp(mb_detected[j], [mb_detected[j].min(), mb_detected[j].max()], [0, 255]).astype('uint8'),
                            np.interp(mb_detected[j], [mb_detected[j].min(), mb_detected[j].max()], [0, 255]).astype('uint8')
                        )
                    ))
                mb, mg, mr = np.zeros((chunks, H, W)), np.zeros((chunks, H, W)), np.zeros((chunks, H, W))
                print(f"{i}/{total_frames} frames cleaned")
            ret, frame = cap.read()
    except KeyboardInterrupt:
        pass
    cap.release()
    out.release()
    out2.release()

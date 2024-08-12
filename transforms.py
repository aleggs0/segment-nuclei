import numpy as np
import cv2
import skimage

def pad_to_multiple(img,lbl,k=32):
    #zero pad in order to make input sizes divisible by k
    shape_img = img.shape
    img=np.pad(img,((0,-shape_img[0]%16),(0,-shape_img[1]%k)))
    lbl=np.pad(lbl,((0,-shape_img[0]%16),(0,-shape_img[1]%k)))
    shape_img = img.shape
    assert shape_img[0]%16==0 & shape_img[1]%k==0
    return img,lbl

def normalize99(Y, lower=1, upper=99, copy=True):
    """
    Normalize the image so that 0.0 corresponds to the 1st percentile and 1.0 corresponds to the 99th percentile.

    Args:
        Y (ndarray): The input image.
        lower (int, optional): The lower percentile. Defaults to 1.
        upper (int, optional): The upper percentile. Defaults to 99.
        copy (bool, optional): Whether to create a copy of the input image. Defaults to True.

    Returns:
        ndarray: The normalized image.
    """
    X = Y.copy() if copy else Y
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    if x99 - x01 > 1e-3:
        X = (X - x01) / (x99 - x01)
    else:
        X[:] = 0
    return X

def random_rotate_and_resize(img, labels, xy=(336, 336), do_3D=False,
                             do_flip=True, rotate=True, unet=False,
                             random_per_image=True):
    Ly, Lx = img.shape
    # generate random augmentation parameters
    flip = np.random.rand() > .5
    theta = (np.random.rand() * np.pi * 2 if rotate
                else np.random.randint(0,4)/4 * np.pi)
    scale = np.random.uniform(low=0.75,high=1.25)
    dxy = np.maximum(0, np.array([Lx * scale - xy[1],
                                    Ly * scale - xy[0]])) #amount of leeway for crop frame, or zero
    dxy = (np.random.rand(2,) - .5) * dxy #random amount to slide the crop frame

    # create affine transform
    cc = np.array([Lx / 2, Ly / 2]) #centre of image
    cc1 = cc - np.array([Lx - xy[1], Ly - xy[0]]) / 2 + dxy #where the original centre of image will be in the output
    pts1 = np.float32([cc, cc + np.array([1, 0]), cc + np.array([0, 1])]) #append unit vectors away from centre
    pts2 = np.float32([
        cc1,
        cc1 + scale * np.array([np.cos(theta), np.sin(theta)]),
        cc1 + scale *
        np.array([np.cos(np.pi / 2 + theta),
                    np.sin(np.pi / 2 + theta)])
    ])
    M = cv2.getAffineTransform(pts1, pts2)
    #this includes a random crop!
    if flip and do_flip:
        img = img[..., ::-1]
        labels = labels[..., ::-1]      
    img = cv2.warpAffine(img, M, (xy[1], xy[0]), flags=cv2.INTER_LINEAR)
    labels = cv2.warpAffine(labels, M, (xy[1], xy[0]), flags=cv2.INTER_NEAREST)
    return img, labels

def add_channel_axis(img,lbl):
    #add a channel axis because pytorch's batchnorm expects one
    return img[None,:],lbl[None,:]

def adjust_gamma(image, gamma=1.0, random_affine=True):
    image=image.astype(np.float64) # should already be float64 anyway
    a=min(0,np.min(image))
    b=max(1,np.max(image))
    image = (image-a)/(b-a)
    image=skimage.exposure.adjust_gamma(image, gamma=gamma)
    image/=((1-a)/(b-a)**gamma-(0-a)/(b-a)**gamma)
    image-=(0-a)**gamma
    #maintains normalize99 properties
    if random_affine:
        #slightly break the normalize99 properties
        image*=np.exp(np.random.normal(0,0.05))
        image+=np.random.normal(0,0.05)
    return image

def make_tiles(imgi, bsize=224, augment=False, tile_overlap=0.1):
    """Make tiles of image to run at test-time.

    Args:
        imgi (np.ndarray): Array of shape (nchan, Ly, Lx) representing the input image.
        bsize (int, optional): Size of tiles. Defaults to 224.
        augment (bool, optional): Whether to flip tiles and set tile_overlap=2. Defaults to False.
        tile_overlap (float, optional): Fraction of overlap of tiles. Defaults to 0.1.

    Returns:
        tuple containing
            - IMG (np.ndarray): Array of shape (ntiles, nchan, bsize, bsize) representing the tiles.
            - ysub (list): List of arrays with start and end of tiles in Y of length ntiles.
            - xsub (list): List of arrays with start and end of tiles in X of length ntiles.
            - Ly (int): Height of the input image.
            - Lx (int): Width of the input image.
    """
    nchan, Ly, Lx = imgi.shape
    if augment:
        bsize = np.int32(bsize)
        # pad if image smaller than bsize
        if Ly < bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, bsize - Ly, Lx))), axis=1)
            Ly = bsize
        if Lx < bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, Ly, bsize - Lx))), axis=2)
        Ly, Lx = imgi.shape[-2:]
        # tiles overlap by half of tile size
        ny = max(2, int(np.ceil(2. * Ly / bsize)))
        nx = max(2, int(np.ceil(2. * Lx / bsize)))
        ystart = np.linspace(0, Ly - bsize, ny).astype(int)
        xstart = np.linspace(0, Lx - bsize, nx).astype(int)

        ysub = []
        xsub = []

        # flip tiles so that overlapping segments are processed in rotation
        IMG = np.zeros((len(ystart), len(xstart), nchan, bsize, bsize), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsize])
                xsub.append([xstart[i], xstart[i] + bsize])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1], xsub[-1][0]:xsub[-1][1]]
                # flip tiles to allow for augmentation of overlapping segments
                if j % 2 == 0 and i % 2 == 1:
                    IMG[j, i] = IMG[j, i, :, ::-1, :]
                elif j % 2 == 1 and i % 2 == 0:
                    IMG[j, i] = IMG[j, i, :, :, ::-1]
                elif j % 2 == 1 and i % 2 == 1:
                    IMG[j, i] = IMG[j, i, :, ::-1, ::-1]
    else:
        tile_overlap = min(0.5, max(0.05, tile_overlap))
        bsizeY, bsizeX = min(bsize, Ly), min(bsize, Lx)
        bsizeY = np.int32(bsizeY)
        bsizeX = np.int32(bsizeX)
        # tiles overlap by 10% tile size
        ny = 1 if Ly <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Ly / bsize))
        nx = 1 if Lx <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Lx / bsize))
        ystart = np.linspace(0, Ly - bsizeY, ny).astype(int)
        xstart = np.linspace(0, Lx - bsizeX, nx).astype(int)

        ysub = []
        xsub = []
        IMG = np.zeros((len(ystart), len(xstart), nchan, bsizeY, bsizeX), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsizeY])
                xsub.append([xstart[i], xstart[i] + bsizeX])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1], xsub[-1][0]:xsub[-1][1]]

    return IMG, ysub, xsub, Ly, Lx

# def _run_tiled(net, imgi, batch_size=8, augment=False, bsize=224, tile_overlap=0.1):
#     """ 
#     Run network on tiles of size [bsize x bsize]
    
#     (faster if augment is False)

#     Args:
#         imgs (np.ndarray): The input image or stack of images of size [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan].
#         batch_size (int, optional): Number of tiles to run in a batch. Defaults to 8.
#         augment (bool, optional): Tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
#         tile_overlap (float, optional): Fraction of overlap of tiles when computing flows. Defaults to 0.1.
#         bsize (int, optional): Size of tiles to use in pixels [bsize x bsize]. Defaults to 224.

#     Returns:
#         y (np.ndarray): output of network, if tiled it is averaged in tile overlaps. Size of [Ly x Lx x 3] or [Lz x Ly x Lx x 3].
#             y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability.
#         style (np.ndarray): 1D array of size 256 summarizing the style of the image, if tiled it is averaged over tiles.
#     """
#     nout = net.nout
#     if imgi.ndim == 4:
#         Lz, nchan = imgi.shape[:2]
#         IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi[0], bsize=bsize,
#                                                         augment=augment,
#                                                         tile_overlap=tile_overlap)
#         ny, nx, nchan, ly, lx = IMG.shape
#         batch_size *= max(4, (bsize**2 // (ly * lx))**0.5)
#         yf = np.zeros((Lz, nout, imgi.shape[-2], imgi.shape[-1]), np.float32)
#         styles = []
#         if ny * nx > batch_size:
#             ziterator = trange(Lz, file=tqdm_out)
#             for i in ziterator:
#                 yfi, stylei = _run_tiled(net, imgi[i], augment=augment, bsize=bsize,
#                                          tile_overlap=tile_overlap)
#                 yf[i] = yfi
#                 styles.append(stylei)
#         else:
#             # run multiple slices at the same time
#             ntiles = ny * nx
#             nimgs = max(2, int(np.round(batch_size / ntiles)))
#             niter = int(np.ceil(Lz / nimgs))
#             ziterator = trange(niter, file=tqdm_out)
#             for k in ziterator:
#                 IMGa = np.zeros((ntiles * nimgs, nchan, ly, lx), np.float32)
#                 for i in range(min(Lz - k * nimgs, nimgs)):
#                     IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(
#                         imgi[k * nimgs + i], bsize=bsize, augment=augment,
#                         tile_overlap=tile_overlap)
#                     IMGa[i * ntiles:(i + 1) * ntiles] = np.reshape(
#                         IMG, (ny * nx, nchan, ly, lx))
#                 ya, stylea = _forward(net, IMGa)
#                 for i in range(min(Lz - k * nimgs, nimgs)):
#                     y = ya[i * ntiles:(i + 1) * ntiles]
#                     if augment:
#                         y = np.reshape(y, (ny, nx, 3, ly, lx))
#                         y = transforms.unaugment_tiles(y)
#                         y = np.reshape(y, (-1, 3, ly, lx))
#                     yfi = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
#                     yfi = yfi[:, :imgi.shape[2], :imgi.shape[3]]
#                     yf[k * nimgs + i] = yfi
#                     stylei = stylea[i * ntiles:(i + 1) * ntiles].sum(axis=0)
#                     stylei /= (stylei**2).sum()**0.5
#                     styles.append(stylei)
#         return yf, np.array(styles)
#     else:
#         IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize=bsize,
#                                                         augment=augment,
#                                                         tile_overlap=tile_overlap)
#         ny, nx, nchan, ly, lx = IMG.shape
#         IMG = np.reshape(IMG, (ny * nx, nchan, ly, lx))
#         niter = int(np.ceil(IMG.shape[0] / batch_size))
#         y = np.zeros((IMG.shape[0], nout, ly, lx))
#         for k in range(niter):
#             irange = slice(batch_size * k, min(IMG.shape[0],
#                                                batch_size * k + batch_size))
#             y0, style = _forward(net, IMG[irange])
#             y[irange] = y0.reshape(irange.stop - irange.start, y0.shape[-3],
#                                    y0.shape[-2], y0.shape[-1])
#             # check size models!
#             if k == 0:
#                 styles = style.sum(axis=0)
#             else:
#                 styles += style.sum(axis=0)
#         styles /= IMG.shape[0]
#         if augment:
#             y = np.reshape(y, (ny, nx, nout, bsize, bsize))
#             y = transforms.unaugment_tiles(y)
#             y = np.reshape(y, (-1, nout, bsize, bsize))

#         yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
#         yf = yf[:, :imgi.shape[1], :imgi.shape[2]]
#         styles /= (styles**2).sum()**0.5
#         return yf, styles

def train_transform(img,lbl,shape_out=336):
    """Preprocess training data
    
    Args:
        img (np.ndarray): a single 2D image
        lbl (np.ndarray): labels with values 0 to 5, of the same shape as img
        shape_out (int, optional): shape of output (output must be square)"""

    #take tile and/or pad to 224
    img = normalize99(img)
    lbl=lbl/5
    img,lbl = random_rotate_and_resize(img,lbl)
    if np.random.rand()>0.5:
        img=adjust_gamma(img,gamma=np.random.rand()+0.5)
    img,lbl= add_channel_axis(img,lbl)
    return img,lbl

def val_transform(img,lbl):
    img = normalize99(img)
    lbl=lbl/5
    img,lbl= pad_to_multiple(img,lbl)
    img,lbl= add_channel_axis(img,lbl)
    return img,lbl

# if __name__=="__main__":
#     import tifffile as tiff
#     img = tiff.imread("data/img/well3_em1_t0037_a0_s0022_img.tif")[::2,::2]
#     lbl = tiff.imread("data/div_lbl/well3_em1_t0037_a0_s0022_div.tif")[::2,::2]
#     x,y = random_rotate_and_resize(img,lbl)
#     tiff.imwrite("x.tif",x)

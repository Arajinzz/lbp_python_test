import numpy as np
import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

matplotlib.use("TkAgg")

def paddedImg(img, shape):
    h, w = img.shape
    padded_img = np.zeros((shape[0] + 4, shape[1] + 4), np.uint8)
    padded_img[2:h+2 , 2:w+2] = img
    return padded_img
    

def getHist(img):
    hist = np.zeros(256, np.int16)
    for pixel in img.flatten():
        hist[pixel] += 1

    return hist


# LBP value Rotation Left
def ROL0(block, i, j):
    lval = 0
    px = block.item((i, j))

    lval += 1 if px < block.item(i-1, j+1) else 0
    lval += 2 if px < block.item(i, j+1) else 0
    lval += 4 if px < block.item(i+1, j+1) else 0
    lval += 8 if px < block.item(i+1, j) else 0
    lval += 16 if px < block.item(i+1, j-1) else 0
    lval += 32 if px < block.item(i, j-1) else 0
    lval += 64 if px < block.item(i-1, j-1) else 0
    lval += 128 if px < block.item(i-1, j) else 0

    return lval


# LBP value Rotation Right Clockwise
def ROR0(block, i, j):
    lval = 0
    px = block.item((i, j))

    lval += 1 if px < block.item(i-1, j-1) else 0
    lval += 2 if px < block.item(i-1, j) else 0
    lval += 4 if px < block.item(i-1, j+1) else 0
    lval += 8 if px < block.item(i, j+1) else 0
    lval += 16 if px < block.item(i+1, j+1) else 0
    lval += 32 if px < block.item(i+1, j) else 0
    lval += 64 if px < block.item(i+1, j-1) else 0
    lval += 128 if px < block.item(i, j-1) else 0

    return lval


def ROR45(block, i, j):
    lval = 0
    px = block.item((i, j))

    lval += 1 if px < block.item(i-1, j-1) else 0
    lval += 2 if px < block.item(i, j - 2) else 0
    lval += 4 if px < block.item(i + 1, j - 1) else 0
    lval += 8 if px < block.item(i+2, j) else 0
    lval += 16 if px < block.item(i+1, j+1) else 0
    lval += 32 if px < block.item(i, j+2) else 0
    lval += 64 if px < block.item(i-1, j+1) else 0
    lval += 128 if px < block.item(i-2, j) else 0

    return lval


# LBP
def LBP(img, oh, ow, LBPFunction):
    h, w = img.shape
    lbp_values = np.zeros((h - 2, w - 2), np.uint8)

    for y in range(2, oh+2):
        for x in range(2, ow+2):
            lbp_values.itemset((y-2, x-2), LBPFunction(img, y, x))

    return lbp_values


def splitBlocks(img, blocksize, Ys, Xs):
    blocks = np.zeros((Ys, Xs, blocksize, blocksize), np.uint8)
    for y in range(Ys):
        for x in range(Xs):
            yoff = y * blocksize
            xoff = x * blocksize
            blocks[y, x] = img[yoff: yoff + blocksize, xoff : xoff + blocksize]
    
    return blocks
    

# https://www.researchgate.net/publication/220809357_Rotation_Invariant_Image_Description_with_Local_Binary_Pattern_Histogram_Fourier_Features
def shiftLeft(img, angle):
    a = int(angle * 8 / 360)
    return ((img & np.uint8((0b1111_1111 << (8 - a)))) >> (8 - a)) | (img << a)



start = time.time()

img = cv2.imread('test2.jpg', cv2.IMREAD_GRAYSCALE)

# to split image into chunks
# hyper Parameters
VISUALIZE = True
BLOCKSIZE = 16
h_img, w_img = img.shape
Xs, Ys = int(math.ceil(w_img / BLOCKSIZE)), int(math.ceil(h_img / BLOCKSIZE))
w_pimg, h_pimg = Xs * 16, Ys * 16

# image padded to become devisible by 16x16 chunks and add 0 borders
padded_img = paddedImg(img, (h_pimg, w_pimg))

lbps = {}
lbps_blocks = {}

lbps['0'] = LBP(padded_img, h_img, w_img, ROR0)
lbps['45'] = shiftLeft(lbps['0'], 45)
lbps['90'] = shiftLeft(lbps['0'], 90)
lbps['135'] = shiftLeft(lbps['0'], 135)
lbps['180'] = shiftLeft(lbps['0'], 180)
lbps['225'] = shiftLeft(lbps['0'], 225)
lbps['270'] = shiftLeft(lbps['0'], 270)
lbps['315'] = shiftLeft(lbps['0'], 315)

#LBP blocks to display histogram of each region
#lbp_blocks = splitBlocks(lbp_img, BLOCKSIZE, Ys, Xs)
for key in lbps:
    lbps_blocks[key] = splitBlocks(lbps[key], BLOCKSIZE, Ys, Xs)

end = time.time()

print(end - start)

#display all lbps
for key in lbps:
    cv2.imshow(key, lbps[key])


if VISUALIZE:
    #DRAW LINES IN PADDED IMAGE
    for vline in range(1, Ys):
        cv2.line(padded_img, (0, vline * BLOCKSIZE), (w_pimg-1, vline * BLOCKSIZE), (0, 0, 0))

    for hline in range(1, Xs):
        cv2.line(padded_img, (hline * BLOCKSIZE, 0), (hline * BLOCKSIZE, h_pimg-1), (0, 0, 0))

    fig, ax = plt.subplots(4, 3)
    gs = ax[0, 0].get_gridspec()

    # remove the underlying axes
    for a in ax[:, 0]:
        a.set_axis_off()

    axbig = fig.add_subplot(gs[:, 0])

    axbig.imshow(padded_img, cmap='gray')
    regionX, regionY = 0, 0
    bins = range(256)
    initvec = np.zeros(256)
    
    lbps_hists = {}
    axes = {}
    lines = {}
    
    #set limits and titles
    axes['0'] = ax[0, 1]
    axes['45'] = ax[0, 2]
    axes['90'] = ax[1, 1]
    axes['135'] = ax[1, 2]
    axes['180'] = ax[2, 1]
    axes['225'] = ax[2, 2]
    axes['270'] = ax[3, 1]
    axes['315'] = ax[3, 2]

    for key in axes:
        axes[key].title.set_text(key)
        axes[key].set_xticklabels([])
        lines[key], = axes[key].plot(bins, initvec)
    
    for key in lbps:
        lbps_hists[key] = np.zeros((Ys, Xs, 256))
        for y in range(Ys):
            for x in range(Xs):
                lbps_hists[key][y, x] = getHist(lbps_blocks[key][y, x]) 


    '''def animate(i):
        #ax[1].clear()
        #line.set_ydata(hists[regionY, regionX])
        #ax[1].relim()
        #ax[1].autoscale_view()
        for rect, y in zip(line, hists[regionY, regionX]):
            rect.set_height(y)
        
        return line'''



    def onclick(event):
        global ix, iy, regionX, regionY
        ix, iy = event.xdata, event.ydata
        regionX, regionY = int(ix / BLOCKSIZE), int(iy/BLOCKSIZE)

        for key in lines:
            lines[key].set_ydata(lbps_hists[key][regionY, regionX])
            axes[key].relim()
            axes[key].autoscale_view()
            

        fig.canvas.draw()
        fig.canvas.flush_events()

        
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    #ani = animation.FuncAnimation(fig, animate, interval=100, blit=True)
    plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()



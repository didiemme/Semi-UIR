import numpy as np
from PIL import Image
from scipy.ndimage import convolve

def UIConM(img):
    R = np.array(img.getchannel('R'), dtype=np.float64)
    G = np.array(img.getchannel('G'), dtype=np.float64)
    B = np.array(img.getchannel('B'), dtype=np.float64)

    patchsz = 5
    m, n = R.shape

    # resize the input image to match the patch size
    if m % patchsz != 0 or n % patchsz != 0:
        m_new = m - m % patchsz + patchsz
        n_new = n - n % patchsz + patchsz
        R = np.resize(R, (m_new, n_new))
        G = np.resize(G, (m_new, n_new))
        B = np.resize(B, (m_new, n_new))
        m, n = m_new, n_new

    k1 = m // patchsz
    k2 = n // patchsz

    AMEER = 0
    AMEEG = 0
    AMEEB = 0

    for i in range(0, m, patchsz):
        for j in range(0, n, patchsz):
            sz = patchsz - 1
            imR = R[i:i+sz+1, j:j+sz+1]
            imG = G[i:i+sz+1, j:j+sz+1]
            imB = B[i:i+sz+1, j:j+sz+1]
            MaxR, MinR = np.max(imR), np.min(imR)
            MaxG, MinG = np.max(imG), np.min(imG)
            MaxB, MinB = np.max(imB), np.min(imB)
            if (MaxR != 0 or MinR != 0) and MaxR != MinR:
                AMEER += np.log((MaxR - MinR) / (MaxR + MinR)) * ((MaxR - MinR) / (MaxR + MinR))
            if (MaxG != 0 or MinG != 0) and MaxG != MinG:
                AMEEG += np.log((MaxG - MinG) / (MaxG + MinG)) * ((MaxG - MinG) / (MaxG + MinG))
            if (MaxB != 0 or MinB != 0) and MaxB != MinB:
                AMEEB += np.log((MaxB - MinB) / (MaxB + MinB)) * ((MaxB - MinB) / (MaxB + MinB))

    AMEER = 1 / (k1 * k2) * np.abs(AMEER)
    AMEEG = 1 / (k1 * k2) * np.abs(AMEEG)
    AMEEB = 1 / (k1 * k2) * np.abs(AMEEB)

    uiconm = AMEER + AMEEG + AMEEB

    return uiconm

def UISM(img):
    Ir = np.array(img.getchannel('R'), dtype=np.float64)
    Ig = np.array(img.getchannel('G'), dtype=np.float64)
    Ib = np.array(img.getchannel('B'), dtype=np.float64)

    hx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    hy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    SobelR = np.abs(convolve(Ir, hx, mode='constant') + convolve(Ir, hy, mode='constant'))
    SobelG = np.abs(convolve(Ig, hx, mode='constant') + convolve(Ig, hy, mode='constant'))
    SobelB = np.abs(convolve(Ib, hx, mode='constant') + convolve(Ib, hy, mode='constant'))

    patchsz = 5
    m, n = Ir.shape

    # resize the input image to match the patch size
    if m % patchsz != 0 or n % patchsz != 0:
        m_new = m - m % patchsz + patchsz
        n_new = n - n % patchsz + patchsz
        SobelR = np.resize(SobelR, (m_new, n_new))
        SobelG = np.resize(SobelG, (m_new, n_new))
        SobelB = np.resize(SobelB, (m_new, n_new))
        m, n = m_new, n_new

    k1 = m // patchsz
    k2 = n // patchsz

    # calculate the EME value
    EMER = 0
    EMEG = 0
    EMEB = 0

    for i in range(0, m, patchsz):
        for j in range(0, n, patchsz):
            sz = patchsz - 1
            imR = SobelR[i:i+sz+1, j:j+sz+1]
            imG = SobelG[i:i+sz+1, j:j+sz+1]
            imB = SobelB[i:i+sz+1, j:j+sz+1]
            if np.max(imR) != 0 and np.min(imR) != 0:
                EMER += np.log(np.max(imR) / np.min(imR))
            if np.max(imG) != 0 and np.min(imG) != 0:
                EMEG += np.log(np.max(imG) / np.min(imG))
            if np.max(imB) != 0 and np.min(imB) != 0:
                EMEB += np.log(np.max(imB) / np.min(imB))

    EMER = 2 / (k1 * k2) * np.abs(EMER)
    EMEG = 2 / (k1 * k2) * np.abs(EMEG)
    EMEB = 2 / (k1 * k2) * np.abs(EMEB)

    lambdaR = 0.299
    lambdaG = 0.587
    lambdaB = 0.114

    uism = lambdaR * EMER + lambdaG * EMEG + lambdaB * EMEB

    return uism

def UIQM(image, c1=0.0282, c2=0.2953, c3=3.5753):
    uicm = UICM(image)
    uism = UISM(image)
    uiconm = UIConM(image)

    uiqm = c1 * uicm + c2 * uism + c3 * uiconm

    return uiqm

def UICM(img):
    R = np.array(img.getchannel('R'), dtype=np.float64)
    G = np.array(img.getchannel('G'), dtype=np.float64)
    B = np.array(img.getchannel('B'), dtype=np.float64)
    RG = R - G
    YB = (R + G) / 2 - B

    K = R.size[0] * R.size[1]

    # for R-G channel
    RG1 = np.sort(RG.flatten())
    alphaL = 0.1
    alphaR = 0.1
    RG1 = RG1[int(alphaL * K):int(K * (1 - alphaR))]
    N = K * (1 - alphaL - alphaR)
    meanRG = np.sum(RG1) / N
    deltaRG = np.sqrt(np.sum((RG1 - meanRG) ** 2) / N)

    # for Y-B channel
    YB1 = np.sort(YB.flatten())
    YB1 = YB1[int(alphaL * K):int(K * (1 - alphaR))]
    meanYB = np.sum(YB1) / N
    deltaYB = np.sqrt(np.sum((YB1 - meanYB) ** 2) / N)

    # UICM
    uicm = -0.0268 * np.sqrt(meanRG ** 2 + meanYB ** 2) + \
        0.1586 * np.sqrt(deltaRG ** 2 + deltaYB ** 2)

    return meanRG, deltaRG, meanYB, deltaYB, uicm

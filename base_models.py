import megengine as mge
import megengine.module as M
import megengine.functional as F

def addLeakyRelu(x):
    return M.Sequential(x, M.LeakyReLU(0.2))

def addPadding(x):
    shape = x.shape
    padding_shape = [(k + 1) // 2 * 2 for k in shape]
    res = mge.zeros(padding_shape, dtype=x.dtype)
    res = res.set_subtensor(x)[:shape[0], :shape[1], :shape[2], :shape[3]]
    return res
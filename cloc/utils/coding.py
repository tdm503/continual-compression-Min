import sys
import json
import math
import pickle
import struct
import numpy as np
import torchvision.transforms.functional as tvf


__all__ = [
    'get_object_size', 'pad_divisible_by', 'crop_divisible_by', 'bd_rate'
]


def get_object_size(obj, unit='bits'):
    """ Get object size in bits

    Args:
        obj: Python object
        unit (str): 'bits' or 'bytes'
    """
    num_bytes = sys.getsizeof(pickle.dumps(obj))
    if unit == 'bits':
        return num_bytes * 8
    elif unit == 'bytes':
        return num_bytes
    else:
        raise ValueError(f'Unknown unit {unit}')


def pack_byte_strings(list_of_strings):
    """ Pack a list of byte strings into a single byte string

    Args:
        list_of_strings (List[str]): a list of byte strings

    Returns:
        str: a single byte string
    """
    # each string corresponds to a latent variable
    lengths = [len(s) for s in list_of_strings] # length of each string
    packed = b''.join(list_of_strings) # concatenate all strings
    # save the lengths of each string as uint32 'I'
    packed = struct.pack(f'{len(lengths)}I', *lengths) + packed
    # save the number of latent variables as a uint8 'B'
    packed = struct.pack(f'B', len(lengths)) + packed
    if False: # debug
        print(f'{len(packed)*8=} bits, {sum(lengths)*8=} bits')
        print(f'{lengths=}')
        decoded = unpack_byte_sting(packed)
        assert decoded == list_of_strings, f'{decoded=} should equal to {list_of_strings=}'
    return packed


def unpack_byte_string(string):
    """ Unpack a byte string into a list of byte strings.
    The input byte string should be packed by `pack_byte_strings()`.

    Args:
        string (str): a byte string packed by `pack_byte_strings()`

    Returns:
        List[str]: a list of byte strings
    """
    # read the number of latent variables
    _len = 1
    num, string = struct.unpack('B', string[:_len])[0], string[_len:]
    # read the lengths of each string
    _len = num * 4
    lengths, string = struct.unpack(f'{num}I', string[:_len]), string[_len:]
    assert sum(lengths) == len(string), f'{sum(lengths)=} should equal to {len(string)=}'
    # split the string into num strings
    edges = np.cumsum((0,) + lengths, dtype=np.uint32)
    strings_all = [string[edges[i]:edges[i+1]] for i in range(num)]
    return strings_all


def pad_divisible_by(img, div=64):
    """ Pad a PIL.Image such that both its sides are divisible by `div`

    Args:
        img (PIL.Image): input image
        div (int, optional): denominator. Defaults to 64.

    Returns:
        PIL.Image: padded image
    """
    h_old, w_old = img.height, img.width
    if (h_old % div == 0) and (w_old % div == 0):
        return img
    h_tgt = round(div * math.ceil(h_old / div))
    w_tgt = round(div * math.ceil(w_old / div))
    # left, top, right, bottom
    padding = (0, 0, (w_tgt - w_old), (h_tgt - h_old))
    padded = tvf.pad(img, padding=padding, padding_mode='edge')
    return padded


def crop_divisible_by(img, div=64):
    ''' Center crop a PIL.Image such that both its sides are divisible by `div`

    Args:
        img (PIL.Image): input image
        div (int, optional): denominator. Defaults to 64.

    Returns:
        PIL.Image: cropped image
    '''
    h_old, w_old = img.height, img.width
    if (h_old % div == 0) and (w_old % div == 0):
        return img
    h_new = div * (h_old // div)
    w_new = div * (w_old // div)
    cropped = tvf.center_crop(img, output_size=(h_new, w_new))
    return cropped


def bd_rate(r1, psnr1, r2, psnr2):
    """ Compute average bit rate difference between RD-2 and RD-1. (RD-1 is the baseline)

    Equivalent to the implementations in:
    https://github.com/Anserw/Bjontegaard_metric/blob/master/bjontegaard_metric.py
    https://github.com/google/compare-codecs/blob/master/lib/visual_metrics.py

    args:
        r1    (list, np.ndarray): baseline rate
        psnr1 (list, np.ndarray): baseline psnr
        r2    (list, np.ndarray): rate 2
        psnr2 (list, np.ndarray): psnr 2
    """
    lr1 = np.log(r1)
    lr2 = np.log(r2)

    # fit each curve by a polynomial
    degree = 3
    p1 = np.polyfit(psnr1, lr1, deg=degree)
    p2 = np.polyfit(psnr2, lr2, deg=degree)
    # compute integral of the polynomial
    p_int1 = np.polyint(p1)
    p_int2 = np.polyint(p2)
    # area under the curve = integral(max) - integral(min)
    min_psnr = max(min(psnr1), min(psnr2))
    max_psnr = min(max(psnr1), max(psnr2))
    auc1 = np.polyval(p_int1, max_psnr) - np.polyval(p_int1, min_psnr)
    auc2 = np.polyval(p_int2, max_psnr) - np.polyval(p_int2, min_psnr)

    # find avgerage difference
    avg_exp_diff = (auc2 - auc1) / (max_psnr - min_psnr)
    avg_diff = (np.exp(avg_exp_diff) - 1) * 100

    if False: # debug
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        l1 = plt.plot(psnr1, lr1, label='PSNR-logBPP 1',
                      marker='.', markersize=12, linestyle='None')
        l2 = plt.plot(psnr2, lr2, label='PSNR-logBPP 2',
                      marker='.', markersize=12, linestyle='None')
        x = np.linspace(min_psnr, max_psnr, num=100)
        # x = np.linspace(min_psnr-10, max_psnr+10, num=100)
        plt.plot(x, np.polyval(p1, x), label='polyfit 1',
                 linestyle='-', color=l1[0].get_color())
        plt.plot(x, np.polyval(p2, x), label='polyfit 2',
                 linestyle='-', color=l2[0].get_color())
        plt.legend(loc='lower right')
        plt.xlim(np.concatenate([psnr1,psnr2]).min()-1, np.concatenate([psnr1,psnr2]).max()+1)
        plt.ylim(np.concatenate([lr1, lr2]).min()-0.1, np.concatenate([lr1, lr2]).max()+0.1)
        # plt.ylim(np.concatenate(lr1, lr2).min(), max(np.concatenate(lr1, lr2)))
        plt.show()
    return avg_diff

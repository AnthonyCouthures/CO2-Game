from itertools import zip_longest
import numpy as np

## invNormal code

def invNormal(low : np.ndarray, high : np.ndarray, loc = None, scale=1, *, size=1, block_size=1024, random_state=None):
    remain = size
    result = []
    if not loc:
        loc = (high + low) / 2
    
    mul = -0.5 * scale**-2

    while remain:
        # draw next block of uniform variates within interval
        x = np.random.uniform(low, high, size=min((remain+5)*2, block_size))
        
        # reject proportional to normal density
        x = x[np.exp(mul*(x-loc)**2) < np.random.rand(*x.shape)]
        
        # make sure we don't add too much
        if remain < len(x):
            x = x[:remain]

        result.append(x)
        remain -= len(x)

    return np.concatenate(result)

## Scipy code addapted

def _check_shape(argshape, size ):
    """
    This is a utility function used by `_rvs()` in the class geninvgauss_gen.
    It compares the tuple argshape to the tuple size.

    Parameters
    ----------
    argshape : tuple of integers
        Shape of the arguments.
    size : tuple of integers or integer
        Size argument of rvs().

    Returns
    -------
    The function returns two tuples, scalar_shape and bc.

    scalar_shape : tuple
        Shape to which the 1-d array of random variates returned by
        _rvs_scalar() is converted when it is copied into the
        output array of _rvs().

    bc : tuple of booleans
        bc is an tuple the same length as size. bc[j] is True if the data
        associated with that index is generated in one call of _rvs_scalar().

    """
    scalar_shape = []
    bc = []
    for argdim, sizedim in zip_longest(argshape[::-1], size[::-1],
                                       fillvalue=1):
        if sizedim > argdim or (argdim == sizedim == 1):
            scalar_shape.append(sizedim)
            bc.append(True)
        else:
            bc.append(False)
    return tuple(scalar_shape[::-1]), tuple(bc[::-1])

def invNormal_rvs(a, b, loc = None, scale = 1, size=None, random_state=None):
    # if a and b are scalar, use _rvs_scalar, otherwise need to create
    # output by iterating over parameters
    if np.isscalar(a) and np.isscalar(b):
        out = invNormal(a, b, size, random_state=random_state)
    elif a.size == 1 and b.size == 1:
        out = invNormal(a.item(), b.item(), size,
                                random_state=random_state)
    else:
        # When this method is called, size will be a (possibly empty)
        # tuple of integers.  It will not be None; if `size=None` is passed
        # to `rvs()`, size will be the empty tuple ().

        a, b = np.broadcast_arrays(a, b)
        # a and b now have the same shape.

        # `shp` is the shape of the blocks of random variates that are
        # generated for each combination of parameters associated with
        # broadcasting a and b.
        # bc is a tuple the same length as size.  The values
        # in bc are bools.  If bc[j] is True, it means that
        # entire axis is filled in for a given combination of the
        # broadcast arguments.
        shp, bc = _check_shape(a.shape, size)

        # `numsamples` is the total number of variates to be generated
        # for each combination of the input arguments.
        numsamples = int(np.prod(shp))

        # `out` is the array to be returned.  It is filled in in the
        # loop below.
        out = np.empty(size)

        it = np.nditer([a, b],
                        flags=['multi_index'],
                        op_flags=[['readonly'], ['readonly']])
        while not it.finished:
            # Convert the iterator's multi_index into an index into the
            # `out` array where the call to _rvs_scalar() will be stored.
            # Where bc is True, we use a full slice; otherwise we use the
            # index value from it.multi_index.  len(it.multi_index) might
            # be less than len(bc), and in that case we want to align these
            # two sequences to the right, so the loop variable j runs from
            # -len(size) to 0.  This doesn't cause an IndexError, as
            # bc[j] will be True in those cases where it.multi_index[j]
            # would cause an IndexError.
            idx = tuple((it.multi_index[j] if not bc[j] else slice(None))
                        for j in range(-len(size), 0))
            out[idx] =  invNormal(it[0], it[1], loc=loc, scale=scale, size=numsamples,
                                        random_state = random_state).reshape(shp)
            it.iternext()

    if size == ():
        out = out.item()
    return out
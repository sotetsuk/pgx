class At:
    """A utility for JAX's x = x.at[i].set(val) API
    >>> x = At(x)[i].set(val)
    """
    def __init__(self, array):
        self.array = array

    def __getitem__(self, index):
        return _IndexUpdateRef(self.array, index)


class _IndexUpdateRef:
    def __init__(self, array, index):
        self.array = array.clone()  # make immutable
        self.index = index

    def set(self, value):
        self.array[self.index] = value
        return self.array

    def add(self, value):
        self.array[self.index] += value
        return self.array

    # TODO: add all methods



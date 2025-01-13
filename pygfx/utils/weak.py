import weakref


class WeakAssociativeContainer:
    """An object similar to a WeakKeyDictionary, except with multiple 'keys'."""

    def __init__(self):
        self.hash2value = {}  # hash -> value
        self.ref2hashes = {}  # ref -> hashes

        def remove(ref, selfref=weakref.ref(self)):  # noqa: B008
            self = selfref()
            if self is not None:
                hashes = self.ref2hashes.pop(ref, ())
                for hash in hashes:
                    self.hash2value.pop(hash, None)

        self._remove = remove

    def get(self, keys, default=None):
        if not isinstance(keys, tuple):
            raise TypeError("WeakAssociativeContainer keys must be tuple.")
        return self.hash2value.get(hash(keys), default)

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            raise TypeError("WeakAssociativeContainer keys must be tuple.")
        return self.hash2value[hash(keys)]

    def __setitem__(self, keys, value):
        if not isinstance(keys, tuple):
            raise TypeError("WeakAssociativeContainer keys must be tuple.")
        h = hash(keys)
        self.hash2value[h] = value
        for key in keys:
            ref = weakref.ref(key, self._remove)
            self.ref2hashes.setdefault(ref, set()).add(h)

    def setdefault(self, keys, default=None):
        try:
            return self[keys]
        except KeyError:
            self[keys] = default
            return default

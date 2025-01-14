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
                for h in hashes:
                    self.hash2value.pop(h, None)

        self._remove = remove

    def get(self, keys: tuple, default: object = None) -> object:
        """Get the value associated with a tuple of keys."""
        if not isinstance(keys, tuple):
            raise TypeError("WeakAssociativeContainer keys must be tuple.")
        return self.hash2value.get(hash(keys), default)

    def get_associated(self, key: object) -> set:
        """Get a set of values associated with the given (singular) key."""
        if isinstance(key, tuple):
            raise TypeError("WeakAssociativeContainer.get_all key must not be tuple.")
        hashes = self.ref2hashes.get(weakref.ref(key), ())
        values = []
        for h in hashes:
            try:
                values.append(self.hash2value[h])
            except KeyError:
                pass
        return set(values)

    def __getitem__(self, keys: tuple):
        if not isinstance(keys, tuple):
            raise TypeError("WeakAssociativeContainer keys must be tuple.")
        return self.hash2value[hash(keys)]

    def __setitem__(self, keys: tuple, value: object):
        if not isinstance(keys, tuple):
            raise TypeError("WeakAssociativeContainer keys must be tuple.")
        h = hash(keys)
        self.hash2value[h] = value
        for key in keys:
            ref = weakref.ref(key, self._remove)
            self.ref2hashes.setdefault(ref, set()).add(h)

    def setdefault(self, keys: tuple, default: object = None) -> object:
        try:
            return self[keys]
        except KeyError:
            self[keys] = default
            return default

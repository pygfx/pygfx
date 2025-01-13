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

    def get(self, *objects):
        h = hash(tuple(objects))
        return self.hash2value.get(h, None)

    def set(self, *objects, value):
        h = hash(tuple(objects))
        self.hash2value[h] = value
        for ob in objects:
            ref = weakref.ref(ob, self._remove)
            self.ref2hashes.setdefault(ref, set()).add(h)
        return value

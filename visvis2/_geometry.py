class Geometry:
    @property
    def index(self):
        pass

    def vertex_buffers(self):
        pass

    def set_vertex_data(self, binding, buffer):
        self._buffers[
            binding
        ] = buffer  # we don't care what the buffer is, renderer creates and uses it


class BoxGeometry(Geometry):
    def __init__(self, renderer, width, height, depth):
        pass


class StanfordBunnyGeometry(Geometry):
    pass


## user code

# color_data = np.array(...)

buffer = renderer.createBuffer(color_data)
# renderer does device.createBuffer(1000,)

geometry.set_vertex_data(0, buffer)

class Material:
    def __init__(self):
        self.dirty = True
        self.shaders = {}
        self.uniforms = None
        self.primitive_topology = None

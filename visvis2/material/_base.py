class Material:
    def __init__(self):
        self.dirty = True
        self.shaders = {}
        self.uniforms = {}
        self.primitive_topology = None

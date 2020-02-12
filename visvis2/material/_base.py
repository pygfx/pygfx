class Material:
    def __init__(self):
        self.dirty = True
        self.shaders = {}
        self.uniforms = {}
        self.primitiveTopology = None

class Geometry:
    def __init__(self):
        self.index = None
        self.vertex_data = []
        # self.storage_data = {} ?

    def get_buffer_info(self):
        return {
            "index": self.index,
            "vertex_data": self.vertex_data,
        }

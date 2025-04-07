import numpy as np
import math

class OrbitCamera:
    def __init__(self, canvas, fov=45, focal_length=1, defocus_angle=0, center=(0, 0, 0), distance=1, azimuth=0, attitude=0, up=(0, 1, 0)):
        self.buffer_data = np.zeros(
            (),
            dtype=[
                ("origin", "float32", (3)),
                ("aspect_ratio", "float32"),
                ("u", "float32", (3)),
                ("fov", "float32"),
                ("v", "float32", (3)),
                ("focal_length", "float32"),
                ("w", "float32", (3)),
                ("defocus_angle", "float32"),
            ],
        )

        self._center = np.array(center, dtype=np.float32)
        self._distance = distance
        self._azimuth = azimuth
        self._attitude = attitude
        self._up = np.array(up, dtype=np.float32)

        self.canvas = canvas
        width, height = canvas.get_physical_size()
        self.fov = fov
        self.aspect_ratio = width / height

        self.focal_length = focal_length
        self.defocus_angle = defocus_angle

        self.calculate_buffer_data()

        self._need_update_buffer = True
        self._need_calculate_buffer = False

        self._last_mouse_state = (-1, -1)
        self._bind_event_handlers()

    @property
    def aspect_ratio(self):
        return self.buffer_data["aspect_ratio"]
    
    @aspect_ratio.setter
    def aspect_ratio(self, aspect_ratio):
        self.buffer_data["aspect_ratio"] = aspect_ratio
        self._need_update_buffer = True

    @property
    def fov(self):
        return math.degrees(self.buffer_data["fov"])
    
    @fov.setter
    def fov(self, fov):
        self.buffer_data["fov"] = math.radians(fov)
        self._need_update_buffer = True

    @property
    def focal_length(self):
        return self.buffer_data["focal_length"]
    
    @focal_length.setter
    def focal_length(self, focal_length):
        self.buffer_data["focal_length"] = focal_length
        self._need_update_buffer = True

    @property
    def defocus_angle(self):
        return math.degrees(self.buffer_data["defocus_angle"])
    
    @defocus_angle.setter
    def defocus_angle(self, defocus_angle):
        self.buffer_data["defocus_angle"] = math.radians(defocus_angle)
        self._need_update_buffer = True

    @property
    def distance(self):
        return self._distance
    
    @distance.setter
    def distance(self, distance):
        self._distance = distance
        self._need_calculate_buffer = True

    @property
    def center(self):
        return self._center
    
    @center.setter
    def center(self, center):
        self._center = np.array(center, dtype=np.float32)
        self._need_calculate_buffer = True

    @property
    def azimuth(self):
        return self._azimuth
    
    @azimuth.setter
    def azimuth(self, azimuth):
        self._azimuth = azimuth % (2 * math.pi)
        self._need_calculate_buffer = True

    @property
    def attitude(self):
        return self._attitude
    
    @attitude.setter
    def attitude(self, attitude):
        self._attitude = max(-math.pi / 2, min(math.pi / 2, attitude))
        self._need_calculate_buffer = True

    @property
    def up(self):
        return self._up
    
    @up.setter
    def up(self, up):
        self._up = np.array(up, dtype=np.float32)
        self._need_calculate_buffer = True


    def zoom(self, delta):
        scale = 1 - delta / 10
        if scale < 0.1:
            scale = 0.1

        self.distance *=  scale
        self.buffer_data["origin"] = self.center - self.distance * self.buffer_data["w"]
        self._need_update_buffer = True

    
    def pan(self, du, dv):
        pan = du * self.buffer_data["u"] + dv * self.buffer_data["v"]
        self.buffer_data["origin"] += pan
        self.center += pan

        self._need_update_buffer = True


    def orbit(self, du, dv):
        at = self.attitude + dv
        edge = math.pi / 2 - 1e-6
        self.attitude = max(-edge, min(edge, at)) # clamp to edge
        self.azimuth += du
        self.azimuth = self.azimuth % (2 * math.pi)

        self._need_calculate_buffer = True


    def calculate_buffer_data(self):
        w = -np.array([
            math.cos(self.attitude) * math.sin(self.azimuth),
            math.sin(self.attitude),
            math.cos(self.attitude) * math.cos(self.azimuth),
        ], dtype=np.float32)


        origin = self.center - self.distance * w
        u = np.cross(self.up, w)
        u /= np.linalg.norm(u)

        v = np.cross(w, u)

        self.buffer_data["origin"]= origin
        self.buffer_data["u"] = u
        self.buffer_data["v"] = v
        self.buffer_data["w"] = w

        self._need_calculate_buffer = False
        self._need_update_buffer = True


    def _bind_event_handlers(self):
        def on_mouse(event):
            event_type = event["event_type"]
            x = event["x"]
            y = event["y"]

            if event_type == "pointer_down":
                self._last_mouse_state = (x, y)

            elif event_type == "pointer_up":
                self._last_mouse_state = (-1, -1)

            elif event_type == "pointer_move":
                last_x, last_y = self._last_mouse_state
                if last_x == -1 or last_y == -1:
                    return

                if event["buttons"]:
                    dx = x - last_x
                    dy = y - last_y

                    self._last_mouse_state = (x, y)
                
                    if 1 in event["buttons"]:
                        self.orbit(dx * 0.005, dy * 0.005)
                
                    if 2 in event["buttons"]:
                        self.pan(-dx * 0.01, dy * 0.01)

        self.canvas.add_event_handler(on_mouse, "pointer_up", "pointer_down", "pointer_move")

        def on_wheel(event):
            dy = - event["dy"] / 100
            self.zoom(dy * 0.1)

        self.canvas.add_event_handler(on_wheel, "wheel")

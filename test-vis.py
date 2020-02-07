import asyncio

import visvis2 as vv


f = vv.Figure()

v = vv.View()
f._views.append(v)  # todo: API?

t1 = vv.Triangle()
t2 = vv.Triangle()

v.scene.children.append(t1)  # todo: API?
v.scene.children.append(t2)
v.scene.children.append(vv.Triangle())
v.scene.children.append(vv.Triangle())
v.scene.children.append(vv.Triangle())


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_forever()



renderer = vv.WgpuRenderer(widget_ish_or_surface_maybe_non_qt_specific)

camera = vv.Camera()

scene = vv.Scene()

controller = vv.QtPanZoomController(camera, widget)

scene.add(t1)

renderer.render(scene, camera)
# run_example = false

import math

from PySide6 import QtWidgets, QtGui
from wgpu.gui.qt import WgpuWidget
import pygfx as gfx

from pygfx.linalg.vector3 import Vector3


class LightViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Light_viewer")
        self.resize(800, 600)
        self.wgpu_widget = WgpuWidget()

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.wgpu_widget, 1)
        main_layout.addSpacing(10)

        btn_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(btn_layout)

        self.light1_checkbox = QtWidgets.QCheckBox("Point Light1")
        self.light1_checkbox.setChecked(True)
        self.light1_checkbox.toggled.connect(self.toggle_light1)
        btn_layout.addWidget(self.light1_checkbox)

        self.light1_color_btn = QtWidgets.QPushButton("Color")
        self.light1_color_btn.clicked.connect(self.set_light1_color)
        btn_layout.addWidget(self.light1_color_btn)

        self.light1_movement = QtWidgets.QCheckBox("Movement")
        btn_layout.addWidget(self.light1_movement)

        btn_layout.addSpacing(10)
        btn_layout.addWidget(QtWidgets.QLabel("-----------------------"))
        btn_layout.addSpacing(10)

        self.light2_checkbox = QtWidgets.QCheckBox("Point Light2")
        self.light2_checkbox.toggled.connect(self.toggle_light2)
        btn_layout.addWidget(self.light2_checkbox)

        self.light2_color_btn = QtWidgets.QPushButton("Color")
        self.light2_color_btn.setEnabled(False)
        self.light2_color_btn.clicked.connect(self.set_light2_color)
        btn_layout.addWidget(self.light2_color_btn)

        self.light2_movement = QtWidgets.QCheckBox("Movement")
        self.light2_movement.setEnabled(False)
        btn_layout.addWidget(self.light2_movement)

        btn_layout.addSpacing(10)
        btn_layout.addWidget(QtWidgets.QLabel("-----------------------"))
        btn_layout.addSpacing(10)

        self.light3_checkbox = QtWidgets.QCheckBox("Directional Light")
        self.light3_checkbox.toggled.connect(self.toggle_light3)
        btn_layout.addWidget(self.light3_checkbox)

        self.light3_color_btn = QtWidgets.QPushButton("Color")
        self.light3_color_btn.setEnabled(False)
        self.light3_color_btn.clicked.connect(self.set_light3_color)
        btn_layout.addWidget(self.light3_color_btn)

        btn_layout.addStretch(1)
        self.setLayout(main_layout)

    def set_light1_color(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.light1.color.hex))
        if color.isValid():
            self.light1_color_btn.setStyleSheet("background-color: %s" % color.name())
            self.light1.color = color.name()
            self.light1_helper.material.color = color.name()

    def set_light2_color(self):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            self.light2_color_btn.setStyleSheet("background-color: %s" % color.name())
            self.light2.color = color.name()
            self.light2_helper.material.color = color.name()

    def set_light3_color(self):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            self.light3_color_btn.setStyleSheet("background-color: %s" % color.name())
            self.light3.color = color.name()
            self.light3_helper.material.color = color.name()

    def toggle_light1(self):
        if self.light1_checkbox.isChecked():
            self.light1.visible = True
            self.light1_color_btn.setEnabled(True)
            self.light1_movement.setEnabled(True)
        else:
            self.light1.visible = False
            self.light1_color_btn.setEnabled(False)
            self.light1_movement.setEnabled(False)

    def toggle_light2(self):
        if self.light2_checkbox.isChecked():
            self.light2.visible = True
            self.light2_color_btn.setEnabled(True)
            self.light2_movement.setEnabled(True)
        else:
            self.light2.visible = False
            self.light2_color_btn.setEnabled(False)
            self.light2_movement.setEnabled(False)

    def toggle_light3(self):
        if self.light3_checkbox.isChecked():
            self.light3.visible = True
            self.light3_color_btn.setEnabled(True)
        else:
            self.light3.visible = False
            self.light3_color_btn.setEnabled(False)

    def init_scene(self):
        renderer = gfx.renderers.WgpuRenderer(self.wgpu_widget)
        scene = gfx.Scene()
        self.scene = scene

        cube = gfx.Mesh(
            gfx.box_geometry(20, 20, 20),
            gfx.MeshFlatMaterial(),
        )
        cube.rotation.set_from_euler(gfx.linalg.Euler(math.pi / 6, math.pi / 6))
        scene.add(cube)

        # Point Light1
        light1 = gfx.PointLight("#0040ff")
        self.light1_color_btn.setStyleSheet("background-color: %s" % light1.color.hex)
        self.light1 = light1
        light1.position.x = 25
        light1.position.y = 20

        light_sp = gfx.sphere_geometry(1)

        self.light1_helper = create_pointlight_helper(light1, light_sp)
        scene.add(light1)

        # Point Light2
        light2 = gfx.DirectionalLight("#80ff80")
        self.light2_color_btn.setStyleSheet("background-color: %s" % light2.color.hex)
        self.light2 = light2
        light2.visible = False
        light2.position.x = -25
        light2.position.y = 20

        self.light2_helper = create_pointlight_helper(light2, light_sp)
        scene.add(light2)

        # Directional light
        light3 = gfx.DirectionalLight("#ffaa00")
        self.light3_color_btn.setStyleSheet("background-color: %s" % light3.color.hex)
        self.light3 = light3
        light3.visible = False
        light3.position.x = -25
        light3.position.y = 20

        self.light3_helper = create_directionallight_helper(light3, 10)
        scene.add(light3)

        camera = gfx.PerspectiveCamera(70, 16 / 9)
        camera.position.z = 60

        controller = gfx.OrbitController(camera.position.clone())
        controller.add_default_event_handlers(renderer, camera)

        t1 = 0
        t2 = 0
        scale = 30

        light1.position.x = math.sin(t1 + math.pi / 3) * scale
        light1.position.y = math.sin(t1 + 1) * 5 + 15
        light1.position.z = math.cos(t1 + math.pi / 3) * scale

        light2.position.x = math.sin(t2 - math.pi / 3) * scale
        light2.position.y = math.sin(t2 + 2) * 5 + 15
        light2.position.z = math.cos(t2 - math.pi / 3) * scale

        def animate():
            # rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
            # cube.rotation.multiply(rot)
            controller.update_camera(camera)

            nonlocal t1, t2, scale

            if self.light1_movement.isChecked() and self.light1.visible:
                t1 += 0.01
                light1.position.x = math.sin(t1 + math.pi / 3) * scale
                light1.position.y = math.sin(t1 + 1) * 5 + 15
                light1.position.z = math.cos(t1 + math.pi / 3) * scale

            if self.light2_movement.isChecked() and self.light2.visible:
                t2 += 0.02
                light2.position.x = math.sin(t2 - math.pi / 3) * scale
                light2.position.y = math.sin(t2 + 2) * 5 + 15
                light2.position.z = math.cos(t2 - math.pi / 3) * scale

            # light1.position.x = math.cos(t) * math.cos(3*t) * scale
            # light1.position.y = math.cos(3*t) * math.sin(t) * scale
            # light1.position.z = math.sin(3*t) * scale

            renderer.render(scene, camera)
            renderer.request_draw()

        renderer.request_draw(animate)


def create_pointlight_helper(light, geometry=None):
    if geometry is None:
        geometry = gfx.sphere_geometry(1)

    helper = gfx.Mesh(
        geometry,
        gfx.MeshBasicMaterial(color=light.color.hex),
    )
    light.add(helper)
    return helper


def create_directionallight_helper(light, length=None):
    helper = gfx.Line(
        gfx.Geometry(
            positions=[
                [0, 0, 0],
                [0, 0, 1],
                [1, 0, 0],
                [1, 0, 1],
                [-1, 0, 0],
                [-1, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [0, -1, 0],
                [0, -1, 1],
            ]
        ),
        gfx.LineArrowMaterial(color=light.color.hex),
    )

    light.add(helper)
    helper.look_at(light.target)
    helper.scale.z = (
        length or Vector3().sub_vectors(light.target, light.position).length()
    )
    return helper


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = LightViewer()
    window.init_scene()
    window.show()
    app.exec_()

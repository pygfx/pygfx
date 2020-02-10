import asyncio

import visvis2 as vv
from visvis2._figure import QtCore, QtGui, QtWidgets  # to make sure we use the same lib


class Main(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(None)
        self.resize(640, 480)

        self._button = QtWidgets.QPushButton("add triangle", self)
        self._button.clicked.connect(self._on_button_click)

        self._figure = vv.Figure(parent=self)
        v = vv.View()
        self._figure._views.append(v)  # todo: API?

        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self._button)
        layout.addWidget(self._figure.widget)

    def _on_button_click(self):
        t = vv.Triangle()
        self._figure.views[0].scene.children.append(t)
        self._figure.widget.update()


m = Main()
m.show()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_forever()

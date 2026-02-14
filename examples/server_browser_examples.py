"""
A little script to serve pygfx examples on localhost so you can try them in the browser.
This is an iteration of the same script in rendercanvas and wgpu-py although some changes:

* swapped in pathlib for os.path (mostly)
* avoided pyscript for now, but likely a good idea to use for webworker recovery etc.

Files are loaded from disk on each request, so you can leave the server running
and just update examples, update pygfx and build the wheel, etc.
"""

import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import flit

import pygfx


# sphinx_gallery_pygfx_test = 'off'

# from here: https://github.com/harfbuzz/uharfbuzz/pull/275 placed in /dist
uharfbuzz_wheel = "uharfbuzz-0.1.dev1+ga19185453-cp310-abi3-pyodide_2025_0_wasm32.whl"
wgpu_wheel = "https://wgpu-py--753.org.readthedocs.build/en/753/_static/wgpu-0.29.0-py3-none-any.whl" # very hacky way to serve this but it does work...
# wgpu_wheel = "wgpu-0.29.0-py3-none-any.whl"
rendercanvas_deps = ["sniffio", "rendercanvas==2.4.2"] #TODO: I put a restriction into the pyproject.toml so it might pick <2.5.0 already.

# the pygfx wheel will be listed after this. it might be possible to still get deps from pyproject.toml
pygfx_deps = [*rendercanvas_deps, wgpu_wheel, uharfbuzz_wheel, "hsluv", "pylinalg", "jinja2"]

root = Path(__file__).parent.parent.absolute()

short_version = ".".join(str(i) for i in pygfx.version_info[:3])
wheel_name = f"pygfx-{short_version}-py3-none-any.whl"

example_files = list((root / "examples").glob("**/*.py"))

def get_html_index():
    """Create a landing page."""

    examples_list = [f"<li><a href='{str(name.relative_to(root / "examples")).replace('.py', '.html')}'>{name.relative_to(root / "examples")!s}</a></li>" for name in example_files]

    html = """<!doctype html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width,initial-scale=1.0">
        <title>pygfx browser examples</title>
        <script type="module" src="https://pyscript.net/releases/2025.11.2/core.js"></script>
    </head>
    <body>

    <a href='/build'>Rebuild the wheel</a><br><br>
    """

    html += "List of examples that might run in Pyodide:\n"
    html += f"<ul>{''.join(examples_list)}</ul><br>\n\n"

    html += "</body>\n</html>\n"
    return html


html_index = get_html_index()


# An html template to show examples using pyscript.
pyscript_graphics_template = """
<!doctype html>
<html>
<head>
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <title>{example_script} via PyScript</title>
    <script type="module" src="https://pyscript.net/releases/2025.11.2/core.js"></script>
</head>

<body>
    <a href="/">Back to list</a><br><br>

    <p>
    {docstring}
    </p>
    <dialog id="loading" style='outline: none; border: none; background: transparent;'>
        <h1>Loading...</h1>
    </dialog>
    <script type="module">
        const loading = document.getElementById('loading');
        addEventListener('py:ready', () => loading.close());
        loading.showModal();
    </script>

    <canvas id="canvas" style="background:#aaa; width: 90%; height: 480px;"></canvas>
    <script type="py" src="{example_script}",
        config='{{"packages": [{dependencies}]}}'>
    </script>
</body>

</html>
"""

# TODO: a pyodide example for the compute examples (so we can capture output?)
# modified from _pyodide_iframe.html from rendercanvas
pyodide_compute_template = """
<!doctype html>
<html>
<head>
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <title>{example_script} via Pyodide</title>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.29.3/full/pyodide.js"></script>
</head>
<base href="/">

<dialog id="loading" style='outline: none; border: none; background: transparent;'>
        <h1>Loading...</h1>
    </dialog>
<body>
    <a href="/">Back to list</a><br><br>
    <p>
    {docstring}
    </p>
    <div id="output" style="white-space: pre-wrap; background:#eee; padding:4px; margin:4px; border:1px solid #ccc;">
        <p>Output:</p>
    </div>
    <canvas id='canvas' style='width:calc(100% - 40px); height:600px; background-color: #ddd;'></canvas>
    <script type="text/javascript">
        async function main() {{
            let loading = document.getElementById('loading');
            loading.showModal();
            try {{
                let example_name = {example_script!r};
                pythonCode = await (await fetch(example_name)).text();
                // this env var is really only used for the pygfx examples - so maybe we make a script for that gallery instead?
                let pyodide = await loadPyodide({{env: {{'PYGFX_DEFAULT_PPAA': 'none' }}}});
                pyodide.setStdout({{
                    batched: (s) => {{
                        // TODO: newline, scrollable, echo to console?
                        document.getElementById("output").innerHTML += "<br>" + s.replace(/</g, "&lt;").replace(/>/g, "&gt;");
                        console.log(s); // so we also have it formatted
                    }}
                }});
                await pyodide.loadPackage("micropip");
                const micropip = pyodide.pyimport("micropip");
                {dependencies}
                // TODO: maybe use https://pyodide.org/en/stable/usage/api/js-api.html#pyodide.loadPackagesFromImports
                await pyodide.loadPackagesFromImports(pythonCode);
                pyodide.setDebug(true);
                let ret = await pyodide.runPythonAsync(pythonCode);
                console.log("Example finished:", ret);
                loading.close();
            }} catch (err) {{
                // TODO: this could be formatted better as this overlaps and is unreadable...
                loading.innerHTML = "Failed to load: " + err;
                console.error(err); // so we have it here too
            }}
        }}
        main();
    </script>
</body>

</html>
"""


def build_wheel():
    toml_filename = (root / "pyproject.toml")
    flit.main(["-f", str(toml_filename.resolve()), "build", "--no-use-vcs", "--format", "wheel"])
    wheel_filename = root / "dist" / wheel_name
    assert wheel_filename.is_file(), f"{wheel_name} does not exist"


def get_docstring_from_py_file(fname):
    filename = root / "examples" / fname
    docstate = 0
    doc = ""
    with open(filename, "rb") as f:
        for line in f:
            line = line.decode()
            if docstate == 0:
                if line.lstrip().startswith('"""'):
                    docstate = 1
            else:
                if docstate == 1 and line.lstrip().startswith(("---", "===")):
                    docstate = 2
                    doc = ""
                elif '"""' in line:
                    doc += line.partition('"""')[0]
                    break
                else:
                    doc += line

    return doc.replace("\n\n", "<br><br>")


class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.respond(200, html_index, "text/html")
        elif self.path == "/build":
            try:
                self.respond(200, "Building wheel...<br>", "text/html")
                build_wheel()
            except Exception as err:
                self.respond(500, str(err), "text/plain")
            else:
                html = f"Wheel build: {wheel_name}<br><br><a href='/'>Back to list</a>"
                self.respond(200, html, "text/html")
        elif self.path.endswith(".whl"):
            requested_path = Path(self.path)
            filename = root / "dist" / requested_path.name
            if os.path.isfile(filename):
                with open(filename, "rb") as f:
                    data = f.read()
                self.respond(200, data, "application/octet-stream")
            else:
                self.respond(404, "wheel not found")
        elif self.path.endswith(".html"):
            # name = self.path.strip("/")
            pyname = self.path.replace(".html", ".py").lstrip("/")
            try:
                doc = get_docstring_from_py_file(pyname)
                deps = [*pygfx_deps, f"/{wheel_name}"]
                html = pyodide_compute_template.format(
                    docstring=doc,
                    example_script=pyname,
                    # todo: refactor this to a list and maybe get other deps from pyodide.loadPackagesFromImports
                    dependencies="\n".join(
                        [f"await micropip.install({dep!r});" for dep in deps]
                    ),
                )
                self.respond(200, html, "text/html")
            except Exception as err:
                self.respond(404, f"example not found: {err}")
        elif self.path.endswith(".py"):
            filename = os.path.join(root, "examples", self.path.strip("/"))
            if os.path.isfile(filename):
                with open(filename, "rb") as f:
                    data = f.read()
                self.respond(200, data, "text/plain")
            else:
                self.respond(404, "py file not found")
        else:
            self.respond(404, "not found")

    def respond(self, code, body, content_type="text/plain"):
        self.send_response(code)
        self.send_header("Content-type", content_type)
        self.end_headers()
        if isinstance(body, str):
            body = body.encode()
        self.wfile.write(body)


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[-1])
        except ValueError:
            pass

    build_wheel()
    print("Opening page in web browser ...")
    webbrowser.open(f"http://localhost:{port}/")
    HTTPServer(("", port), MyHandler).serve_forever()



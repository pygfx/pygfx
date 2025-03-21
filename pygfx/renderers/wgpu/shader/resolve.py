import re


varying_types = ["f32", "vec2<f32>", "vec3<f32>", "vec4<f32>"]
varying_types = (
    varying_types
    + [t.replace("f", "i") for t in varying_types]
    + [t.replace("f", "u") for t in varying_types]
)

re_varying_getter = re.compile(r"[^\w]varyings\.(\w+)", re.UNICODE)
re_varying_setter = re.compile(r"varyings\.(\w+)(\.\w+)?\s*\=", re.UNICODE)
builtin_varyings = {"position": "vec4<f32>"}


def resolve_shadercode(wgsl):
    """Apply all shader resolve opearions."""
    return _resolve(wgsl, VaryingResolver(), OutputResolver())


def resolve_varyings(wgsl):
    return _resolve(wgsl, VaryingResolver())


def resolve_output(wgsl):
    return _resolve(wgsl, OutputResolver())


def _resolve(wgsl, *resolvers):
    assert isinstance(wgsl, str)

    # Split into lines, which is easier to process. Ensure it ends with newline in the end.
    lines = wgsl.splitlines()
    while lines and not lines[0]:
        lines.pop(0)
    if not (lines and lines[-1] == ""):
        lines.append("")

    current_func_kind = None

    # Apply resolve functions. They modify the lines
    for linenr, ori_line in enumerate(lines):
        line = ori_line.split("//", 1)[0].strip()
        if not line:
            continue

        if line.startswith("fn "):
            prevline = lines[linenr - 1].strip() if linenr > 0 else ""
            current_func_kind = "normal"
            if prevline.startswith("@vertex"):
                current_func_kind = "vertex"
            elif prevline.startswith("@fragment"):
                current_func_kind = "fragment"
            elif prevline.startswith("@compute"):
                current_func_kind = "compute"

        for resolver in resolvers:
            resolver.process_line(linenr, line, current_func_kind)

    for resolver in resolvers:
        resolver.update_lines(lines)

    # Return as str
    return "\n".join(lines)


class VaryingResolver:
    """Resolve varyings in the given wgsl:
    * Detect varyings being used.
    * Check that these are also set.
    * Remove assignments of varyings that are not used.
    * Include the Varyings struct.
    """

    def __init__(self):
        # Prepare dicts that map name to list-of-linenr. And a tupe dict.
        self.assigned_varyings = {}
        self.used_varyings = {}
        self.types = {}  # varying types

        # We try to find the function that first uses the Varyings struct.
        self.struct_linenr = None

        # Go over all lines to:
        # - find the lines where a varying is set
        # - collect the types of these varyings
        # - collect all used varyings
        # - find where the vertex-shader starts

    def process_line(self, linenr, line, current_func_kind):
        # Look for varyings being set (in the vertex shader)
        # Note that this implicitly matches from start of the strpped line.
        in_vertex_shader = current_func_kind == "vertex"

        # Put the Varyings struct right above where its first used
        if self.struct_linenr is None:
            if "Varyings" in line:
                self.struct_linenr = linenr if current_func_kind else 0

        # Look for varyings being set (in vertex shader)
        if in_vertex_shader:
            match = re_varying_setter.match(line)
            if match:
                # Get parts
                name = match.group(1)
                attr = match.group(2)
                # Handle builtin
                if name in builtin_varyings:
                    self.used_varyings[name] = []
                    self.types[name] = builtin_varyings[name]
                # Find type
                type = line[match.end() :].split("(")[0].strip().replace(" ", "")
                if type not in varying_types:
                    type = ""
                # Triage
                if attr:
                    pass  # Not actually a type but an attribute access
                elif not type:
                    raise TypeError(
                        f"Varying {name!r} assignment needs an explicit cast (of a correct type), e.g. `varying.{name} = f32(3.0);`:\n{line}"
                    )
                elif name in self.types and type != self.types[name]:
                    raise TypeError(
                        f"Varying {name!r} assignment does not match expected type {self.types[name]}:\n{line}"
                    )
                else:
                    self.types[name] = type
                # Store position
                self.assigned_varyings.setdefault(name, []).append(linenr)

        # Look for varyings being get (anywhere)
        if "varyings." in line:
            for match in re_varying_getter.finditer(" " + line):
                name = match.group(1)
                this_varying_is_set_on_this_line = linenr in self.assigned_varyings.get(
                    name, []
                )
                if this_varying_is_set_on_this_line:
                    pass
                elif in_vertex_shader:
                    # If varyings are used in another way than setting, in the vertex shader,
                    # we should either consider them "used", or possibly break the shader if
                    # the used varying is disabled. So let's just not allow it.
                    raise TypeError(
                        f"Varying {name!r} is read in the vertex shader, but only writing is allowed:\n{line}"
                    )
                else:
                    self.used_varyings.setdefault(name, []).append(linenr)

    def update_lines(self, lines):
        assigned_varyings = self.assigned_varyings
        used_varyings = self.used_varyings
        types = self.types
        struct_linenr = self.struct_linenr

        # Check if all used varyings are assigned
        for name in used_varyings:
            if name not in assigned_varyings:
                line = lines[used_varyings[name][0]]
                raise TypeError(f"Varying {name!r} is read, but not assigned:\n{line}")

        # Comment-out the varying setter if its unused elsewhere in the shader
        for name, linenrs in assigned_varyings.items():
            if name not in used_varyings:
                for linenr in linenrs:
                    line = lines[linenr]
                    indent = line[: len(line) - len(line.lstrip())]
                    lines[linenr] = indent + "// unused: " + line[len(indent) :]
                    # Deal with multiple-line assignments
                    line_s = line.strip()
                    while not line_s.endswith(";"):
                        linenr += 1
                        line_s = lines[linenr].strip()
                        unexpected = "fn ", "struct ", "var ", "let ", "}"
                        if line_s.startswith(unexpected) or linenr == len(lines) - 1:
                            raise TypeError(
                                f"Varying {name!r} assignment seems to be missing a semicolon:\n{line}"
                            )
                        lines[linenr] = indent + "// " + line_s

        # Build and insert the struct
        if struct_linenr is not None:
            # Refine position
            while struct_linenr > 0 and not lines[struct_linenr].lstrip().startswith(
                "fn"
            ):
                struct_linenr -= 1
            if struct_linenr > 0 and lines[struct_linenr - 1].lstrip().startswith("@"):
                struct_linenr -= 1
            # First divide into slot-based and builtins
            used_varyings = set(used_varyings)
            used_builtins = used_varyings.intersection(builtin_varyings)
            used_slots = used_varyings.difference(used_builtins)
            used_slots = list(sorted(used_slots))
            # Build struct
            struct_lines = ["struct Varyings {"]
            for slotnr, name in enumerate(used_slots):
                struct_lines.append(f"    @location({slotnr}) {name} : {types[name]},")
            for name in sorted(used_builtins):
                struct_lines.append(f"    @builtin({name}) {name} : {types[name]},")
            struct_lines.append("};\n")
            # Apply indentation and insert
            line = lines[struct_linenr]
            indent = line[: len(line) - len(line.lstrip())]
            new_lines = [indent + line for line in struct_lines] + [line]
            lines[struct_linenr] = "\n".join(new_lines)
        else:
            assert not used_varyings, "woops, did not expect used_varyings here"


re_depth_setter = re.compile(r"out\.depth\s*\=")


class OutputResolver:
    """When out.depth is set (in the fragment shader), adjust the FragmentOutput
    to accept depth.
    """

    def __init__(self):
        # Detect whether the depth is set in the shader. We're going to assume
        # this is in the fragment shader. We check for "out.depth =".
        # Background: by default the depth is based on the geometry (set
        # by vertex shader and interpolated). It is possible for a fragment
        # shader to write the depth instead. If this is done, the GPU cannot
        # do early depth testing; the fragment shader must be run for the
        # depth to be known.
        self.depth_is_set = False
        self.struct_linenr = None

    def process_line(self, linenr, line, current_func_kind):
        in_fragment_shader = current_func_kind == "fragment"

        if self.struct_linenr is None:
            if line.startswith("struct FragmentOutput {"):
                self.struct_linenr = linenr
        if in_fragment_shader:
            if re_depth_setter.match(line):
                self.depth_is_set = True

    def update_lines(self, lines):
        depth_is_set = self.depth_is_set
        struct_linenr = self.struct_linenr

        if depth_is_set:
            if struct_linenr is None:
                raise TypeError("FragmentOutput definition not found.")
            depth_field = "    @builtin(frag_depth) depth : f32,"
            line = lines[struct_linenr]
            indent = line[: len(line) - len(line.lstrip())]
            new_lines = [line, indent + depth_field]
            lines[struct_linenr] = "\n".join(new_lines)

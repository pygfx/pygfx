import re


varying_types = ["f32", "vec2<f32>", "vec3<f32>", "vec4<f32>"]
varying_types = (
    varying_types
    + [t.replace("f", "i") for t in varying_types]
    + [t.replace("f", "u") for t in varying_types]
)

re_varying_getter = re.compile(r"[\s,\(\[]varyings\.(\w+)", re.UNICODE)
re_varying_setter = re.compile(r"\A\s*?varyings\.(\w+)(\.\w+)?\s*?\=")
builtin_varyings = {"position": "vec4<f32>"}


def resolve_varyings(wgsl):
    """Resolve varyings in the given wgsl:
    * Detect varyings being used.
    * Check that these are also set.
    * Remove assignments of varyings that are not used.
    * Include the Varyings struct.
    """
    assert isinstance(wgsl, str)

    # Split into lines, which is easier to process. Ensure it ends with newline in the end.
    lines = wgsl.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop(-1)
    lines.append("")

    # Prepare dicts that map name to list-of-linenr. And a tupe dict.
    assigned_varyings = {}
    used_varyings = {}
    types = {}  # varying types

    # We try to find the function that first uses the Varyings struct.
    struct_insert_pos = None

    # Go over all lines to:
    # - find the lines where a varying is set
    # - collect the types of these varyings
    for linenr, line in enumerate(lines):
        match = re_varying_setter.match(line)
        if match:
            # Get parts
            name = match.group(1)
            attr = match.group(2)
            # Handle builtin
            if name in builtin_varyings:
                used_varyings[name] = []
                types[name] = builtin_varyings[name]
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
            elif name in types and type != types[name]:
                raise TypeError(
                    f"Varying {name!r} assignment does not match expected type {types[name]}:\n{line}"
                )
            else:
                types[name] = type
            # Store position
            assigned_varyings.setdefault(name, []).append(linenr)

    # Go over all lines to:
    # - collect all used varyings
    # - find where the vertex-shader starts
    in_vertex_shader = False
    current_func_linenr = 0
    for linenr, line in enumerate(lines):
        line = line.strip()
        # Detect when we enter a new function
        if line.startswith("fn "):
            current_func_linenr = linenr
            if line.startswith("fn vs_main"):
                in_vertex_shader = True
            else:
                in_vertex_shader = False
        # Remove comments (shader code has no strings that can contain slashes)
        line = line.split("//")[0]
        if "Varyings" in line and struct_insert_pos is None:
            struct_insert_pos = current_func_linenr
        # Everything we find here is a match (prepend a space to allow an easier regexp)
        for match in re_varying_getter.finditer(" " + line):
            name = match.group(1)
            this_varying_is_set_on_this_line = linenr in assigned_varyings.get(name, [])
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
                used_varyings.setdefault(name, []).append(linenr)

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
    if struct_insert_pos is not None:
        # Maybe we should move up a bit
        if struct_insert_pos > 0:
            if lines[struct_insert_pos - 1].lstrip().startswith("@"):
                struct_insert_pos -= 1
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
        line = lines[struct_insert_pos]
        indent = line[: len(line) - len(line.lstrip())]
        struct_lines = [indent + line for line in struct_lines]
        lines.insert(struct_insert_pos, "\n".join(struct_lines))
    else:
        assert not used_varyings, "woops, did not expect used_varyings here"

    # Return modified code
    return "\n".join(lines)


re_depth_setter = re.compile(r"\A\s*?out\.depth\s*?\=")


def resolve_depth_output(wgsl):
    """When out.depth is set (in the fragment shader), adjust the FragmentOutput
    to accept depth.
    """
    assert isinstance(wgsl, str)

    # Split into lines, which is easier to process. Ensure it ends with newline in the end.
    lines = wgsl.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop(-1)
    lines.append("")

    # Detect whether the depth is set in the shader. We're going to assume
    # this is in the fragment shader. We check for "out.depth =".
    # Background: by default the depth is based on the geometry (set
    # by vertex shader and interpolated). It is possible for a fragment
    # shader to write the depth instead. If this is done, the GPU cannot
    # do early depth testing; the fragment shader must be run for the
    # depth to be known.
    depth_is_set = False
    struct_linenr = -1
    for linenr, line in enumerate(lines):
        if line.lstrip().startswith("struct FragmentOutput {"):
            struct_linenr = linenr
        elif re_depth_setter.match(line):
            depth_is_set = True
            if struct_linenr >= 0:
                break

    if depth_is_set:
        if struct_linenr < 0:
            raise TypeError("FragmentOutput definition not found.")
        depth_field = "    @builtin(frag_depth) depth : f32,"
        line = lines[struct_linenr]
        indent = line[: len(line) - len(line.lstrip())]
        lines.insert(struct_linenr + 1, indent + depth_field)

    return "\n".join(lines)

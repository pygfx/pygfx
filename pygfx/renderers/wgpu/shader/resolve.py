import re
import logging


logger = logging.getLogger("pygfx")
warned_for = set()


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

    curly_braces_level = 0
    current_func_kind = None

    # Apply resolve functions. They modify the lines
    for linenr, ori_line in enumerate(lines):
        line = ori_line.split("//", 1)[0].strip()
        if not line:
            continue

        if line.startswith("fn "):
            prevline = lines[linenr - 1].strip() if linenr > 0 else ""
            current_func_kind = "normal"
            if prevline.startswith("@"):
                if prevline.startswith("@vertex"):
                    current_func_kind = "vertex"
                elif prevline.startswith("@fragment"):
                    current_func_kind = "fragment"
                elif prevline.startswith("@compute"):
                    current_func_kind = "compute"
            curly_braces_level = 0

        if current_func_kind:
            curly_braces_level += line.count("{") - line.count("}")
            if curly_braces_level <= 0:
                current_func_kind = None

        for resolver in resolvers:
            resolver.process_line(linenr, line, current_func_kind)

    for resolver in resolvers:
        resolver.update_lines(lines)

    # Return as str
    return "\n".join(lines)


def indent_from_line(line):
    return line[: len(line) - len(line.lstrip())]


varying_types = ["f32", "vec2<f32>", "vec3<f32>", "vec4<f32>"]
varying_types = (
    varying_types
    + [t.replace("f", "i") for t in varying_types]
    + [t.replace("f", "u") for t in varying_types]
)

re_varying_getter = re.compile(r"[^\w]varyings\.(\w+)", re.UNICODE)
re_varying_setter = re.compile(r"varyings\.(\w+)(\.\w+)?\s*\=", re.UNICODE)
builtin_varyings = {"position": "vec4<f32>"}


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
        # Note that this implicitly matches from the start of the stripped line.
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

        # Look for varyings being get (in any function)
        if current_func_kind and "varyings." in line:
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
                    indent = indent_from_line(line)
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
            indent = indent_from_line(line)
            new_lines = [indent + line for line in struct_lines] + [line]
            lines[struct_linenr] = "\n".join(new_lines)
        else:
            assert not used_varyings, "woops, did not expect used_varyings here"


re_out_create = re.compile(r"var\s*out\s*\:\s*FragmentOutput\s*;")
re_out_create_old = re.compile(r"var\s*out\s*\=\s*get_fragment_output\(")
re_out_field = re.compile(r"out\.(\w+)\s*\=", re.UNICODE)
re_out_return = re.compile(r"return\s*out\s*;", re.MULTILINE)


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

        self.struct_linenr = None
        self.assigned_fields = None  # list of (field-name, linenr)
        self.assigned_fields_list = []  # list of assigned_fields

    def process_line(self, linenr, line, current_func_kind):
        in_fragment_shader = current_func_kind == "fragment"

        if not in_fragment_shader:
            self.assigned_fields = None

        if self.struct_linenr is None:
            if line.startswith("struct FragmentOutput {"):
                self.struct_linenr = linenr

        if in_fragment_shader:
            if self.assigned_fields is None:
                # Detect var out: FragmentOutput
                if re_out_create.match(line) or re_out_create_old.match(line):
                    self.assigned_fields = [("__create", linenr)]
                    self.assigned_fields_list.append(self.assigned_fields)
            else:
                # Detect fields being set
                match = re_out_field.match(line)
                if match:
                    name = match.group(1)
                    self.assigned_fields.append((name, linenr))
                # Detect return, because that's the best place to insert our call
                match = re_out_return.match(line)
                if match:
                    self.assigned_fields.append(("return", linenr))
                    self.assigned_fields = None

    def update_lines(self, lines):
        struct_linenr = self.struct_linenr
        assigned_fields_list = self.assigned_fields_list

        # Collect virtual fields, specified in comments inside the FragmentOutput definition.
        # These will be the args for apply_virtual_fields_of_fragment_output()
        # E.g. // virtualfield foo : f32 = vec4<f32>(0.0)
        virtual_field_prefix = "// virtualfield "
        virtual_fields = []
        if struct_linenr is not None:
            linenr = struct_linenr
            while linenr < len(lines) - 1:
                linenr += 1
                line = lines[linenr].lstrip()
                if line.startswith(virtual_field_prefix):
                    name_type_value = line[len(virtual_field_prefix) :].split("//")[0]
                    name, colon, type_value = name_type_value.partition(":")
                    type, eq, value = type_value.partition("=")
                    assert colon and eq, (
                        "A virtualfield needs to be of the form 'name : type = defaultvalue'. {name!r} does not"
                    )
                    virtual_fields.append(
                        (name.strip(), type.strip(), value.strip(" ;"))
                    )
                else:
                    break

        depth_is_set = False

        # Handle each place where a FragmentOutput is returned
        for assigned_fields in assigned_fields_list:
            create_linenr = next(i for name, i in assigned_fields if name == "__create")
            create_indent = indent_from_line(lines[create_linenr])

            # Handle get_fragment_output deprecated usage. Not pretty, but it works and is temporary.
            # TODO: after one or two releases, remove the deprecated get_fragment_output case.
            create_line = lines[create_linenr]
            if re_out_create_old.match(create_line.lstrip()):
                if "get_fragment_output" not in warned_for:
                    warned_for.add("get_fragment_output")
                    logger.warning("get_fragment_output() in WGSL is deprecated")
                new_create_line = create_indent + "var out: FragmentOutput;"
                colorset_line = (
                    create_indent
                    + "out.color ="
                    + create_line.split("=", 1)[1].replace(");", ").color;")
                )
                lines[create_linenr - 1] += "\n" + new_create_line
                lines[create_linenr] = colorset_line
                assigned_fields.append(("color", create_linenr))
                create_linenr -= 1

            # Collect virtual argument values
            args = []
            for name, type, default_value in virtual_fields:
                linenrs = [i for n, i in assigned_fields if n == name]
                if not linenrs:
                    args.append(default_value)
                else:
                    prefixed_name = f"out_virtualfield_{name}"
                    args.append(prefixed_name)
                    lines[create_linenr] += (
                        f"\n{create_indent}var {prefixed_name}: {type};"
                    )

                    for linenr in linenrs:
                        lines[linenr] = lines[linenr].replace(
                            f"out.{name}", f"{prefixed_name}", 1
                        )

            # Process special arguments
            if any(i for name, i in assigned_fields if name == "depth"):
                depth_is_set = True

            # Apply the virtual fields. The call to the apply-function is
            # inserted right before 'return out', or right after the last struct
            # field is set (if we did not detect a return).
            if args:
                linenr = max(i for _, i in assigned_fields)
                args.insert(0, "&out")
                line = lines[linenr]
                is_return_line = line.lstrip().startswith("return")
                indent = indent_from_line(line)
                extra_line = f"{indent}apply_virtual_fields_of_fragment_output({', '.join(args)});"
                new_lines = [extra_line, line] if is_return_line else [line, extra_line]
                lines[linenr] = "\n".join(new_lines)

        # If depth is set, add the depth field to the struct definition.
        if depth_is_set:
            if struct_linenr is None:
                raise TypeError("FragmentOutput definition not found.")
            depth_field = "    @builtin(frag_depth) depth : f32,"
            line = lines[struct_linenr]
            indent = indent_from_line(line)
            new_lines = [line, indent + depth_field]
            lines[struct_linenr] = "\n".join(new_lines)

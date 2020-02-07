from .wgpu import BaseWGPU


def help(name_part, look_in_docs=False):
    """ Print constants, enums, structs, and functions that contain the given name_part.
    """

    name_part = name_part.lower()

    # Find items
    items = []
    for name, val in BaseWGPU.__dict__.items():
        if name.startswith("__"):
            continue
        if name_part in name.lower():
            items.append((name, val))
        elif look_in_docs and hasattr(val, "__doc__") and isinstance(val.__doc__, str):
            if name_part in val.__doc__.lower():
                items.append((name, val))

    # Order
    items_ordered = {"constants": [], "enums": [], "structs": [], "functions": []}
    for name, val in items:
        if isinstance(val, int):
            if name.split("_")[-1].upper() == name.split("_")[-1]:
                items_ordered["constants"].append(name)
            else:
                items_ordered["enums"].append(name)
        elif name.startswith("create_"):
            items_ordered["structs"].append(name)
        else:
            items_ordered["functions"].append(name)

    # Display
    print(f"Searching for {name_part!r} ...")
    for key in items_ordered.keys():
        print(f"Found {len(items_ordered[key])} {key}")
        for name in items_ordered[key]:
            print(f"    {name}")


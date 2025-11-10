from typing import Any


# TODO: generalize and make below a package
def get_nested_values(buckets: list[dict[str, Any]], kv: dict[str, str]):
    values = []
    for bucket in buckets:
        value = {}
        for k, v in kv.items():
            value[k] = get_all_values(bucket, v)
        values.append(value)
    return values


def get_all_values(root, key):
    values = []
    if "%v" in key:
        i = 0
        while True:
            keys = key.replace("%v", str(i)).split(".")
            _, el, sub_value = get_by_path(root, keys)
            if sub_value == None:
                break
            values.append(sub_value)
            i += 1
    else:
        _, el, value = get_by_path(root, key.split("."))
        if not el:
            return value
    return values


def get_values_buckets(buckets: list[dict[str, Any]], key: str):
    values = []
    keys = key.split(".")
    for i, key in enumerate(keys):
        if "~1" in key:
            keys[i] = key.replace("~1", ".")
    for bucket in buckets:
        _, el, value = get_by_path(bucket, keys)
        if el == None:
            values.append(value)
    return values


def navigate(sequence, initial):
    it = iter(sequence)

    value = initial
    for i, element in enumerate(it):
        try:
            value = value.get(element, None)
            if value == None:
                return i, element, None
        except AttributeError:
            if isinstance(value, list):
                try:
                    value = value[int(element)]
                except IndexError:
                    return i, element, None

    return None, None, value


def get_by_path(root: dict[str, Any], keys: list[str]):
    return navigate(keys, root)


def set_at_path(root: dict[str, Any], value: Any, keys: list[str]):
    for i, key in enumerate(keys):
        if type(root) is dict:
            if i == len(keys) - 1:
                root[key] = value
                break
            elif not root.get(key):
                if keys[i + 1] == "-":
                    root[key] = []
                else:
                    root[key] = {}
            root = root[key]
        elif type(root) is list:
            if key == "-":
                if value == len(keys) - 1:
                    root.append(value)
                    break
                root.append({})
                root = root[len(root) - 1]
            else:
                root = root[int(key)]


# def set(data: dict[str, Any], value: Any, hierarchy: list[str]):
#     for key in range(0, len(hierarchy)):
#         path = hierarchy[key]
#         print(data)
#         if type(data) is dict:
#             if key == len(hierarchy) - 1:
#                 data[key] = value
#             elif not data.get(path, None):
#                 data[path] = {}
#         elif isinstance(data, list):
#             if path == "-":
#                 if key < 1:
#                     raise ValueError("unable to append new array index at root of path")
#                 if value == len(hierarchy) - 1:
#                     data.append(value)
#                 else:
#                     new_obj = {}
#                     data.append(new_obj)
#                     data = new_obj
#             else:
#                 try:
#                     index = int(path)
#                 except ValueError:
#                     raise ValueError(
#                         f"failed to resolve path segment '{value}': found array but segment value '{path}' could not be parsed into array index"
#                     )

#                 if index < 0 or index >= len(data):
#                     raise IndexError(
#                         f"failed to resolve path segment '{value}': index '{index}' is out of bounds for array of size {len(data)}"
#                     )

#                 if value == len(hierarchy) - 1:
#                     data[index] = value
#                 else:
#                     if obj[index] is None:
#                         raise ValueError(
#                             f"failed to resolve path segment '{value}': field '{path}' was not found"
#                         )
#                     data = data[index]

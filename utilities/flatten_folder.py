import os.path


def get_filenames(root, suffix=None):
    file_path_list = []
    if not suffix:
        suffix = ''
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in [f for f in filenames if f.endswith(suffix)]:
            file_path_list.append(os.path.join(dirpath, filename))
    return file_path_list

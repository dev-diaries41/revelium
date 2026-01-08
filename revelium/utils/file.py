import os 

def get_new_dirname(dir_path: str, prefix: str):
    os.makedirs(dir_path, exist_ok=True)
    highest = -1
    dirs = os.listdir(dir_path)
    if len(dirs) == 0:
        return os.path.join(dir_path, prefix + "0")
    for d in dirs:
        if not d.startswith(prefix):
            continue
        n = int(d.strip(prefix))
        highest = n if n > highest else highest
    return os.path.join(dir_path, prefix + str(highest + 1))


def get_new_filename(dir_path: str, prefix: str, ext):
    os.makedirs(dir_path, exist_ok=True)
    highest = -1
    files = os.listdir(dir_path)
    if len(files) == 0:
        return os.path.join(dir_path,  prefix + "0" + ext)
    for f in files:
        if not f.startswith(prefix):
            continue
        n = int(os.path.splitext(f)[0].strip(prefix))
        highest = n if n > highest else highest
    return os.path.join(dir_path, prefix + str(highest + 1) + ext)

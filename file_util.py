import os


__all__ = ['mkdirs']


def exists(filename):
    return os.path.exists(filename)


def mkdirs(filename, all_dir=False, mode=0o755):
    dirs = os.path.dirname(filename) if not all_dir else filename
    if not exists(dirs):
        os.makedirs(dirs, mode=mode)
    return True


if __name__ == "__main__":
    mkdirs("./a/b/c")
    mkdirs("./aa/bb/cc", all_dir=True)

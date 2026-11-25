"""
Utils for animl-r
"""

from animl import __version__


def get_version():
    """Returns the version of animl-py."""
    return __version__


def check_exiftool():
    """Checks if exiftool is installed and accessible."""
    import exiftool
    try:
        with exiftool.ExifToolHelper() as et:
            return et.version
    except Exception as e:
        print(f"Error checking exiftool: {e}")
        return False


def check_onnx_cuda():
    """Checks if ONNX Runtime can actually run on a CUDA GPU (see cuda_usable)."""
    from animl.utils.general import cuda_usable
    return cuda_usable()

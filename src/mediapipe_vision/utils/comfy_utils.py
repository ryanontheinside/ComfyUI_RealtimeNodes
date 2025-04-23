"""Utilities specific to ComfyUI integration."""

class AlwaysEqualProxy(str):
    """Helper class to allow connecting multiple specific input types to a single input node.
       Borrowed from https://github.com/theUpsider/ComfyUI-Logic
    """
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False 
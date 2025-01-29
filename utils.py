class AlwaysEqualProxy(str):
    #borrowed from https://github.com/theUpsider/ComfyUI-Logic
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False
class ImageDimensions:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}
    
    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "count")
    FUNCTION = "get_dimensions"
    CATEGORY = "image/info"

    def get_dimensions(self, image):
        count = image.shape[0]  # Batch size
        height = image.shape[1] 
        width = image.shape[2]
        print(f"Image dimensions: {width}x{height} (batch size: {count})")
        return (width, height, count)

NODE_CLASS_MAPPINGS = {
    "ImageDimensions": ImageDimensions
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageDimensions": "Get Image Dimensions"
}
import torch
import torch.nn.functional as F

class HistogramStretch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "clip_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05}),
                "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance"
    CATEGORY = "image/enhance"

    def enhance(self, image, clip_percent, amount):
        # Automatically stretch histogram for better contrast
        B, H, W, C = image.shape
        x = image.permute(0, 3, 1, 2)  # to BCHW
        
        # Calculate histogram bounds
        low = torch.quantile(x, clip_percent, dim=2, keepdim=True)
        low = torch.quantile(low, clip_percent, dim=3, keepdim=True)
        
        high = torch.quantile(x, 1.0 - clip_percent, dim=2, keepdim=True)
        high = torch.quantile(high, 1.0 - clip_percent, dim=3, keepdim=True)
        
        # Stretch contrast with amount control
        enhanced = x + amount * ((x - low) / (high - low) - x)
        enhanced = torch.clamp(enhanced, 0, 1)
        
        return (enhanced.permute(0, 2, 3, 1),)

class AdaptiveSharpen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "edge_threshold": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance"
    CATEGORY = "image/enhance"

    def enhance(self, image, max_amount, edge_threshold):
        # Content-aware sharpening using Laplacian
        B, H, W, C = image.shape
        x = image.permute(0, 3, 1, 2)  # to BCHW
        
        # Laplacian kernel for edge detection
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        
        # Detect edges
        edges = F.conv2d(x, laplacian.expand(C, 1, 3, 3), padding=1, groups=C)
        edges = torch.abs(edges)
        
        # Adaptive amount based on edge strength
        amount = torch.sigmoid(edges * edge_threshold) * max_amount
        
        # Apply sharpening
        enhanced = x + amount * edges
        enhanced = torch.clamp(enhanced, 0, 1)
        
        return (enhanced.permute(0, 2, 3, 1),)

class ChannelBalance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance"
    CATEGORY = "image/enhance"

    def enhance(self, image, amount):
        # Automatic RGB channel balancing
        B, H, W, C = image.shape
        
        # Calculate mean for each channel
        means = image.mean(dim=(0, 1, 2), keepdim=True)
        target_mean = means.mean()
        
        # Adjust each channel to match target mean with amount control
        balance = (target_mean / means)
        enhanced = image + amount * (image * balance - image)
        enhanced = torch.clamp(enhanced, 0, 1)
        
        return (enhanced,)

class MultiScaleDetail:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "small_amount": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.005}),
                "medium_amount": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.005}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance"
    CATEGORY = "image/enhance"

    def enhance(self, image, small_amount, medium_amount):
        # Multi-scale detail enhancement with zero-sum kernels
        B, H, W, C = image.shape
        x = image.permute(0, 3, 1, 2)  # to BCHW
        
        # Small details (3x3) - zero-sum kernel
        small_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        small_details = F.conv2d(x, small_kernel.expand(C, 1, 3, 3), padding=1, groups=C)
        
        # Medium details (5x5) - zero-sum kernel
        medium_kernel = torch.tensor([[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1], [-1, -1, -1, -1, -1]], dtype=torch.float32, device=x.device).view(1, 1, 5, 5)
        medium_kernel = medium_kernel - medium_kernel.mean()  # Ensure zero-sum
        medium_details = F.conv2d(x, medium_kernel.expand(C, 1, 5, 5), padding=2, groups=C)
        
        # Combine details with adjustable weighting
        enhanced = x + small_amount * small_details + medium_amount * medium_details
        enhanced = torch.clamp(enhanced, 0, 1)
        
        return (enhanced.permute(0, 2, 3, 1),)

class UnsharpMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 10.0, "step": 0.1}),
                "amount": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sharpen"
    CATEGORY = "image/enhance"

    def sharpen(self, image, blur_radius, amount):
        # Input: (B, H, W, C)
        B, H, W, C = image.shape
        x = image.permute(0, 3, 1, 2)  # to BCHW

        # Gaussian kernel (separable)
        def gaussian_kernel1d(radius, sigma):
            size = int(2 * radius + 1)
            coords = torch.arange(size) - radius
            kernel = torch.exp(-coords**2 / (2 * sigma**2))
            kernel /= kernel.sum()
            return kernel.to(x.device)

        radius = int(blur_radius)
        kernel = gaussian_kernel1d(radius, sigma=blur_radius)

        # Create vertical and horizontal kernels
        kernel_v = kernel.view(1, 1, -1, 1)
        kernel_h = kernel.view(1, 1, 1, -1)

        # Blur H and W
        blurred = F.conv2d(x, kernel_v.expand(C, 1, -1, 1), padding=(0, radius), groups=C)
        blurred = F.conv2d(blurred, kernel_h.expand(C, 1, 1, -1), padding=(radius, 0), groups=C)

        # Unsharp mask
        sharpened = x + amount * (x - blurred)
        sharpened = torch.clamp(sharpened, 0, 1)

        return (sharpened.permute(0, 2, 3, 1),)

class ICAS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0}),
                "radius": ("INT", {"default": 2, "min": 1, "max": 5}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance"
    CATEGORY = "image/enhance"

    def enhance(self, image, amount, radius):
        # Image Contrast Adaptive Sharpen
        B, H, W, C = image.shape
        x = image.permute(0, 3, 1, 2)  # to BCHW

        # Efficient Gaussian blur using separable kernels
        def gaussian_kernel1d(radius, sigma):
            size = int(2 * radius + 1)
            coords = torch.arange(size) - radius
            kernel = torch.exp(-coords**2 / (2 * sigma**2))
            kernel /= kernel.sum()
            return kernel.to(x.device)

        # Create and apply Gaussian blur
        kernel = gaussian_kernel1d(radius, sigma=radius)
        kernel_v = kernel.view(1, 1, -1, 1)
        kernel_h = kernel.view(1, 1, 1, -1)
        
        # Apply separable blur
        blurred = F.conv2d(x, kernel_v.expand(C, 1, -1, 1), padding=(0, radius), groups=C)
        blurred = F.conv2d(blurred, kernel_h.expand(C, 1, 1, -1), padding=(radius, 0), groups=C)

        # Calculate local contrast
        local_contrast = torch.abs(x - blurred)
        local_contrast = F.avg_pool2d(local_contrast, kernel_size=3, stride=1, padding=1)

        # Adaptive sharpening mask
        mask = torch.sigmoid(local_contrast * 10.0)  # Adjust sensitivity
        mask = mask * amount

        # Apply sharpening
        sharpened = x + mask * (x - blurred)
        sharpened = torch.clamp(sharpened, 0, 1)

        return (sharpened.permute(0, 2, 3, 1),)

class FastBilateralDetail:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0}),
                "radius": ("INT", {"default": 3, "min": 1, "max": 7}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance"
    CATEGORY = "image/enhance"

    def enhance(self, image, amount, radius):
        # Fast bilateral filter-based detail enhancement
        B, H, W, C = image.shape
        x = image.permute(0, 3, 1, 2)  # to BCHW

        # Efficient bilateral filter approximation using separable Gaussian
        def gaussian_kernel1d(radius, sigma):
            size = int(2 * radius + 1)
            coords = torch.arange(size) - radius
            kernel = torch.exp(-coords**2 / (2 * sigma**2))
            kernel /= kernel.sum()
            return kernel.to(x.device)

        # Create kernels
        kernel = gaussian_kernel1d(radius, sigma=radius)
        kernel_v = kernel.view(1, 1, -1, 1)
        kernel_h = kernel.view(1, 1, 1, -1)

        # Apply separable Gaussian blur
        blurred = F.conv2d(x, kernel_v.expand(C, 1, -1, 1), padding=(0, radius), groups=C)
        blurred = F.conv2d(blurred, kernel_h.expand(C, 1, 1, -1), padding=(radius, 0), groups=C)

        # Calculate edge strength using Sobel
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(x, sobel_x.expand(C, 1, 3, 3), padding=1, groups=C)
        grad_y = F.conv2d(x, sobel_y.expand(C, 1, 3, 3), padding=1, groups=C)
        edge_strength = torch.sqrt(grad_x**2 + grad_y**2)
        edge_strength = F.avg_pool2d(edge_strength, kernel_size=3, stride=1, padding=1)

        # Create adaptive mask based on edge strength
        mask = torch.sigmoid(edge_strength * 5.0)  # Adjust sensitivity
        mask = mask * amount

        # Apply detail enhancement
        enhanced = x + mask * (x - blurred)
        enhanced = torch.clamp(enhanced, 0, 1)

        return (enhanced.permute(0, 2, 3, 1),)

class WaveletDetail:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0}),
                "scale": ("INT", {"default": 2, "min": 1, "max": 4}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance"
    CATEGORY = "image/enhance"

    def enhance(self, image, amount, scale):
        # Wavelet-based detail enhancement
        B, H, W, C = image.shape
        x = image.permute(0, 3, 1, 2)  # to BCHW

        # Haar wavelet decomposition
        def haar_decompose(x):
            # Downsample
            x_low = F.avg_pool2d(x, kernel_size=2, stride=2)
            # Upsample and subtract for detail
            x_high = x - F.interpolate(x_low, size=x.shape[2:], mode='bilinear', align_corners=False)
            return x_low, x_high

        # Multi-scale decomposition
        details = []
        current = x
        for _ in range(scale):
            current, detail = haar_decompose(current)
            # Upsample detail to match original size
            detail = F.interpolate(detail, size=(H, W), mode='bilinear', align_corners=False)
            details.append(detail)

        # Enhance details
        enhanced = x
        for detail in details:
            enhanced = enhanced + amount * detail

        enhanced = torch.clamp(enhanced, 0, 1)
        return (enhanced.permute(0, 2, 3, 1),)

class FrequencySelective:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0}),
                "radius": ("INT", {"default": 3, "min": 1, "max": 7}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance"
    CATEGORY = "image/enhance"

    def enhance(self, image, amount, radius):
        # Frequency-selective detail enhancement
        B, H, W, C = image.shape
        x = image.permute(0, 3, 1, 2)  # to BCHW

        # Efficient Gaussian blur using separable kernels
        def gaussian_kernel1d(radius, sigma):
            size = int(2 * radius + 1)
            coords = torch.arange(size) - radius
            kernel = torch.exp(-coords**2 / (2 * sigma**2))
            kernel /= kernel.sum()
            return kernel.to(x.device)

        # Create kernels
        kernel = gaussian_kernel1d(radius, sigma=radius)
        kernel_v = kernel.view(1, 1, -1, 1)
        kernel_h = kernel.view(1, 1, 1, -1)

        # Apply separable Gaussian blur
        blurred = F.conv2d(x, kernel_v.expand(C, 1, -1, 1), padding=(0, radius), groups=C)
        blurred = F.conv2d(blurred, kernel_h.expand(C, 1, 1, -1), padding=(radius, 0), groups=C)

        # Frequency separation
        high_freq = x - blurred
        low_freq = blurred

        # Enhance high frequencies
        enhanced = low_freq + amount * high_freq
        enhanced = torch.clamp(enhanced, 0, 1)

        return (enhanced.permute(0, 2, 3, 1),)

class GuidedDetail:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0}),
                "radius": ("INT", {"default": 3, "min": 1, "max": 7}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance"
    CATEGORY = "image/enhance"

    def enhance(self, image, amount, radius):
        # Guided filter-based detail enhancement
        B, H, W, C = image.shape
        x = image.permute(0, 3, 1, 2)  # to BCHW

        # Efficient Gaussian blur using separable kernels
        def gaussian_kernel1d(radius, sigma):
            size = int(2 * radius + 1)
            coords = torch.arange(size) - radius
            kernel = torch.exp(-coords**2 / (2 * sigma**2))
            kernel /= kernel.sum()
            return kernel.to(x.device)

        # Create kernels
        kernel = gaussian_kernel1d(radius, sigma=radius)
        kernel_v = kernel.view(1, 1, -1, 1)
        kernel_h = kernel.view(1, 1, 1, -1)

        # Apply separable Gaussian blur
        blurred = F.conv2d(x, kernel_v.expand(C, 1, -1, 1), padding=(0, radius), groups=C)
        blurred = F.conv2d(blurred, kernel_h.expand(C, 1, 1, -1), padding=(radius, 0), groups=C)

        # Calculate local variance
        local_mean = F.avg_pool2d(x, kernel_size=2*radius+1, stride=1, padding=radius)
        local_var = F.avg_pool2d(x**2, kernel_size=2*radius+1, stride=1, padding=radius) - local_mean**2

        # Create adaptive mask based on local variance
        mask = torch.sigmoid(local_var * 10.0)  # Adjust sensitivity
        mask = mask * amount

        # Apply detail enhancement
        enhanced = x + mask * (x - blurred)
        enhanced = torch.clamp(enhanced, 0, 1)

        return (enhanced.permute(0, 2, 3, 1),)

class ImageConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "direction": (["horizontal", "vertical"],),
                "spacing": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concat"
    CATEGORY = "image/enhance"

    def concat(self, image1, image2, direction, spacing):
        # Ensure both images have the same number of channels
        if image1.shape[-1] != image2.shape[-1]:
            raise ValueError("Images must have the same number of channels")

        # Add spacing
        if spacing > 0:
            if direction == "horizontal":
                spacer = torch.zeros((image1.shape[0], image1.shape[1], spacing, image1.shape[3]), device=image1.device)
                return (torch.cat([image1, spacer, image2], dim=2),)
            else:
                spacer = torch.zeros((image1.shape[0], spacing, image1.shape[2], image1.shape[3]), device=image1.device)
                return (torch.cat([image1, spacer, image2], dim=1),)

        # Concatenate without spacing
        if direction == "horizontal":
            return (torch.cat([image1, image2], dim=2),)
        else:
            return (torch.cat([image1, image2], dim=1),)

NODE_CLASS_MAPPINGS = {
    "HistogramStretch": HistogramStretch,
    "AdaptiveSharpen": AdaptiveSharpen,
    "ChannelBalance": ChannelBalance,
    "MultiScaleDetail": MultiScaleDetail,
    "UnsharpMask": UnsharpMask,
    "ICAS": ICAS,
    "FastBilateralDetail": FastBilateralDetail,
    "WaveletDetail": WaveletDetail,
    "FrequencySelective": FrequencySelective,
    "GuidedDetail": GuidedDetail,
    "ImageConcat": ImageConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HistogramStretch": "Histogram Stretch",
    "AdaptiveSharpen": "Adaptive Sharpen",
    "ChannelBalance": "Channel Balance",
    "MultiScaleDetail": "Multi-Scale Detail",
    "UnsharpMask": "Unsharp Mask",
    "ICAS": "Image Contrast Adaptive Sharpen",
    "FastBilateralDetail": "Fast Bilateral Detail",
    "WaveletDetail": "Wavelet Detail",
    "FrequencySelective": "Frequency Selective",
    "GuidedDetail": "Guided Detail",
    "ImageConcat": "Image Concat",
}



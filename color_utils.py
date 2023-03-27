import torch
import cv2
import numpy as np
from typing import *


def color_mod(img: torch.Tensor, rgb: torch.Tensor, num_bins: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Modify the color of the image and point cloud to further enhance pose estimation quality
    use histogram equalization for ycbcr

    Args:
        img: (H, W, 3) torch tensor containing image RGB values
        rgb: (N, 3) torch tensor containing point cloud RGB values
        num_bins: number of bins to use when making histograms

    Returns:
        img: (H, W, 3) torch tensor containing modified image RGB values
        rgb: (N, 3) torch tensor containing modified point cloud RGB values
    """

    orig_device = img.device
    # Process image first
    H, W, _ = img.shape

    img = img.reshape(-1, 3)
    tgt_img = img[(img * 255).long().sum(-1) > 0]

    # Convert to YCbCr
    tgt_img = cv2.cvtColor((tgt_img * 255.).cpu().numpy().astype(np.uint8).reshape(1, -1, 3),
                            cv2.COLOR_RGB2YCR_CB).squeeze()
    mod_rgb = cv2.cvtColor((rgb * 255.).cpu().numpy().astype(np.uint8).reshape(1, -1, 3),
                            cv2.COLOR_RGB2YCR_CB).squeeze()

    tgt_img = torch.from_numpy(tgt_img) / 255.
    mod_rgb = torch.from_numpy(mod_rgb) / 255.

    img_y_hist = torch.bincount((tgt_img[:, 0] * (num_bins - 1)).long(), minlength=num_bins).float()
    rgb_y_hist = torch.bincount((mod_rgb[:, 0] * (num_bins - 1)).long(), minlength=num_bins).float()

    tot_y_hist = img_y_hist + rgb_y_hist
    tot_y_hist /= tot_y_hist.sum()

    # Cumulative sum for generating equalized image
    tot_y_hist = torch.cumsum(tot_y_hist, 0)

    tgt_img[:, 0] = torch.take(tot_y_hist, (tgt_img[:, 0] * (num_bins - 1)).long())

    tgt_img = cv2.cvtColor((tgt_img * 255.).numpy().astype(np.uint8).reshape(1, -1, 3), cv2.COLOR_YCR_CB2RGB)
    tgt_img = torch.from_numpy(tgt_img).reshape(-1, 3) / 255.

    img[(img * 255).long().sum(-1) > 0] = tgt_img.to(orig_device)
    img = img.reshape(H, W, 3)

    img = img.to(orig_device)

    # Process point cloud rgb
    mod_rgb[:, 0] = torch.take(tot_y_hist, (mod_rgb[:, 0] * (num_bins - 1)).long())

    mod_rgb = cv2.cvtColor((mod_rgb * 255.).numpy().astype(np.uint8).reshape(1, -1, 3), cv2.COLOR_YCR_CB2RGB)
    mod_rgb = torch.from_numpy(mod_rgb).reshape(-1, 3) / 255.

    rgb = mod_rgb.to(orig_device)

    return img, rgb


def histogram(img: torch.Tensor, mask: torch.Tensor, channels: List[int] = [32, 32, 32], normalize=True) -> torch.Tensor:
    """
    Returns a color histogram of an input image

    Args:
        img: (H, W, 3) or (B, H, W, 3) torch tensor containing RGB values
        mask: (H, W) or (B, H, W) torch tensor with mask values
        channels: List of length 3 containing number of bins per each channel
        normalize: If True, normalizes histogram

    Returns:
        hist: Histogram of shape (*channels)
    """

    # Make the color of an image to be in range (0, 255)
    tgt_img = img.clone().detach()
    final_mask = mask.clone().detach()
    max_rgb = torch.LongTensor([255] * 3).to(tgt_img.device)
    bin_size = torch.ceil(max_rgb.float() / torch.tensor(channels).float().to(tgt_img.device)).long()

    if tgt_img.max() <= 1:
        tgt_img = (tgt_img * max_rgb.reshape(-1, 3)).long()
    
    if len(img.shape) == 3:
        tgt_rgb = tgt_img[torch.nonzero(final_mask.long(), as_tuple=True)].long() # (N, 3) torch tensor
        tgt_rgb = tgt_rgb // bin_size.reshape(-1, 3)

        tgt_rgb = tgt_rgb[:, 0] + channels[0] * tgt_rgb[:, 1] + channels[0] * channels[1] * tgt_rgb[:, 2]

        hist = torch.bincount(tgt_rgb, minlength=channels[0] * channels[1] * channels[2]).float()
        hist = hist.reshape(*channels)

        if normalize:
            # normalize histogram
            hist = hist / hist.sum()
    else:  # Batched input
        eps = 1e-6
        tgt_img = tgt_img // bin_size.reshape(-1, 3)
        tgt_img = tgt_img[..., 0] + channels[0] * tgt_img[..., 1] + channels[0] * channels[1] * tgt_img[..., 2]  # (B, H, W)
        tgt_img *= final_mask.float()
        tgt_img = tgt_img.reshape(tgt_img.shape[0], -1).long()  # (B, H * W)
        hist = torch.zeros([tgt_img.shape[0], channels[0] * channels[1] * channels[2]], device=tgt_img.device, dtype=torch.long).scatter_add(
            dim=-1, index=tgt_img, src=torch.ones_like(tgt_img, dtype=torch.long))  # (B, C)
        hist[:, 0] -= (~final_mask).reshape(tgt_img.shape[0], -1).sum(-1)  # Subtract zeros from final mask
        hist = hist.float()

        if normalize:
            hist_sum = hist.sum(-1)
            hist = hist / (hist_sum.reshape(-1, 1) + eps)  # Normalize
        hist = hist.reshape([hist.shape[0], *channels])

    return hist


def histogram_intersection(hist_1: torch.Tensor, hist_2: torch.Tensor) -> float:
    """
    Computes intersection between two histrograms

    Args:
        hist_1: torch tensor containing first histogram
        hist_2: torch tensor containing second histogram
    
    Returns:
        intersection: Amount of intersection between hist_1 and hist_2
    """

    assert hist_1.shape == hist_2.shape

    if len(hist_1.shape) == 3:
        intersection = torch.min(hist_1, hist_2).sum()
    else:  # Batched case: returns batched intersections
        hist_1 = hist_1.reshape(hist_1.shape[0], -1)
        hist_2 = hist_2.reshape(hist_2.shape[0], -1)
        intersection = torch.min(hist_1, hist_2)  # (B, C)
        intersection = intersection.sum(dim=-1)  # (B, )
    
    return intersection

def color_match(img: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """
    Match the color of the image and point cloud to further enhance pose estimation quality

    Args:
        img: (H, W, 3) torch tensor containing image RGB values
        rgb: (N, 3) torch tensor containing point cloud RGB values

    Returns:
        img: (H, W, 3) torch tensor containing modified image RGB values
    """

    def _interp(x, xp, fp, period=360):
        """
        Linear interpolation for monotonically increasing sample points.

        Returns the linear interpolant to a function
        with given discrete data points (`xp`, `fp`), evaluated at `x`.
        """

        asort_xp = torch.argsort(xp)
        xp = xp[asort_xp]
        fp = fp[asort_xp]

        xp = torch.cat([xp[-1:] - period, xp, xp[0:1] + period])
        fp = torch.cat([fp[-1:], fp, fp[0:1]])

        interpolant = torch.zeros_like(x)

        for i in range(len(x)):
            big_ind = len(xp) - (x[i:i+1] < xp).sum()
            small_ind = big_ind - 1

            inds = torch.arange(x.shape[0])
            interpolant[i] = ((x[i] - xp[small_ind]) * fp[big_ind] + (xp[big_ind] - x[i]) * fp[small_ind]) / (xp[big_ind] - xp[small_ind])

        return interpolant


    def _match_cumulative_cdf(source, template, weight):
        """
        Return modified source array so that the cumulative density function of
        its values matches the cumulative density function of the template.
        """

        src_values, src_unique_indices = torch.unique(source, return_inverse=True, sorted=True)
        tmp_values, tmp_counts = torch.unique(template, return_counts=True, sorted=True)
        src_counts = torch.bincount((source * 255).int(), weight)
        # caculate normalized quantiles for each array
        src_quantiles = torch.cumsum(src_counts, 0)
        src_quantiles = src_quantiles / src_quantiles[-1]
        tmp_quantiles = torch.cumsum(tmp_counts, 0) / len(template)

        interp_a_values = _interp(src_quantiles, tmp_quantiles, tmp_values)

        return interp_a_values[src_unique_indices].reshape(source.shape)


    def match_histograms(image, reference, weight):
        """
        Adjust an image so that its cumulative histogram matches that of another.
        The adjustment is applied separately for each channel.
        """
        matched = torch.empty(image.shape).to(image.device)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(image[..., channel], reference[..., channel], weight)
            matched[..., channel] = matched_channel
        
        return matched

    orig_device = img.device
    # Process image first
    H, W, _ = img.shape

    h_inds = torch.tensor([i for i in range(H) for _ in range(W)]).float().to(orig_device)
    sin_weight = torch.sin(h_inds / H * np.pi)

    img = img.clone().detach().reshape(-1, 3)
    tgt_img = img[(img * 255).long().sum(-1) > 0]
    tgt_sin_weight = sin_weight[(img * 255).long().sum(-1) > 0]

    # Match image with respect to point cloud
    # mod_img = match_histograms(tgt_img.cpu().numpy(), rgb.cpu().numpy(), multichannel=True)
    mod_img = match_histograms(tgt_img, rgb, tgt_sin_weight)

    img[(img * 255).long().sum(-1) > 0] = mod_img.clone().detach()
    img = img.reshape(H, W, 3)

    return img

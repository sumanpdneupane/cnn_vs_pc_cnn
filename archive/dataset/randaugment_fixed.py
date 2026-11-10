from torchvision import transforms


class RandAugment64(transforms.RandAugment):
    """RandAugment with shifts sized for a 64 px image, even when it runs on a padded image."""

    def _augmentation_space(self, num_bins, image_size):
        return super()._augmentation_space(num_bins, (64, 64))

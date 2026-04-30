
# STUDENT's UCO: 514143

############################################################################################################
# Write short answers to the questions. Please do not exceed a total of 250 words for all of your answers. #
############################################################################################################

1. Project code:

Data loading: The entire dataset is preloaded into RAM to eliminate I/O bottlenecks during training. The dataset is split 70/20/10 (train/val/test). Training augmentations — 512x512 random crops, ShiftScaleRotate, HorizontalFlip, RandomBrightnessContrast, and ImageNet normalization — improve generalization; validation uses normalization only.

Class imbalance: A custom MixedCropTransform biases 50% of crops toward rare classes (object, human) to increase their representation. Additionally, per-class frequency weights w_c = 1/log(1.02 + freq_c) are passed to the loss, penalizing errors on underrepresented classes more.

Loss: HybridSegmentationLoss = weighted CrossEntropyLoss + DiceLoss. CE provides stable pixel-wise gradients; Dice directly optimizes the overlap metric and is robust to imbalance. Void class (0) is ignored.

Optimizer & scheduler: AdamW with weight decay regularizes effectively. Linear warmup (4% of epochs) stabilizes early training; polynomial decay (power=0.9) gradually reduces the learning rate for fine convergence.

Metric: mIoU from an accumulated confusion matrix, excluding the void class — the standard metric for semantic segmentation.


2. What model architecture did you use? Why did you choose it?

I used MobileNetV3-Large with a DeepLabV3+ decoder (ASPP + skip connections).

MobileNetV3-Large was the practical choice given hardware constraints—it's efficient yet maintains strong feature extraction. Cityscapes requires multi-scale understanding (cars, pedestrians, poles), which ASPP provides through dilated convolutions. The encoder-decoder skip connection (stride-8 to stride-32) preserves object boundaries while capturing semantic context. This balanced computational efficiency with segmentation quality for the dataset.




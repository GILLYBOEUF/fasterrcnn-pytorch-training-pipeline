import torchvision
import torch.nn as nn

from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator

def create_model(num_classes=2, pretrained=True, coco_model=False):
    if pretrained:
        # Load the pretrained KeypointRCNN large features
        backbone = torchvision.models.mobilenet_v2(weights='DEFAULT').features
    else:
        backbone = torchvision.models.mobilenet_v2().features
    # KeypointRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280,
    # so we need to add it here
    backbone.out_channels = 1280
    
    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect 
    # ratios.
    anchor_generator = AnchorGenerator(
        sizes=(32, 64, 128, 256, 512), 
        aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0)
        )

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be ['0']. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)

    keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                                output_size=14,
                                                                sampling_ratio=2)
    model = KeypointRCNN(backbone,
                            num_classes=num_classes,
                            num_keypoints=6, # We only want to detect the lower body keypoints
                            rpn_anchor_generator=anchor_generator,
                            box_roi_pool=roi_pooler,
                            keypoint_roi_pool=keypoint_roi_pooler)

    if coco_model: # Return the COCO pretrained model for COCO classes.
        return model, coco_model
    
    return model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)
import torchvision
import torch
import torch.nn as nn

from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes=2, pretrained=True, coco_model=False):
    import models
    from models.model_summary import summary
    # Load Faster RCNN pre-trained model
    faster_model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )
    
    # Get the number of input features 
    in_features = faster_model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    faster_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    weights = "/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/fasterrcnn-pytorch-training-pipeline/best_model.pth"
    DEVICE = "cpu"
    # Load the pretrained checkpoint.
    checkpoint = torch.load(weights, map_location=DEVICE) 
    keys = list(checkpoint['model_state_dict'].keys())
    ckpt_state_dict = checkpoint['model_state_dict']
    # Get the number of classes from the loaded checkpoint.
    old_classes = ckpt_state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]

    # Load weights.
    faster_model.load_state_dict(ckpt_state_dict)
    
    backbone = faster_model.backbone

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
                                                        output_size=3,
                                                        sampling_ratio=2)

    keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                                output_size=14,
                                                                sampling_ratio=2)
    key_model = KeypointRCNN(backbone,
                            num_classes=num_classes,
                            num_keypoints=6, # We only want to detect the lower body keypoints
                            rpn_anchor_generator=anchor_generator,
                            box_roi_pool=roi_pooler,
                            keypoint_roi_pool=keypoint_roi_pooler)
    
    summary(key_model)

    return key_model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=2, pretrained=True, coco_model=True)
    summary(model)
import torchvision

def get_model(device):
    # Load a pre-trained SSD Lite model with MobileNetV3 backbone.
    # The model is trained on the COCO dataset and ready for fine-tuning or inference.
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights='DEFAULT')
    
    # Set the model to evaluation mode and move it to the specified device (CPU or GPU).
    # Evaluation mode disables layers like dropout and batch normalization updates.
    model = model.eval().to(device)
    
    # Return the prepared model.
    return model

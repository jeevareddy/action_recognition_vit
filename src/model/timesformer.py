import torch.nn as nn
from transformers import TimesformerForVideoClassification

class Timesformer(nn.Module):
    def __init__(self, n_classes, device ="cpu", model_path = "facebook/timesformer-base-finetuned-k400"):
        super(Timesformer, self).__init__()        
        self.backbone = TimesformerForVideoClassification.from_pretrained(model_path, output_hidden_states=True, output_attentions=True)
        self.classifier_in_feature_dim = self.backbone.classifier.in_features
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.backbone.classifier = nn.Linear(self.classifier_in_feature_dim, n_classes)
        self.backbone.to(device)
        
        # self.classifier = nn.Linear(self.feature_dim, n_classes)

    def forward(self, x):        
        return self.backbone(x)# The above code is not doing anything as it consists of two comments. The
        # first comment is "# Python" which indicates that the code is written in
        # Python programming language. The second comment is "atten" which is just a
        # string and does not have any significance in the code.
        
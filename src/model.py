import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel
from torchvision.models import ResNet50_Weights


class MultimodalLongTailClassifier(nn.Module):
    def __init__(self, num_classes, image_token_dim=768, num_attention_heads=8, dropout=0.1):
        """
        Args:
            num_classes: number of classes for classification
            image_token_dim: image token dimension, e.g. 768
            num_attention_heads: number of attention heads in cross-attention
            dropout: dropout rate for attention layers
        """
        super(MultimodalLongTailClassifier, self).__init__()

        # 1. using pre-trained ResNet-50 as image backbone
        # resnet = models.resnet50(pretrained=True)
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # remove the last two layers (avgpool and fc) to get the backbone
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-2])
        # project the image features to image_token_dim
        self.image_proj = nn.Linear(2048, image_token_dim)

        # 2. using pre-trained BERT as text encoder
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        # BERT hidden size 768
        text_hidden_size = self.text_encoder.config.hidden_size

        # 3. using cross-attention to fuse image and text, [CLS] token as query
        # using nn.MultiheadAttention，set batch_first=True to deal [B, seq_len, d]
        self.cross_attn = nn.MultiheadAttention(embed_dim=image_token_dim,
                                                num_heads=num_attention_heads,
                                                dropout=dropout,
                                                batch_first=True)

        # 4. fusion layer to combine [CLS] token and cross-attention output
        self.fusion_layer = nn.Linear(text_hidden_size + image_token_dim, image_token_dim)

        # 5. classifier layer
        self.classifier = nn.Sequential(
            nn.LayerNorm(image_token_dim),
            nn.Linear(image_token_dim, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        """
        Args:
            image: [B, C, H, W] image tensor input
            input_ids: [B, seq_len], tokenized input text
            attention_mask: corresponding attention mask，shape: [B, seq_len]
        Returns:
            logits: [B, num_classes] classification logits
        """
        # ----- iamge branch -----
        # get features, shape: [B, 2048, 7, 7]
        img_feat = self.resnet_backbone(image)
        B, C, H, W = img_feat.shape
        # reshape to [B, 49, 2048]（7*7=49）
        img_feat = img_feat.view(B, C, H * W).permute(0, 2, 1)
        # project to image_token_dim, shape: [B, 49, 768]
        image_tokens = self.image_proj(img_feat)  # [B, 49, 768]

        # ----- text branch -----
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # get the last hidden state of BERT, shape: [B, seq_len, 768]
        text_tokens = text_outputs.last_hidden_state

        # get the [CLS] token embedding
        cls_text = text_tokens[:, 0:1, :]  # 形状 [B, 1, 768]

        # ----- fusing layer -----
        # cross-attention between [CLS] token and image tokens
        # cross_attended shape: [B, 1, 768]
        cross_attended, _ = self.cross_attn(query=cls_text, key=image_tokens, value=image_tokens)

        # concatenate [CLS] token and cross-attended image tokens
        fused_feature = torch.cat([cls_text, cross_attended], dim=-1)  # [B, 1, 768+768]
        fused_feature = self.fusion_layer(fused_feature)  # [B, 1, 768]
        fused_feature = fused_feature.squeeze(1)  # [B, 768]

        # ----- prediction -----
        logits = self.classifier(fused_feature)  # [B, num_classes]
        return logits

if __name__ == "__main__":
    num_classes = 50
    model = MultimodalLongTailClassifier(num_classes=num_classes)

    dummy_image = torch.randn(2, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 30522, (2, 16))
    dummy_attention_mask = torch.ones(2, 16, dtype=torch.long)

    logits = model(dummy_image, dummy_input_ids, dummy_attention_mask)
    print("Logits shape:", logits.shape)
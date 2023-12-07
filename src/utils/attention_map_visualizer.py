from einops import rearrange, repeat
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from PIL import Image

# Source: https://github.com/yiyixuxu/TimeSformer-rolled-attention/blob/main/visualizing_space_time_attention.ipynb


def combine_divided_attention(attn_t, attn_s):
    # time attention
    # average time attention weights across heads
    attn_t = attn_t.mean(dim=1)
    # add cls_token to attn_t as an identity matrix since it only attends to itself
    I = torch.eye(attn_t.size(-1)).unsqueeze(0)
    attn_t = torch.cat([I, attn_t], 0)
    # adding identity matrix to account for skipped connection
    attn_t = attn_t + torch.eye(attn_t.size(-1))[None, ...]
    # renormalize
    attn_t = attn_t / attn_t.sum(-1)[..., None]

    # space attention
    # average across heads
    attn_s = attn_s.mean(dim=1)
    # adding residual and renormalize
    attn_s = attn_s + torch.eye(attn_s.size(-1))[None, ...]
    attn_s = attn_s / attn_s.sum(-1)[..., None]

    # combine the space and time attention
    attn_ts = torch.tensor(np.einsum('tpk, ktq -> ptkq', attn_s, attn_t))

    # average the cls_token attention across the frames
    # splice out the attention for cls_token
    attn_cls = attn_ts[0, :, :, :]
    # average the cls_token attention and repeat across the frames
    attn_cls_a = torch.mean(attn_cls, axis=0)
    attn_cls_a = repeat(attn_cls_a, 'p t -> j p t', j=8)
    # add it back
    attn_ts = torch.cat([attn_cls_a.unsqueeze(0), attn_ts[1:, :, :, :]], 0)
    return (attn_ts)


class DividedAttentionRollout():
    def __init__(self, model, **kwargs):
        self.model = model
        self.hooks = []

    def get_attn_t(self, module, input, output):
        self.time_attentions.append(output.detach().cpu())

    def get_attn_s(self, module, input, output):
        self.space_attentions.append(output.detach().cpu())

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def __call__(self, input_tensor):
        # input_tensor = create_video_input(path_to_video)
        self.model.zero_grad()
        self.time_attentions = []
        self.space_attentions = []
        self.attentions = []
        for name, m in self.model.named_modules():
            if 'temporal_attention.attention.attn_drop' in name:
                self.hooks.append(m.register_forward_hook(self.get_attn_t))
            elif 'attention.attention.attn_drop' in name:
                self.hooks.append(m.register_forward_hook(self.get_attn_s))
        preds = self.model(input_tensor)
        for h in self.hooks:
            h.remove()
        self.attentions.extend(
            combine_divided_attention(attn_t, attn_s)
            for attn_t, attn_s in zip(self.time_attentions, self.space_attentions)
        )
        p, t = self.attentions[0].shape[0], self.attentions[0].shape[1]
        result = torch.eye(p*t)
        for attention in self.attentions:
            attention = rearrange(attention, 'p1 t1 p2 t2 -> (p1 t1) (p2 t2)')
            result = torch.matmul(attention, result)
        mask = rearrange(result, '(p1 t1) (p2 t2) -> p1 t1 p2 t2', p1=p, p2=p)
        mask = mask.mean(dim=1)
        mask = mask[0, 1:, :]
        width = int(mask.size(0)**0.5)
        mask = rearrange(mask, '(h w) t -> h w t', w=width).numpy()
        mask = mask / np.max(mask)
        return (mask)


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def create_masks(masks_in, np_imgs):
    masks = []
    for mask, img in zip(masks_in, np_imgs):
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask = show_mask_on_image(img, mask)
        masks.append(mask)
    return (masks)


transform_plot = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    lambda x: rearrange(x*255, 'c h w -> h w c').numpy().astype('uint8')
])


def plot_attention(model, video):
    att_roll = DividedAttentionRollout(model)
    masks = att_roll(video.unsqueeze(0))
    np_imgs = [transform_plot(p) for p in video]
    masks = create_masks(list(rearrange(masks, 'h w t -> t h w')), np_imgs)
    plt.subplots(2, 8, figsize=(24, 6))
    for num, x in enumerate(np_imgs):
        img = Image.fromarray(x)
        plt.subplot(2, 8, num+1)
        plt.title(str(num+1))
        plt.axis('off')
        plt.imshow(img)
    for num, x in enumerate(masks):
        img = Image.fromarray(x)
        plt.subplot(2, 8, num+9)
        plt.title(str(num+1))
        plt.axis('off')
        plt.imshow(img)

from segment_anything import sam_model_registry
import torch
from torch import nn
import torch.nn.functional as F
import os

class SAM_pred(nn.Module):
    def __init__(self, checkpoint_path='./checkpoint/sam_vit_h_4b8939.pth', model_type='vit_h'):
        super().__init__()
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please download it from https://github.com/facebookresearch/segment-anything#model-checkpoints")
        self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam_model.eval()
        self.freeze_sam()

    def freeze_sam(self):
        image_encoder_layers = list(self.sam_model.image_encoder.children())
        for layer in image_encoder_layers[:-1]:
            for param in layer.parameters():
                param.requires_grad = False
        for param in image_encoder_layers[-1].parameters():
            param.requires_grad = True

        prompt_encoder_layers = list(self.sam_model.prompt_encoder.children())
        for layer in prompt_encoder_layers[:-1]:
            for param in layer.parameters():
                param.requires_grad = False
        for param in prompt_encoder_layers[-1].parameters():
            param.requires_grad = True

        mask_decoder_layers = list(self.sam_model.mask_decoder.children())
        for layer in mask_decoder_layers[:-1]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in mask_decoder_layers[-1].parameters():
            param.requires_grad = True

        for name, param in self.sam_model.named_parameters():
            if 'pos_embed' in name: 
                param.requires_grad = False


    def forward_img_encoder(self, query_img):
        query_img = F.interpolate(query_img, (1024,1024), mode='bilinear', align_corners=True)
        query_feats = self.sam_model.image_encoder(query_img)
        return query_feats

    def print_grad_status(self):
        print("Gradient status of each layer in image_encoder:")
        for name,param in self.sam_model.image_encoder.state_dict().items():
            print(name,param.requires_grad)

    def get_prompt(self, protos, points_mask=None):
        if points_mask is not None :
            point_mask = points_mask

            postivate_pos = (point_mask.squeeze(0).nonzero().unsqueeze(0) + 0.5) * 64 -0.5
            postivate_pos = postivate_pos[:,:,[1,0]]
            point_label = torch.ones(postivate_pos.shape[0], postivate_pos.shape[1]).to(postivate_pos.device)
            point_prompt = (postivate_pos, point_label)
        else:
            point_prompt = None
        protos = protos
        return  protos, point_prompt

    def forward_prompt_encoder(self, points=None, boxes=None, protos=None, masks=None):
        q_sparse_em, q_dense_em = self.sam_model.prompt_encoder(
                points=points,
                boxes=None,
                protos=protos,
                masks=None)
        return  q_sparse_em, q_dense_em
    
    def forward_mask_decoder(self, query_feats, q_sparse_em, q_dense_em, ori_size=(512,512)):
        low_res_masks, iou_predictions, attened_embedding, upscaled_embedding = self.sam_model.mask_decoder(
                image_embeddings=query_feats,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=q_sparse_em,
                dense_prompt_embeddings=q_dense_em,
                multimask_output=False)
        low_masks = F.interpolate(low_res_masks, size=ori_size, mode='bilinear', align_corners=True)
        
        binary_mask = torch.where(low_masks > 0, 1, 0)
        return low_masks, binary_mask, attened_embedding, upscaled_embedding

    def forward(self, query_feats, q_sparse_em, q_dense_em, ori_size):
            
        low_masks, binary_mask, attened_embedding, upscaled_embedding = self.forward_mask_decoder(query_feats, q_sparse_em, q_dense_em, ori_size)

        return low_masks, binary_mask.squeeze(1), attened_embedding, upscaled_embedding
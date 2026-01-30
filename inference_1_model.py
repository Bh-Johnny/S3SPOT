import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from S3SPOT.model.Reweight_adapter import Reweight_adapter
from data.LabelDatasetwithGT import PairedImageDataset
from SAM2pred import SAM_pred

IMG_SIZE = 1024
ENC_FM_SIZE = 64
DEC_FM_SIZE = 64
PATCH_SIZE = IMG_SIZE // DEC_FM_SIZE  # 16

def greedy_assignment(score_matrix):

    rows, cols = score_matrix.shape
    flat_indices = np.argsort(score_matrix, axis=None)[::-1]
    
    used_rows = set()
    used_cols = set()
    
    row_ind = []
    col_ind = []
    max_matches = min(rows, cols)
    
    for idx in flat_indices:
        r, c = np.unravel_index(idx, (rows, cols))
        
        if r not in used_rows and c not in used_cols:
            row_ind.append(r)
            col_ind.append(c)
            used_rows.add(r)
            used_cols.add(c)
            
            if len(row_ind) >= max_matches:
                break
                
    return np.array(row_ind), np.array(col_ind)

def inference(args):
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load Models
    print("Loading SAM model...")
    sam_model = SAM_pred()
    sam_model.eval()
    sam_model.to(device)

    print(f"Loading Adapter from {args.checkpoint}...")
    model = Reweight_adapter(args)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)

    # 3. Dataset & DataLoader
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    dataset = PairedImageDataset(root_dir=args.datapath, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    assert dataloader.batch_size == 1, "Inference must use batch_size=1 to handle variable point counts correctly."

    # 4. Output Directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting Inference...")
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader)):
            
            # Unpack Data
            target_img = data[0].to(device)
            deid_img = data[1].to(device)
            # child_img = data[2] 
            parsing_mask = data[3] # [1, H, W]
            folder_name = data[4][0] # Tuple inside list, take first element
            
            save_name = f"{batch_idx:04d}_{folder_name}.png"

            B, C, h, w = target_img.shape # B should be 1

            # Step 1: Image Embeddings (Frozen SAM)
            img_embeddings = sam_model.forward_img_encoder(target_img)
            deid_img_embeddings = sam_model.forward_img_encoder(deid_img)

            # Step 2: High Points (Coarse location)
            parsing_mask_downsampled = (parsing_mask[:, 8::16, 8::16]).float().to(device)  # [1, 64, 64]
            sim = F.cosine_similarity(img_embeddings, deid_img_embeddings, dim=1).squeeze()
            sim = sim * parsing_mask_downsampled

            max_values, max_indices = torch.max(sim.view(B, -1), dim=1)
            ys = max_indices // sim.size(2) # 注意这里 sim.size(2) 应该是 64
            xs = max_indices % sim.size(2)

            input_points = torch.stack([xs, ys], dim=1)  # [B, 2]
            input_points = input_points * IMG_SIZE / ENC_FM_SIZE
            input_points = input_points.unsqueeze(1)  # [B, 1, 2]
            input_labels = torch.ones(B, 1).to(device)
            high_points = (input_points, input_labels)

            # Step 3: Prompt Encoder & Decoder (First Pass)
            hp_sparse_em, hp_dense_em = sam_model.forward_prompt_encoder(
                points=high_points, boxes=None, protos=None, masks=None
            )
            
            _, _, attened_img_embedding, _ = sam_model(img_embeddings, hp_sparse_em, hp_dense_em, (h, w))
            _, _, attened_deid_embedding, _ = sam_model(deid_img_embeddings, hp_sparse_em, hp_dense_em, (h, w))

            # Step 4: Greedy Matching 
            # Normalize features
            img_norms = torch.norm(attened_img_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            deid_norms = torch.norm(attened_deid_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            
            normalized_att_img_feats = attened_img_embedding / img_norms
            normalized_att_deid_feats = attened_deid_embedding / deid_norms
            
            # Reshape: [B, C, H, W] -> [B, N, C]
            att_img_feats = normalized_att_img_feats.permute(0, 2, 3, 1).reshape(B, -1, C)
            att_deid_feats = normalized_att_deid_feats.permute(0, 2, 3, 1).reshape(B, -1, C)

            # Similarity Matrix [B, N, N]
            S = torch.bmm(att_deid_feats, att_img_feats.transpose(1, 2))

            S_b = S[0].detach().cpu().numpy()
            parsing_mask_b_flat = parsing_mask_downsampled[0].reshape(-1).bool().cpu().numpy()
            
            S_forward_b = S_b[parsing_mask_b_flat]

            row_ind, col_ind = greedy_assignment(S_forward_b)
            
            sim_scores_f = S_forward_b[row_ind, col_ind]
            
            indices_forward = (torch.tensor(row_ind, dtype=torch.int64, device=device),
                               torch.tensor(col_ind, dtype=torch.int64, device=device))
            sim_scores_f_tensor = torch.tensor(sim_scores_f, device=device)

            reduced_points_num = len(sim_scores_f_tensor)
            sim_sorted, sim_idx_sorted = torch.sort(sim_scores_f_tensor, descending=True)
            sim_filter = sim_idx_sorted[:reduced_points_num]
            
            points_matched_inds = indices_forward[1][sim_filter]
            points_matched_inds_set = list(set(points_matched_inds.cpu().tolist()))
            
            # Index -> Pixel
            face_points_temp = []
            scale = IMG_SIZE / DEC_FM_SIZE
            offset = scale // 2
            
            for idx in points_matched_inds_set:
                w_idx = idx % DEC_FM_SIZE
                h_idx = idx // DEC_FM_SIZE
                px = w_idx * scale + offset
                py = h_idx * scale + offset
                if px < IMG_SIZE and py < IMG_SIZE:
                    face_points_temp.append([float(px), float(py)])
            
            face_point = np.array(face_points_temp)

            # Occlusion Points (Grid - Face)
            all_centers = []
            num_patches = int(IMG_SIZE // PATCH_SIZE)
            for r in range(num_patches):
                for c in range(num_patches):
                    cx = c * PATCH_SIZE + PATCH_SIZE // 2
                    cy = r * PATCH_SIZE + PATCH_SIZE // 2
                    all_centers.append((int(cx), int(cy)))

            if len(face_point) > 0:
                face_point_int_set = set((int(p[0]), int(p[1])) for p in face_point)
            else:
                face_point_int_set = set()
                
            occlusion_point_set = set(all_centers) - face_point_int_set
            occlusion_point = np.array(list(occlusion_point_set))

            #  (Check against parsing mask)
            if len(face_point) > 0:
                face_indices = torch.from_numpy(face_point).long()
                face_indices[:, 0] = face_indices[:, 0].clamp(0, IMG_SIZE-1)
                face_indices[:, 1] = face_indices[:, 1].clamp(0, IMG_SIZE-1)
                valid_face = parsing_mask[0, face_indices[:, 1], face_indices[:, 0]].bool().cpu()
                filtered_face = face_indices[valid_face].to(device)
            else:
                filtered_face = torch.empty((0, 2), device=device, dtype=torch.long)

            if len(occlusion_point) > 0:
                occ_indices = torch.from_numpy(occlusion_point).long()
                occ_indices[:, 0] = occ_indices[:, 0].clamp(0, IMG_SIZE-1)
                occ_indices[:, 1] = occ_indices[:, 1].clamp(0, IMG_SIZE-1)
                valid_occ = parsing_mask[0, occ_indices[:, 1], occ_indices[:, 0]].bool().cpu()
                filtered_occ = occ_indices[valid_occ].to(device)
            else:
                filtered_occ = torch.empty((0, 2), device=device, dtype=torch.long)

            if len(filtered_face) == 0 and len(filtered_occ) == 0:
                print(f"Warning: No points found for {save_name}, skipping...")
                continue

            curr_points = torch.cat((filtered_face, filtered_occ), dim=0).unsqueeze(0) # [1, L, 2]
            curr_labels = torch.cat((
                torch.ones(len(filtered_face), dtype=torch.int64, device=device),
                torch.zeros(len(filtered_occ), dtype=torch.int64, device=device)
            ), dim=0).unsqueeze(0) # [1, L]

            # Step 5: Prompt Encoder (Second Pass with refined points)
            q_sparse_em, q_dense_em = sam_model.forward_prompt_encoder(
                points=(curr_points, curr_labels),
                boxes=None, protos=None, masks=None
            )

            # Step 6: Adapter Forward
            adjusted_prompt_em, attn_weights = model(attened_img_embedding, q_sparse_em, attn_mask=None)

            # Step 7: SAM Decoder (Final Prediction)
            low_masks, _, _, _ = sam_model(img_embeddings, adjusted_prompt_em, q_dense_em, (h, w))
            
            logit_mask = low_masks.squeeze() # [H, W]

            # Sigmoid -> Threshold -> 0/255
            pred_mask = (torch.sigmoid(logit_mask) > 0.5).cpu().numpy().astype(np.uint8) * 255
            
            save_path = os.path.join(args.output_dir, save_name)
            Image.fromarray(pred_mask).save(save_path)
            
            if args.save_vis:
                orig_img = target_img[0].permute(1, 2, 0).cpu().numpy()
                orig_img = (orig_img * 255).astype(np.uint8)
                vis_img = orig_img.copy()
                vis_img[pred_mask > 0, 0] = 255 # R channel
                
                vis_path = os.path.join(args.output_dir, f"vis_{save_name}")
                Image.fromarray(vis_img).save(vis_path)

    print("Inference Finished!")

def main():
    parser = argparse.ArgumentParser(description='Inference Script')
    parser.add_argument('--datapath', type=str, default='./dataset/Label_data/Face_1013_pair_train+val/val', help='Path to validation/test data')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save masks')
    parser.add_argument('--save_vis', action='store_true', help='Save visualization (overlay) along with masks')
    
    parser.add_argument('--bsz', type=int, default=1) # Inference fixed to 1
    # parser.add_argument('--num_heads', type=int, default=...) 
    
    args = parser.parse_args()
    
    inference(args)

if __name__ == '__main__':
    main()
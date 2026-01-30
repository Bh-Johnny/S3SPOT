import os
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from S3SPOT.model.Reweight_adapter import Reweight_adapter
from common.simple_logger import SimpleLogger
from data.LabelDatasetwithGT import PairedImageDataset
from SAM2pred import SAM_pred

# --- Constants ---
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

def run_one_epoch(args, device, model, sam_model, dataloader, epoch, mode, optimizer=None, scheduler=None):
    if mode == 'train':
        model.train()
    else:
        model.eval()
        
    total_loss = 0
    
    if args.local_rank == 0:
        data_iter = tqdm(dataloader, desc=f"{mode.capitalize()} Epoch {epoch}", ncols=100)
    else:
        data_iter = dataloader

    context = torch.enable_grad() if mode == 'train' else torch.no_grad()
    
    with context:
        num_batches = 0
        for batch_idx, data in enumerate(data_iter):
            
            target_img = data[0].to(device)
            deid_img = data[1].to(device)
            # child_img = data[2] # Unused
            parsing_mask = data[3]
            folder_name = data[4]

            B, C, h, w = target_img.shape

            # 1. Image Embeddings (Frozen SAM)
            with torch.no_grad():
                img_embeddings = sam_model.module.forward_img_encoder(target_img)
                deid_img_embeddings = sam_model.module.forward_img_encoder(deid_img)

            # 2. High Points (Coarse location)
            parsing_mask_downsampled = (parsing_mask[:, 8::16, 8::16]).float().to(device)  # [B, 64, 64]
            sim = F.cosine_similarity(img_embeddings, deid_img_embeddings, dim=1).squeeze()
            sim = sim * parsing_mask_downsampled

            max_values, max_indices = torch.max(sim.view(B, -1), dim=1)
            ys = max_indices // sim.size(2)
            xs = max_indices % sim.size(2)

            input_points = torch.stack([xs, ys], dim=1)  # [B, 2]
            input_points = input_points * IMG_SIZE / ENC_FM_SIZE
            input_points = input_points.unsqueeze(1)  # [B, 1, 2]
            input_labels = torch.ones(B, 1).to(device)
            high_points = (input_points, input_labels)

            # 3. Prompt Encoder & Decoder
            hp_sparse_em, hp_dense_em = sam_model.module.forward_prompt_encoder(
                points=high_points, boxes=None, protos=None, masks=None
            )
            
            _, _, attened_img_embedding, _ = sam_model.module(img_embeddings, hp_sparse_em, hp_dense_em, (h, w))
            _, _, attened_deid_embedding, _ = sam_model.module(deid_img_embeddings, hp_sparse_em, hp_dense_em, (h, w))

            # 4. Greedy
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

            input_points_list = []
            input_labels_list = []
            face_point_list = []     # For Loss calculation
            occlusion_point_list = [] # For Loss calculation

            # --- Loop over batch items to perform Greedy Matching ---
            for b in range(B):
                S_b = S[b].detach().cpu().numpy()
                parsing_mask_b_flat = parsing_mask_downsampled[b].reshape(-1).bool().cpu().numpy()
                
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
                
                #  (Index -> Pixel)
                points_matched_inds_set = list(set(points_matched_inds.cpu().tolist()))
                
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

                if len(face_point) > 0:
                    face_indices = torch.from_numpy(face_point).long()
                    # Clamp to avoid index out of bounds
                    face_indices[:, 0] = face_indices[:, 0].clamp(0, IMG_SIZE-1)
                    face_indices[:, 1] = face_indices[:, 1].clamp(0, IMG_SIZE-1)
                    
                    valid_face = parsing_mask[b, face_indices[:, 1], face_indices[:, 0]].bool().cpu() # Check on CPU or Device
                    filtered_face = face_indices[valid_face].to(device)
                else:
                    filtered_face = torch.empty((0, 2), device=device, dtype=torch.long)

                if len(occlusion_point) > 0:
                    occ_indices = torch.from_numpy(occlusion_point).long()
                    occ_indices[:, 0] = occ_indices[:, 0].clamp(0, IMG_SIZE-1)
                    occ_indices[:, 1] = occ_indices[:, 1].clamp(0, IMG_SIZE-1)
                    
                    valid_occ = parsing_mask[b, occ_indices[:, 1], occ_indices[:, 0]].bool().cpu()
                    filtered_occ = occ_indices[valid_occ].to(device)
                else:
                    filtered_occ = torch.empty((0, 2), device=device, dtype=torch.long)

                face_point_list.append(filtered_face)
                occlusion_point_list.append(filtered_occ)

                input_points = torch.cat((filtered_face, filtered_occ), dim=0)
                input_labels = torch.cat((
                    torch.ones(len(filtered_face), dtype=torch.int64, device=device),
                    torch.zeros(len(filtered_occ), dtype=torch.int64, device=device)
                ), dim=0)

                input_points_list.append(input_points)
                input_labels_list.append(input_labels)

            # --- Calculate Loss for the Batch ---
            loss_B = 0
            
            for b in range(B):
                if len(input_points_list[b]) == 0:
                    continue

                # 5. Prompt Encoder (Second Pass with refined points)
                curr_points = input_points_list[b].unsqueeze(0) # [1, L, 2]
                curr_labels = input_labels_list[b].unsqueeze(0) # [1, L]
                
                q_sparse_em, q_dense_em = sam_model.module.forward_prompt_encoder(
                    points=(curr_points, curr_labels),
                    boxes=None, protos=None, masks=None
                )

                # 6. Adapter Forward
                attened_img_embedding_b = attened_img_embedding[b:b+1]
                adjusted_prompt_em, attn_weights = model.module(attened_img_embedding_b, q_sparse_em, attn_mask=None)

                # 7. SAM Decoder (Final Prediction)
                img_embeddings_b = img_embeddings[b:b+1]
                low_masks, _, _, _ = sam_model.module(img_embeddings_b, adjusted_prompt_em, q_dense_em, (h, w))
                
                logit_mask = low_masks.squeeze(0) # [1, H, W] -> [H, W] (assuming 1 mask output)
                if logit_mask.dim() == 3: logit_mask = logit_mask.squeeze(0)

                # 8. Loss Calculation 
                pred_mask_binary_soft = torch.sigmoid(10 * logit_mask) # Soft binary for gradients

                face_indices = face_point_list[b]
                occ_indices = occlusion_point_list[b]

                # Recall Loss (Face)
                if len(face_indices) > 0:
                    face_probs = pred_mask_binary_soft[face_indices[:, 1], face_indices[:, 0]]
                    loss_recall = -face_probs.mean()
                    face_punish = F.relu(0.5 - face_probs).mean()
                else:
                    loss_recall = torch.tensor(0.0, device=device)
                    face_punish = torch.tensor(0.0, device=device)

                # Recall Loss (Occlusion)
                if len(occ_indices) > 0:
                    occ_probs = pred_mask_binary_soft[occ_indices[:, 1], occ_indices[:, 0]]
                    loss_recall_occ = occ_probs.mean()
                else:
                    loss_recall_occ = torch.tensor(0.0, device=device)

                loss_b = loss_recall + loss_recall_occ + 0.5 * face_punish
                
                loss_B += loss_b

            # Normalize loss by batch size if needed 
            loss_B = loss_B / B

            # Optimization Step
            if mode == 'train':
                optimizer.zero_grad()
                loss_B.backward()
                optimizer.step()
                
            total_loss += loss_B.item()
            num_batches += 1
            
            if mode == 'train' and batch_idx % args.log_interval == 0 and args.local_rank == 0:
                print(f"[Epoch {epoch}] Batch {batch_idx} Loss: {loss_B.item():.4f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        if args.local_rank == 0:
            print(f"Epoch {epoch} Average {mode.capitalize()} Loss: {avg_loss}")

        if mode == 'train' and scheduler is not None:
            scheduler.step()
            
        return avg_loss

def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.manual_seed(321)

def cleanup():
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='General Model Training with DDP')
    parser.add_argument('--datapath', type=str, default='./dataset/Label_data/Face_1013_pair_train+val/')
    parser.add_argument('--bsz', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--local_rank', type=int, default=-1)
    # Add other args as needed
    parser.add_argument('--patience', type=int, default=5) 
    parser.add_argument('--min_delta', type=float, default=1e-4)
    args = parser.parse_args()

    rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    setup(rank, world_size)

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        log_dir = './logs/train_general_logs'
        os.makedirs(log_dir, exist_ok=True)
        logger = SimpleLogger(log_dir)
    else:
        logger = None

    # Load SAM
    sam_model = SAM_pred()
    sam_model.eval()
    sam_model.to(device)
    sam_model = DDP(sam_model, device_ids=[rank])

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = PairedImageDataset(root_dir=os.path.join(args.datapath, 'train'), transform=transform)
    val_dataset = PairedImageDataset(root_dir=os.path.join(args.datapath, 'val'), transform=transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=False, num_workers=args.nworker, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.bsz, shuffle=False, num_workers=args.nworker, sampler=val_sampler)

    # Model
    model = Reweight_adapter(args)
    model.to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(model.module.transformer_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    dist.barrier()

    for epoch in range(1, args.epochs + 1):
        if rank == 0:
            print(f"\nEpoch {epoch}/{args.epochs}")

        train_sampler.set_epoch(epoch)

        train_loss = run_one_epoch(args, device, model, sam_model, train_loader, epoch, 'train', optimizer, scheduler)
        val_loss = run_one_epoch(args, device, model, sam_model, val_loader, epoch, 'val')
        
        if rank == 0:
            logger.update(epoch, train_loss, val_loss)
            
            model_save_dir = './checkpoint/train_general'
            os.makedirs(model_save_dir, exist_ok=True)
            model_save_path = os.path.join(model_save_dir, f'model_epoch_{epoch}.pth')
            torch.save(model.module.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    if rank == 0:
        logger.info('==================== Finished Training ====================')

    cleanup()

if __name__ == '__main__':
    main()
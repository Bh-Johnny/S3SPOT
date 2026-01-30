import os
import argparse
import traceback
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# Local imports
from S3SPOT.model.Reweight_adapter import Reweight_adapter
from common.simple_logger import SimpleLogger
from data.LabelDataset import PairedImageDataset
from SAM2pred import SAM_pred

# --- Constants ---
IMG_SIZE = 1024
ENC_FM_SIZE = 64  # Encoder Feature Map Size
DEC_FM_SIZE = 64  # Decoder Feature Map Size
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

def save_visualization(epoch, mask, folderpath, input_points, attn_weights, folder_name, prefix=''):
    full_path = os.path.join(folderpath, folder_name)
    os.makedirs(full_path, exist_ok=True)

    # 1. Save Mask Image
    mask_binary = np.where(mask > 0.5, 1, 0)
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask_binary.shape[-2:]
    # Broadcast color to mask shape
    mask_overlay = mask_binary.reshape(h, w, 1) * color.reshape(1, 1, -1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 1: Mask
    ax1.imshow(mask_overlay)
    ax1.set_title(f'Mask Epoch {epoch}')
    ax1.axis('off')

    # Plot 2: Points with Attention Weights
    points = input_points[0].cpu().numpy()
    weights = attn_weights.cpu().numpy()
    
    # Handle CLS token or mismatch
    if len(weights) > len(points):
        weights = weights[:len(points)]
    elif len(weights) < len(points):
        points = points[:len(weights)]
    
    sizes = weights * 50000  # Scale factor for visualization
    scatter = ax2.scatter(points[:, 0], points[:, 1], s=sizes, c=weights, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ax=ax2, label='Attention Weight')
    ax2.set_xlim(0, w)
    ax2.set_ylim(0, h)
    ax2.invert_yaxis()
    ax2.set_title('Points with Weights')

    plt.tight_layout()
    plt.savefig(os.path.join(full_path, f'{prefix}combined_{epoch}.png'))
    plt.close(fig) # Important: Close figure to free memory

    # Save raw mask
    mask_pil = Image.fromarray((mask_binary.squeeze() * 255).astype(np.uint8))
    mask_pil.save(os.path.join(full_path, f'{prefix}mask_{epoch}.png'))

def get_initial_high_point(img_embeddings, ref_img_embeddings, parsing_mask, device):
    # Downsample mask to feature map size (1024 -> 64)
    # 8::16 means start at 8, step 16 (center of the patch)
    parsing_mask_downsampled = (parsing_mask[8::16, 8::16]).float().to(device) # 64x64

    sim = F.cosine_similarity(img_embeddings, ref_img_embeddings, dim=1).squeeze()
    sim = sim * parsing_mask_downsampled
    
    max_index = torch.argmax(sim)
    y, x = divmod(max_index.item(), sim.size(1))
    
    # Scale back to image size
    input_points = torch.tensor([[x, y]]).unsqueeze(0).to(device)
    input_points = input_points * IMG_SIZE / ENC_FM_SIZE
    input_labels = torch.tensor([1]).unsqueeze(0).to(device)
    
    return input_points, input_labels

def generate_matching_points(sam_model, img_embeddings, ref_img_embeddings, high_points, parsing_mask, device, h, w):
    # 1. Prompt Encoder Forward
    hp_sparse_em, hp_dense_em = sam_model.forward_prompt_encoder(
        points=high_points, boxes=None, protos=None, masks=None
    )
    
    # 2. SAM Decoder Forward (to get attended embeddings)
    _, _, attened_img_embedding, _ = sam_model(img_embeddings, hp_sparse_em, hp_dense_em, (h, w))
    _, _, attened_ref_embedding, _ = sam_model(ref_img_embeddings, hp_sparse_em, hp_dense_em, (h, w))

    parsing_mask_downsampled = (parsing_mask[8::16, 8::16]).float().to(device)

    # 3. Calculate Similarity Matrix
    # Flatten features: (C, H, W) -> (N, C)
    att_img_feats = F.normalize(attened_img_embedding, p=2, dim=1).squeeze().permute(1, 2, 0).reshape(DEC_FM_SIZE*DEC_FM_SIZE, -1)
    att_ref_feats = F.normalize(attened_ref_embedding, p=2, dim=1).squeeze().permute(1, 2, 0).reshape(DEC_FM_SIZE*DEC_FM_SIZE, -1)

    S = att_ref_feats @ att_img_feats.t()
    
    # 4. Filter by Mask and Greedy Assignment
    mask_bool_flat = parsing_mask_downsampled.reshape(-1).bool()
    S_forward = S[mask_bool_flat] # Only consider points inside the mask

    indices_forward = greedy_assignment(S_forward.detach().cpu().numpy())
    indices_forward = [torch.as_tensor(idx, dtype=torch.int64, device=device) for idx in indices_forward]
    
    # 5. Filter Top Matches
    sim_scores_f = S_forward[indices_forward[0], indices_forward[1]]
    reduced_points_num = len(sim_scores_f) # Use all valid matches
    sim_sorted, sim_idx_sorted = torch.sort(sim_scores_f, descending=True)
    sim_filter = sim_idx_sorted[:reduced_points_num]
    
    points_matched_inds = indices_forward[1][sim_filter]
    points_matched_inds_set = list(set(points_matched_inds.cpu().tolist()))
    
    # 6. Convert Indices to Coordinates
    face_points_list = []
    scale = IMG_SIZE / DEC_FM_SIZE
    offset = scale // 2
    
    for idx in points_matched_inds_set:
        w_idx = idx % DEC_FM_SIZE
        h_idx = idx // DEC_FM_SIZE
        
        px = w_idx * scale + offset
        py = h_idx * scale + offset
        
        if px < IMG_SIZE and py < IMG_SIZE:
            face_points_list.append([float(px), float(py)])
            
    face_point = np.array(face_points_list)

    # 7. Generate Occlusion Points (Grid points NOT in Face Points)
    # [FIX] Use integer arithmetic for robust set subtraction
    all_centers = []
    num_patches = int(IMG_SIZE // PATCH_SIZE)
    for r in range(num_patches):
        for c in range(num_patches):
            cx = c * PATCH_SIZE + PATCH_SIZE // 2
            cy = r * PATCH_SIZE + PATCH_SIZE // 2
            all_centers.append((int(cx), int(cy)))
            
    if len(face_point) > 0:
        face_point_int_tuples = set((int(p[0]), int(p[1])) for p in face_point)
    else:
        face_point_int_tuples = set()

    occlusion_point_set = set(all_centers) - face_point_int_tuples
    occlusion_point = np.array(list(occlusion_point_set))

    # 8. Filter points by Parsing Mask again (Double check)
    if len(face_point) > 0:
        face_indices = torch.from_numpy(face_point).long()
        face_indices[:, 0] = face_indices[:, 0].clamp(0, IMG_SIZE-1)
        face_indices[:, 1] = face_indices[:, 1].clamp(0, IMG_SIZE-1)
        valid_face = parsing_mask[face_indices[:, 1], face_indices[:, 0]].bool()
        filtered_face = face_indices[valid_face]
    else:
        filtered_face = torch.empty((0, 2), device=device, dtype=torch.long)

    if len(occlusion_point) > 0:
        occ_indices = torch.from_numpy(occlusion_point).long()
        occ_indices[:, 0] = occ_indices[:, 0].clamp(0, IMG_SIZE-1)
        occ_indices[:, 1] = occ_indices[:, 1].clamp(0, IMG_SIZE-1)
        valid_occ = parsing_mask[occ_indices[:, 1], occ_indices[:, 0]].bool()
        filtered_occ = occ_indices[valid_occ]
    else:
        filtered_occ = torch.empty((0, 2), device=device, dtype=torch.long)
    
    # 9. Prepare Final Outputs
    final_points = torch.cat((filtered_face, filtered_occ), dim=0).unsqueeze(0).to(device)
    
    face_labels = torch.ones(len(filtered_face), dtype=torch.long, device=device)
    occ_labels = torch.zeros(len(filtered_occ), dtype=torch.long, device=device)
    final_labels = torch.cat((face_labels, occ_labels), dim=0).unsqueeze(0)
    
    return (final_points, final_labels), attened_img_embedding, filtered_face, filtered_occ

def train_on_single_image(args, device, model, sam_model, data, optimizer, scheduler, num_epochs, image_idx):
    model.train()
    
    target_img = data[0].to(device)
    ref_img = data[1].to(device)
    parsing_mask = data[2].squeeze(0).to(device) # 1024x1024
    folder_name = data[3][0]
    
    B, C, h, w = target_img.shape
    
    # Pre-calculate embeddings (Frozen SAM)
    with torch.no_grad():
        img_embeddings = sam_model.forward_img_encoder(target_img)
        ref_img_embeddings = sam_model.forward_img_encoder(ref_img)
        
        high_points = get_initial_high_point(img_embeddings, ref_img_embeddings, parsing_mask, device)
        
        initial_points, attened_img_embedding, face_indices, occ_indices = generate_matching_points(
            sam_model, img_embeddings, ref_img_embeddings, high_points, parsing_mask, device, h, w
        )
        
        q_sparse_em, q_dense_em = sam_model.forward_prompt_encoder(
            points=initial_points, boxes=None, protos=None, masks=None
        )
    
    best_loss = float('inf')
    epochs_no_improve = 0
    
    # Initialize return values to avoid UnboundLocalError
    final_loss = 0.0
    final_mask_not_at_face = 0.0
    final_loss_recall_occ = 0.0

    for epoch in range(num_epochs):
        # Forward Pass
        adjusted_prompt_em, attn_weights = model(attened_img_embedding, q_sparse_em)
        
        low_masks, _, _, _ = sam_model(img_embeddings, adjusted_prompt_em, q_dense_em, (h, w))
        logit_mask = low_masks
        pred_mask_binary_soft = torch.sigmoid(10 * logit_mask).squeeze()
        
        # --- Loss Calculation ---
        # 1. Recall Loss (Face Points should be 1)
        if len(face_indices) > 0:
            face_probs = pred_mask_binary_soft[face_indices[:, 1], face_indices[:, 0]]
            loss_recall = -face_probs.mean()
            face_punish = F.relu(0.5 - face_probs).mean()
        else:
            loss_recall = torch.tensor(0.0, device=device)
            face_punish = torch.tensor(0.0, device=device)
        
        # 2. Recall Loss (Occlusion Points should be 0)
        if len(occ_indices) > 0:
            occ_probs = pred_mask_binary_soft[occ_indices[:, 1], occ_indices[:, 0]]
            loss_recall_occ = occ_probs.mean()
        else:
            loss_recall_occ = torch.tensor(0.0, device=device)
        
        loss = loss_recall + loss_recall_occ + 0.5 * face_punish
        
        # Metric for logging
        if len(face_indices) > 0:
            mask_not_at_face = (1 - torch.sigmoid(2 * (face_probs - 0.5))).sum()
        else:
            mask_not_at_face = torch.tensor(0.0)

        # Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update return values
        final_loss = loss.item()
        final_mask_not_at_face = mask_not_at_face.item()
        final_loss_recall_occ = loss_recall_occ.item()

        # Visualization
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Img {image_idx}, Ep {epoch}, Loss: {loss.item():.4f}, Recall: {-loss_recall.item():.4f}, Occ: {loss_recall_occ.item():.4f}")
            
            # [FIX] Handle attention weights slicing safely
            mean_attn_weights = attn_weights.squeeze(0)[0] 
            # Assuming the first token is a CLS token or similar that we skip
            vis_weights = mean_attn_weights[1:].detach().cpu() if len(mean_attn_weights) > 1 else mean_attn_weights.detach().cpu()

            # save_visualization(f'{epoch}', torch.sigmoid(logit_mask).detach().cpu().numpy(), 
                    #   os.path.join(args.vis_path, folder_name), 
                    #   initial_points[0], # Pass points only
                    #   vis_weights, 
                    #   str(folder_name))

        # Early Stopping
        if best_loss - loss.item() > args.min_delta:
            best_loss = loss.item()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return final_loss, final_mask_not_at_face, final_loss_recall_occ

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./dataset/Label_data/Face_1013_pair/')
    parser.add_argument('--vis_path', type=str, default='./datavis/Label_data/Nmodel2')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        device = torch.device('cpu')

    logger = SimpleLogger('./logs')

    # Load Frozen SAM
    sam_model = SAM_pred()
    sam_model.eval()
    sam_model.to(device)
    for param in sam_model.parameters():
        param.requires_grad = False

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = PairedImageDataset(root_dir=args.datapath, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    print("Start Processing...")
    
    for idx, data in enumerate(dataloader):
        print(f"\nProcessing image {idx+1}/{len(dataloader)}")

        # Initialize Adapter per image (One-Shot / TTA)
        model = Reweight_adapter(args)
        model.to(device)

        optimizer = optim.AdamW(model.transformer_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        try:
            loss, metric_a, metric_b = train_on_single_image(
                args, device, model, sam_model, data, optimizer, scheduler, args.epochs, idx
            )
            logger.update(idx, loss, 0, metric_a, metric_b)
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            traceback.print_exc()

    logger.info('==================== Finished Training ====================')
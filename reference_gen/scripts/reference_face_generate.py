import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from src.utils import torch_utils
from src.datasets.dataset import TO_TENSOR, NORMALIZE
from src.utils.alignmengt import crop_faces, calc_alignment_coefficients
from S3SPOT.reference_gen.src.options.gen_ref_options import GenerateReferencePipelineOptions
from src.pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo, vis_parsing_maps
from src.models.networks import Net3

@torch.no_grad()
def face_reconstruction_pipeline(source_path, opts, save_dir, need_crop=True, verbose=False):
    """
    Pipeline for Face Reconstruction using E4S (Net3).
    
    It passes the source image through the Encoder (RGI) to get style vectors,
    and then through the Generator to reconstruct the face.

    Args:
        source_path (str): Path to the source image.
        opts (Namespace): Configuration arguments.
        save_dir (str): Directory to save the results.
        need_crop (bool): Whether to detect and crop the face from the original image.
        verbose (bool): Whether to save intermediate visualization files.
    """
    filename = os.path.basename(source_path).split('.')[0]
    output_dir = os.path.join(save_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing: {filename}")
    
    if need_crop:
        # Define a list of tuples for the crop function: [(name, path)]
        source_file_list = [(filename, source_path)]
        
        # Crop faces to 1024x1024 (standard alignment size)
        crops, orig_images, quads, inv_transforms = crop_faces(
            image_size=1024, 
            image_paths=source_file_list, 
            scale=1.0, 
            use_fa=False
        )
        
        # We assume one face per image for this demo
        aligned_img = crops[0].convert("RGB")
        orig_image = orig_images[0]
        inv_trans_coeffs = calc_alignment_coefficients(
            quads[0] + 0.5, 
            [[0, 0], [0, 1024], [1024, 1024], [1024, 0]]
        )
    else:
        # If already cropped/aligned, just load and resize
        aligned_img = Image.open(source_path).convert("RGB").resize((1024, 1024))
        orig_image = aligned_img # No pasting back needed if not cropped from larger img

    # Generate the 12-class segmentation mask
    mask_np = faceParsing_demo(
        faceParsing_model, 
        aligned_img, 
        convert_to_seg12=True, 
        model_name=opts.faceParser_name
    )
    np.save(os.path.join(save_dir, f"{filename}_parsing_mask.npy"), mask_np)

    # Resize to 512x512 for the network (standard E4S input resolution)
    network_input_size = (512, 512)
    img_for_net = aligned_img.resize(network_input_size, Image.BILINEAR)
    mask_for_net = Image.fromarray(mask_np).resize(network_input_size, Image.NEAREST)

    # Convert Image to Tensor [1, 3, 512, 512]
    img_tensor = transforms.Compose([TO_TENSOR, NORMALIZE])(img_for_net)
    img_tensor = img_tensor.to(opts.device).float().unsqueeze(0)

    # Convert Mask to Tensor [1, 1, 512, 512]
    mask_tensor = transforms.Compose([TO_TENSOR])(mask_for_net)
    mask_tensor = (mask_tensor * 255).long().to(opts.device).unsqueeze(0)
    
    # Convert Mask to One-Hot encoding [1, 12, 512, 512]
    onehot_tensor = torch_utils.labelMap2OneHot(mask_tensor, num_cls=opts.num_seg_cls)

    # Extract Style Vectors (RGI - Region-Global Inversion)
    style_vectors, _ = net.get_style_vectors(img_tensor, onehot_tensor)
    
    if verbose:
        torch.save(style_vectors, os.path.join(save_dir, f"{filename}_style_vec.pt"))

    # Calculate Style Codes (Mapping Network)
    style_codes = net.cal_style_codes(style_vectors)

    # Generate Image (Decoder)
    # Reconstruct the face using the extracted codes and the original mask
    # constant_input is usually a learned constant or zeros, shape [1, 512, 32, 32]
    constant_input = torch.zeros(1, 512, 32, 32).to(opts.device)
    
    recon_face_tensor, _, _ = net.gen_img(constant_input, style_codes, onehot_tensor)

    # Convert output tensor back to PIL Image
    recon_img = torch_utils.tensor2im(recon_face_tensor[0])
    
    # Resize back to 1024 (alignment size) to match original quality if pasting
    recon_img = recon_img.resize((1024, 1024), Image.BILINEAR)
    
    save_path = os.path.join(save_dir, f"Reference_{filename}.png")

    if need_crop:
        # Paste the reconstructed face back into the original background
        recon_img = recon_img.convert('RGBA')
        pasted_image = orig_image.copy().convert('RGBA')
        
        # Use the inverse transformation matrix calculated during alignment
        projected = recon_img.transform(
            orig_image.size, 
            Image.PERSPECTIVE, 
            inv_trans_coeffs, 
            Image.BILINEAR
        )
        
        # Simple pasting (alpha composite). 
        # Note: For better results, you might want to use the soft blending logic 
        # (create_masks, smooth_face_boundry) from the swap pipeline here.
        mask_projected = projected.split()[3] # Get alpha channel
        pasted_image.paste(projected, (0, 0), mask=mask_projected)
        pasted_image.save(save_path)
    else:
        # Just save the aligned reconstruction
        recon_img.save(save_path)

    print(f"Reconstruction saved to: {save_path}")


if __name__ == "__main__":

    try:
        opts = GenerateReferencePipelineOptions().parse()
    except NameError:
        # Fallback if options class isn't available in this context
        from argparse import Namespace
        opts = Namespace(
            device='cuda',
            checkpoint_path='./reference_gen/pretrained_ckpts/e4s/iteration_300000.pt', # Update path
            faceParser_name='default',
            num_seg_cls=12,
            output_dir='./reference_gen/example/output'
        )

    # Initialize Face Parser
    if opts.faceParser_name == "default":
        faceParser_ckpt = "./reference_gen/pretrained_ckpts/face_parsing/79999_iter.pth"
        config_path = ""
    elif opts.faceParser_name == "segnext":
        faceParser_ckpt = "./reference_gen/pretrained_ckpts/face_parsing/segnext.small.best_mIoU_iter_140000.pth"
        config_path = "./reference_gen/pretrained_ckpts/face_parsing/segnext.small.512x512.celebamaskhq.160k.py"
    
    faceParsing_model = init_faceParsing_pretrained_model(opts.faceParser_name, faceParser_ckpt, config_path)

    # Initialize E4S Network (Net3)
    net = Net3(opts)
    net = net.to(opts.device)
    
    if os.path.exists(opts.checkpoint_path):
        save_dict = torch.load(opts.checkpoint_path)
        # Handle potential module prefix from DataParallel
        state_dict = torch_utils.remove_module_prefix(save_dict["state_dict"], prefix="module.")
        net.load_state_dict(state_dict)
        net.latent_avg = save_dict['latent_avg'].to(opts.device)
        print("E4S Model loaded successfully.")
    else:
        print(f"Warning: Checkpoint not found at {opts.checkpoint_path}")

    # Run Generation
    # Replace with your image path
    original_image_path = "./reference_gen/example/test_image.jpg" 
    
    if os.path.exists(original_image_path):
        face_reconstruction_pipeline(
            source_path=original_image_path,
            opts=opts,
            save_dir=opts.output_dir,
            need_crop=True, # Set to True if input is a full raw photo
            verbose=True
        )
    else:
        print("Please provide a valid image path.")
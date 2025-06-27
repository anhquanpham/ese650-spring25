from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import imageio
import cv2
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
dataset_root = '/content/drive/MyDrive/data/'

"""
Code implemented in Colab Notebook
"""

def load_colmap_data(dataset_root):
    r"""
    After using colmap2nerf.py to convert the colmap intrinsics and extrinsics,
    read in the transform_colmap.json file

    Expected Returns:
      An array of resized imgs, normalized to [0, 1]
      An array of poses, essentially the transform matrix
      Camera parameters: H, W, focal length

    NOTES:
      We recommend you resize the original images from 800x800 to lower resolution,
      i.e. 200x200 so it's easier for training. Change camera parameters accordingly
    """
    ################### YOUR CODE START ###################
    # Construct the path to the JSON file containing the transformation data.
    transforms_file_path = os.path.join(dataset_root, 'transforms_colmap.json')

    # Open and load the JSON file
    with open(transforms_file_path) as file:
        transform_data = json.load(file)

    # Define the original image dimensions and the new target dimensions for resizing.
    original_height, original_width = 800, 800
    resized_height, resized_width = 200, 200

    # Extract the camera's horizontal field of view (in radians) from the first frame.
    cam_angle_x = transform_data['frames'][0]['camera_angle_x']

    # Compute the original focal length using the pinhole camera model.
    focal_original = 0.5 * original_width / np.tan(0.5 * cam_angle_x)

    # Adjust the focal length for the resized image dimensions.
    focal_resized = focal_original * (resized_width / original_width)

    # Initialize lists for storing image tensors and camera poses.
    image_tensors = []
    camera_poses = []

    # Define the folder containing training images.
    train_images_folder = os.path.join(dataset_root, 'images', 'train')

    # Process each frame defined in the JSON file.
    for frame in transform_data['frames']:
        # Build the full image path. Notice that file_path is a list, so we take the first element.
        img_file_path = os.path.join(train_images_folder, frame['file_path'][0])

        # If the image file does not exist, output a warning and skip to the next frame.
        if not os.path.exists(img_file_path):
            print(f"Image {frame['file_path'][0]} not found")
            continue

        # Read the image using OpenCV.
        image = cv2.imread(img_file_path)
        if image is None:
            print(f"Image {frame['file_path'][0]} not found")
            continue

        # Convert image from BGR (OpenCV default) to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to the new dimensions.
        resized_image = cv2.resize(image, (resized_width, resized_height))

        # Convert image to a float tensor and normalize pixel values to [0, 1].
        image_tensor = torch.tensor(resized_image / 255.0, dtype=torch.float32)
        image_tensors.append(image_tensor)

        # Convert the transformation matrix into a float tensor and add to the poses list.
        camera_pose = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        camera_poses.append(camera_pose)

    # Stack individual image tensors and camera poses into single tensors.
    image_tensors = torch.stack(image_tensors)
    camera_poses = torch.stack(camera_poses)

    # Return the images, poses, and camera parameters.
    return image_tensors, camera_poses, [resized_height, resized_width, focal_resized]
    ################### YOUR CODE END ###################


def get_rays(image_height, image_width, focal_length, transform_w2c):
    r"""Compute rays passing through each pixels

    Expected Returns:
      ray_origins: A tensor of shape (H, W, 3) denoting the centers of each ray.
      ray_directions: A tensor of shape (H, W, 3) denoting the direction of each
        ray. ray_directions[i][j] denotes the direction (x, y, z) of the ray
        passing through the pixel at row index `i` and column index `j`.
    """
    ################### YOUR CODE START ###################
    # Set the computation device: CUDA if available, otherwise CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure transform matrix is a torch.Tensor on the correct device.
    if isinstance(transform_w2c, np.ndarray):
        transform_w2c = torch.from_numpy(transform_w2c).to(device)
    elif isinstance(transform_w2c, torch.Tensor):
        transform_w2c = transform_w2c.to(device)
    else:
        print("transform_w2c type error")

    # Ensure focal_length, image_width, image_height are scalar numbers.
    if not isinstance(focal_length, (int, float)):
        focal_length = focal_length.item()
    if not isinstance(image_width, (int, float)):
        image_width = image_width.item()
    if not isinstance(image_height, (int, float)):
        image_height = image_height.item()

    # Create a meshgrid of pixel coordinates.
    # Note: torch.meshgrid is used with linspace values spanning the width and height.
    grid_i, grid_j = torch.meshgrid(
        torch.linspace(0, image_width - 1, int(image_width), device=device),
        torch.linspace(0, image_height - 1, int(image_height), device=device)
    )

    # Transpose and flatten the grids to match the pixel coordinate order.
    # Compute the ray directions in the camera coordinate system.
    # The expressions adjust pixels relative to the center and scale by the focal length.
    cam_ray_directions = torch.stack([
        (grid_i.t().flatten() - image_width * 0.5) / focal_length,
        -(grid_j.t().flatten() - image_height * 0.5) / focal_length,
        -torch.ones_like(grid_j.t().flatten(), device=device)
    ], -1)

    # Transform the ray directions from camera coordinates to world coordinates.
    # Multiply the ray direction with the rotation part of the transform (upper-left 3x3).
    world_ray_directions = torch.sum(cam_ray_directions[..., None, :] * transform_w2c[:3, :3], axis=-1)

    # The ray origin for all rays is the translation part of the transform.
    world_ray_origins = transform_w2c[:3, -1].expand(world_ray_directions.shape)



    # Reshape the flat ray directions and origins into (H, W, 3) tensors.
    ray_directions = world_ray_directions.view(int(image_height), int(image_width), 3)
    ray_origins = world_ray_origins.view(int(image_height), int(image_width), 3)

    return ray_origins, ray_directions
    ################### YOUR CODE END ###################


def sample_points_from_rays(ray_origins, ray_directions, near_depth, far_depth, num_samples):
    r"""Compute a set of 3D points given the bundle of rays

    Expected Returns:
      sampled_points: axis of the sampled points along each ray, shape (H, W, num_samples, 3)
      depth_values: sampled depth values along each ray, shape (H, W, num_samples)
    """
    ################### YOUR CODE START ###################
    # Determine the computation device from ray origins.
    device = ray_origins.device

    # Extract image height and width from the shape of ray origins.
    H, W, _ = ray_origins.shape

    # Create a set of uniformly spaced depth values between near_depth and far_depth.
    uniform_depths = torch.linspace(near_depth, far_depth, num_samples).to(device)

    # Expand uniform depths to match image dimensions (H, W) and clone for potential modification.
    depth_values = uniform_depths[None, None, :].expand(H, W, num_samples).clone()

    # Define a scale for the random perturbation (jitter) along depth.
    jitter_scale = 0.3

    # If more than two samples are present, introduce randomness to the intermediate depth values.
    if num_samples > 2:
        # Compute jitter for intermediate depth values (excluding first and last depth samples).
        mid_depths = depth_values[:, :, 1:-1] + jitter_scale * torch.rand((H, W, num_samples-2), device=device) * ((far_depth - near_depth) / (num_samples - 1))
        # Replace the original intermediate depth values with the jittered values.
        depth_values[:, :, 1:-1] = mid_depths

    # Calculate the sampled 3D points along each ray.
    # This is done by taking the ray origin and adding the ray direction scaled by the depth values.
    sampled_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]

    return sampled_points, depth_values

    ################### YOUR CODE END ###################


def positional_encoding(input_tensor, max_freq_log2, include_input=True):
    r"""Apply positional encoding to the input. (Section 5.1 of original paper)
    We use positional encoding to map continuous input coordinates into a
    higher dimensional space to enable our MLP to more easily approximate a
    higher frequency function.

    Expected Returns:
      pos_out: positional encoding of the input tensor.
               (H*W*num_samples, (include_input + 2*freq) * 3)
    """
    ################### YOUR CODE START ###################
    # Generate frequency bands as powers of 2: [1, 2, 4, ..., 2^(max_freq_log2 - 1)]
    frequency_bands = 2 ** torch.arange(max_freq_log2, device=input_tensor.device, dtype=input_tensor.dtype)

    # List to store the sinusoidal components corresponding to each frequency band.
    sinusoidal_components = []

    # Iterate over each frequency and compute sine and cosine values.
    for freq in frequency_bands:
        sinusoidal_components.append(torch.sin(input_tensor * freq))
        sinusoidal_components.append(torch.cos(input_tensor * freq))

    # Concatenate the sinusoidal encodings along the last dimension.
    pos_encoded = torch.cat(sinusoidal_components, dim=-1)

    # Optionally include the original input in the positional encoding.
    if include_input:
        pos_encoded = torch.cat([input_tensor, pos_encoded], dim=-1)

    return pos_encoded
    ################### YOUR CODE END ###################


def volume_rendering(
    radiance_field: torch.Tensor,
    ray_origins: torch.Tensor,
    depth_samples: torch.Tensor
) -> Tuple[torch.Tensor]:
    r"""Differentiably renders a radiance field, given the origin of each ray in the
    bundle, and the sampled depth values along them.

    Args:
      radiance_field: at each query location (X, Y, Z), our model predict
        RGB color and a volume density (sigma), shape (H, W, num_samples, 4)
      ray_origins: origin of each ray, shape (H, W, 3)
      depth_values: sampled depth values along each ray, shape (H, W, num_samples)

    Expected Returns:
      rgb_map: rendered RGB image, shape (H, W, 3)
    """
    ################### YOUR CODE START ###################
    # Infer the device from the radiance field tensor.
    device = radiance_field.device

    # Retrieve image dimensions and the number of samples along each ray.
    H, W, num_samples, _ = radiance_field.shape

    # Apply activation functions:
    # - Sigmoid for RGB colors to constrain values between 0 and 1.
    # - ReLU for sigma (volume density) to ensure non-negative density.
    rgb_values = torch.sigmoid(radiance_field[..., :3])
    density = torch.relu(radiance_field[..., 3])

    # Compute differences between consecutive depth samples.
    # Initialize delta with a large number for the last sample.
    delta_depth = 1e10 * torch.ones_like(depth_samples, device=device)
    # For all but the last sample, compute the difference between consecutive depth values.
    delta_depth[..., :-1] = torch.diff(depth_samples, dim=-1)

    # Compute the alpha value per sample. Alpha represents the probability of a ray terminating at a given sample.
    # The formula used here is based on the volume rendering equation.
    alpha = 1.0 - torch.exp(-density * delta_depth)

    # Compute the cumulative product of (1 - alpha) along each ray.
    # This represents the accumulated transparency (i.e. survival probability) up to each sample.
    cumulative_transmittance = torch.cumprod(1.0 - alpha, dim=-1)
    # Shift the cumulative product to right (using roll) so that it aligns with the sample intervals.
    cumulative_transmittance = torch.roll(cumulative_transmittance, 1, dims=-1)

    # The first element of cumulative transmittance should be 1 (full visibility) so set it explicitly.
    cumulative_transmittance[..., 0] = 1.0

    # Compute the weight for each sample as the product of alpha and cumulative transmittance.
    sample_weights = alpha * cumulative_transmittance

    # Integrate the colors along the ray by summing the weighted RGB colors.
    rgb_map = torch.sum(sample_weights[..., None] * rgb_values, dim=-2)

    return rgb_map
    ################### YOUR CODE END ###################


class TinyNeRF(torch.nn.Module):
    def __init__(self, pos_dim, fc_dim=64):
      r"""Initialize a tiny nerf network, which composed of linear layers and
      ReLU activation. More specifically: linear - relu - linear - relu - linear
      - relu -linear. The module is intentionally made small so that we could
      achieve reasonable training time

      Args:
        pos_dim: dimension of the positional encoding output
        fc_dim: dimension of the fully connected layer
      """
      super().__init__()

      self.nerf = nn.Sequential(
                    nn.Linear(pos_dim, fc_dim),
                    nn.ReLU(),
                    nn.Linear(fc_dim, fc_dim),
                    nn.ReLU(),
                    nn.Linear(fc_dim, fc_dim),
                    nn.ReLU(),
                    nn.Linear(fc_dim, 4)
                  )

    def forward(self, x):
      r"""Output volume density and RGB color (4 dimensions), given a set of
      positional encoded points sampled from the rays
      """
      x = self.nerf(x)
      return x


def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def nerf_step_forward(height, width, focal_length, trans_matrix,
                            near_point, far_point, num_depth_samples_per_ray,
                            chunksize, model):
    r"""Perform one iteration of training, which take information of one of the
    training images, and try to predict its rgb values

    Args:
      height: height of the image
      width: width of the image
      focal_length: focal length of the camera
      trans_matrix: transformation matrix, which is also the camera pose
      near_point: threshhold of nearest point
      far_point: threshold of farthest point
      num_depth_samples_per_ray: number of sampled depth from each rays in the ray bundle
      get_minibatches_function: function to cut the ray bundles into several chunks
        to avoid out-of-memory issue

    Expected Returns:
      rgb_predicted: predicted rgb values of the training image
    """
    ################### YOUR CODE START ###################

    # Compute ray origins and directions using the given camera parameters.
    ray_origins, ray_directions = get_rays(height, width, focal_length, trans_matrix)

    # Sample 3D points along each ray and obtain corresponding depth values.
    sampled_points, depth_values = sample_points_from_rays(ray_origins, ray_directions, near_point, far_point, num_depth_samples_per_ray)

    # Retrieve image dimensions and number of samples per ray from the sampled points.
    img_height, img_width, num_depth_samples, _ = sampled_points.shape

    # Flatten the sampled points so that each ray's points become a single batch for positional encoding.
    # New shape: (img_height * img_width, num_depth_samples, 3)
    flat_sampled_points = sampled_points.reshape(-1, num_depth_samples, 3)

    # Define the number of frequency bands for positional encoding and set the flag to include the original input.
    num_frequency_bands = 10
    include_original_input = True

    # Apply positional encoding to the flattened 3D points.
    positional_encoded_points = positional_encoding(flat_sampled_points, num_frequency_bands, include_original_input)


    ################### YOUR CODE END ###################

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    batches = get_minibatches(positional_encoded_points, chunksize=16384)
    predictions = []
    for batch in batches:
      predictions.append(model(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0) # (H*W*num_samples, 4)

    # "Unflatten" the radiance field.
    unflattened_shape = [height, width, num_depth_samples_per_ray, 4] # (H, W, num_samples, 4)
    radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape) # (H, W, num_samples, 4)

    ################### YOUR CODE START ###################
    # TODO: Perform differentiable volume rendering to re-synthesize the RGB image. # (H, W, 3)
    rgb_predicted = volume_rendering(radiance_field, ray_origins, depth_values)
    return rgb_predicted
    ################### YOUR CODE END ###################


def train(images, poses, hwf, near_point,
          far_point, num_depth_samples_per_ray,
          num_iters, model, chunksize, DEVICE="cuda"):
    r"""Training a tiny nerf model

    Args:
      images: all the images extracted from dataset (including train, val, test)
      poses: poses of the camera, which are used as transformation matrix
      hwf: [height, width, focal_length]
      near_point: threshhold of nearest point
      far_point: threshold of farthest point
      num_depth_samples_per_ray: number of sampled depth from each rays in the ray bundle
      num_iters: number of training iterations
      model: predefined tiny NeRF model
    """
    H, W, focal_length = hwf
    H = int(H)
    W = int(W)
    n_train = images.shape[0]

    # Optimizer parameters
    lr = 5e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.8) # Scheduler

    # Seed RNG, for repeatability
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Display parameters
    display_frequency = 50
    losses = []
    iters = []
    directory_path = 'logs'

    for _ in tqdm(range(num_iters)):
      # Randomly pick a training image as the target, get rgb value and camera pose
      train_idx = np.random.randint(n_train)
      train_img_rgb = images[train_idx, ..., :3]
      train_pose = poses[train_idx]

      # Run one iteration of TinyNeRF and get the rendered RGB image.
      rgb_predicted = nerf_step_forward(H, W, focal_length,
                          train_pose, near_point,
                          far_point, num_depth_samples_per_ray,
                          chunksize, model)

      # Compute mean-squared error between the predicted and target images
      loss = torch.nn.functional.mse_loss(rgb_predicted, train_img_rgb)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      scheduler.step()  # Update learning rate

      # Display training progress
      if _ % display_frequency == 0:

            iters.append(_)
            # Output loss
            print(f"Iter: {_}, Loss: {loss.item()}")
            losses.append(loss.item())

            plt.figure(figsize=(15, 5))

            # Display the loss curve
            plt.subplot(1, 3, 1)
            plt.plot(iters, losses)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Training Loss Progress')

            # Display the predicted image
            plt.subplot(1, 3, 2)
            predicted_img = rgb_predicted.cpu().detach().numpy()
            predicted_img = np.clip(predicted_img, 0, 1)
            plt.imshow(predicted_img)
            plt.title('Model Predicted Image')

            #  Display the ground truth image
            plt.subplot(1, 3, 3)
            ground_truth_img = train_img_rgb.cpu().detach().numpy()
            ground_truth_img = np.clip(ground_truth_img, 0, 1)
            plt.imshow(ground_truth_img)
            plt.title('Actual Ground Truth Image')

            plt.savefig(dataset_root + directory_path + '/training_progress_' + str(_) + '.png')
            plt.show()

    print('Finish training')

def spherical_to_cartesian_converter(radius, azimuth, polar_angle):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Args:
      radius: The distance from the origin.
      azimuth: The angle in the xy-plane measured from the x-axis.
      polar_angle: The angle from the z-axis.

    Returns:
      x_coord, y_coord, z_coord: The corresponding Cartesian coordinates.
    """
    x_coord = radius * torch.sin(polar_angle) * torch.cos(azimuth)
    y_coord = radius * torch.sin(polar_angle) * torch.sin(azimuth)
    z_coord = radius * torch.cos(polar_angle)
    return x_coord, y_coord, z_coord

def camera_look(eye_position, target_point, world_up=np.array([0, 1, 0])):
    """
    Generate look transformation matrix for a camera.

    This function computes a 4x4 transformation matrix that transforms world
    coordinates to a camera coordinate system where the camera is positioned at
    `eye_position` and is pointed toward `target_point`.

    Args:
      eye_position: A tensor representing the camera's position in world coordinates.
      target_point: A tensor representing the point in space that the camera is looking at.
      world_up: An array representing the world's up direction (default is [0, 1, 0]).

    Returns:
      A 4x4 look-at transformation matrix as a torch tensor.
    """
    # Convert tensors to numpy arrays for computation.
    eye_position = eye_position.numpy()
    target_point = target_point.numpy()

    # Compute the forward (view) direction vector from target to camera.
    forward_vec = eye_position - target_point
    forward_vec /= np.linalg.norm(forward_vec)

    # Compute the right vector as the cross product of the up vector and forward direction.
    right_vec = np.cross(world_up, forward_vec)
    right_vec /= np.linalg.norm(right_vec)

    # Compute the corrected up vector from the forward and right vectors.
    up_vec = np.cross(forward_vec, right_vec)

    # Build the rotation part of the transformation matrix.
    rotation_transform = np.eye(4)
    rotation_transform[:3, :3] = np.stack([right_vec, up_vec, forward_vec], axis=-1)

    # Build the translation part of the transformation matrix.
    translation_transform = np.eye(4)
    translation_transform[:3, 3] = -eye_position

    # Combine rotation and translation to obtain the final look-at matrix.
    look_at_transform = rotation_transform @ translation_transform
    return torch.tensor(look_at_transform, dtype=torch.float)


def generate_images(test_model, camera_hwf, near_point, far_point, num_depth_samples_per_ray, chunksize, poses, imgs): # Added 'imgs' as input
    """Generates and displays predicted images alongside original images and viewpoint info."""

    H, W, _ = camera_hwf # Unpack camera parameters
    H, W = int(H), int(W)

    # Create a directory to save the plots if it doesn't exist
    output_dir = os.path.join(dataset_root, 'generated_images') # Path for saving images
    os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist

    # --- From Training Poses ---
    random_indices = np.random.choice(poses.shape[0], 10, replace=False)
    test_pose = poses[random_indices]
    for i in range(10):
        # 1. Render using NeRF
        predicted_img = nerf_step_forward(H, W, camera_hwf[2], test_pose[i], near_point, far_point, num_depth_samples_per_ray, chunksize, test_model)

        # 2. Get Original Image
        original_img = imgs[random_indices[i]].cpu().detach().numpy() # Get original image corresponding to the pose

        # 3. Viewpoint Info (Camera Pose)
        viewpoint_info = test_pose[i].cpu().detach().numpy() # Get camera pose

        # --- Display ---
        plt.figure(figsize=(12, 4))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(np.clip(original_img, 0, 1))
        plt.title("Original Image")

        # Predicted Image
        plt.subplot(1, 3, 2)
        plt.imshow(np.clip(predicted_img.cpu().detach().numpy(), 0, 1))
        plt.title("Predicted Image")

        # Viewpoint Info
        plt.subplot(1, 3, 3)
        plt.text(0.05, 0.95, f"Camera Pose:\n{viewpoint_info}", fontsize=8, va='top') # Display camera pose as text
        plt.axis('off') # Hide axes for the text subplot

        plt.suptitle("Comparison (Training Pose)")
        plt.savefig(os.path.join(output_dir, f'training_view_{i}.png'))  # Save the plot
        plt.show()

    # --- From Novel Poses ---
    # --- From Novel Poses (perturbed training poses) ---
    test_poses = []
    for i in range(20):  # Generate 20 novel views
        # 1. Select a random training pose
        random_idx = np.random.randint(poses.shape[0])
        base_pose = poses[random_idx]


        # 2. Perturb the pose slightly
        # You can adjust the perturbation magnitude (scale)
        perturbation_scale = 0.7
        perturbation = torch.randn(3, device=base_pose.device) * perturbation_scale # Create perturbation on the same device as base_pose
        # Apply perturbation to the translation part of the pose
        perturbed_pose = base_pose.clone()
        perturbed_pose[:3, 3] += perturbation

        test_poses.append(perturbed_pose)

    for i in range(20):
        # 1. Render using NeRF
        predicted_img = nerf_step_forward(H, W, camera_hwf[2], test_poses[i], near_point, far_point, num_depth_samples_per_ray, chunksize, test_model)

        # 2. Viewpoint Info (Camera Pose)
        viewpoint_info = test_poses[i].cpu().detach().numpy()

        # --- Display ---
        plt.figure(figsize=(8, 4))

        # Predicted Image
        plt.subplot(1, 2, 1)
        plt.imshow(np.clip(predicted_img.cpu().detach().numpy(), 0, 1))
        plt.title("Predicted Image (Novel View)")

        # Viewpoint Info
        plt.subplot(1, 2, 2)
        plt.text(0.05, 0.95, f"Camera Pose:\n{viewpoint_info}", fontsize=8, va='top')
        plt.axis('off')

        plt.suptitle("Novel View Rendering")
        plt.savefig(os.path.join(output_dir, f'novel_view_{i}.png'))  # Save the plot
        plt.show()

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    imgs, poses, camera_hwf = load_colmap_data(dataset_root)

    # Test if the images are loaded correctly
    plt.imshow(imgs[0])
    plt.show()

    if isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs)
    elif isinstance(imgs, torch.Tensor):
        pass  # do nothing
    else:
        print("T_wc type error")

    if isinstance(poses, np.ndarray):
        poses = torch.from_numpy(poses)
    elif isinstance(poses, torch.Tensor):
        pass  # do nothing
    else:
        print("T_wc type error")

    imgs = imgs.to(DEVICE)
    poses = poses.to(DEVICE)

    # Define the parameters
    near_point = 2
    far_point = 6
    num_depth_samples_per_ray = 164



    chunksize = 8192 #* 2
    max_iters = 2001
    # Positional encoding parameters
    max_freq_log2 = 10
    include_input = True
    pos_dim = 3 * (1 + 2 * max_freq_log2) if include_input else 3 * 2 * max_freq_log2

    # Initialize the model
    model = TinyNeRF(pos_dim).to(DEVICE)

    # Train the model
    train(imgs, poses, camera_hwf, near_point, far_point, num_depth_samples_per_ray, max_iters, model, chunksize=chunksize, DEVICE=DEVICE)

    # Save the model
    torch.save(model.state_dict(), dataset_root + 'model/quanpham_TinyNerf.pt')

    # Load the model for testing
    test_model = TinyNeRF(pos_dim).to(DEVICE)
    test_model.load_state_dict(torch.load(dataset_root + 'model/quanpham_TinyNerf.pt'))
    test_model.eval() # Set the model to evaluation mode.

    # Generate images using the separate function
    generate_images(test_model, camera_hwf, near_point, far_point, num_depth_samples_per_ray, chunksize, poses, imgs)



if __name__ == "__main__":
    main()
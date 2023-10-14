import torch
import torch.nn.functional as F

def create_irregular_area(starting_point, target_size, image_size):
    # Create a tensor with a single starting point
    mask = torch.zeros(image_size, image_size)
    mask[starting_point[0], starting_point[1]] = 1.0
    
    # Apply a Gaussian filter
    filter_size = int(target_size ** 0.5) # Assuming areas are roughly square
    if filter_size % 2 == 0: filter_size += 1
    gauss_filter = torch.randn(filter_size, filter_size)
    mask = F.conv2d(mask[None, None, ...], gauss_filter[None, None, ...], padding=filter_size//2)
    
    # Threshold to create an area of the desired size
    mask = (mask > mask.flatten().topk(target_size)[0][-1]).float()
    
    return mask

def create_areas_in_tensor(batch_size, image_size, N, S):
    tensor = torch.zeros(batch_size, 1, image_size, image_size)
    
    for _ in range(N):
        # Randomly select starting points for each sample in the batch
        starting_points = torch.randint(0, image_size, (batch_size, 2))
        
        # Create a mask for each sample
        masks = [create_irregular_area(starting_point, S, image_size) for starting_point in starting_points]
        
        # Add the masks to the tensor, avoiding overlaps
        try:
            tensor += torch.stack(masks).view(batch_size, 1, image_size, image_size)
        except:
            print('image_size==',image_size)
            print(batch_size)
        tensor += torch.stack(masks).view(batch_size, 1, image_size, image_size)
    
    # Clamp values to ensure they're either 0 or 1
    tensor.clamp_(0, 1)
    
    return tensor
"""
Contrastive direction loss for CLIP fine-tuning on camera motion
Pulls together similar directions, pushes apart opposite directions
"""
import torch
import torch.nn.functional as F
import re


# Direction definitions - map direction labels to text variants
DIRECTION_PAIRS = {
    'left': ['left', 'leftward', 'leftwards'],
    'right': ['right', 'rightward', 'rightwards'],
    'up': ['up', 'upward', 'upwards'],
    'down': ['down', 'downward', 'downwards'],
    'forward': ['forward', 'forwards', 'ahead'],
    'backward': ['backward', 'backwards', 'back'],
}

# Opposite direction pairs - these should be pushed APART in embedding space
# E.g., "pan left" should be far from "pan right"
OPPOSITE_PAIRS = [
    ('left', 'right'),
    ('up', 'down'),
    ('forward', 'backward'),
]


def parse_camera_directions(text):
    """
    Extract camera direction labels from text
    Returns: list of direction labels found in text
    """
    text_lower = text.lower()
    directions = []
    
    for dir_key, dir_variants in DIRECTION_PAIRS.items():
        for variant in dir_variants:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(variant) + r'\b'
            if re.search(pattern, text_lower):
                if dir_key not in directions:
                    directions.append(dir_key)
                break
    
    return directions


def get_direction_labels(texts):
    """
    Batch process texts to extract direction labels
    Args:
        texts: list of caption strings
    Returns:
        directions: list of lists, each containing direction labels for that text
    """
    return [parse_camera_directions(text) for text in texts]


def contrastive_direction_loss(text_embeddings, texts, temperature=0.07, margin=0.2):
    """
    Contrastive loss that:
    1. Pulls together embeddings with similar directions (same type: pan/tilt/dolly)
    2. Pushes apart embeddings with opposite directions (left vs right, up vs down, etc.)
    
    Args:
        text_embeddings: (batch_size, embed_dim) CLIP text embeddings
        texts: list of text captions
        temperature: temperature for contrastive loss
        margin: margin for pushing apart opposites
    
    Returns:
        loss: scalar contrastive loss
        stats: dict with loss components for logging
    """
    batch_size = text_embeddings.shape[0]
    device = text_embeddings.device
    
    # Parse directions from texts
    direction_labels = get_direction_labels(texts)
    
    # Normalize embeddings
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    # Add epsilon to prevent division by zero in normalization
    # Check for NaN/Inf in embeddings
    if torch.isnan(text_embeddings).any() or torch.isinf(text_embeddings).any():
        print("WARNING: NaN or Inf detected in text embeddings before similarity computation")
        return torch.tensor(0.0, device=device), {
            'direction_loss/total': 0.0,
            'direction_loss/same': 0.0,
            'direction_loss/opposite': 0.0,
            'direction_loss/num_same_pairs': 0.0,
            'direction_loss/num_opposite_pairs': 0.0,
        }
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(text_embeddings, text_embeddings.T)  # (bs, bs)
    
    # Create masks for different relationships
    opposite_mask = torch.zeros(batch_size, batch_size, device=device)
    same_mask = torch.zeros(batch_size, batch_size, device=device)
    
    for i in range(batch_size):
        for j in range(i+1, batch_size):
            dirs_i = set(direction_labels[i])
            dirs_j = set(direction_labels[j])
            
            if not dirs_i or not dirs_j:
                continue
            
            # Check if they share ANY direction (same direction = pull together)
            shared_directions = dirs_i & dirs_j
            if shared_directions:
                # Same direction: pull together
                same_mask[i, j] = 1.0
                same_mask[j, i] = 1.0
                continue
            
            # Check for opposite directions (push apart)
            is_opposite = False
            for dir1, dir2 in OPPOSITE_PAIRS:
                if (dir1 in dirs_i and dir2 in dirs_j) or (dir2 in dirs_i and dir1 in dirs_j):
                    is_opposite = True
                    opposite_mask[i, j] = 1.0
                    opposite_mask[j, i] = 1.0
                    break
    
    # Loss for same direction pairs: maximize similarity (pull together)
    # E.g., "pan left" and "move leftward" should have high similarity
    same_loss = torch.tensor(0.0, device=device)
    num_same = same_mask.sum()
    if num_same > 0:
        # Want high similarity (close to 1), so minimize negative similarity
        same_loss = -((similarity_matrix * same_mask).sum() / num_same)
    
    # Loss for opposite pairs: minimize similarity (push apart)
    # E.g., "pan left" and "pan right" should have low similarity
    opposite_loss = torch.tensor(0.0, device=device)
    num_opposite = opposite_mask.sum()
    if num_opposite > 0:
        # Want low similarity (below -margin)
        # Use hinge loss: max(0, similarity + margin)
        opposite_similarities = similarity_matrix * opposite_mask
        opposite_loss = F.relu(opposite_similarities + margin).sum() / num_opposite
    
    # Combined loss
    total_loss = same_loss + opposite_loss
    
    # Check for NaN in final loss
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"WARNING: NaN/Inf in direction loss. same_loss={same_loss.item():.4f}, opposite_loss={opposite_loss.item():.4f}")
        print(f"  num_same={num_same.item()}, num_opposite={num_opposite.item()}")
        print(f"  similarity_matrix range: [{similarity_matrix.min().item():.4f}, {similarity_matrix.max().item():.4f}]")
        total_loss = torch.tensor(0.0, device=device)
    
    # Ensure all stats are valid floats
    stats = {
        'direction_loss/total': total_loss.item() if not torch.isnan(total_loss) else 0.0,
        'direction_loss/same': same_loss.item() if not torch.isnan(same_loss) else 0.0,
        'direction_loss/opposite': opposite_loss.item() if not torch.isnan(opposite_loss) else 0.0,
        'direction_loss/num_same_pairs': (num_same.item() / 2) if num_same > 0 else 0.0,
        'direction_loss/num_opposite_pairs': (num_opposite.item() / 2) if num_opposite > 0 else 0.0,
    }
    
    return total_loss, stats


def direction_classification_loss(text_embeddings, texts, direction_classifier):
    """
    Optional: Multi-label classification loss for direction prediction
    
    Args:
        text_embeddings: (batch_size, embed_dim) CLIP text embeddings
        texts: list of text captions
        direction_classifier: nn.Module that maps embeddings to 6-dim logits
    
    Returns:
        loss: BCE loss for multi-label classification
        stats: dict with accuracy metrics
    """
    batch_size = text_embeddings.shape[0]
    device = text_embeddings.device
    
    # Parse directions and create labels
    direction_labels = get_direction_labels(texts)
    direction_order = ['left', 'right', 'up', 'down', 'forward', 'backward']
    
    # Create multi-label targets (batch_size, 6)
    targets = torch.zeros(batch_size, 6, device=device)
    for i, dirs in enumerate(direction_labels):
        for dir_name in dirs:
            if dir_name in direction_order:
                idx = direction_order.index(dir_name)
                targets[i, idx] = 1.0
    
    # Predict
    logits = direction_classifier(text_embeddings)  # (batch_size, 6)
    
    # BCE loss
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
    
    # Accuracy
    predictions = (torch.sigmoid(logits) > 0.5).float()
    accuracy = (predictions == targets).float().mean()
    
    stats = {
        'direction_cls/loss': loss.item(),
        'direction_cls/accuracy': accuracy.item(),
    }
    
    return loss, stats


# Example usage and testing
if __name__ == '__main__':
    # Test direction parsing
    test_texts = [
        "Camera pans left",
        "Camera pans right",
        "Camera moves forward and tilts up",
        "Camera tilts down while moving backward",
        "The camera slowly pans leftward",
    ]
    
    print("=== Direction Parsing Test ===")
    for text in test_texts:
        dirs = parse_camera_directions(text)
        print(f"{text:50s} -> {dirs}")
    
    print("\n=== Contrastive Loss Test ===")
    # Create dummy embeddings
    batch_size = 5
    embed_dim = 512
    text_embeddings = torch.randn(batch_size, embed_dim)
    
    loss, stats = contrastive_direction_loss(text_embeddings, test_texts)
    print(f"Loss: {loss.item():.4f}")
    for key, val in stats.items():
        print(f"  {key}: {val:.4f}")

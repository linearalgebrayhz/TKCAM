"""
Advanced CLaTr Metrics Implementation
Based on "Exceptional Trajectory" paper and motion generation evaluation
"""

import numpy as np
from scipy import linalg
from sklearn.metrics import f1_score, precision_score, recall_score


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Fréchet Distance (FID-like metric) between two Gaussian distributions.
    
    Adapted from FID implementation for trajectory embeddings (CLaTr-FID).
    
    Args:
        mu1: Mean of first distribution (GT trajectories)
        sigma1: Covariance of first distribution
        mu2: Mean of second distribution (Generated trajectories)
        sigma2: Covariance of second distribution
        eps: Small value to add to diagonal for numerical stability
    
    Returns:
        Fréchet distance between the two distributions
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, f"Mean vectors have different shapes: {mu1.shape} vs {mu2.shape}"
    assert sigma1.shape == sigma2.shape, f"Covariance matrices have different shapes"
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        print(f"WARNING: FID calculation produces singular product; adding {eps} to diagonal of cov estimates")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid


def compute_clatr_fid(gt_embeddings, gen_embeddings):
    """
    Compute CLaTr-FID: Fréchet Distance between GT and Generated trajectory embeddings.
    
    Args:
        gt_embeddings: [N, D] Ground truth trajectory embeddings from CLaTr encoder
        gen_embeddings: [M, D] Generated trajectory embeddings from CLaTr encoder
    
    Returns:
        fid: CLaTr-FID score (lower is better)
    """
    # Check if embeddings are valid
    assert len(gt_embeddings.shape) == 2, f"GT embeddings must be 2D, got {gt_embeddings.shape}"
    assert len(gen_embeddings.shape) == 2, f"Gen embeddings must be 2D, got {gen_embeddings.shape}"
    assert gt_embeddings.shape[1] == gen_embeddings.shape[1], \
        f"Embedding dimensions must match: GT={gt_embeddings.shape[1]}, Gen={gen_embeddings.shape[1]}"
    
    print(f"  Computing FID: GT shape={gt_embeddings.shape}, Gen shape={gen_embeddings.shape}")
    
    # Compute statistics
    mu_gt = np.mean(gt_embeddings, axis=0)
    sigma_gt = np.cov(gt_embeddings, rowvar=False)
    
    mu_gen = np.mean(gen_embeddings, axis=0)
    sigma_gen = np.cov(gen_embeddings, rowvar=False)
    
    # Debug: Check if covariance matrices are reasonable
    print(f"  GT covariance trace: {np.trace(sigma_gt):.2f}, condition number: {np.linalg.cond(sigma_gt):.2e}")
    print(f"  Gen covariance trace: {np.trace(sigma_gen):.2f}, condition number: {np.linalg.cond(sigma_gen):.2e}")
    
    # If condition number is too high, add regularization
    max_cond = 1e10
    if np.linalg.cond(sigma_gt) > max_cond or np.linalg.cond(sigma_gen) > max_cond:
        print(f"  WARNING: Ill-conditioned covariance, adding regularization")
        eps = 1e-6
        sigma_gt += np.eye(sigma_gt.shape[0]) * eps
        sigma_gen += np.eye(sigma_gen.shape[0]) * eps
    
    fid = calculate_frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen)
    
    return fid


def compute_coverage(gt_embeddings, gen_embeddings, k=10):
    """
    Compute Coverage: percentage of GT data "covered" by generated data.
    
    For each GT sample, find its k-nearest neighbors in generated samples.
    Coverage = percentage of GT samples that have at least one close generated neighbor.
    
    Args:
        gt_embeddings: [N, D] Ground truth embeddings
        gen_embeddings: [M, D] Generated embeddings
        k: Number of nearest neighbors to consider
    
    Returns:
        coverage: Coverage score (0-100, higher is better)
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Normalize embeddings
    gt_norm = gt_embeddings / (np.linalg.norm(gt_embeddings, axis=1, keepdims=True) + 1e-8)
    gen_norm = gen_embeddings / (np.linalg.norm(gen_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # For each GT sample, find k nearest generated samples
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(gen_norm)
    distances, indices = nbrs.kneighbors(gt_norm)
    
    # Consider a GT sample "covered" if its nearest generated neighbor is close enough
    # Use median distance as threshold
    min_distances = distances[:, 0]  # Closest neighbor for each GT sample
    threshold = np.median(min_distances)
    
    covered = np.sum(min_distances < threshold)
    coverage = (covered / len(gt_embeddings)) * 100
    
    return coverage


def compute_clatr_clip_score(text_embeddings, traj_embeddings):
    """
    Compute CLaTr-CLIP: Average cosine similarity between paired text and trajectory.
    
    Similar to CLIP-Score but for text-trajectory pairs.
    Uses CLaTr's similarity computation (normalized cosine similarity).
    
    Args:
        text_embeddings: [N, D] Text embeddings from CLaTr
        traj_embeddings: [N, D] Trajectory embeddings from CLaTr (paired with text)
    
    Returns:
        clip_score: Average similarity score (0-100, higher is better)
    """
    # Normalize (same as CLaTr's get_sim_matrix)
    text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
    traj_norm = traj_embeddings / (np.linalg.norm(traj_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix (full pairwise)
    sim_matrix = text_norm @ traj_norm.T
    
    # CLaTr converts cosine similarity [-1, 1] to score [0, 1]
    score_matrix = sim_matrix / 2.0 + 0.5
    
    # Take diagonal (paired text-trajectory similarities)
    paired_scores = np.diag(score_matrix)
    
    clip_score = np.mean(paired_scores)
    return clip_score * 100  # Convert to percentage


def compute_all_advanced_metrics(gt_traj_latents, gen_traj_latents, 
                                  text_latents_gt, text_latents_gen=None,
                                  max_samples_for_fid=5000):
    """
    Compute all advanced CLaTr metrics.
    
    Args:
        gt_traj_latents: [N, D] Ground truth trajectory embeddings
        gen_traj_latents: [M, D] Generated trajectory embeddings
        text_latents_gt: [N, D] Text embeddings for GT trajectories
        text_latents_gen: [M, D] Text embeddings for generated trajectories (optional)
        max_samples_for_fid: Maximum number of samples to use for FID (for numerical stability)
    
    Returns:
        metrics: Dictionary with all advanced metrics
    """
    metrics = {}
    
    # Subsample if needed for FID computation (avoid numerical issues with large N)
    if len(gt_traj_latents) > max_samples_for_fid:
        print(f"  Subsampling {max_samples_for_fid} from {len(gt_traj_latents)} samples for FID")
        indices = np.random.choice(len(gt_traj_latents), max_samples_for_fid, replace=False)
        gt_traj_fid = gt_traj_latents[indices]
        gen_traj_fid = gen_traj_latents[indices]
    else:
        gt_traj_fid = gt_traj_latents
        gen_traj_fid = gen_traj_latents
    
    # CLaTr-FID
    try:
        fid = compute_clatr_fid(gt_traj_fid, gen_traj_fid)
        metrics['CLaTr-FID'] = float(fid)
    except Exception as e:
        print(f"Warning: Failed to compute CLaTr-FID: {e}")
        import traceback
        traceback.print_exc()
        metrics['CLaTr-FID'] = None
    
    # Coverage
    try:
        coverage = compute_coverage(gt_traj_latents, gen_traj_latents, k=10)
        metrics['Coverage'] = float(coverage)
    except Exception as e:
        print(f"Warning: Failed to compute Coverage: {e}")
        metrics['Coverage'] = None
    
    # CLaTr-CLIP (GT)
    try:
        clip_score_gt = compute_clatr_clip_score(text_latents_gt, gt_traj_latents)
        metrics['CLaTr-CLIP-GT'] = float(clip_score_gt)
    except Exception as e:
        print(f"Warning: Failed to compute CLaTr-CLIP-GT: {e}")
        metrics['CLaTr-CLIP-GT'] = None
    
    # CLaTr-CLIP (Generated)
    if text_latents_gen is not None:
        try:
            clip_score_gen = compute_clatr_clip_score(text_latents_gen, gen_traj_latents)
            metrics['CLaTr-CLIP-Gen'] = float(clip_score_gen)
        except Exception as e:
            print(f"Warning: Failed to compute CLaTr-CLIP-Gen: {e}")
            metrics['CLaTr-CLIP-Gen'] = None
    
    # Diversity (average pairwise distance in generated samples)
    try:
        gen_norm = gen_traj_latents / (np.linalg.norm(gen_traj_latents, axis=1, keepdims=True) + 1e-8)
        pairwise_sim = gen_norm @ gen_norm.T
        # Exclude diagonal (self-similarity)
        mask = ~np.eye(len(pairwise_sim), dtype=bool)
        avg_sim = pairwise_sim[mask].mean()
        diversity = (1 - avg_sim) * 100  # Higher diversity = lower similarity
        metrics['Diversity'] = float(diversity)
    except Exception as e:
        print(f"Warning: Failed to compute Diversity: {e}")
        metrics['Diversity'] = None
    
    return metrics


def print_advanced_metrics(metrics, title="Advanced CLaTr Metrics"):
    """Pretty print advanced metrics"""
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    if metrics.get('CLaTr-FID') is not None:
        print(f"CLaTr-FID:      {metrics['CLaTr-FID']:.2f} (lower is better)")
    
    if metrics.get('Coverage') is not None:
        print(f"Coverage:       {metrics['Coverage']:.2f}% (higher is better)")
    
    if metrics.get('CLaTr-CLIP-GT') is not None:
        print(f"CLaTr-CLIP-GT:  {metrics['CLaTr-CLIP-GT']:.2f}% (text-traj alignment for GT)")
    
    if metrics.get('CLaTr-CLIP-Gen') is not None:
        print(f"CLaTr-CLIP-Gen: {metrics['CLaTr-CLIP-Gen']:.2f}% (text-traj alignment for Generated)")
    
    if metrics.get('Diversity') is not None:
        print(f"Diversity:      {metrics['Diversity']:.2f}% (higher is better)")
    
    print("="*80)

"""
gu_ensemble.py - Generative Uncertainty Enhanced Ensemble

MOTIVATION: 
    Flow Matching can produce high-quality samples on average, but some noise priors 
    lead to low-quality predictions that may "mislead" the ensemble results.
    We aim to enhance flow-ensemble by filtering out low-quality predictions (from bad noise priors).

CORE IDEA:
    Use "generative uncertainty" (prediction variance across multiple models for the same noise)
    as a quality measure. Low variance = high confidence = high quality noise prior.

PIPELINE:
    1. GENERATE: Create N random noise priors
    2. EVALUATE: For each noise, get predictions from M auxiliary models → compute entropy (variance)
    3. FILTER: Keep top (1-filter_ratio) noises with lowest entropy
    4. ENSEMBLE: Use main model's predictions for filtered noises, weighted by inverse entropy

OPTIMIZATION NOTE:
    To avoid redundant computation, if main_model is included in auxiliary_models,
    we reuse its predictions from the evaluation phase instead of sampling again.

References:
    - "Generative Uncertainty in Diffusion Models" [NeurIPS 2024] (https://www.alphaxiv.org/overview/2502.20946v2)


# 我想要实现一个功能：随机选择一个标准高斯噪声，通过多个不同训练好的RF模型生成多个预测，然后计算这些预测的熵值，作为该噪声的质量度量。
# 接着，我想要过滤掉那些熵值较高（即质量较低）的噪声，只保留熵值较低的噪声用于最终的加权平均集成。
# 最终，我希望使用这些高质量噪声生成的预测，结合它们的熵值的逆作为权重，进行加权平均，得到最终的集成预测结果。
# 具体地说，我需要在这个文件中实现以下几个步骤：
# 1. 循环：随机选择一个噪声 -> 多个RF模型 -> 多个预测 -> 计算熵值
# 2. 过滤：根据熵值排序 -> 过滤掉低质量噪声（预设比例？例如，20%）-> 使用熵值的逆作为加权平均的权重（但权重的和应该要处理成sum为1吧？）
# 3. 集成：将高质量噪声输入到我们最好的RF模型中（可能之前已经计算过）-> 执行加权平均集成
# 请在这个文件中实现函数用于输出筛选后高质量的噪声，以及他们的对应权重，以便在app_uq_ensemble.py中使用。
# 对应地，app_uq_ensemble.py中ensemble_sample函数需要修改为能够兼容这个idea，即接收高质量的噪声及其权重，然后输出所有噪声经过RF模型后生成的N个预测。因此，RectifiedFlow类中的sample函数也需要相应修改，支持输入噪声。
# 最后，在app_uq_ensemble.py中，我们需要综合以上各个函数和文件，在test阶段，也就是main模块内完成最终的加权集成预测，从而实验验证这个idea是否能够提升预测效果和不确定性估计效果。
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


def compute_prediction_entropy(predictions: torch.Tensor, dim: int = 2) -> torch.Tensor:
    """
    Compute the entropy (uncertainty) of predictions from multiple models for each noise.
    
    Uses the variance of predictions as a proxy for entropy in continuous space.
    Higher variance = higher entropy = lower quality noise.
    
    Args:
        predictions: Tensor of shape [n_models, batch_size, 1+dim] 
                     containing predictions from multiple models for a single noise
        dim: Spatial dimension
    
    Returns:
        entropy: Tensor of shape [batch_size] representing entropy for each sample
    """
    # Compute variance across models (dim=0) for each spatio-temporal dimension
    variance = predictions.var(dim=0)  # [batch_size, 1+dim]
    # Sum variance across all dimensions (time + space) as total entropy
    entropy = variance.sum(dim=-1)  # [batch_size]
    return entropy


def generate_noises(n_noises: int, batch_size: int, channels: int, seq_length: int, device: torch.device) -> List[torch.Tensor]:
    """
    Generate a list of random noise tensors.
    
    Args:
        n_noises: Number of noise tensors to generate
        batch_size: Batch size for each noise
        channels: Number of channels (typically 1)
        seq_length: Sequence length (1 + dim)
        device: Device to create tensors on
    
    Returns:
        List of noise tensors, each of shape [batch_size, channels, seq_length]
    """
    return [torch.randn(batch_size, channels, seq_length, device=device) for _ in range(n_noises)]


def evaluate_noise_quality(models: List,
                           noises: List[torch.Tensor],
                           cond: torch.Tensor,
                           dim: int = 2) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
    """
    Evaluate the quality of each noise by computing entropy across multiple model predictions.
    
    Args:
        models: List of trained RF models (RF_Model_all instances)
        noises: List of noise tensors [n_noises x (batch_size, channels, seq_length)]
        cond: Conditioning information [batch_size, 1, cond_dim]
        dim: Spatial dimension
    
    Returns:
        entropies: Tensor of shape [n_noises, batch_size] containing entropy for each noise-sample pair
        all_predictions: List[List[Tensor]] - predictions[noise_idx][model_idx] = [batch_size, 1+dim]
                        Organized for easy lookup by noise and model index
    """
    n_noises = len(noises)
    n_models = len(models)
    batch_size = noises[0].shape[0]

    entropies = []
    all_predictions = []  # [n_noises][n_models] = [batch_size, 1+dim]

    for noise in noises:
        # Collect predictions from all models for this noise
        model_predictions = []  # [n_models] predictions for this noise
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model.diffusion.sample(batch_size=batch_size, cond=cond, noise=noise)
                # pred: [batch_size, 1, 1+dim]
                model_predictions.append(pred[:, 0, :])  # [batch_size, 1+dim]

        all_predictions.append(model_predictions)  # Keep as list for individual access

        # Stack predictions for entropy computation: [n_models, batch_size, 1+dim]
        stacked_preds = torch.stack(model_predictions, dim=0)

        # Compute entropy for this noise
        entropy = compute_prediction_entropy(stacked_preds, dim)  # [batch_size]
        entropies.append(entropy)

    # Stack entropies: [n_noises, batch_size]
    entropies = torch.stack(entropies, dim=0)

    return entropies, all_predictions


def filter_and_weight_noises(noises: List[torch.Tensor],
                             entropies: torch.Tensor,
                             filter_ratio: float = 0.2,
                             use_weighting: bool = True) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Filter out low-quality noises and compute weights based on inverse entropy.
    
    Args:
        noises: List of noise tensors
        entropies: Tensor of shape [n_noises, batch_size] containing entropy values
        filter_ratio: Ratio of noises to filter out (0.0 to 1.0), e.g., 0.2 means remove top 20% high-entropy noises
        use_weighting: Whether to use inverse-entropy weighting (True) or uniform weights (False)
    
    Returns:
        filtered_noises: List of high-quality noise tensors
        weights: Normalized weights for each filtered noise [n_filtered]
        keep_indices: Indices of kept noises (for looking up cached predictions)
    """
    n_noises = len(noises)

    # Average entropy across batch for ranking noises
    mean_entropies = entropies.mean(dim=1)  # [n_noises]

    # Determine number of noises to keep
    n_keep = max(1, int(n_noises * (1 - filter_ratio)))

    # Get indices of top-k lowest entropy noises (highest quality)
    _, sorted_indices = torch.sort(mean_entropies)
    keep_indices = sorted_indices[:n_keep]

    # Filter noises
    filtered_noises = [noises[i] for i in keep_indices]
    kept_entropies = mean_entropies[keep_indices]

    if use_weighting:
        # Compute weights as inverse of entropy (add small epsilon for stability)
        epsilon = 1e-8
        inverse_entropy = 1.0 / (kept_entropies + epsilon)
        # Normalize weights to sum to 1
        weights = inverse_entropy / inverse_entropy.sum()
    else:
        # Uniform weights
        weights = torch.ones(n_keep, device=kept_entropies.device) / n_keep

    # 添加验证断言
    assert len(filtered_noises) == n_keep, f"Expected {n_keep} filtered noises, got {len(filtered_noises)}"
    assert weights.shape[0] == n_keep, f"Expected {n_keep} weights, got {weights.shape[0]}"
    assert torch.isclose(weights.sum(), torch.tensor(1.0)), f"Weights should sum to 1, got {weights.sum()}"
    
    return filtered_noises, weights, keep_indices


def weighted_ensemble_mean(predictions_temporal: List[torch.Tensor], predictions_spatial: List[torch.Tensor],
                           weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute weighted ensemble mean of predictions.
    
    Args:
        predictions_temporal: List of temporal predictions [n_samples x (batch_size, 1)]
        predictions_spatial: List of spatial predictions [n_samples x (batch_size, dim)]
        weights: Weights for each prediction [n_samples], should sum to 1
    
    Returns:
        weighted_temporal_mean: [batch_size, 1]
        weighted_spatial_mean: [batch_size, dim]
    """
    # Stack predictions: [n_samples, batch_size, ...]
    stacked_temporal = torch.stack(predictions_temporal, dim=0)  # [n_samples, batch_size, 1]
    stacked_spatial = torch.stack(predictions_spatial, dim=0)  # [n_samples, batch_size, dim]

    # Reshape weights for broadcasting: [n_samples, 1, 1]
    weights = weights.view(-1, 1, 1).to(stacked_temporal.device)

    # Weighted mean
    weighted_temporal_mean = (stacked_temporal * weights).sum(dim=0)  # [batch_size, 1]
    weighted_spatial_mean = (stacked_spatial * weights).sum(dim=0)  # [batch_size, dim]

    return weighted_temporal_mean, weighted_spatial_mean


def quality_filtered_ensemble(model,
                              auxiliary_models: Optional[List] = None,
                              batch_size: int = 16,
                              cond: torch.Tensor = None,
                              n_noises: int = 100,
                              dim: int = 2,
                              filter_ratio: float = 0.2,
                              use_weighting: bool = True,
                              device: torch.device = None,
                              main_model_index_in_auxiliary: Optional[int] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Main function for quality-filtered ensemble sampling.
    
    This function orchestrates the complete pipeline:
    1. Generate random noises
    2. Evaluate noise quality using auxiliary models (if provided)
    3. Filter out low-quality noises
    4. Generate predictions using the main model with filtered noises
       (OPTIMIZATION: reuse cached predictions if main model is in auxiliary list)
    5. Return predictions with quality-based weights
    
    Args:
        model: Main trained RF model for final predictions
        auxiliary_models: List of auxiliary models for quality evaluation (can include main model)
                         If None, falls back to naive ensemble (no filtering)
        batch_size: Batch size
        cond: Conditioning information
        n_noises: Number of initial noises to generate
        dim: Spatial dimension
        filter_ratio: Ratio of noises to filter out
        use_weighting: Whether to use inverse-entropy weighting
        device: Device for tensor operations
        main_model_index_in_auxiliary: If main model is in auxiliary_models, provide its index
                                       to reuse cached predictions and avoid redundant sampling.
                                       If None, main model will sample independently.
    
    Returns:
        sampled_temporal_all: List of temporal predictions from filtered noises
        sampled_spatial_all: List of spatial predictions from filtered noises
        weights: Normalized weights for weighted averaging
    """
    if device is None:
        device = next(model.parameters()).device

    # Get model structure info
    channels = model.diffusion.channels
    seq_length = model.diffusion.seq_length

    # Generate initial noises
    noises = generate_noises(n_noises, batch_size, channels, seq_length, device)

    if auxiliary_models is None or len(auxiliary_models) < 2:
        # Fallback to naive ensemble: no quality filtering
        filtered_noises = noises
        weights = torch.ones(n_noises, device=device) / n_noises
        keep_indices = None
        all_predictions = None
    else:
        # Evaluate noise quality using auxiliary models
        entropies, all_predictions = evaluate_noise_quality(auxiliary_models, noises, cond, dim)

        # Filter and weight noises
        filtered_noises, weights, keep_indices = filter_and_weight_noises(noises, entropies, filter_ratio, use_weighting)

    # Generate final predictions using main model with filtered noises
    sampled_temporal_all = []
    sampled_spatial_all = []

    # Check if we can reuse cached predictions from main model
    can_reuse = (main_model_index_in_auxiliary is not None and all_predictions is not None and keep_indices is not None)

    if can_reuse:
        # OPTIMIZATION: Reuse predictions from evaluate_noise_quality() instead of recomputing
        for noise_idx in keep_indices:
            noise_idx = noise_idx.item()  # Convert tensor to int
            # all_predictions[noise_idx][main_model_index_in_auxiliary] = [batch_size, 1+dim]
            cached_pred = all_predictions[noise_idx][main_model_index_in_auxiliary]
            sampled_temporal_all.append(cached_pred[:, :1])  # [batch_size, 1]
            sampled_spatial_all.append(cached_pred[:, -dim:])  # [batch_size, dim]
    else:
        # Standard path: sample main model for each filtered noise
        model.eval()
        with torch.no_grad():
            for noise in filtered_noises:
                sampled_seq = model.diffusion.sample(batch_size=batch_size, cond=cond, noise=noise)
                sampled_temporal_all.append(sampled_seq[:, 0, :1])
                sampled_spatial_all.append(sampled_seq[:, 0, -dim:])

    return sampled_temporal_all, sampled_spatial_all, weights


# def find_main_model_index(main_model, auxiliary_models: List) -> Optional[int]:
#     """
#     Find if main_model is in auxiliary_models list by comparing model identities.

#     NOTE: This function uses Python object identity comparison (is).
#     If models are loaded separately (even from the same checkpoint), they will be
#     different objects and this will return None. For path-based comparison,
#     use find_main_model_index_by_path() instead.

#     Args:
#         main_model: The main model for final predictions
#         auxiliary_models: List of auxiliary models for quality evaluation

#     Returns:
#         Index of main_model in auxiliary_models if found, None otherwise
#     """
#     if auxiliary_models is None:
#         return None

#     for idx, aux_model in enumerate(auxiliary_models):
#         if main_model is aux_model:  # Identity comparison
#             return idx

#     return None


def find_main_model_index_by_path(main_model_path: str, aux_model_paths: List[str]) -> Optional[int]:
    """
    Find if main model's checkpoint path is in auxiliary model paths list.
    
    This is the recommended way to detect if main model is in auxiliary models
    when models are loaded separately (which is the common case).
    
    Args:
        main_model_path: Path to the main model checkpoint
        aux_model_paths: List of paths to auxiliary model checkpoints
    
    Returns:
        Index of main_model_path in aux_model_paths if found, None otherwise
    """
    if aux_model_paths is None or main_model_path is None:
        return None

    # Normalize paths for comparison (handle different path separators, etc.)
    import os
    main_path_normalized = os.path.normpath(os.path.abspath(main_model_path))

    for idx, aux_path in enumerate(aux_model_paths):
        aux_path_normalized = os.path.normpath(os.path.abspath(aux_path))
        if main_path_normalized == aux_path_normalized:
            return idx

    return None

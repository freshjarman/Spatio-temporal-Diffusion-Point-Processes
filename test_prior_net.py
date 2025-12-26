"""
test_prior_net.py - Unit tests for PriorNet (HAP) and RectifiedFlow with adaptive prior

HAP (History Adaptive Prior) Design:
    - μ is supervised by ground truth via NLL Loss
    - σ represents calibrated prediction uncertainty
    - No KL regularization (we WANT informative prior)

Run: python test_prior_net.py
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DSTPP.PriorNet import PriorNet
from DSTPP.RectifiedFlow import RectifiedFlow
from DSTPP.RF_Diffusion import RF_Diffusion


def test_prior_net_shapes():
    """Test PriorNet input/output shapes."""
    print("\n" + "=" * 50)
    print("Test 1: PriorNet shape consistency")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters matching project config
    cond_dim = 64  # d_model
    output_dim = 3  # 1 (time) + 2 (location)
    batch_size = 16

    prior_net = PriorNet(
        cond_dim=cond_dim,
        output_dim=output_dim,
        hidden_dim=128,
    ).to(device)

    # Test with 2D input [batch, cond_dim]
    cond_2d = torch.randn(batch_size, cond_dim, device=device)
    mean, std = prior_net(cond_2d)

    assert mean.shape == (batch_size, output_dim), f"Expected mean shape {(batch_size, output_dim)}, got {mean.shape}"
    assert std.shape == (batch_size, output_dim), f"Expected std shape {(batch_size, output_dim)}, got {std.shape}"
    print(f"✓ 2D input: cond {cond_2d.shape} -> mean {mean.shape}, std {std.shape}")

    # Test with 3D input [batch, 1, cond_dim] (squeezed internally)
    cond_3d = torch.randn(batch_size, 1, cond_dim, device=device)
    mean, std = prior_net(cond_3d)

    assert mean.shape == (batch_size, output_dim), f"Expected mean shape {(batch_size, output_dim)}, got {mean.shape}"
    assert std.shape == (batch_size, output_dim), f"Expected std shape {(batch_size, output_dim)}, got {std.shape}"
    print(f"✓ 3D input: cond {cond_3d.shape} -> mean {mean.shape}, std {std.shape}")

    # Test sample
    z = prior_net.sample(cond_2d)
    assert z.shape == (batch_size, output_dim), f"Expected sample shape {(batch_size, output_dim)}, got {z.shape}"
    print(f"✓ Sample shape: {z.shape}")

    # Test log_prob
    log_prob = prior_net.log_prob(z, cond_2d)
    assert log_prob.shape == (batch_size, ), f"Expected log_prob shape {(batch_size,)}, got {log_prob.shape}"
    print(f"✓ Log prob shape: {log_prob.shape}")

    # Test log_prob_decomposed
    log_prob, log_prob_t, log_prob_s = prior_net.log_prob_decomposed(z, cond_2d)
    assert log_prob.shape == (batch_size, ), f"Expected log_prob shape {(batch_size,)}, got {log_prob.shape}"
    assert log_prob_t.shape == (batch_size, ), f"Expected log_prob_t shape {(batch_size,)}, got {log_prob_t.shape}"
    assert log_prob_s.shape == (batch_size, ), f"Expected log_prob_s shape {(batch_size,)}, got {log_prob_s.shape}"
    print(f"✓ Decomposed log prob shapes: total {log_prob.shape}, t {log_prob_t.shape}, s {log_prob_s.shape}")

    # Test NLL loss (HAP core feature)
    x_real = torch.randn(batch_size, output_dim, device=device)
    nll = prior_net.nll_loss(x_real, cond_2d)
    assert nll.shape == (batch_size, ), f"Expected nll shape {(batch_size,)}, got {nll.shape}"
    print(f"✓ NLL loss shape: {nll.shape}, mean: {nll.mean().item():.4f}")

    # Test NLL loss decomposed
    nll, nll_t, nll_s = prior_net.nll_loss_decomposed(x_real, cond_2d)
    assert nll.shape == (batch_size, ), f"Expected nll shape {(batch_size,)}, got {nll.shape}"
    print(f"✓ Decomposed NLL: total {nll.mean():.4f}, t {nll_t.mean():.4f}, s {nll_s.mean():.4f}")

    print("\n✅ Test 1 PASSED: All shapes correct")
    return True


def test_nll_loss_behavior():
    """Test that NLL loss behaves correctly (HAP core test)."""
    print("\n" + "=" * 50)
    print("Test 2: NLL loss behavior (HAP core)")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prior_net = PriorNet(
        cond_dim=64,
        output_dim=3,
        hidden_dim=128,
    ).to(device)

    # Random conditioning
    cond = torch.randn(100, 64, device=device)
    mean, std = prior_net(cond)

    # When x_real == mean, NLL should be smaller
    nll_at_mean = prior_net.nll_loss(mean, cond)

    # When x_real is far from mean, NLL should be larger
    x_far = mean + 5.0 * std
    nll_far = prior_net.nll_loss(x_far, cond)

    print(f"NLL at mean: {nll_at_mean.mean().item():.4f}")
    print(f"NLL far from mean: {nll_far.mean().item():.4f}")

    assert nll_far.mean() > nll_at_mean.mean(), "NLL should be larger when x is far from mean"
    print("✓ NLL increases when x is far from mean")

    # Test that NLL decomposes correctly
    nll_total, nll_t, nll_s = prior_net.nll_loss_decomposed(mean, cond)
    reconstructed = nll_t + nll_s  # Should roughly equal nll_total (minus constant terms)
    print(f"NLL decomposition check: total={nll_total.mean():.4f}, t+s≈{reconstructed.mean():.4f}")

    print("\n✅ Test 2 PASSED: NLL loss behaves correctly")
    return True


def test_std_constraints():
    """Test that std is constrained to [min_std, max_std]."""
    print("\n" + "=" * 50)
    print("Test 3: PriorNet std constraints")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    min_std, max_std = 0.1, 2.0
    prior_net = PriorNet(
        cond_dim=64,
        output_dim=3,
        hidden_dim=128,
        min_std=min_std,
        max_std=max_std,
    ).to(device)

    # Test with extreme inputs
    cond = torch.randn(100, 64, device=device) * 10  # Large random values
    _, std = prior_net(cond)

    std_min = std.min().item()
    std_max = std.max().item()

    print(f"Std range: [{std_min:.4f}, {std_max:.4f}]")
    print(f"Expected: [{min_std}, {max_std}]")

    assert std_min >= min_std - 1e-6, f"Std should be >= {min_std}, got {std_min}"
    assert std_max <= max_std + 1e-6, f"Std should be <= {max_std}, got {std_max}"

    print("\n✅ Test 3 PASSED: Std is properly constrained")
    return True


def test_rectified_flow_with_prior_net():
    """Test RectifiedFlow integration with PriorNet."""
    print("\n" + "=" * 50)
    print("Test 4: RectifiedFlow + PriorNet integration")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    batch_size = 8
    seq_length = 3  # 1 (time) + 2 (location)
    cond_dim = 64
    timesteps = 50

    # Create RF_Diffusion
    rf_diffusion = RF_Diffusion(
        n_steps=timesteps,
        dim=seq_length,
        condition=True,
        cond_dim=cond_dim,
    ).to(device)

    # Create PriorNet
    prior_net = PriorNet(
        cond_dim=cond_dim,
        output_dim=seq_length,
        hidden_dim=128,
    ).to(device)

    # Create RectifiedFlow with PriorNet (HAP)
    rf = RectifiedFlow(
        rf_diffusion,
        seq_length=seq_length,
        timesteps=timesteps,
        sampling_timesteps=10,
        prior_net=prior_net,
        prior_loss_weight=0.1,  # HAP: renamed from kl_weight
    ).to(device)

    print(f"HAP/PriorNet enabled: {rf.prior_net is not None}")

    # Create mock data
    # x: [batch, 1, seq_length] in [0, 1]
    x = torch.rand(batch_size, 1, seq_length, device=device)
    # cond: [batch, 1, 3*cond_dim] (enc_temporal, enc_spatial, enc_joint)
    cond = torch.randn(batch_size, 1, 3 * cond_dim, device=device)

    print(f"Input x shape: {x.shape}")
    print(f"Input cond shape: {cond.shape}")

    # Test forward
    print("\n--- Forward pass (training) ---")
    loss = rf(x, cond)
    print(f"Loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss should be finite"

    # Test forward_with_details
    print("\n--- Forward with details ---")
    loss, loss_t, loss_s, prior_loss = rf.forward_with_details(x, cond)
    print(f"Total loss: {loss.item():.4f}")
    print(f"FM Temporal loss: {loss_t.item():.4f}")
    print(f"FM Spatial loss: {loss_s.item():.4f}")
    print(f"Prior NLL loss (HAP): {prior_loss.item():.4f}")
    assert torch.isfinite(loss), "Loss should be finite"
    assert prior_loss.item() > 0, "Prior NLL loss should be positive when using PriorNet"

    # Test sample
    print("\n--- Sampling ---")
    with torch.no_grad():
        samples = rf.sample(batch_size=batch_size, cond=cond, steps=5)
    print(f"Samples shape: {samples.shape}")
    print(f"Samples range: [{samples.min().item():.4f}, {samples.max().item():.4f}]")
    assert samples.shape == (batch_size, 1, seq_length), f"Expected {(batch_size, 1, seq_length)}, got {samples.shape}"
    # Note: Samples may not be in [0, 1] for untrained model, just check finite
    assert torch.isfinite(samples).all(), "Samples should be finite"

    print("\n✅ Test 4 PASSED: RectifiedFlow + HAP integration works")
    return True


def test_rectified_flow_without_prior_net():
    """Test RectifiedFlow without PriorNet (standard Gaussian)."""
    print("\n" + "=" * 50)
    print("Test 5: RectifiedFlow without PriorNet (baseline)")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 8
    seq_length = 3
    cond_dim = 64
    timesteps = 50

    rf_diffusion = RF_Diffusion(
        n_steps=timesteps,
        dim=seq_length,
        condition=True,
        cond_dim=cond_dim,
    ).to(device)

    # No PriorNet (standard Gaussian prior)
    rf = RectifiedFlow(
        rf_diffusion,
        seq_length=seq_length,
        timesteps=timesteps,
        sampling_timesteps=10,
        prior_net=None,
        prior_loss_weight=0.1,  # Has no effect when prior_net is None
    ).to(device)

    print(f"HAP/PriorNet enabled: {rf.prior_net is not None}")

    x = torch.rand(batch_size, 1, seq_length, device=device)
    cond = torch.randn(batch_size, 1, 3 * cond_dim, device=device)

    # Test forward
    loss = rf(x, cond)
    print(f"Loss: {loss.item():.4f}")

    # Test forward_with_details
    loss, loss_t, loss_s, prior_loss = rf.forward_with_details(x, cond)
    print(f"Prior NLL loss (should be 0): {prior_loss.item():.6f}")
    assert prior_loss.item() == 0.0, "Prior NLL loss should be 0 when not using PriorNet"

    # Test sample
    with torch.no_grad():
        samples = rf.sample(batch_size=batch_size, cond=cond, steps=5)
    print(f"Samples shape: {samples.shape}")

    print("\n✅ Test 5 PASSED: RectifiedFlow works without PriorNet")
    return True


def test_gradient_flow():
    """Test that gradients flow through PriorNet (HAP training)."""
    print("\n" + "=" * 50)
    print("Test 6: Gradient flow through PriorNet (HAP training)")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4
    seq_length = 3
    cond_dim = 64

    rf_diffusion = RF_Diffusion(
        n_steps=50,
        dim=seq_length,
        condition=True,
        cond_dim=cond_dim,
    ).to(device)

    prior_net = PriorNet(
        cond_dim=cond_dim,
        output_dim=seq_length,
        hidden_dim=64,
    ).to(device)

    rf = RectifiedFlow(
        rf_diffusion,
        seq_length=seq_length,
        timesteps=50,
        prior_net=prior_net,
        prior_loss_weight=0.5,  # Larger weight to ensure gradients are significant
    ).to(device)

    # Record initial params
    initial_params = {name: p.clone() for name, p in prior_net.named_parameters()}

    # Optimizer
    optimizer = torch.optim.Adam(rf.parameters(), lr=1e-3)

    # Training step
    x = torch.rand(batch_size, 1, seq_length, device=device)
    cond = torch.randn(batch_size, 1, 3 * cond_dim, device=device)

    optimizer.zero_grad()
    loss = rf(x, cond)
    loss.backward()
    optimizer.step()

    # Check that PriorNet params have changed
    params_changed = False
    for name, p in prior_net.named_parameters():
        if not torch.allclose(p, initial_params[name]):
            params_changed = True
            break

    print(f"PriorNet parameters updated: {params_changed}")
    assert params_changed, "PriorNet parameters should be updated during training"

    print("\n✅ Test 6 PASSED: Gradients flow through PriorNet")
    return True


def test_nll_calculation():
    """Test NLL calculation with adaptive prior (HAP)."""
    print("\n" + "=" * 50)
    print("Test 7: NLL calculation with HAP adaptive prior")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4
    seq_length = 3
    cond_dim = 64

    rf_diffusion = RF_Diffusion(
        n_steps=50,
        dim=seq_length,
        condition=True,
        cond_dim=cond_dim,
    ).to(device)

    prior_net = PriorNet(
        cond_dim=cond_dim,
        output_dim=seq_length,
        hidden_dim=64,
    ).to(device)

    rf = RectifiedFlow(
        rf_diffusion,
        seq_length=seq_length,
        timesteps=50,
        prior_net=prior_net,
        prior_loss_weight=0.1,  # HAP
    ).to(device)

    x = torch.rand(batch_size, 1, seq_length, device=device)
    cond = torch.randn(batch_size, 1, 3 * cond_dim, device=device)

    # Use euler method for faster testing
    with torch.no_grad():
        nll, nll_t, nll_s = rf.calculate_neg_log_likelihood(x, cond, method='euler')

    print(f"NLL: {nll:.4f}")
    print(f"NLL temporal: {nll_t:.4f}")
    print(f"NLL spatial: {nll_s:.4f}")

    # NLL should be finite
    assert not (nll != nll), "NLL should not be NaN"  # Check for NaN

    print("\n✅ Test 7 PASSED: NLL calculation works")
    return True


def test_prior_learns_target():
    """Test that PriorNet can learn to predict targets via NLL loss (HAP validation)."""
    print("\n" + "=" * 50)
    print("Test 8: PriorNet learns to predict targets (HAP validation)")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prior_net = PriorNet(
        cond_dim=64,
        output_dim=3,
        hidden_dim=128,
    ).to(device)

    optimizer = torch.optim.Adam(prior_net.parameters(), lr=1e-3)

    # Simple mapping task: cond -> x_real = linear_transform(cond)
    W = torch.randn(3, 64, device=device) * 0.1

    initial_nll = None
    for epoch in range(100):
        cond = torch.randn(64, 64, device=device)
        x_real = cond @ W.T  # [64, 3]

        optimizer.zero_grad()
        nll = prior_net.nll_loss(x_real, cond).mean()
        nll.backward()
        optimizer.step()

        if epoch == 0:
            initial_nll = nll.item()

    final_nll = nll.item()
    print(f"Initial NLL: {initial_nll:.4f}")
    print(f"Final NLL: {final_nll:.4f}")
    print(f"Improvement: {(initial_nll - final_nll) / initial_nll * 100:.1f}%")

    assert final_nll < initial_nll, "NLL should decrease during training"

    # Check prediction accuracy
    with torch.no_grad():
        cond_test = torch.randn(100, 64, device=device)
        x_test = cond_test @ W.T
        mean, std = prior_net(cond_test)
        mae = (mean - x_test).abs().mean().item()
        print(f"Final MAE: {mae:.4f}")
        print(f"Final mean std (uncertainty): {std.mean().item():.4f}")

    print("\n✅ Test 8 PASSED: PriorNet learns to predict targets via NLL")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running HAP (History Adaptive Prior) Unit Tests")
    print("=" * 60)

    tests = [
        test_prior_net_shapes,
        test_nll_loss_behavior,
        test_std_constraints,
        test_rectified_flow_with_prior_net,
        test_rectified_flow_without_prior_net,
        test_gradient_flow,
        test_nll_calculation,
        test_prior_learns_target,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

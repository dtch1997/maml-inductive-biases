"""
Sanity check for the gradient mask MAML approach.

Uses a toy 2D problem instead of a neural network:
  - theta = [a, b], starts at zero
  - Inner loss (the "attack"): push theta toward target [1, 1]
      L_inner = (a - 1)^2 + (b - 1)^2
  - Outer loss: penalize dimension 0 only (we want to block learning in dim 0)
      L_outer = a^2  (lower = better, meaning a should stay near 0)

The mask phi = [phi_a, phi_b] gates inner gradients:
  g_eff = sigmoid(phi) * g

Expected behavior:
  - phi_a should decrease (block dim 0 gradient, keep a near 0)
  - phi_b should stay high or increase (allow dim 1 gradient, let b learn)

This tests:
  1. The phi gradient formula is correct
  2. The sign is right (phi decreases for dimensions that hurt the outer loss)
  3. The mask actually learns to be selective

Usage:
    python3 test_mask_gradient.py
"""

import torch


def test_single_step_gradient():
    """Verify phi gradient formula against autograd for a single inner step."""
    print("=== Test 1: Verify gradient formula (single inner step) ===")

    lr = 0.1
    phi = torch.tensor([3.0, 3.0], requires_grad=True)  # sigmoid ≈ 0.95

    # Simulate one inner step
    theta_0 = torch.zeros(2)
    target = torch.tensor([1.0, 1.0])

    # Inner gradient: d/dtheta (theta - target)^2 = 2(theta - target)
    g = 2 * (theta_0 - target)  # = [-2, -2]

    # Masked update: theta* = theta_0 - lr * sigmoid(phi) * g
    theta_star = theta_0 - lr * torch.sigmoid(phi) * g

    # Outer loss: only penalize dim 0
    outer_loss = theta_star[0] ** 2

    # Autograd gradient
    outer_loss.backward()
    autograd_grad = phi.grad.clone()

    # Manual formula:
    # d(outer_loss)/d(phi_i) = d(outer_loss)/d(theta*_i) * (-lr * sigmoid'(phi_i) * g_i)
    with torch.no_grad():
        d_outer_d_theta_star = torch.tensor([2 * theta_star[0].item(), 0.0])
        sig = torch.sigmoid(phi)
        sig_deriv = sig * (1 - sig)
        manual_grad = d_outer_d_theta_star * (-lr * sig_deriv * g)

    print(f"  theta* = {theta_star.detach().tolist()}")
    print(f"  outer_loss = {outer_loss.item():.6f}")
    print(f"  autograd grad(phi) = {autograd_grad.tolist()}")
    print(f"  manual   grad(phi) = {manual_grad.tolist()}")
    print(f"  match: {torch.allclose(autograd_grad, manual_grad, atol=1e-6)}")

    # Check sign: phi_a grad should be negative (we want to DECREASE phi_a to block dim 0)
    # Why? Outer loss = a^2. Inner loop pushes a away from 0. Blocking dim 0 keeps a near 0.
    # So decreasing phi_a (closing the gate) reduces outer loss → gradient should be negative... wait:
    # Actually, we need to check. If a > 0 after inner loop, outer_loss = a^2, and reducing phi_a
    # would reduce a, reducing outer loss. So d(loss)/d(phi_a) > 0? Let me check:
    #
    # theta* = 0 - 0.1 * sigmoid(phi) * (-2) = 0.2 * sigmoid(phi)
    # a* = 0.2 * sigmoid(phi_a) > 0
    # outer_loss = (0.2 * sigmoid(phi_a))^2
    # d(loss)/d(phi_a) = 2 * 0.2 * sigmoid(phi_a) * 0.2 * sigmoid'(phi_a) > 0
    # So gradient is POSITIVE → Adam subtracts → phi_a decreases → gate closes. Correct!
    print(f"  phi_a grad sign: {'positive (correct)' if autograd_grad[0] > 0 else 'NEGATIVE (wrong!)'}")
    print(f"  phi_b grad sign: {'zero (correct)' if abs(autograd_grad[1]) < 1e-6 else f'nonzero={autograd_grad[1]:.6f}'}")
    print()


def test_multi_step_gradient():
    """Verify the first-order approximation for multiple inner steps."""
    print("=== Test 2: First-order approx vs autograd (3 inner steps) ===")

    lr = 0.1
    inner_steps = 3

    # --- Autograd (exact) ---
    phi_exact = torch.tensor([3.0, 3.0], requires_grad=True)
    theta = torch.zeros(2)
    target = torch.tensor([1.0, 1.0])

    for _ in range(inner_steps):
        g = 2 * (theta - target)
        theta = theta - lr * torch.sigmoid(phi_exact) * g

    outer_loss = theta[0] ** 2
    outer_loss.backward()
    exact_grad = phi_exact.grad.clone()

    # --- First-order approximation ---
    with torch.no_grad():
        phi_fo = torch.tensor([3.0, 3.0])
        theta_fo = torch.zeros(2)
        acc_g = torch.zeros(2)

        for _ in range(inner_steps):
            g = 2 * (theta_fo - target)
            theta_fo = theta_fo - lr * torch.sigmoid(phi_fo) * g
            acc_g += g

        # Outer gradient w.r.t. theta*
        d_outer = torch.tensor([2 * theta_fo[0].item(), 0.0])

        sig = torch.sigmoid(phi_fo)
        sig_deriv = sig * (1 - sig)
        fo_grad = d_outer * (-lr * sig_deriv * acc_g)

    print(f"  theta* (exact)  = {theta.detach().tolist()}")
    print(f"  theta* (approx) = {theta_fo.tolist()}")
    print(f"  exact   grad(phi) = {exact_grad.tolist()}")
    print(f"  1st-ord grad(phi) = {fo_grad.tolist()}")
    print(f"  relative error: {((exact_grad - fo_grad).norm() / exact_grad.norm()).item():.4f}")
    print(f"  same sign: {(exact_grad.sign() == fo_grad.sign()).all().item()}")
    print()


def test_mask_learns():
    """Run full optimization loop, verify mask becomes selective."""
    print("=== Test 3: Does the mask learn to be selective? ===")

    lr = 0.1
    inner_steps = 3
    outer_lr = 0.1
    n_outer_steps = 200

    phi = torch.tensor([3.0, 3.0])  # sigmoid ≈ 0.95
    target = torch.tensor([1.0, 1.0])

    phi_optimizer = torch.optim.Adam([phi], lr=outer_lr)
    phi.requires_grad = True

    for step in range(n_outer_steps):
        # Inner loop: push theta toward target with masked gradients
        theta = torch.zeros(2)
        acc_g = torch.zeros(2)

        with torch.no_grad():
            for _ in range(inner_steps):
                g = 2 * (theta - target)
                theta = theta - lr * torch.sigmoid(phi) * g
                acc_g += g

        # Outer loss: penalize dim 0 only
        # Use the exact autograd version for optimization
        phi_for_grad = phi.detach().clone().requires_grad_(True)
        theta_grad = torch.zeros(2)
        for _ in range(inner_steps):
            g = 2 * (theta_grad - target)
            theta_grad = theta_grad - lr * torch.sigmoid(phi_for_grad) * g
        outer_loss = theta_grad[0] ** 2
        outer_loss.backward()

        phi_optimizer.zero_grad()
        phi.grad = phi_for_grad.grad.clone()
        phi_optimizer.step()

        if step % 50 == 0 or step == n_outer_steps - 1:
            sig = torch.sigmoid(phi)
            print(
                f"  step {step:4d} | loss={outer_loss.item():.6f} "
                f"sigmoid(phi)=[{sig[0].item():.4f}, {sig[1].item():.4f}] "
                f"theta*=[{theta[0].item():.4f}, {theta[1].item():.4f}]"
            )

    sig_final = torch.sigmoid(phi)
    print(f"\n  Final mask: sigmoid(phi) = {sig_final.tolist()}")
    print(f"  dim 0 (should be blocked): {sig_final[0].item():.4f}")
    print(f"  dim 1 (should be open):    {sig_final[1].item():.4f}")
    selective = sig_final[0] < 0.3 and sig_final[1] > 0.7
    print(f"  Selective: {'YES' if selective else 'NO'}")
    print()


def test_fo_mask_learns():
    """Same as test 3, but using the first-order approximation for phi gradients.
    This is what the actual training script does."""
    print("=== Test 4: Does the first-order approximation also learn? ===")

    lr = 0.1
    inner_steps = 3
    outer_lr = 0.1
    n_outer_steps = 200

    phi = torch.tensor([3.0, 3.0], requires_grad=True)
    target = torch.tensor([1.0, 1.0])

    phi_optimizer = torch.optim.Adam([phi], lr=outer_lr)

    for step in range(n_outer_steps):
        # Inner loop (no grad tracking, like the real script)
        theta = torch.zeros(2)
        acc_g = torch.zeros(2)

        with torch.no_grad():
            for _ in range(inner_steps):
                g = 2 * (theta - target)
                theta = theta - lr * torch.sigmoid(phi) * g
                acc_g += g

        # Outer gradient w.r.t. theta*
        d_outer_d_theta = torch.tensor([2 * theta[0].item(), 0.0])

        # First-order phi gradient
        phi_optimizer.zero_grad()
        with torch.no_grad():
            sig = torch.sigmoid(phi)
            sig_deriv = sig * (1 - sig)
            phi.grad = d_outer_d_theta * (-lr * sig_deriv * acc_g)

        phi_optimizer.step()

        if step % 50 == 0 or step == n_outer_steps - 1:
            sig = torch.sigmoid(phi)
            outer_loss = theta[0] ** 2
            print(
                f"  step {step:4d} | loss={outer_loss.item():.6f} "
                f"sigmoid(phi)=[{sig[0].item():.4f}, {sig[1].item():.4f}] "
                f"theta*=[{theta[0].item():.4f}, {theta[1].item():.4f}]"
            )

    sig_final = torch.sigmoid(phi)
    print(f"\n  Final mask: sigmoid(phi) = {sig_final.tolist()}")
    print(f"  dim 0 (should be blocked): {sig_final[0].item():.4f}")
    print(f"  dim 1 (should be open):    {sig_final[1].item():.4f}")
    selective = sig_final[0] < 0.3 and sig_final[1] > 0.7
    print(f"  Selective: {'YES' if selective else 'NO'}")


if __name__ == "__main__":
    test_single_step_gradient()
    test_multi_step_gradient()
    test_mask_learns()
    test_fo_mask_learns()

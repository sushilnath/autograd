import numpy as np
from autograd_forward import GradNode, sin, cos, exp, log

def run_tests():
    print("Running Verification Tests...\n")
    tolerance = 1e-6

    # Simple tests
    # f(x) = x at x = 7
    x = GradNode(7.0)
    y = x
    assert abs(y.x - 7.0) < tolerance, f"Identity Value failed: Got {y.x}"
    assert abs(y.dx - 1.0) < tolerance, f"Identity Deriv failed: Got {y.dx}"
    print("Test 0 (Identity) Passed")

    # --- Test 0.1: Neg/Power/Mul/Add ---
    # f(x) = -x^2 + 3x at x = 5
    # Analytical f(5) = -25 + 15 = -10
    # Analytical f'(x) = -2x + 3 -> f'(5) = -10 + 3 = -7
    x = GradNode(5.0)
    y = -x**2 + 3 * x
    assert abs(y.x - (-25 + 15)) < tolerance, f"Neg/Power/Mul/Add Value failed: Got {y.x}"
    assert abs(y.dx - (-2 * 5 + 3)) < tolerance, f"Neg/Power/Mul/Add Deriv failed: Got {y.dx}"
    print("Test 0.1 (Neg/Power/Mul/Add) Passed")

    #Test 0.2 for __rmul__
    # --- Test 0.2: Reverse Multiplication ---
    # f(x) = 4 * x at x = 3
    # Analytical f(3) = 12
    # Analytical f'(x) = 4 -> f'(3) = 4
    x = GradNode(3.0)
    y = 4 * x
    assert abs(y.x - 12.0) < tolerance, f"Reverse Multiplication Value failed: Got {y.x}"
    assert abs(y.dx - 4.0) < tolerance, f"Reverse Multiplication Deriv failed: Got {y.dx}"
    print("Test 0.2 (Reverse Multiplication) Passed")

    # Test very large values
    # --- Test 0.3: Large Values ---
    # f(x) = x^2 + 2x + 1 at x = 1e6
    # Analytical f(1e6) = (1e6)^2 + 2*(1e6) + 1 = 1.000002e12
    # Analytical f'(x) = 2x + 2 -> f'(1e6) = 2*(1e6) + 2 = 2000002
    x = GradNode(1e6)
    y = x**2 + 2*x + 1
    assert abs(y.x - 1.000002e12) < tolerance * 1e12, f"Large Values Value failed: Got {y.x}"
    assert abs(y.dx - 2000002.0) < tolerance * 1e6, f"Large Values Deriv failed: Got {y.dx}"
    print("Test 0.3 (Large Values) Passed")

    # Test polynomial division
    # f = (x^2 + 1) / (x + 1) at x = 3
    # Analytical f(3) = (9 + 1) / (3 + 1) = 10 / 4 = 2.5
    # Analytical f'(x) = [(2x)(x + 1) - (x^2 + 1)(1)] / (x + 1)^2
    # f'(3) = [(6)(4) - (10)(1)] / (4^2) = (24 - 10) / 16 = 14 / 16 = 0.875
    x = GradNode(3.0)
    y = (x**2 + 1) / (x + 1)
    assert abs(y.x - 2.5) < tolerance, f"Poly Division Value failed: Got {y.x}"
    assert abs(y.dx - 0.875) < tolerance, f"Poly Division Deriv failed: Got {y.dx}"
    print("Test  0.4 (Polynomial Division) Passed")


    # --- Test 1: Polynomial ---
    # f(x) = x^3 + 2x + 5 at x = 2
    # Analytical f(2) = 8 + 4 + 5 = 17
    # Analytical f'(x) = 3x^2 + 2 -> f'(2) = 12 + 2 = 14
    x = GradNode(2.0)
    y = x**3 + x*2 + 5
    
    assert abs(y.x - 17.0) < tolerance, f"Poly Value failed: Got {y.x}"
    assert abs(y.dx - 14.0) < tolerance, f"Poly Deriv failed: Got {y.dx}"
    print("Test 1 (Polynomial) Passed")

    # --- Test 2: Quotient Rule ---
    # f(x) = x / (x + 1) at x = 1
    # Analytical f(1) = 0.5
    # Analytical f'(x) = 1/(x+1)^2 -> f'(1) = 1/4 = 0.25
    x = GradNode(1.0)
    y = x / (x + 1)

    assert abs(y.x - 0.5) < tolerance, f"Quotient Value failed: Got {y.x}"
    assert abs(y.dx - 0.25) < tolerance, f"Quotient Deriv failed: Got {y.dx}"
    print("Test 2 (Quotient Rule) Passed")

    # --- Test 3: Chain Rule with Trig ---
    # f(x) = sin(x^2) at x = sqrt(pi/2) approx 1.2533
    # Analytical f(x) = sin(pi/2) = 1.0
    # Analytical f'(x) = cos(x^2) * 2x -> cos(pi/2) * ... = 0.0
    val = np.sqrt(np.pi / 2)
    x = GradNode(val)
    y = sin(x**2)

    assert abs(y.x - 1.0) < tolerance, f"Chain Value failed: Got {y.x}"
    assert abs(y.dx - 0.0) < tolerance, f"Chain Deriv failed: Got {y.dx}"
    print("Test 3 (Chain Rule/Trig) Passed")

    # --- Test 4: Product + Log ---
    # f(x) = x * log(x) at x = e
    # Analytical f(e) = e * 1 = e
    # Analytical f'(x) = 1*log(x) + x*(1/x) = log(x) + 1 -> f'(e) = 1 + 1 = 2
    x = GradNode(np.e)
    y = x * log(x)

    assert abs(y.x - np.e) < tolerance, f"Log/Product Value failed: Got {y.x}"
    assert abs(y.dx - 2.0) < tolerance, f"Log/Product Deriv failed: Got {y.dx}"
    print("Test 4 (Log/Product) Passed")

    print("\nAll correctness checks passed!")

if __name__ == "__main__":
    run_tests()

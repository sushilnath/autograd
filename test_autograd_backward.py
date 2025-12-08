import numpy as np
from autograd_backward import Node, sin, cos, log

def check_grad(node, expected_val, test_name):
    try:
        assert abs(node.grad - expected_val) < 1e-6
        print(f"[PASS] {test_name}")
    except AssertionError:
        print(f"[FAIL] {test_name} -> Expected {expected_val}, Got {node.grad}")
    except Exception as e:
        print(f"[ERR ] {test_name} -> {e}")

def run_comprehensive_tests():
    print("--- Starting 20-Point Test Suite ---\n")

    # 1. Basic Addition
    # y = x + 4 -> dy/dx = 1
    x = Node(10.0)
    y = x + 4
    y.backprop()
    check_grad(x, 1.0, "Basic Addition")

    # 2. Basic Multiplication
    # y = x * 3 -> dy/dx = 3
    x = Node(5.0)
    y = x * 3
    y.backprop()
    check_grad(x, 3.0, "Basic Multiplication")

    # 3. Basic Subtraction (via Addition of negative)
    # y = x + (-2) -> dy/dx = 1
    x = Node(5.0)
    y = x + Node(-2.0)
    y.backprop()
    check_grad(x, 1.0, "Basic Subtraction")

    # 4. Basic Division
    # y = x / 2 -> dy/dx = 0.5
    x = Node(10.0)
    y = x / 2
    y.backprop()
    check_grad(x, 0.5, "Basic Division")

    # 5. Reverse Division (Constant / Node)
    # y = 6 / x -> y = 6 * x^-1 -> dy/dx = -6 / x^2
    # at x = 2: -6 / 4 = -1.5
    x = Node(2.0)
    y = 6 / x
    y.backprop()
    check_grad(x, -1.5, "Reverse Division (6 / x)")

    # 6. Power Rule
    # y = x ^ 3 -> dy/dx = 3x^2
    # at x = 3: 3 * 9 = 27
    x = Node(3.0)
    y = x ** 3
    y.backprop()
    check_grad(x, 27.0, "Power Rule")

    # 7. Negation
    # y = -x -> dy/dx = -1
    x = Node(5.0)
    y = -x
    y.backprop()
    check_grad(x, -1.0, "Negation")

    # 8. Sine
    # y = sin(x) -> dy/dx = cos(x)
    # at x = 0: cos(0) = 1
    x = Node(0.0)
    y = sin(x)
    y.backprop()
    check_grad(x, 1.0, "Sine Function")

    # 9. Cosine
    # y = cos(x) -> dy/dx = -sin(x)
    # at x = pi/2: -sin(pi/2) = -1
    x = Node(np.pi / 2)
    y = cos(x)
    y.backprop()
    check_grad(x, -1.0, "Cosine Function")

    # 10. Logarithm (Natural Log)
    # y = log(x) -> dy/dx = 1/x
    # at x = 2: 0.5
    x = Node(2.0)
    y = log(x)
    y.backprop()
    check_grad(x, 0.5, "Logarithm")

    # 11. The "Diamond" (Variable Reuse - Sum)
    # y = x + x -> y = 2x -> dy/dx = 2
    x = Node(3.0)
    y = x + x
    y.backprop()
    check_grad(x, 2.0, "Diamond Sum (Reuse Variable)")

    # 12. Variable Reuse - Multiplication
    # y = x * x -> y = x^2 -> dy/dx = 2x
    # at x = 4: 8
    x = Node(4.0)
    y = x * x
    y.backprop()
    check_grad(x, 8.0, "Self Multiplication")

    # 13. Complex Chain Rule
    # y = sin(x^2) -> dy/dx = cos(x^2) * 2x
    # at x = sqrt(pi): cos(pi) * 2sqrt(pi) = -1 * 2sqrt(pi)
    val = np.sqrt(np.pi)
    x = Node(val)
    y = sin(x ** 2)
    y.backprop()
    expected = -2 * val
    check_grad(x, expected, "Chain Rule: sin(x^2)")

    # 14. Reverse Multiplication (Constant * Node)
    # y = 3 * x -> dy/dx = 3
    x = Node(10.0)
    y = 3 * x
    y.backprop()
    check_grad(x, 3.0, "__rmul__ (3 * x)")

    # 15. Reverse Addition (Constant + Node)
    # y = 10 + x -> dy/dx = 1
    x = Node(5.0)
    y = 10 + x
    y.backprop()
    check_grad(x, 1.0, "__radd__ (10 + x)")

    # 16. Zero Gradient Check
    # y = x * 0 -> dy/dx = 0
    x = Node(5.0)
    y = x * 0
    y.backprop()
    check_grad(x, 0.0, "Multiplication by Zero")

    # 17. The "Self-Canceling" Graph
    # y = x - x -> y = 0 -> dy/dx = 0
    x = Node(5.0)
    y = x + (-x)
    y.backprop()
    check_grad(x, 0.0, "Self Canceling (x - x)")

    # 18. Multi-Variable (Partial Derivatives)
    # f(a, b) = a * b
    # df/da = b, df/db = a
    a = Node(2.0)
    b = Node(3.0)
    y = a * b
    y.backprop()
    check_grad(a, 3.0, "Multi-Var Partial (a)")
    check_grad(b, 2.0, "Multi-Var Partial (b)")

    # 19. Deep Recursion / Long Chain
    # y = (((x + 1) + 1) ... + 1)  (100 times)
    # dy/dx = 1
    x = Node(0.0)
    y = x
    for _ in range(100):
        y = y + 1
    y.backprop()
    check_grad(x, 1.0, "Deep Recursion (Chain Length 100)")

    # 20. The "Kitchen Sink" Equation
    # y = x^2 + sin(x) + log(x) + 5/x
    # dy/dx = 2x + cos(x) + 1/x - 5/x^2
    # at x = 1:
    # dy/dx = 2(1) + cos(1) + 1 - 5 = -2 + 0.5403 = -1.4597
    x = Node(1.0)
    y = x**2 + sin(x) + log(x) + 5/x
    y.backprop()
    expected = 2 + np.cos(1) + 1 - 5
    check_grad(x, expected, "Kitchen Sink Equation")

    # 21. ReLU (Active)
    # y = relu(x) where x > 0. dy/dx should be 1.
    x = Node(5.0)
    y = x.relu()
    y.backprop()
    check_grad(x, 1.0, "ReLU Active (Positive Input)")

    # 22. ReLU (Inactive)
    # y = relu(x) where x <= 0. dy/dx should be 0.
    x = Node(-5.0)
    y = x.relu()
    y.backprop()
    check_grad(x, 0.0, "ReLU Inactive (Negative Input)")

    # 23. ReLU Chain Rule
    # y = relu(x^2) at x = -2.
    # Inside is 4 (positive), so ReLU passes gradient.
    # dy/dx = 1 * d(x^2)/dx = 2x = -4
    x = Node(-2.0)
    y = (x**2).relu()
    y.backprop()
    check_grad(x, -4.0, "ReLU Chain Rule")

def test_multiple_inputs():
    print("--- Testing Multiple Inputs ---")
    
    # 1. Define Multiple Inputs
    x = Node(3.0) # Input 1
    y = Node(4.0) # Input 2
    z = Node(5.0) # Input 3
    
    # 2. Combine them in one graph
    f = x**2 + y*z
    
    # 3. Call backprop on the single output
    f.backprop()
    
    # 4. Check gradients on ALL inputs
    print(f"Output f: {f.x}") # 9 + 20 = 29
    
    print(f"x.grad (Expect 2x = 6.0): {x.grad}")
    print(f"y.grad (Expect z  = 5.0): {y.grad}")
    print(f"z.grad (Expect y  = 4.0): {z.grad}")
    
    assert x.grad == 6.0
    assert y.grad == 5.0
    assert z.grad == 4.0
    print("Success!\n")

if __name__ == "__main__":
    run_comprehensive_tests()

    test_multiple_inputs()
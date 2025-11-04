import numpy as np


# def op_shift(S: np.ndarray, offsets: np.ndarray, target_dim: int) -> np.ndarray:
#     """
#     Shift the shapelets according to given offsets.
#
#     Args:
#         S: Shapelet matrix of shape (K, q)
#         offsets: Offset vector of shape (K,)
#         target_dim: Target dimension for the shifted matrix
#
#     Returns:
#         Shifted shapelet matrix of shape (K, target_dim)
#     """
#     K, q = S.shape
#     res = np.zeros((K, target_dim))
#
#     # MATLAB: res(IDX) = S';
#     # In Python, we'll do this directly by correctly placing each shapelet
#     for k in range(K):
#         if offsets[k] + q <= target_dim:
#             res[k, offsets[k]:(offsets[k] + q)] = S[k, :]
#
#     return res

def op_shift(S: np.ndarray, offsets: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Shift the shapelets according to given offsets.

    Args:
        S: Shapelet matrix of shape (K, q)
        offsets: Offset vector of shape (K,)
        target_dim: Target dimension for the shifted matrix

    Returns:
        Shifted shapelet matrix of shape (K, target_dim)
    """
    K, q = S.shape
    res = np.zeros((K, target_dim))  # Changed from (target_dim, K) to (K, target_dim)

    # Place each shapelet in the correct position
    for k in range(K):
        start_idx = offsets[k]
        end_idx = start_idx + q
        if end_idx <= target_dim:
            res[k, start_idx:end_idx] = S[k, :]

    return res


def test_op_shift():
    """Test op_shift function with a small example and print detailed outputs"""
    # Create small test case
    K, q = 2, 3
    target_dim = 6
    S = np.array([[1, 2, 3],
                  [4, 5, 6]])
    offsets = np.array([0, 2])

    print("Input S shape:", S.shape)
    print("S:\n", S)
    print("Offsets:", offsets)

    result = op_shift(S, offsets, target_dim)

    print("\nResult shape:", result.shape)
    print("Result:\n", result)

    # Verify the output
    expected_result = np.array([
        [1, 2, 3, 0, 0, 0],  # First shapelet starts at offset 0
        [0, 0, 4, 5, 6, 0]  # Second shapelet starts at offset 2
    ])

    print("\nExpected shape:", expected_result.shape)
    print("Expected result:\n", expected_result)

    if np.allclose(result, expected_result):
        print("\nTest PASSED: Output matches expected result")
    else:
        print("\nTest FAILED: Output differs from expected result")
        print("Difference:\n", result - expected_result)

    return result


# Run test
if __name__ == "__main__":
    print("Testing op_shift function...")
    test_result = test_op_shift()
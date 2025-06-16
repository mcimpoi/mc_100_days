
def maximumDifference(nums):
    """
    This function calculates the maximum difference between two elements in the list
    such that the larger element comes after the smaller element.
    If no such pair exists, it returns -1.
    """
    min_value = float('inf')
    max_diff = -1

    for num in nums:
        if num < min_value:
            min_value = num
        elif num - min_value > max_diff:
            max_diff = num - min_value

    return max_diff if max_diff > 0 else -1

def maxDifferenceNaive(nums):
    """
    A naive implementation that checks all pairs to find the maximum difference.
    This is less efficient than the optimal solution.
    """
    max_diff = -1
    n = len(nums)

    for i in range(n):
        for j in range(i + 1, n):
            if nums[j] > nums[i]:
                max_diff = max(max_diff, nums[j] - nums[i])

    return max_diff 

if __name__ == "__main__":
    test_cases = [
        [1, 2, 3, 4, 5], 
        [5, 4, 3, 2, 1],
        [7, 1, 5, 3, 6, 4],
        [9, 4, 3, 2],
        [7, 1, 5, 4],
        [1,5, 2, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [10, 20, 30, 40, 50],
        [1],
    ]
    expected_outputs = [4, -1, 5, -1, 4, 9, 8, 40, -1]

    for i, case in enumerate(test_cases):
        assert maximumDifference(case) == expected_outputs[i]
        assert maxDifferenceNaive(case) == expected_outputs[i]

class Solution:
    def possibleStringCount(self, word: str) -> int:
        idx = 0
        res = 1
        while idx < len(word):
            idx1 = idx + 1
            while idx1 < len(word) and word[idx] == word[idx1]:
                idx1 += 1
            res += idx1 - idx - 1
            idx = idx1
        return res


if __name__ == "__main__":
    sol = Solution()
    print(f"aaabbb - Actual: {sol.possibleStringCount('aaabbb')} expected: 5")

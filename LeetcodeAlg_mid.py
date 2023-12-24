from typing import *
class ListNode:
    def __init__(self, val=0, next=None) -> None:
        self.val = val
        self.next = next
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
class GraphNode:
    def __init__(self, val: int = 0, neighbors: 'List[Node]' = None) -> None:
        self.val = val
        self.neighbors = neighbors
        
# 1.2 Add Two Numbers ============================================================ https://leetcode.com/problems/add-two-numbers/
# Problem: Given two non-empty linked list [l1], [l2]. Two linked lists represent two non-negative integers, where each digit is 
#          stored as node in reverse order. Sum up both linked list and return the sum as a linked list.
#          Ex 342+456 = 807
#             [2] -> [4] -> [3] == [7] -> [0] -> [8]
#             [6] -> [5] -> [4] 
# Description: Create a dummy node to store the sum. Iterate throught [l1] and [l2], keep tracking current node of [l1] and [l2]
#              and [carry]. If [l1], [l2] or carry are not zero or not None, means result has more digit, then create a now node
#              one result linked list to store the sum of [l1], [l2] and carry. The value of new node is [sum]%10, and value of 
#              carry is [sum]//10
# Time complexity: O(n)
def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    carry = 0
    head = cur = ListNode()
    while l1 or l2 or carry:
        cur.next = ListNode()
        if l1:
            cur.next.val += l1.val
            l1 = l1.next
        if l2:
            cur.next.val += l2.val
            l2 = l2.next
        cur.next.val += carry
        carry = cur.next.val//10
        cur.next.val %= 10
        cur = cur.next
    return head.next

# 2.3 Longest Substring Without Repeating Characters ======================= https://leetcode.com/problems/longest-substring-without-repeating-characters/
# Problem: Given a string [s], find the length of "longest substring" without repeating character
#          Ex: "abcadcbb" => "bcad" length = 4
# Description: Maintain dictionary [track], that tracks latest index of each character. [start] tracks the beginning of the current substring.
#              [max_length] tracks the result length. 
#              Iterate through character in [s]. If a character is not in [track], record the character and its index, and update [max_length]
#              with current length "i-start+1". If a character is already in [track] and [start] comes before the last occurance of the character,  
#              means current substring has duplicates. Set [start] to the next index of the duplicate to start a new substring. For every character
#              record index of latest occurrence
# In short: Iterate through [s], track [start] of current substring, and [lastShown] index of character. If encounter a [c] whose [lastShown] comes 
#           after [start], then current substing has duplicate of [c]
# Time complexity: O(N)
def lengthOfLongestSubstring(s: str) -> int:
    track = {}
    max_length = start = 0          # [start] represent the starting index of current substring
    for i, c in enumerate(s):
        # duplicate detected in current substring
        if c in track and start <= track[c]:                
            start = track[c] + 1                   # start a new substring right after last occurence of [c]
        # no duplicate so far
        else:                                                         
            max_length = max(max_length, i - start + 1)     # calculate current length and update [max_length]
        track[c] = i                  # record latest occurrence of each character   
    return max_length

# 3.5 Longest Palindromic Substring ===================================== https://leetcode.com/problems/longest-palindromic-substring/
# Problem: Given a string [s], return the longest substring that is a palindrome
# Description: Iterate through every character in [s] with index [i]. Use [i] as middle of palindrome substring, and expand on both sides of [i], 
#              if character on both side of [i] are same, then expand more. Stop expanding and return the palindrome when character on both sides
#              are differernt or hit end of [s]. Maintain [res] as longest palindrome substring, and compare with returned substring.
# Time compleixty: O(n^2)
def longestPalindrome(s):
    res = ""
    for i in range(len(s)):
        # substring starts with single character, and check odd length substrings
        temp = longestPalindrome_help(s, i, i)
        res = temp if len(temp)>len(res) else res
        # substring starts with two characters, and check even length substrings
        temp = longestPalindrome_help(s, i, i+1)
        res = temp if len(temp)>len(res) else res
    return res

def longestPalindrome_help(s, l, r):
    while l>=0 and r<len(s) and s[l] == s[r]:   
        l, r = l-1, r+1     # expand substring on both sides
    return s[l+1: r]            # return the last valid palindrome

# 4.6 ZigZag Conversion ====================================================== https://leetcode.com/problems/zigzag-conversion/
# Problem: Given a string [s], and an integer [numRows]. Display the [s] in ZigZag pattern with [numRows] rows. Return the a 
#          string that extract the pattern row by row. 
#          Example: "PAYPALISHIRING" numRows=4, returns "PINALSIGYAHRPI"
#                   P       I       N
#                   A     L S    I  G
#                   Y  A    H  R
#                   P       I
# Description: Maintain a string list [res] of size [numRows], each element in [res] represents a row in zigzag pattern. Track
#              [index] and [step], that correspondingly represent the index of [res] and the direction that [index] moves in [res].  
#              If [index] hit first row or last row of pattern(index == 0 or index == numRows-1), change direction by negate 
#              [step]
# Time complexity: O(n)
# Space complexity: O(n)
def convert(s: str, numRows: int) -> str:
    res = [""]*numRows          # create list, each element represent a row in pattern
    index, step = 0, 1
    for c in s:
        res[index] += c
        index = (index+step)%numRows        # move to next [index]
        if index == 0 or index == numRows-1:
            step *= -1                      # negate step, if [index] hit top/bottom boundary
    return "".join(res)

# 5.8 String to Integer ======================================================= https://leetcode.com/problems/string-to-integer-atoi/
# Problem: Given a string [s], convert it to a 32-bit signed integer. The leading and tailling space should be ignored, the string 
#          may have sign '-' or '+' at beginning, any alphabets and special characters that doesn't form valid integer will end the
#          convertion and the rest of string is ignored. If the result is between [-2**31, 2**31-1], return result. If result is less
#          than -2**31, return -2**31. If result is larger than 2**31-1, return 2**31-1.
# Description: Strip the leading and tailing spaces from [s]. Maintain a [res] to store converted integer number, and [sign] to store
#              negative or positive sign. Iterate through characters in [s]. If a char is number "isnumeric()", add it to [res]. If a 
#              char is not number, and it is '-' or '+' at index 0, then change [sign] accordingly. If the char is not at index 0, 
#              then stop iteration. Compare [res] with "-2**31" and "2**31-1" and return accordingly
def myAtoi(s: str) -> int:
    if len(s)<=0:
        return 0
    s = s.strip()           # strip leading and tailing spaces
    res, sign = 0, 1
    for i in range(len(s)):
        if s[i].isnumeric():
            res *= 10
            res += ord(s[i]) - 48
        else:
            if i==0:
                if s[i] == '-': 
                    sign = -1
                elif s[i] == '+': 
                    sign = 1   
                else:
                    break
            else:
                break
    res = int(res)*sign         # get converted integer
    if res > 2**31-1:           # compare with 32-bit integer boundary
        return 2**31-1
    elif res < -2**31:
        return -2**31
    else:
        return res

# 6.11 Container with most water ================================================ https://leetcode.com/problems/container-with-most-water/
# Problem: Given a list [height] with non-negative integers. Each integer represent a thin wall with given height. Find two walls that if 
#          a horizontal line forms a container, such that the container contains the most water
#          Ex: [1,8,6,2,5,4,8,3,7], draw the horizontal line between index 1 and index 8, with height = 7. Maximum amount = 7*7 = 49
# Description: Maintain two pointers [left] and [right], start from both end of the list. Compare height of height[left] and heifht[right],
#              move the smaller point towards middle. Maintain the maximum size of container [res]. Calculat the size of container, where
#              size = min(left, right) * (right-left). Update [res] according to calculated size
# Time complexity: O(N), N=len(height)
def maxArea(self, height: List[int]) -> int:
    left, right = 0, len(height)-1
    res = min(height[left], height[right])*right
    while left<right:
        if height[left]<height[right]:
            left += 1
        else:
            right -= 1
        res = max(res, min(height[left], height[right])*(right-left))
    return res

# 7.12 Integer to Roman ======================================================== https://leetcode.com/problems/integer-to-roman/
# Problem: Given an integer convert it to a string of Roman number. The char of convertion is below
#          I:1 V:5 X:10 L:50 C:100 D:500 M:1000
#          Note, If a smaller symbol comes before a larger, means subtraction. IV:4 IX:9 XL:40 XC:90 CD:500 CM:900
# Description: Maintain a "reversed" roman string [res] as result, maintain two lists contain Roman number of [ones](I, X, C, M) and 
#              number of [fives](V, L, D). Track index [i] of [ones], where start from "I". Retrieve numbers from lowest digit by %10. 
#              every mod by 10 remove one digit from [num] thus increase [i] by 1. The module is denoted as [cnt]
#              If [cnt] is less than 4, add "ones[i]"*[cnt] to [res]. 
#              If [cnt] is equals to 4, add "ones[i]" and "fives[i]" to construct 4.
#              If [cnt] greater than 4 and less than 9, add a "fives[i]" and "ones[i]" * (cnt-5)
#              If [cnt] is equals to 9, add "ones[i]" and "ones[i+1]" to construct 5
#              Each iteration, [num] is decreased by 10 times, and [i] is increased by 1, At the end, reverse [res] and return
# Time complexity: O(lg(n))
def intToRoman(num: int) -> str:
    ones = ["I", "X", "C", "M"]
    fives = ["V", "L", "D"]
    res, i = [], 0
    while num > 0:
        cnt = num%10
        if cnt < 4:
            res.append(cnt*ones[i])
        elif cnt == 4:
            res.append(ones[i]+fives[i])
        elif 4<cnt<9:
            res.append(fives[i]+ones[i]*(cnt-5))
        else:
            res.append(ones[i]+ones[i+1])
        i+=1
        num//=10
    return "".join(res[::-1])

# 8.15 3SUM ====================================================================== https://leetcode.com/problems/3sum/
# Problem: Given an array of integers [nums], Find three unique elements in the array such that their sum is Zero. For each 
#          combination an element can be used no more than once. Return all unique triplets in the array that sum up to zero. 
#          The result can NOT contain dulicate triplets
# Description: Sort array, iterate every element in sorted array. In each iteration nums[i], need to find two elements that 
#              sum up to "-nums[i]" denote as [target]. In order to find such two number, use two pointers [left] and [right] 
#              that [left] starts from [i+1], [right] starts from right-end of array. Since the array is sorted, move [left] 
#              to right if their sum is smaller than [target]. If sum is larger than [target], then move [right] to left.
#              If sum equals to [target], means a combination is found, three numbers are [i], [left] and [right]. Add the
#              triplet to [res] list.
#              This manner can avoid using same element multiple times, but can't avoid using duplicate elements in the list.
#              Thus, we need to skip duplicate elements for [i], [left] and [right]. If [i] == [i-1], skip [i]. If [left] ==
#              [left-1], skip [left]. If [right] == [right+1], skip [right]
# Time complexity: O(n^2)
def threeSum(nums: List[int]) -> List[List[int]]:
    nums.sort()
    res = []
    length = len(nums)
    for i in range(length-2):
        if i>0 and nums[i]==nums[i-1]:          # skip duplicates on [i]
            continue
        left, right = i+1, length-1
        target = -nums[i]
        while left < right:                     # find [left] and [right] that sum up to target
            s = nums[left] + nums[right]
            if s<target:
                left+=1
            elif s>target:
                right-=1
            else:
                res.append([nums[i], nums[left], nums[right]])
                while left<right and nums[left] == nums[left+1]:        # skip duplicates on [left]
                    left+=1
                while left<right and nums[right-1] == nums[right]:      # skip duplicates on [right]
                    right-=1
                left+=1
                right-=1
    return res

# 9.16 3Sum Closest ================================================================= https://leetcode.com/problems/3sum-closest/
# Problem: Given a list of integers [nums], and a integer number [target]. Find three number is [nums] that their sum is closest
#          to [target]. Return the sum of the three numbers.
# Description: Use similar strategy as "3Sum". Sort [nums] and iterate through it, For each iteration with index [i], maintain 
#              two pointers [j] and [k], that start from [i+1] and [length-1] respectely. Sum up [i], [j], [k] and compare with
#              [target]. If [sum]<[target], move [j] to right to increase [sum]. If [sum]>[target], move [k] to left to decrease
#              [sum]. If [sum]=[target], then return [sum] since it is the closest. For each combination of [i], [j], [k], track
#              the smallest difference between [target] and [sum], and maintain [sum]
# In short: Sort [nums], iterate through [nums] with [i], use [left] and [right] pointer to find sum=0. Skip duplicates on [i],
#           and skip duplicates on [left] and [right] when a solution is found
# Time complexity: O(n^2)
def threeSumClosest(nums: List[int], target: int) -> int:
    nums.sort()
    res, diff = sum(nums[0:3]), float("inf")
    length = len(nums)
    for i in range(len(nums)):
        j, k = i+1, length-1
        while j<k:
            s = nums[i] + nums[j] + nums[k]
            if s<target:
                j+=1
            elif s>target:
                k-=1
            else:
                return target
            if abs(s-target)<diff:
                res = s
                diff = abs(target-s)
    return res
        
# 10.17 Letter Combinations of a Phone Number ====================== https://leetcode.com/problems/letter-combinations-of-a-phone-number/
# Problem: Consider a cell phone keyboard, numbers from 2 to 9 have alphabets binded to them, where 2="abc", 3="def", 4="ghi",  ="jkl",
#          6="mno", 7="pqrs", 8="tuv", 9="wxyz". Given a string with [digits] from 2 to 9, construct a list contains all possible 
#          combinations that [digits] can represent
#          Ex: "23", 2="abc" 3="def" ===> ["ad","ae","af","bd","be","bf","cd","ce","cf"]
# Description: Use DFS backtrack. Create a [letters] list which convert number to its corresponding alphabets, create a list [res]
#              as result list. The recursive method takes [digits], [letters], [path], and [res] as parameter, where [path] maintains 
#              the characters in a single path down the DFS tree. Each recursive call take the first number in [digits], and iterate 
#              through characters in its corresponding alphabets. Each iteration calls the recursive method to construct back-track
#              pass result of [digits] as parameter, and append current charater to [path]. When no more number left in [digits], append
#              [path] to [res]
# Time complexity: O(4^n), n=len(digits). Each number in [digits] will trigger 3 or 4 recursive calls depending on number of alphabets 
# Space complexity: O(4^n)                associated with them. Thus 4^n
def letterCombinations(digits: str) -> List[str]:
    if not digits:
        return []
    letters = ["","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"]         # a number-letter conversion, 0 and 1 are not used
    res = []
    letterCombinations_helper(digits, letters, "", res)
    return res

def letterCombinations_helper(digits, letters, path, res):
    if not digits:                      # no more digits left, a full path is constructed
        res.append(path)
    else:
        for c in letters[int(digits[0])]:   
            letterCombinations_helper(digits[1:], letters, path+c, res)     # each recursive call take a character and add to path

# 11.18 4Sum ========================================================================================= https://leetcode.com/problems/4sum/
# Problem: Given an integer aray [nums], and an integer [target]. Find four numbers in [nums], such that their sum equals to [target]. 
#          Find all unique quadruplets and return as 2D array, where each quadruplet is a sub-array
# Description: Recursively reduce "4sum" to "3sum" and furture reduce to "2sum". Use back-track recursion apply DFS, maintain a list [temp] 
#              that store numbers on DFS path, track [target] of current level and [N] representing how many numbers need to pick. 
#              For each recursive call, add first number in [nums] to [temp] to construct DFS tree. Subtract nums[0] from [target], since
#              nums[0] is picked and new [target] shoud exclud it. Reduce [N] by 1 since nums[0] is picked. Pass rest of list nums[1:] to next 
#              recursion.
#              When N=2 invoke "2sum", maintain 2 pointers [left] and [right], which start at beginning and end of list. Keep moving [left]
#              and [right] to middle and compare their sum with [target](target here is reduced). When their sum equals to [target], add
#              nums[left] and nums[right] along with [temp] into result list [res].
# Time complexity: O(n^3)
def fourSum(nums: List[int], target: int) -> List[List[int]]:
    def NSum(nums, target, N, temp, res):
        # Early ternimation improve performance
        #   when there's no enough elements in list
        #   when the elements in list are too small or too large to reach target
        if len(nums) < 2 or target < nums[0]*N or target > nums[-1]*N:  
            return
        # Perform twoSums
        if N == 2:
            left, right = 0, len(nums)-1
            while left<right:
                s = nums[left]+nums[right]
                if s<target:
                    left+=1
                elif s>target:
                    right-=1
                else:
                    res.append(temp+[nums[left], nums[right]])
                    while left<right and nums[left] == nums[left-1]:        # skip duplicates on [left]
                        left += 1
                    while left<right and nums[right-1] == nums[right]:      # skip duplicates on [right]
                        right -= 1
                    left += 1
                    right -= 1
        # reduce N
        else:
            for i in range(len(nums)-N+1):
                if i==0 or nums[i-1] != nums[i]:            # skip duplicates on [i], but invoke at least once 
                    NSum(nums[i+1:], target-nums[i], N-1, temp+[nums[i]], res)
    res = []
    NSum(sorted(nums), target, 4, [], res)
    return res

# Description: Create a dictionary [record], Iterate [nums] with pointer [i] and [j], maintain pairs of numbers from and their sum.
#              Specificall, the "value" is collection of pairs of number index (multiple pairs might sum up tp same number),
#              the "key" is how much more do numbers needed to reach [target]. Iterate [nums] again with pointer [i] and [j],
#              search for their sum [s] = nums[i]+nums[j]. That pairs in [record] can be paired with nums[i] and nums[j] to sum
#              up to [target]. There can have multiple pairs in "record[key]", iterate though "record[key]" and construct a 
#              result by appending nums[i] and nums[j] after existing pairs. Ensure [i] and [j] comes after exsting pairs, to 
#              eliminate duplicates. Add each result to [res] set, and convert it to list before returning
# Time Compleixty: O(n^2)
import collections
def fourSum_2(nums: List[int], target: int) -> List[List[int]]:
    nums.sort()
    # eliminate corner cases
    if len(nums) < 4 or 4*nums[0] > target or 4*nums[-1] < target: return []    
    # create dictionary where "value" is set() of tuples (index_1, index_2), "key" is target-nums[i]-nums[j]
    record = collections.defaultdict(set)           
    for i in range(len(nums)-3):
        for j in range(i+1, len(nums)):
            need = target-nums[i]-nums[j]
            record[need].add((i,j))

    res = set()
    # iterate [nums] again, find other two numbers that can sum up to "key" of dictionary
    for i in range(len(nums)-1):
        for j in range(i+1, len(nums)):
            s = nums[i]+nums[j]
            if s in record:                 # find other two numbers can pair with the two numbers in dictionaray
                for pair in record[s]:          # there are multiple pairs in record[s]
                    if pair[1]<i:                       # new pair should comes after existing pairs 
                        res.add((nums[pair[0]], nums[pair[1]], nums[i], nums[j]))
    return [list(e) for e in res]                       # convert set() to list() before returning

# 12.19 Remove Nth Node From End of List ========================================== https://leetcode.com/problems/remove-nth-node-from-end-of-list/
# Problem: Given the head of a linked list [head], and an integer [n]. Remove [n]th node from end of list, and return the head.
#          Ex 1 -> 2 -> 3 -> 4 -> 5, n=2  ====> remove second last element "4", 1 -> 2 -> 3 -> 5
# Descrption: Use two pointers [fast] and [slow], where [slow] start from head, [fast] start [n] nodes after head. Both node traverse at same pace,
#             when [fast] hit the end, [slow] points to the node need to be removed. Insert a dummy header before [head], and return [dummy.next]
#             eventually. Traverse [fast] and [slow], when [fast.next] is None, then [slow] is pointing to the node before the node need to be 
#             removed. Assign [slow.next] to [slow.next.next] to remove the node
# Time complexity: O(n)
def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    dummy = ListNode(0, head)
    fast = slow = dummy             # use [slow] and [fast], to find the node before the node to be removed
    for _ in range(n+1):
        fast = fast.next
    while fast.next:
        slow = slow.next
        fast = fast.next
    slow.next = slow.next.next      # remove target node
    return dummy.next

# 13.22 Generate Parentheses =========================================================== https://leetcode.com/problems/generate-parentheses/
# Problem: Generate [n] pairs of parentheses, return all valid combinantions of well-formed parentheses.
# Description: Maintain a string [temp] representing a valid parentheses with [n] pairs. During the construction of [temp], maintain [open]
#              and [close] represent number of opening and closing parentheses in [temp]. Keep constructing [temp] by adding opening or 
#              closing parentheses to [temp], and increase count of [open] and [close] accordingly. [close] can exceed [open] otherwise 
#              the combination is not valid. When [open]==[close]==[n], means a completed and valid combination is formed, append [temp] 
#              to result list. 
# Time complexity: O((2n choose n)/(n+1)) aka Catalan number
def generateParenthesis(n: int) -> List[str]:
    def generateParenthesis_helper(open, close, temp, res):
        if open == close == n:                  # completed well-formed parentheses, add it to [res]
            res.append(temp)
            return
        if close > open:                # more closing than opening, it is invalid and terminate this path
            return
        if open < n:
            generateParenthesis_helper(open+1, close, temp+"(", res)        # add an opening parenthesis
        if close < n:
            generateParenthesis_helper(open, close+1, temp+")", res)        # add a closing parenthesis
    res = []
    generateParenthesis_helper(0, 0, "", res)       # initially zero opening and closing parentheses 
    return res

# 14.24 Swap Nodes in Pairs =============================================================== https://leetcode.com/problems/swap-nodes-in-pairs/
# Problem: Given the [head] of a linked list, sap position of every adjacent nodes and return the head
#          Ex: 1->2->3->4->5  ====>   2->1->4->3->5
# Description: Maintain three pointers [first], [second], and [third], where [second] and [third] are being swapped, and [first] is the anchor
#              to link [second] and [third] back to list after swapping. After each swap, move [first] two position forward, and reassign both
#              [second] and [third], until remaining list has less than 2 nodes. Consider the size of given linked list is smaller than 2, 
#              no swap is  neede, return [head] directly
# Time complexity: O(n)
def swapPairs(head: ListNode) -> ListNode:  
    if not head or not head.next:               # less than 2 nodes in list
        return head
    dummy = first = ListNode(0, head)
    while first.next and first.next.next:       # swap until remaining nodes is less than 2
        second = first.next
        third = second.next
        second.next = third.next
        first.next = third
        third.next = second
        first = first.next.next
    return dummy.next

# 15.29 Divide Two Integers ================================================================= https://leetcode.com/problems/divide-two-integers/
# Problem: Given two integers [dividend] and [divisor], divide two integers without using multiplication, division or mod operations. Return the
#          quotient after dividing [dividend] by [divisor], truncate fraction/decimal part if any. If result is out of bound of 32-bit integer,
#          return the value of bound -2^31 or 2^31-1
# Description: Check the sign of quotient, by comapring signs of [dividend] and [divisor]. If they have same sign, then quotient is positive, 
#              otherwise quotient is negative. Use "abs()"" to make [dividend] and [divisor] positive, then subtract [divisor] from [dividend] 
#              in a loop, until [dividend] is less than [divisor], and count how many [divisor] are subtracted with [cnt]. Meanwhile, double
#              [divisor] by "divisor<<=1" to reduce number of iterations. Also, since [divisor] is douled, [cnt] shoule be doubled every 
#              iteration by "cnt<<=1", because double amount of [divisor] is subtracted. At the end of loop, add "sign" to [res], and compare
#              [res] with boundary then return 
# Time complexity: O(log(dividend, divisor)), outter and innter loop both O(log(dividend, divisor))
def divide(dividend: int, divisor: int) -> int:
    positive = (dividend < 0) == (divisor < 0)          # extract sign of result
    dividend, divisor = abs(dividend), abs(divisor)
    res = 0
    while dividend >= divisor:              # outter loop keep subtracting [divisor] until [dividend] is smaller than [divisor]
        temp, cnt = divisor, 1                  # rest current divisor, and cnt
        while dividend >= temp:                 # inner loop that double [divisor] every iteration
            dividend -= temp
            res += cnt
            cnt <<= 1
            temp <<= 1          
    if not positive:                        # assign "sign" to result
        res = 0-res
    return min(max(-2147483648, res), 2147483647)       # compare with 32-bit boundary

# 16.31 Next Permutation ====================================================================== https://leetcode.com/problems/next-permutation/
# Problem: Given a list of integers [nums], rearrange order of integers the generate next greater permutation, aka a number that contains exact 
#          digits, new number is greater than [nums] and no other possible permutation is smaller than new number and larger than [nums]. 
#          Rearrange [nums] in-place
# Description: The next permutation is slightly larger. Find the last ascending between adjacent digits, swap the ascending number with a 
#              number that is slightly larger than it can make the number larger. Reverse the descending part which comes after the swapped 
#              number, this will make that part become the smallest permutation. Since the higher digit is larger than before, and the rest 
#              digits are the smallest possible, the new number is the next permutation
#              Maintain a pointer [i] starts from end of list, iterate towards front and find the last ascending position that "[i-1]<[i]".
#              Maintain another pointer [j] starts from end of list, iterate towards front and find the digit that is slightly larger than
#              [i-1]. Swap [i-1] and [j], and reverse all digits after [i-1].
# Time complexity: O(n), n=len(nums)
def nextPermutation(nums: List[int]) -> None:
    i = j = len(nums)-1
    while i>0 and nums[i-1]>=nums[i]:
        i -= 1
    if i==0:                # [nums] is the last permutation, reverse all digits
        nums.reverse()
        return
    while nums[i-1]>=nums[j]:       # find the digit slightly larger than [i-1]
        j -= 1
    nums[i-1], nums[j] = nums[j], nums[i-1]     # swap 
    nums[i:] = reversed(nums[i:])             # reverse digits after [i-1]

# 17.33 Search in Rotated Sorted Array ========================================= https://leetcode.com/problems/search-in-rotated-sorted-array/
# Problem: Given an integer array [nums], which sorted in ascending order. However, the elements are shifted/rotated by "k" time, value of "k"
#          is unknown. Ex [1,2,3,4,5,6,7] is rotated by 4 and become [4,5,6,7,1,2,3]. Given the integer array [nums] and an integer [target],
#          find and return the index of [target] in [nums], if [target] is not presented return -1
# Description: Use binary search. Maintain [l] and [r] as two ends of list, [m] as middle index of list. 
#              If [mid] == target, [target] is found and return [mid].
#              If [l] <= [mid], means sublist from [l] to [mid] are sorted in ascending order. 
#                   If [target] is located between [l] and [mid], just need to apply regular binary search to find [target]
#                   Else, [target] is located between [mid] and [r], reassign 'l=mid+1" to look for sorted sublist
#              Else, sublist between [mid] and [r] are sorted in ascending order.
#                   If [target] is located between [mid] and [r], apply regular binary search to find [target]
#                   Else. [target] is located between [l] and [mid], reassgin "r=mid-1" to look for sorted sublist
# Time complexity: O(logN), N=len(nums)
def search(nums: List[int], target: int) -> int:
    l, r = 0, len(nums)-1
    while l<=r:
        mid = (l+r)//2
        if nums[mid] == target:                 # target is found
            return mid  
        elif nums[l] <= nums[mid]:              # left half is sorted
            if nums[l] <= target <= nums[mid]:      # target is in the sorted half
                r = mid-1
            else:                                   #  target is not in the sorted half
                l = mid+1
        else:                                   # right half is sorted
            if nums[mid] <= target <= nums[r]:      # target is in the sorted half
                l = mid+1
            else:                                   # target is not in the sorted half
                r = mid-1
    return -1

# 18.34 Find First and Last Position of Element in Sorted Array ==== https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
# Problem: Given a sorted integer list [nums], Given a integer [target]. Find the starting and ending index of [target] in [nums]. 
#          Return starting and ending indices as list [s, e]. If [target] is not presented return [-1, -1]
# Description: Implement "bisect_left" and "bisect_right" to find insert index of target. After find insertion index [l] and [r], 
#              reduce [r] by 1 since we need to find the last index of [target] instead of insertion index. Validate [l] and [r]
#              that they should not be out of bound. Elements of index [l] and [r] should both equal to [target]. If any validation
#              is false, should return [-1, -1].
def searchRange(nums: List[int], target: int) -> List[int]:
    l = insertLeft(nums, target)
    r = insertRight(nums, target)-1             # minus 1 to get last index of [target]
    if 0<=l<len(nums) and 0<=r<len(nums) and nums[l] == nums[r] == target:      # validate [l] and [r]
        return [l,r]
    else:
        return [-1, -1]

def insertLeft(a, x):       # bisect_left
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo

def insertRight(a, x):      # bisect_right
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo

# 19.36 Valid Sudoku ================================================================================= https://leetcode.com/problems/valid-sudoku/
# Problem: Given a 9*9 list representing a sudoku board. Some cells are filled with number strings, some cells are dots "." as they are not filled.
#          Validate the given sudoku [board] return True or False with following rules
#          1. each row must contain digits 1-9 without duplicates
#          2. each column must contain digits 1-9 without duplicates
#          3. each 3*3 sub-box must contain 1-9 without duplicates
# Description: Maintain a set [record] to record occurrence of each cell, where "(2)1" means "2" appears at column 1, "1(2)" means "2" appears at
#              row 1, "0(2)2" means top-right square contains "2". Iterate though each cell of [board] and each cell will add 3 records to [record].
#              If any cell adds less than 3 records, means there are duplicates then return False. 
# Time complexity: O(n), n=81
def isValidSudoku(board: List[List[str]]) -> bool:
    record = set()
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] != ".":                  # only check filled cells and skip empty cells
                prevSize = len(record)
                row = str(i)+"("+board[i][j]+")"                    # row record
                col = "("+board[i][j]+")"+str(j)                    # col record
                square = str(i//3)+"("+board[i][j]+")"+str(j//3)    # square record
                record.add(row)
                record.add(col)
                record.add(square)
                if len(record)-prevSize != 3:           # must add 3 records otherwise there are duplicates
                    return False
    return True

# 20.39 Combination Sum =========================================================================== https://leetcode.com/problems/combination-sum/
# Problem: Given a list of "unique" integers [candidates] and a integer [target]. Pick numbers from [candidates] to sum up to [target]. A number in
#          [candidates] can be used as many times. Return all unique combinations of numbers from [candidates] that sum up to [target].
# Description: Dynamic programming, Create a lsit [dp] that index [t] store combinations that sum up to [t]. In order to find combination that
#              sum up to [t+c], just need to grab all combination of dp[t] and append [c] after them.
#              Sort [candidates] in ascending order. Iterate though [t] from 1 to target+1, find combination that sum up to [t], and store them 
#              in dp[t]. For each [t], iterate though [candidates]. For each [c] of [candidates], if [c]>[t], means the rest of [candidates] 
#              all greater than [t], not need to continue. If [c]==[t], means [c] itself sum up to [t], append [c] into dp[t]. Else, find complement
#              of [c] which is dp[t-c], append [c] after combinations in dp[t-c] and store at dp[t].
#              The combination that sum up to [t] are stored in the last element of [dp], return dp[-1] at the end
# Time complexity: O(target^2*n), n=len(candidates) OR O(nlogn), if len(candidate) is very large
# Space complexity: O(target*3), len(dp) == len(dp[i]) == len(dp[i][i]) == target
def combinationSum(candidates: List[int], target: int) -> List[List[int]]:
    candidates.sort()
    dp = [[] for _ in range(target+1)]
    for t in range(1, target+1):                # dp[t] saves all combinations have t sum
        for c in candidates:
            if c > t: break                         # from now on, all sum > t, break out
            if c == t:                              # the element value equals to current target t
                dp[t].append([c])
                break         
            # to ensure no duplicate, the later coming item should be strictly greater than previous ones, make the result an ascending order. 
            for path in dp[t-c]:
                if c>=path[-1]:
                    dp[t].append(path+[c])
    return dp[-1]

# Description: Create a list [dp], where each index [i] is a list of combinations that sum up to [i]. Iterate though each [candidate], find 
#              all possible sums that can be achieved by adding other candidates to [candidate], and record the combination in dp[i]. By  
#              iterate though from [candidate] to [target], we can find the combination that has [candidate] involved.
def combinationSum_2(candidates: List[int], target: int) -> List[List[int]]:
    dp = [[[]] for _ in range(target+1)]
    for candidate in candidates:
        for i in range(candidate, target + 1):
                if i-candidate == 0:
                    dp[i].append([candidate])
                else:
                    for sublist in dp[i-candidate]:
                        if sublist:
                            dp[i].append(sublist+[candidate])
    return [comb for comb in dp[target] if comb]

# 21.40 Combination Sum II ==================================================================== https://leetcode.com/problems/combination-sum-ii/
# Problem: Given a list of integer [candidates] (may have duplicates), and an integet [target]. Find all unique combinations where combinations
#          sum up to [target]. Each number in candidates can be used no more than once.
# Description: DFS backtrack, on a sorted [candidates]. Each iteration track the starting index [start], and maintain [candidates] in recursion.
#              skip dupliate elements if "candidates[i]==candidates[i-1]". Deduct [target] when "candidates[i]" is added to a combination. If
#              [target]==0, the sum is reached and add current combination to result. If target<0, the sum is exceeded and return to stop 
#              recursion. 
# Time complexity: O(2**n) [n] is size of [candidates]. The total number of subsets of [candidates]
def combinationSum2(candidates: List[int], target: int) -> List[List[int]]:
    def helper(pool, temp, target, res):
        if target<0:
            return
        elif target == 0:
            res.append(temp)
            return
        for i in range(len(pool)):
            if i>0 and pool[i] == pool[i-1]:
                continue
            helper(pool[i+1:], temp+[pool[i]], target-pool[i], res)
    res = []
    helper(sorted(candidates), [], target, res)
    return res

# 22.43 Multiply Strings ==================================================================== https://leetcode.com/problems/multiply-strings/
# Problem: Given two non-negative integer strings [num1] and [num2]. return the product of them as string.
# Description: Multiplication is done by multiplying a digit in [num1] with every digit in [num2]. Maintain [res] as result, iterate digits
#              in [num1] and [num2] in reversed order. Multiple each digit from [num1] is multiplied with every digit in [num2], and maintain
#              [multi1] and [mulit2] to represent digital position. Production of each multiplication is added to [res]
# Time complexity: O(n*m), n=len(num1) m=len(num2)
def multiply(num1: str, num2: str) -> str:
    def strToInt(char):                 # function to convert each digit to int
        return ord(char)-ord("0")

    res = 0
    multi1 = 1                          # represent digital position of [n1]
    for n1 in reversed(num1):
        multi2 = 1                          # represent digital position of [n2]
        for n2 in reversed(num2):
            res += strToInt(n1)*strToInt(n2)*multi1*multi2
            multi2*=10
        multi1*=10
    return str(res)

# 23.45 Jump Game II =========================================================================== https://leetcode.com/problems/jump-game-ii/
# Problem: Given a list of non-negative integers [nums], each element represent how many index you can jump from that location. For example,
#          nums[0] = 3 then from nums[0], can jump to nums[1], nums[2] or nums[3]. Assume you start from index 0, return the minimal [jumps] 
#          to reach the end of [nums]
# Description: Greedy. Iterate through elements in [nums]. Maintain [furthest] as the furthest index reached so far, which is update for 
#              element in [nums] to be max(furthest, i+nums[i]). Maintain [lastJump] as the index reached in last jump, when iteration 
#              reaches [lastJump], update [lastJump] as [furthest] to make another jump. After making a new jump, increase [jumps] by 1.
#              If [furthest] touches or exceeds the end of [nums], return [jumps]
# Time Complexity: O(n), n=len(nums)
def jump(nums: List[int]) -> int:
    if len(nums)<=1:                                # corner case, no need to jump, already start at the end of [nums]
        return 0
    furthest = lastJump = jumps = 0
    for i in range(len(nums)):
        furthest = max(furthest, i+nums[i])         # update [furthest] for every elements
        if i == lastJump:                           # when [i] touches last jump location, make a jump to [furthest]
            lastJump = furthest
            jumps+=1
        if lastJump >= len(nums)-1:                 # reaches the end of [nums] return [jumps]
            return jumps    

# 24.46 Permutations ============================================================================== https://leetcode.com/problems/permutations/
# Problem: Given a list of "distinct" integers, return all possible permutations in an array
# Description: DFS backtracking. Use recursive function, maintain [pool] as integers that are not added to permutation yet, maintain [temp] as 
#              current permutation, maintain [res] as the result list where each element is a permutation. In back track, extract an integer
#              a time from [pool] to [temp]. Pass the rest [pool] to next recursive call, and append extracted integer to [temp]. If [pool]
#              is empty, means all integer are added to [temp], then append [temp] to [res]
# Time complexity: O(n!)
def permute(nums: List[int]) -> List[List[int]]:
    res = []
    permute_helper(nums, [], res)
    return res
    
def permute_helper(pool, temp, res):
    if pool == []:
        res.append(temp)
    else:
        for i in range(len(pool)):
            permute_helper(pool[:i]+pool[i+1:], temp+[pool[i]], res)

# 25.47 Permutations II ========================================================================== https://leetcode.com/problems/permutations-ii/
# Problem: Given a list of integers [nums] with duplicates. Return all possbile "unique" permutations in a list.
# Description: Back track on keys of Counter(nums) as [counter]. [counter] doesn't have duplicate keys, back track on Counter(nums) helps get rid 
#              of duplicates. Maintain a list [temp] holding current perumutation, and [res] as result list holding permutations. Each recursion 
#              iterate though key of [counter]. If counter[key] > 0, means there are [key]s can be appended to [temp]. Reduce counter[key] by 1
#              after appending to [temp]. After each recusive call, restore counter[key] by adding 1 back to it. When [temp] has same length of 
#              [nums], means all nums are added to [temp]. Add [temp] to [res]
# Time complexity: O(n^2)
from collections import Counter
def permuteUnique(nums: List[int]) -> List[List[int]]:
    def permuteUnique_helper(numCnt, temp, res):
        if len(temp) == len(nums):          # all numbers in [num] are added to [res], this permutation is finished
            res.append(temp)
            return
        for num in numCnt.keys():
            if numCnt[num] > 0:             # should has at least one [num] left
                numCnt[num] -= 1                # reduce counter[num] by 1, since it is added to [temp]
                permuteUnique_helper(numCnt, temp+[num], res)
                numCnt[num] += 1                # add counter[num] back, after current recursion

    res = []
    permuteUnique_helper(Counter(nums), [], res)
    return res

# 26.48 Rotate Image =============================================================================== https://leetcode.com/problems/rotate-image/
# Problem: Given a "n*n" [matrix], rotate the matrix by 90 degree clockwise. Modify the [matrix] "in-place"
# Description: Rotate a matrix layer by layer, since top row is rotated to right-most column, right-most column is rotated to buttom row, buttom
#              row is rotated to left-most column, left-most column is rotated to top row. Consider there two row and two column as outter-most 
#              layer. A "n*n" matrix contains "n//2" layers, say the outter-most layer is "0"th layer. For "i"th layer, the element are from 
#              column "i" to column "n-i-1". When rotate an element, other three elements of other side of layer are also involved, where top 
#              [i][j] moves to right [n-1-j][i], right [n-1-j][i] moves to buttom [n-1-i][size-1-j], buttom [n-1-i][n-1-j] moves to left [j][n-1-i].
#              Repeat the four-switches for elements in top row of a layer, and repeat for every layer
# Time compleixty: O(n^2), n=len(matrix)
def rotate(matrix: List[List[int]]) -> None:
    n = len(matrix)
    for i in range(n//2):
        for j in range(i, n-1-i):
            temp = matrix[i][j]
            matrix[i][j] = matrix[n-1-j][i]
            matrix[n-1-j][i] = matrix[n-1-i][n-1-j]
            matrix[n-1-i][n-1-j] = matrix[j][n-1-i]
            matrix[j][n-1-i] = temp

# 27.49 Group Anagrams =========================================================================== https://leetcode.com/problems/group-anagrams/
# Problem: Given an array of strings [strs], group the "anagrams" together.
#          Anagarms: are words that contain same characters.
#          Ex: ["eat","tea","tan","ate","nat","bat"] ===> [["bat"],["nat","tan"],["ate","eat","tea"]]
# Description: For each word in [strs], create an integer array [num_list] with 26 zeros, representing 26 alphates. Iterate characters in [word], 
#              add one to corresponding element in list. Convert [num_list] to "tuple" and use it as "key" of dictionary [record], and append 
#              [word] to the value of record[key]. Return the value of [record] at the end
# Time complexity: O(NK), N=len(strs) K=length of each word
from collections import defaultdict
def groupAnagrams(strs: List[str]) -> List[List[str]]:
    record = defaultdict(list)              # maintain dictionary, the value is "list"
    for s in strs:
        num_list = [0]*26                   # creact list of 26 elements
        for c in s:
            num_list[ord(c)-97] += 1            # convert character to corresponding index in [num_list]
        record[tuple(num_list)].append(s)   # convert to "tuple" as key in dictionary
    return record.values()

# 28.50 Pow(x, n) ==================================================================================== https://leetcode.com/problems/powx-n/
# Problem: Implement pow(x, n), which calculate [x] raised to power [n]. It is possible that "n<0", Ex: pow(2, -10) = 1/1024
# Description: Recursion. Because of x^n == x^(n/2) * x^(n/2), then continuously reduce [n] by half, and multiply two halves. If [n] is odd,
#              x^n == x^(n/2) * x^(n/2) * x, then reduce [x] by half, and multiply two halves and an extra [x]. When [n] hit zero n==0,
#              return 1, because any number x^0 == 1
# Time compleixty: O(logN)
def myPow(x: float, n: int) -> float:
    if n>0:
        return myPow_helper(x, n)
    else:
        return 1/myPow_helper(x, -n)
    
def myPow_helper(x, n):
    if n == 0:
        return 1
    half = myPow_helper(x, n//2)            # reduce [n] by half
    if n%2==0:
        return half*half                    # if n is even, multiply two halves
    else:
        return half*half*x                  # if n is odd, multiply two halves and an extra [x]

# 29.54 Spiral Matrix ============================================================================= https://leetcode.com/problems/spiral-matrix/
# Problem: Givne a "m*n" matrix, return all elements in "sprial order".
#          Ex [[1, 2, 3]    ===>  [1, 2, 3, 6, 9, 8, 7, 4, 5]
#              [4, 5, 6]    
#              [7, 8, 9]]   
# Description: Define four boundaries [top], [bottom], [left], [right]. Bring down [top] by 1, after traversing a row on top. Move [right] to 
# #            left by 1, after traversing a column on right side. Raise up [bottom] by 1, after traversing a row at bottom. Move [left] to 
#              right by 1, after traversFing a column on left side. When traversing a row, traverse elements between [left] and [right], when
#              traversing a column, traverse elements between [top] to [bottom]. Check if boundary overlap after traversing a row or column, 
#              exit loop if any boundaries overlaps each other
# Time complexity: O(n*m)
def spiralOrder(matrix: List[List[int]]) -> List[int]:
    left, right, top, bottom = 0, len(matrix[0])-1, 0, len(matrix)-1        # define 4 boundaries
    res = []                
    while True:
        for i in range(left, right+1):              # traverse top row
            res.append(matrix[top][i])
        top+=1                                          # lower [top]        
        if top>bottom: return res                       # check boundary overlap
        for j in range(top, bottom+1):              # traverse right column
            res.append(matrix[j][right])
        right-=1                                        # move [right] to left
        if left>right: return res                       # check boundary overlap
        for i in reversed(range(left, right+1)):    # traverse bottom row
            res.append(matrix[bottom-1][i])
        bottom-=1                                       # raise [bottom]
        if top>bottom: return res                       # check boundary overlap
        for j in reversed(range(top, bottom)):      # traverse left column
            res.append(matrix[j][left])
        left+=1                                         # move [left] to right
        if left>right: return res                       # check boundary overlap

# 30.55 Jump Game =================================================================================== https://leetcode.com/problems/jump-game/
# Problem: Given a non-negative integer list [nums], where each element represent maximum indeices you can jump from there. Starting from index
#          zero, determine if you are able to reach the last index.
# Description: Greedy. Maintain a [furthest] represent the maximum index can be reached so far. Iterate though [nums], the furthest can reach 
#              is "i+nums[i]", update [furthest] if any larger index can be reached. If "i > furthest" for any index, meaning index [i] can't
#              be reached, return False. If iteration hit the end of [nums], meaning the end is reached, return True
# Time complexity: O(n), n=len(nums)
def canJump(nums: List[int]) -> bool:
    furthest = 0
    for i in range(len(nums)):
        if furthest<i:
            return False
        furthest = max(furthest, i+nums[i])
    return True

# 31.56 Merge Intervals ======================================================================== https://leetcode.com/problems/merge-intervals/
# Problem: Given a 2D array [intervals], where each element represents an interval [start, end], merge all overlapping intervals, and return an
#          array of non-overlapping intervals 
#          Ex: [[2,6],[1,3],[15,18],[8,10]] ===> [[1,6],[8,10],[15,18]]
#          [1,3] and [2,6] overlaps, can be merged
# Description: Sort [invertals] by "starting boundary" in ascending order. Maintain a [res] array as result initial with "intervals[0]" in it.
#              Iterate through element in [intervals], let [prev] be the last element in [res] and [cur] the element of current interation. If
#              "prev[1] >= cur[0]", then [prev] and [cur] overlaps. Extends the "end" of [prev] in [res] to be max(prev[1], cur[1]). If there 
#              is no overlap between [cur] and [prev], append [cur] to [res] directly
# Time complexity: O(NlogN), N=len(intervals)
def merge(intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda ele:ele[0])       # sort by "starting boundray"
    res = [intervals[0]]                        # [res] array initial with first element of [intervals]
    for cur in intervals[1:]:
        prev = res[-1]                      
        if prev[1]>=cur[0]:                 # detect overlap of [prev] and [cur]
            prev[1] = max(cur[1], prev[1])      # extend "end" of [prev]
        else:
            res.append(cur)                 # no overlap, add [cur] directly
    return res

# 32.57 Insert Interval ========================================================================= https://leetcode.com/problems/insert-interval/
# Problem: Given a set of non-overlapping [intervals], insert a new interval [newInterval], merge overlap after inserting if any.
#          The given [intervals] is sorted initially in ascending order by their "starting"
# Descrption: If [newInterval] overlaps with elements in [intervals], there must exist that "newInterval[0] <= intervals[i][1]", where 
#             [newInterval] starts from "intervals[i]". Or there must exist that "newInterval[1] < intervals[j][0]", where [newInterval]
#             end with "intervals[j-1]". Thus, every intervals between [i] and [j] should be merged.
#             Use binary search to find [i] and [j]. And decide the boundary of merged intervals [merge_start] and [merge_end], where
#             "[merge_start] = min(intervals[i][0], newInterval[0])", and "[merge_end] = min(intervals[j][1], newInterval[1])". Combien
#             [merged] with the rest of [intervals] and return
# Time complexity: O(N), N=len(intervals) combine [merged] with rest of [intervals] takes O(N)
def insert(intervals, newInterval):
    if len(intervals) == 0:                     # corner case, [intervals] is empty
        return [newInterval]
        
    if newInterval[1] < intervals[0][0]:        # corner case, [newInterval] is smaller than every item in [intervals]
        return [newInterval] + intervals
    
    if newInterval[0] > intervals[-1][1]:       # corner case, [newInterval] is larger than every item in [intervals]
        return intervals + [newInterval]
    
    first = insert_helper(intervals, newInterval[0], True)          # find starting point of merge
    second = insert_helper(intervals, newInterval[1], False)        # find ending point of merge
    
    merge_start = min(intervals[first][0], newInterval[0])
    merge_start = max(intervals[second-1][1], newInterval[1])
    merged = [merge_start, merge_start]                             # get [merged] interval
    return intervals[:first] + [merged] + intervals[second:]        # combine [merged] with rest of [intervals]

def insert_helper(intervals, target, findLeft=True):
    left = 0
    right = len(intervals) 
    if findLeft:                                # bisect_left
        while left < right:
            mid = left + (right - left) // 2
            if intervals[mid][1] >= target:
                right = mid
            else:
                left = mid + 1
        return left
    else:                                       # bisect_right
        while left < right:
            mid = left + (right - left) // 2
            if intervals[mid][0] > target:
                right = mid
            else:
                left = mid + 1
        return left

# 33.59 Spiral Matrix II ====================================================================== https://leetcode.com/problems/spiral-matrix-ii/ 
# Problem: Given a positive integer [n], generate a "n*n" matrix fill with integers start from 1 to n^2 in sprial order
#          Ex n = 3  ==>  1 -> 2 -> 3
#                                   |
#                         8 -> 9    4
#                         |         |
#                         7 <- 6 <- 5
# Description: Track boundary [left], [right], [top], [bottom]. While [left]<[right] and [top]<[bottom], traval along the boundary and fill 
#              numbers. Shrink boundary while traveling. After travel "left" to "right", reduce [top]. After travel "top" to "bottom", reduce
#              [right]. After travel from "right" to "left", reduce [bottom]. After travel from "bottom" to "top" reduce [left]
# Time complexity: O(n*n)
def generateMatrix(n: int) -> List[List[int]]:
    res = [[0 for j in range(n)] for i in range(n)]     # create a empty n*n matrix initially
    left, right, top, bottom = 0, n, 0, n               # initialize boundary
    n = 1                                               # initialize number to be filled
    while left<right and top<bottom:
        for col in range(left, right):                      # left to right
            res[top][col] = n
            n+=1
        top += 1
        if top == bottom: return res
        for row in range(top, bottom):                      # top to bottom
            res[row][right-1] = n
            n+=1
        right -= 1
        if left == right: return res
        for col in reversed(range(left, right)):            # right to left
            res[bottom-1][col] = n
            n+=1
        bottom -= 1
        if top == bottom: return res
        for row in reversed(range(top, bottom)):            # bottom to top
            res[row][left] = n
            n+=1
        left += 1
        if left==right: return res
    return res

# 34.61 Rotate List ================================================================================= https://leetcode.com/problems/rotate-list/
# Problem: Given a [head] of a linked list, and a positive integer [k]. Rotate elements in linked list to the right by [k] places
#          Ex: 1->2->3->4->5, k=3
#              rotate once: 5->1->2->3->4 
#              rotate twice: 4->5->1->2->3
#              rotate triple: 3->4->5->1->2
# Description: Iterate though the list to get [size]. Get actual [rotates] need by [k%size]. Find the [rotates]th last element from end of list,
#              Disconnect elements [rotates] and [rotates+1], make [rotates+1] the new [head], and connect [tail] element to old [head]
# Time Complexity: O(N)
def rotateRight(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    if not head:
        return None
    lastElement = head
    length = 1
    while ( lastElement.next ):
        lastElement = lastElement.next
        length += 1
    k = k % length
    lastElement.next = head
    tempNode = head
    for _ in range( length - k - 1 ):
        tempNode = tempNode.next
    answer = tempNode.next
    tempNode.next = None
    return answer

# 35.62 Unique Paths ====================================================================================== https://leetcode.com/problems/unique-paths/
# Problem: Given a m*n grid, a robot starts from top-left corner. The robot can only move down or right, how many possible unqiue path to go to bottom
#          right corner.  1 <= m, n <= 100
# Description: Permutation with duplicates. The problem can be converted to permutation of (m-1) downs and (n-1) rights. According to the formula of 
#              permutation with duplications: [(m-1)+(n-1)]! / [(m-1)! * (n-1)!]
# Time compleixty: O(m+n)
import math
def uniquePaths(m: int, n: int) -> int:
    m -= 1
    n -= 1
    return math.factorial(m+n)//math.factorial(m)//math.factorial(n)

# Description: Dynamic Programming, Maintain a 2d matrix that each element dp[i][j] represents number of unique path to goto [i][j]. In order to goto 
#              [i][j], the robot can only move from [i][j-1] or from [i-1][j]. Thus, the value of [i][j] is the sum of [i-1][j] and [i][j-1]. The first
#              row and first column are all "1"s, since there is only one unique path along first row and first column. Return dp[-1][-1] at the end
def uniquePaths_2(m: int, n: int) -> int:
    dp = [[1 for i in range(n)] for j in range(m)]
    for r in range(1, m):
        for c in range(1, n):
            dp[r][c] = dp[r][c-1] + dp[r-1][c]
    return dp[-1][-1]

# 36.63 Unique Paths II ================================================================================== https://leetcode.com/problems/unique-paths-ii/
# Problem: Given a [m*n] array represents a grid, where "0" element represent a available path, and "1" represent a obstacleG that robot can not pass
#          The robot start from top-left corner and try to reach bottom-right corner, the robot only moves down or right. Find how many unique paths
#           would there be
# Description: Dynamic programming. Maintain a [m*n] matrix that each element dp[i][j] represent number of unique path to reach [i][j]. In order to 
#              goto [i][j], the robot can move from [i][j-1] or from [i-1][j]. Thus, the value of [i][j] is the sum of [i-1][j] and [i][j-1]. However,
#              if [i][j] is an obstacle, then there is no path, meaning [i][j] = 0. Initially, first row and first column of [dp] are 1, since there
#              is only one way to reach those cells. Unless there's a obstacle on the path, then cells after that obstacle are not reachable.
# Time complexity: O(m*n)
def uniquePathsWithObstacles(obstacleGrid: List[List[int]]) -> int: 
    dp = [[0 for _ in obstacleGrid[i]] for i in range(len(obstacleGrid))]
    for i in range(len(obstacleGrid)):      # initial value for first column
        if obstacleGrid[i][0] == 1:
            break
        dp[i][0] = 1
    for j in range(len(obstacleGrid[0])):   # intial value for first row
        if obstacleGrid[0][j] == 1:
            break
        dp[0][j] = 1
    for i in range(1, len(obstacleGrid)):           
        for j in range(1, len(obstacleGrid[i])):
            if obstacleGrid[i][j] == 1:
                dp[i][j] = 0
            else:
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[-1][-1]

# 37.64 Minimum Path Sum ============================================================================ https://leetcode.com/problems/minimum-path-sum/
# Problem: Given a [m*n] grid filled with non-negative integers, where each element of grid represent to cost to go through that cell. Find a path
#          from top-left corner to bottom-right corner with minimal cost, and return the minimal cost. You can only move down or right for each step
# Description: Dynamic Programming. Maintain a [m*n] matrix that each element dp[i][j] represent the minimal cost to reach that cell. In order to 
#              go to bottom-right corner, and only move down or right, you can either move from [i-1][j] to [i][j] or move from [i][j-1] to [i][j].
#              Thus, [i][j] = min([i-1][j], [i][j-1]) + [i][j]. Return dp[-1][-1] at the end
# Time complexity: O(m*n)
def minPathSum(grid: List[List[int]]) -> int:
    m,n = len(grid), len(grid[0])
    for i in range(1, m):
        grid[i][0] += grid[i-1][0]
    for i in range(1, n):
        grid[0][i] += grid[0][i-1]
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i-1][j], grid[i][j-1])
    return grid[-1][-1]

# 38.71 Simplify Path ============================================================================ https://leetcode.com/problems/simplify-path/
# Problem: Given a string [path], which is a "absolute path" to a file or dirctory in a Unix-style file system, convert it to simplified 
#          "canonical path". In Unix-style file system, a period "." refer to current directory, a double period ".." refers to directory up a
#          level, multiple slashes "//" are treated as single slash "/". Any other format are treated as file/directory name.
#          A "Canonical path" should have following format
#          1. path starts with a slash "/"
#          2. any two directories are separated by a slash "/"
#          3. path does not end with slash "/"
#          4. the path only contains directories on the path from root directory to current file/directory
#          Ex: "/a/./b/..//../c/"  ==>  "/c"
# Description: Use a "stack" to track history of direcories. Split [path] by slash "/". Maintain a [stack] and iterate the splited array, 
#              if ".." pop last element from stack, if "." or empty string "", do nothing and move on to next, else append element to stack.
#              The [stack] contains the "canonical path" as an array. Combine elements in array to a string with slash between each element, 
#              and add one more slash at beginning. Return the constructed string
# Time complexity: O(n)
def simplifyPath(path: str) -> str:
    stack = []                  # maintain a stack to track directory history
    path = path.split("/")
    for direcotry in path:
        if direcotry == "..":           # move up level if ".."
            if stack:
                stack.pop()
        elif direcotry == "." or direcotry == "":       # do nothing if "." or empty string
            continue
        else:                               # add directory to stack
            stack.append(direcotry)
    return "/"+"/".join(stack)              # combine elements of stack to build canonical path

# 39.73 Set Matrix Zeroes ========================================================================== https://leetcode.com/problems/set-matrix-zeroes/
# Problem: Given a "m*n" [matrix]. If a element in matrix is "0", then set its entire row and column to "0" in-place
#          Consider solve the problem with constant space
# Description: Use first row and first column as "marker" to denote which row/coulmn need to be set to "0".Iterate through the elements in [matrix], 
#              if the element at [i][j] is "0", set the [i][0] and [0][j] to "0". [i][0] and [0][j] will be the marker to set entire [i]th row and 
#              entire [j]th column to "0" in the second iteration. Iterate the elements in [matrix] the second time, for element [i][j], if [i][0] 
#              or [0][j] equals to "0", meaning [i][j] is on the row or column to be set to "0". Since we use first row and first column as "marker",
#              need to set first row/column separately. Use two booleans [firstRowZero] and [firstColZero], detect if first row/column has "0", and 
#              set entire corresponding row/column to "0"
# Tiem compleixty: O(m*n)
def setZeroes(matrix: List[List[int]]) -> None:
    m, n = len(matrix), len(matrix[0])
    # detect "0" in first row and first column
    firstRowZero = firstColZero = False     
    for i in range(m):
        if matrix[i][0] == 0:
            firstColZero = True
            break
    for j in range(n):
        if matrix[0][j] == 0:
            firstRowZero = True
            break
    # find "0" and set corresponding first row/column to "0" as marker
    for i in range(1,m):
        for j in range(1,n):
            if matrix[i][j] == 0:
                matrix[i][0] = matrix[0][j] = 0
    # set entire row/column to "0", if marker is "0"
    for i in range(1, m):
        for j in range(1,n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    # set entire first row/column to "0", if there exist "0" originally
    if firstRowZero:
        matrix[0] = [0]*n
    if firstColZero:
        for j in range(m):
            matrix[j][0] = 0

# 40.74 Search a 2D Matrix ============================================================================ https://leetcode.com/problems/search-a-2d-matrix/
# Problem: Given a "m*n" [matrix], where numbers in a row are sorted, and the first element of each row is alway larger than the last element of previous
#          row. Given a number [target], check if [target] exists in [matrix], return True or False accordingly
# Description: Treat the [matrix] as a sorted list, and use binary search. The first element is index 0 and last element is index "m*n-1". For element at
#              index [i], its index in [matrix] is [i//n][i%n]. Because [i//n] is the row index and [i%n] is the column index. Use binary search and get
#              [mid] element and compare it with [target], swap index [left] and [right] with [mid] accordingly
# Time compleixty: O(log(m*n))
def searchMatrix(matrix: List[List[int]], target: int) -> bool:
    m,n=len(matrix), len(matrix[0])
    left, right = 0, m*n-1              # binary search, [left] starts from top-left, [right] starts from bottom-right
    while left<=right:
        mid = (left+right)//2
        num = matrix[mid//n][mid%n]     # convert [mid] to the row and column of matrix
        if num < target:
            left = mid+1
        elif num > target:
            right = mid-1
        else:
            return True
    return False

# 41.75 Sort Colors ========================================================================================= https://leetcode.com/problems/sort-colors/
# Problem: Given a list of integers, and each element can only be 0, 1 or 2, where 0 is "red", 1 is "white", 3 is "blue". Sort the list in-place, so that 
#          "red" come before "white", "white" come before "blue". Consider a solution with O(n) time complexity and O(1) space complexity
# Description: Dutch national flag problem, which is a problem to partition a list into three groups. Maintain pinters [red] and [blue]. [red] starts from
#              index 0 to represent the boundary between [red] and [white], meaning index [0, red) are "red" elements. [blue] starts from last index
#              to represent to boundary between [white] and [blue], meaning index (blue, len-1] are "blue" elements. Maintain another pointer [cur], 
#              move [cur] from index 0 towards the end of list. 
#              If [cur] is "red", swap [red] with [cur] make [red] contains "red" element and increase both of them, since element on [red] must be "white"
#              If [cur] is "white", no swap and increase [cur]
#              If [cur] is "blue", swap [blue] with [cur] make [blue] coontains 'blue" element and decrease [blue], no increament on [cur], since the 
#              element swap to [cur] is unknown
# Time Compelexity: O(n)
# Space complexity: O(1)
def sortColors(nums: List[int]) -> None:
    red, blue = 0, len(nums)-1      # maintain index boundary of "red", "white" and "blue"
    cur = 0
    while cur<=blue:                # stop when [cur] hit [blue] because element after [blue] are all "blue"s
        if nums[cur] == 0:              # hit "red" element
            nums[cur], nums[red] = nums[red], nums[cur]
            cur += 1
            red += 1
        elif nums[cur] == 1:            # hit "white" element
            cur += 1
        else:                           # hit "blue" element
            nums[cur], nums[blue] = nums[blue], nums[cur]
            blue -= 1

# 42.77 Combinations =========================================================================================== https://leetcode.com/problems/combinations/
# Problem: Given two integers [n] and [k], return all combinations with length of [k], and consists of number of from [1, n]
# Description: DFS. Create a helper function, that takes three parameters [cur], [temp] and [res]. [cur] start from 1 and increases up to [n],
#              representing the current number need to be added to a combination. [temp] tracks the current combination sub-list. [res] is the result
#              2D list contains all combinations. If len(temp) is [k], current combination is finished and append [temp] to [res], return to stop this 
#              DFS. Recusive case, use a "for-loop" iterate from 1 to [n] and invoke "helper" function, the invoked "helper" function take i+1 as [cur],
#              append [i] tp [temp], and pass down [res].
# Time complexity: O(n^k)
def combine(n: int, k: int) -> List[List[int]]:
    def combine_helper(cur, temp, res):
        if len(temp) == k:                          # append [temp] to [res] when size is achieved
            res.append(temp)
            return
        for i in range(cur, n+1):
            combine_helper(i+1, temp+[i], res)      # increase [cur] by 1, append [i] to [temp]
    res = []
    combine_helper(1, [], res)
    return res
    
# 43.78 Subsets ============================================================================================== https://leetcode.com/problems/subsets/
# Problem: Given a list of unique integers [nums], return all possible subsets (power set) as 2d list. The solution should not contain duplication.
# Description: DFS, create a helper function that tracks [pool] as the element need to be appended to a subset, and [temp] represent the current 
#              subset. Invoke the helper function in a "for-loop" that itarete through elements in [pool] to form DFS. Each invoke of helper function, 
#              append [temp] to [res], since every [temp] is a valid subset, and remove the first element from [pool] and insert it into [temp]. If no 
#              more element left in [pool], return to stop the dfs path
# Time complexity: O(2^n)
def subsets(nums: List[int]) -> List[List[int]]:
    def subsets_helper(pool, temp, res):
        res.append(temp)                                    # append [temp] to [res], since every [temp] is a subset
        for i in range(len(pool)):
            subsets_helper(pool[i+1:], temp+[pool[i]], res)     # remove first element from [pool], append first element to [temp]
    res = []
    subsets_helper(nums, [], res)
    return res

# Description: Dynamic Programming, start with only an empty set "[]" in [res]. For every element in [nums], we can either add it to every existing 
#              subsets or leave the subsets as they are. For the first element, if we add the element we have [ele_1] as a new subset, and if we don't 
#              add the empty set remain the same. Now we have [res] = [[], [ele_1]]. For the second element [ele_2] to every existing subsets, we 
#              have [ele_2], [ele_1, ele_2], if we don't add, subsets remain same, then [res] = [[], [ele_1], [ele_2], [ele_1, ele_2]]. Do this for 
#              every elements in [nums].
# Time compleixty: O(2^n)
def subsets_2(nums: List[int]) -> List[List[int]]:
    res = [[]]
    for n in nums:
        # take every existing subsets in [res] append with [n]
        # take every existing subsets in [res] but don't add [n]
        # concatenate "added" and "not added" subsets
        res += [subset + [n] for subset in res]     
    return res

# 44.79 Word Search ====================================================================================== https://leetcode.com/problems/word-search/
# Prolem: Given a "m*n" grid [board] with English characters, and a string [word]. Determine if [word] can be constructed from characters from [grid]
#         by connecting adjacent cells horizontally or vertically, where a cell can only be used once. Return True if [word] is constructed, otherwise
#         return False
# Description: DFS, iterate through every cell of [grid] that make each cell as the starting point to find [word]. Create a "helper function" use DFS
#              to find [word]. "helper function" maintain the current index [i] and [j], if [i] and [j] are within the range of [board], check the 
#              char in cell [i][j]. If the char is same as first char of [word]. Remove the first char of [word] to search for next char,  mark char
#              at [i][j] to "#" as it can not be used twice. Continue DFS on upper, bottom, left, and right cell by changing parameter of [i] and [j],
#              pass the modified [board] and [word] to new recursive calls. Restore value of [i][j] after all four recursive calls, return value of 
#              four recursive calls with OR gate. "helper function", return True when [word] is empty, meaning [word] is found in [board]
# Time complexity: O((n*m)^2)
def exist(board, word):
    for i in range(len(board)):                 # make every cell as the starting point
        for j in range(len(board[i])):
            if exist_helper(i, j, board, word):     # invoke DFS helper function
                return True
    return False
    
def exist_helper(i, j, board, word):
    if not word:                    # word is empty, [word] is found
        return True
    if i<0 or i>=len(board):        # [i] out of bound
        return False
    if j<0 or j>=len(board[i]):     # [j] out of bound
        return False
    if board[i][j] != word[0]:      # current cell [i][j] doesn't match
        return False
                                    # current [i][j] matches first char of [word]
    temp = board[i][j]                          # record value of [i][j] for restore 
    board[i][j] = "#"                           # mark [i][j] as visited
    word = word[1:]                             # remove first char of [word]
    # DFS on four directions
    down = exist_helper(i+1, j, board, word)    
    right = exist_helper(i, j+1, board, word) 
    up = exist_helper(i-1, j, board, word) 
    left = exist_helper(i, j-1, board, word)
    board[i][j] = temp                          # restore value of [i][j]
    return down or right or up or left

# 45.80 Remove Duplicates from Sorted Array II ================================= https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/ 
# Problem: Given a sorted integer array [nums], remove duplicates "in-place" such that duplicates appears at most twice, and return the new length.
#          Consider modify the array "in-place" with O(1) extra space
# Description: Maintain pointers [slow] and [fast], both start from index 2. [slow] track the tail of result list, [fast] finding non-duplicate element
#              and replace duplicate element on [slow]. And the value of [slow] alway less than or equals to [fast]. If [slow-2] == [fast], means the 
#              element [slow-2], [slow-1] and [slow] must share same value, thus the value of [slow] must be replaced, then move [fast] to find a new
#              value that doest not equal to [slow-2] and replace it. If [slow-2] < [fast], an eligible value is found on [fast], [fast] should replace
#              [slow] to remove duplicate, and move both [slow] and [fast] to next element. End iteration when [fast] hit end of list.
# Time Compleixty: O(n)
def removeDuplicates(nums: List[int]) -> int:
    slow = fast = 2
    while fast < len(nums):         # end iteration when [fast] hit end
        if nums[slow-2] < nums[fast]:       # [fast] can be used to replace [slow]
            nums[slow] = nums[fast]
            slow += 1                       # duplicates is eliminated for current [slow], move [slow] to next
        fast += 1                   # move [fast] to find a non-duplicate element to replace, also move [fast] after replacement
    return slow             

# 46.81 Search in Rotated Sorted Array II ==================================== https://leetcode.com/problems/search-in-rotated-sorted-array-ii/
# Problem: Given an integer list [nums], sorted in ascending order with duplicated elements. [nums] is rotated at an unknown pivot index, such as
#          [0,1,1,2,3] rotate at pivot index 2 becomes [1,2,3,0,1].
#          Given an integer [target], determine if [target] exists in [nums]. Return True or False accordingly
# Description: Leverage algorithm of "Search in Rotated Sorted Array"(without duplicated elements). Maintain pointer [l] and [r] that starts from
#              index [0] and index [len(nums)-1], get [mid] = ([l]+[r])//2, and apply binary search on sorted sublist. If [l]<=[mid], that left 
#              sublist from [l] to [mid] is sorted, if [target] is in between [l] and [mid], shink [r] = [mid]-1, if [target] doesn't fall in the 
#              range of [l] and [mid], move [l] = [mid]+1 to search the other sublist. If [mid]<=[r], that right sublist from [mid] to [r] is 
#              sorted, if [target] is in between [mid] and [r], shink [l]=[mid]+1, if [target] doesn't fall in the range of [mid] and [r], move
#              [r] = [mid]-1 to search the other sublist. And if [mid]==[target], return True. 
#              Since [nums] contains duplicates, we should skip duplicates. While [l] == [mid], keep moving [l] to right, until index [l] equals 
#              to index [mid] or [l]!=[mid]
# Time complexity: O(logn)
def search_2(nums: List[int], target: int) -> bool:
    l, r = 0, len(nums)-1
    while l<=r:
        mid = (l+r)//2
        if nums[mid] == target:                 # target is found
            return True  
        while l < mid and nums[l] == nums[mid]:     # move [l] to skip duplicates
            l += 1
        if nums[l] <= nums[mid]:                # left half is sorted
            if nums[l] <= target < nums[mid]:      # target is in the sorted half, move [r] to [mid] to continue search this sublist
                r = mid-1
            else:                                   #  target is not in the sorted half, move [l] to [mid] to search other sublist
                l = mid+1
        else:                                   # right half is sorted
            if nums[mid] < target <= nums[r]:      # target is in the sorted half, move [l] to [mid] to continue search this sublist
                l = mid+1
            else:                                   # target is not in the sorted half, move [r] to [mid] to search other sublist
                r = mid-1
    return False

# 47.82 Remove Duplicates from Sorted List II ============================ https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/
# Problem: Given the [head] of a sorted linked list, delete all node that appear more than once, and leave distince node in the list with their
#          original order. Return the head of modified linked list
# Description: Dummy node and Pointers. Maintain a pointer [tail] that tracks the end of linked list with duplicates removed. [tail] is a dummy
#              node initially, maintina [dummy] and [dummy.next] is returned as new head. Maintain [start] and [end] as the "starting" and 
#              "ending" node with same value. Move [end] to find the last element that have same value as [start], if [end] is [start.end] means 
#              there is no duplicate for current node, append the node [start] to [tail]. Move [tail] to next node, and move [start] to [end] to 
#              start new iteration. Stop the loop when [end] hit the end, assign None the [tail] and return [dummy.next]
# Time complexity: O(n)
def deleteDuplicates(self, head: ListNode) -> ListNode:
    dummy = tail = ListNode(0, None)
    start = end = head
    while end:
        while end and start.val == end.val:         # move [end] to find the end of duplicates
            end = end.next
        if start.next == end:           # no duplicate found, append [start] to [tail]
            tail.next = start
            tail = tail.next
            start = end
        else:                           # duplicate found
            start = end
    tail.next = None                    # append None to the end of linked list
    return dummy.next

# 48.86 Partition List ====================================================================== https://leetcode.com/problems/partition-list/
# Problem: Given a [head] of linked list, and a value [x]. Partition the nodes of linked list such that all nodes less than [x] comes before
#          the nodes greater than or equals to [x]. Preserve the original order within each partitions
# Description: Sperate nodes into two linked lists. Create two new dummy nodes of linked list, [dummyLess] is the dummy node that leading 
#              the "less than" portion, [dummyGreat] is the dummy node that leading the "greater or equal to" portion. Iterate through nodes
#              in original linked list, compare node with [x] and append into two linked list accordingly. Append None to the end of  
#              [dummyGreater], append the head of [dummyGreater] to [dummyList], and return the head of [dummyLess]
# Time Complexity: O(n)
def partition(head: ListNode, x: int) -> ListNode:
    dummyLess = tailLess = ListNode(0, None)            # create dummyLess to collect nodes less than [x], and maintain tail to add new node
    dummyGreat = tailGreat = ListNode(0, None)          # create dummyGreat node to collect nodes less than [x], maintain tail to add new node
    while head:                                 # iterate every node in [head] 
        if head.val<x:                              # if node.val < x, append it to [tailLess]
            tailLess.next = head
            tailLess = head
        else:                                       # if node.val >= x, append it ot [tailGreat]
            tailGreat.next = head
            tailGreat = head
        head = head.next                        
    tailLess.next = dummyGreat.next             # append "greater" portion to the end of "less" portion
    tailGreat.next = None                       # append None to the end of "greater" potion
    return dummyLess.next               # return head of "less" portion

# 49.89 Gray Code ============================================================================= https://leetcode.com/problems/gray-code/
# Problem: The gray code is a binary number system where two adjacent values differ in only one bit.
#          Given an integer [n] representing the number of bits in the gray code, return a sequence of gray code.
#          A gray code always starts with 0, Example n = 3
#          [000, 001, 011, 010, 110, 111, 101, 100] is a sequence of gray code
# Description: Iterate i from (0 to n), and maintain sequence as [res]. Every iteration, take existing elements in [res] in reversed order,
#              and add 2**i to every element and append them to [res]. The reason is explained below
#              Inspect sequence of n=3,
#              1. 000
#              2. 001
#              3. 011      010
#              4. 110      111     101     100
#              Initially, only 0 is in sequence, sequence is [0]
#              Take elements reversely from sequence, Add 2**0 to every elements, append them to sequence, sequence is [0, 1]
#              Take elements reversely from sequence, Add 2**1 to every elements, append them to sequence, sequence is [0, 1, 11, 10] 
#              Take elements reversely from sequence, Add 2**2 to every elements, append them to sequence, sequence is [0, 1, 11, 10, 110, 111, 101, 100] 
# Time complexity: O(2^n)
def grayCode(n: int) -> List[int]: 
    res = [0]
    for i in range(n):
        res += [val+2**i for val in reversed(res)]
    return res

# 50.90 Subsets II ========================================================================== https://leetcode.com/problems/subsets-ii/
# Problem: Given an integer list [nums], which may contain duplicates. Return all possible unique subsets of [nums].
# Description: DFS. Use backtracking to construct subsets same as "subset I". Sort [nums] before passing into backtracking, backtrack 
#              function maintains list of elements [nums] that have not been picked yet, the list of elements [temp] that are picked 
#              to construct a subset, and the [res] list that contains all possible subsets. Each backtrack call, append current subset
#              [temp] to [res], skip indices that nums[i-1] == nums[i]. Because taking equal value in the same DFS tree level creates 
#              duplicates subset. And invoke next level backtrack, that current element is removed from non-picked numbers, add current
#              element is appended to [temp]
# Time complexityL O(2**n)
def subsetsWithDup(nums: List[int]) -> List[List[int]]:
    def subsetWithDup_helper(nums, temp, res):
        res.append(temp)
        for i in range(len(nums)):
            if i>0 and nums[i-1]==nums[i]:
                continue
            else:
                subsetWithDup_helper(nums[i+1:], temp+[nums[i]], res)
    res = []
    subsetWithDup_helper(sorted(nums), [], res)
    return res

# 51.91 Decode Ways ========================================================================== https://leetcode.com/problems/decode-ways/
# Problem: Given a string [s], which contains integers that encoded from letter A-Z. The encoding mapping is A=1, B=2 ... Z=26. There exists
#          some digits that the same combination of digits can be decoded to multiple alphabet letters. Such as, [s]=11106 can be mapped to
#          1 1 10 6=>AAJF, and 11 10 6=>KJF. Note "06" cannot be mapped to "F", since "06" is different from "6".
#          Given the string [s] and return number of possible ways to decode [s] 
# Description: Dynamic Programming. Maintain a list [dp], where dp[i] represent the number of ways to decode substring s[:i]. 
#              If 0 < s[i-1] <= 9,it means s[i-1] itself can be decoded to a letter from A to I, the number of ways to decode s[:i-1] should
#              be inheritant to decode s[:i], so dp[i] += dp[i-1]. If 10<=s[i-2:i]<=26, means s[i-2] and s[i-1] can be decoded to a letter
#              from J to Z, the number of ways to decode s[:i-2] should also be inheritant to decode s[:i], thus dp[i] += dp[i-2]. Create [dp] 
#              with len(s)+1 element, initially all set to zero. dp[0] is the base case and dp[0]=1. dp[1] is the number of ways to decode 
#              first letter, if s[0] is "0" and "0" cannot be decode to any letter, so dp[1] = 0 if s[0]=="0", if s[0] != "0" there is one way 
#              to decode a single letter, thus dp[1]=1 if s[0]!="0".
#              Itearte through [s] from index 2 to len(s)+1, dupdate dp[i] as the number of ways to decode s[:i]. Return dp[-1] at the end
# Time complexity: O(n)
def numDecodings(s: str) -> int:
    dp = [0 for _ in range(len(s)+1)]           # create dp with len(s)+1 elements
    dp[0] = 1                                       # initial base offset dp[0] 
    dp[1] = 0 if s[0]=="0" else 1                   # initial number of ways to decode s[0]
    for i in range(2, len(s)+1):              
        if 0 < int(s[i-1]) <= 9:                    # s[i-1] can be decoded   
            dp[i] += dp[i-1]
        if 10 <= int(s[i-2:i]) <= 26:               # s[i-2] and s[i-1] combined can be decoded
            dp[i] += dp[i-2]
    return dp[-1]

# Description: Dynamic Programming, with Two variable tracking dp[i-2] and dp[i-1]. If take one character and it is between 1 and 9, use dp[i-1].
#              If take two character and it's between 10 and 26, use dp[i-2]
# Space Complexity: O(1)
def numDecodings(s: str) -> int:
    preTwo = 1
    preOne = 1 if s[0]!="0" else 0
    for i in range(1, len(s)):
        takeOne = preOne if 0<int(s[i])<10 else 0
        takeTwo = preTwo if 9<int(s[i-1:i+1])<27 else 0
        preTwo = preOne
        preOne = takeOne+takeTwo
    return preOne

# 52.92 Reverse Linked List II ======================================================== https://leetcode.com/problems/reverse-linked-list-ii/
# Problem: Given the [head] of a linked list, and two integers [left] and [right], where [left]<=[right]. Reverse the node from position [left]
#          to position [right], return the head of reversed list.
# Description: Create a [dummy] node where "dummy.next = head". Iterate from [dummy] to find the last node before position [left], denote it 
#              as [beg], the loop should iterate [left-1] times. 
#              After found [beg], use algorithm of "Reverse Linked List". Maintain three pointers [prev], [cur], and [next], where initally [prev]  
#              is None, [cur] points the node at position [left]. Iterate a loop [right-left+1] times, in which [next] is assigned with [cur.next]
#              [cur.next] points to [prev], and shift both [prev] and [cur] to next position.
#              Connect the revsered linked list with the reset of the node. At the end of loop, [prev] should points to the node at position [right] 
#              [cur] should points to the node right after [prev]. Thus, [beg.next.next] which is the last node of reversed list, should points to
#              [cur], which is the first node after reversed part. And [beg.next] should points to [prev] the connect to the first node of reversed
#              list.
# Time complexity: O(n)
def reverseBetween(head: ListNode, left: int, right: int) -> ListNode: 
    if left==right:                     # corner case, no need to reverse if left==right
        return head
    dummy = ListNode(0, head)
    beg = dummy
    for _ in range(left-1):             # find the node before position [left]
        beg = beg.next
    # algorithm of Reverse Linked List
    prev = None
    cur = beg.next
    for _ in range(right-left+1):       # reverse list
        next = cur.next
        cur.next = prev
        prev = cur
        cur = next  
    beg.next.next = cur                 # connect tail of reversed list to the node comes after 
    beg.next = prev                     # connect head of reversed list to the node comes before
    return dummy.next
    
# 53.93 Restore IP Addresses ================================================================== https://leetcode.com/problems/restore-ip-addresses/
# Problem: Given an integer string [s] containing only digits, return all possible valid IP address constructed from [s].
#          A valid IP consists of four parts separated by period ".", and each part is a integer number between 0 and 255. Each part can't have 
#          leading zero, but zero by itself is OK.
#          Ex: 1022310119 => 10.223.101.19 is a valid IP address
#                         => 102.231.0.11 is a valid IP address, since zero by itself is OK
#                         => 1.023.101.19 is invalidm "023" has leading zero
# Descritpion: DFS backtracking. Maintain [pool] as substring not taken, [temp] list track the valid IP sub-domains taken from [pool], [res] a set
#              that contains constructed valid IP addresses. For each recursive call, If [temp] contains four elements and no more char left in 
#              [pool], means a valid IP address is created, add [temp] to [res] in IP format. If len(temp)>4, number of domain is excessed, return
#              to stop this traversal. The recursive case, use for loop iterate [i] from 1 to 3, and validate pool[:i] to be inserted into [temp],
#              if pool[:i] is empty or excess 255, or pool[:i] contains multiple chars with leading zero, return this traversal because it is not 
#              possbile to get a valid domain for this recursive level.
#              Invoke new recursive call with pool[i:] to eliminate taken substring. temp+[take] that [take] is the valid domain pool[:i], and add
#              it to current constructed IP address, also pass [res] to next level
def restoreIpAddresses(s: str) -> List[str]:
    res = set()
    restoreIpAddresses_helper(s, [], res)
    return [el for el in res]

def restoreIpAddresses_helper(pool, temp, res):
    if len(temp)==4 and len(pool)==0:                           # valid IP address is created, [temp] has four domains and no char left in [pool]
        res.add(".".join(temp))                                     # add valid IP to [res]
        return
    if len(temp)>4:                                             # number of domains is exceeded
        return
    for i in range(1,4):                                        # take first three char from [pool] and valid them
        take = pool[:i]
        if not take or int(take)>255:                               # [take] is empty or exceeded 255
            return
        if i!=1 and take[0] == "0":                                 # [take] contains leading zero
            return
        restoreIpAddresses_helper(pool[i:], temp+[take], res)

# 54.95 Unique Binary Search Trees II ================================================ https://leetcode.com/problems/unique-binary-search-trees-ii/
# Problem: Given an integer n, return all the structurally unique binary search tree, which contains nodes from 1 to n. Return the answer in any
#          order
# Description: DFS backtracking. [start] and [end] are value boundary of a root, and take [i] at root. Then, [start, i) belongs to left tree, and
#              (i, end] belongs to right tree. Recursively backtracking to create a list with all valid trees on both left and right, then connect
#              them with root one by one and append to result list
#              Create a recursive helper function, that take [start] and [end] as the node value should be created within it.
#              Iterate [i] from [start] to [end], where [i] is the root of tree that contains node between [start] to [end]. According to BST
#              property, left subtree of [i] contains nodes from [start] to [i-1], and right subtree contains nodes from [i+1] to [end]. Thus,
#              recursively call helper function on (start, i-1) and (i+1, end). Both recursive call will return a list of nodes which are 
#              all valid BS subtrees created. Connect all nodes returned from (start, i-1) as the left node of [i], and connect all nodes returned
#              from (i+1, end) as the right node of [i]. Append node [i] to a list [res], which is returned and to be used by upper level.
# Time complexity: see Catalan Number
def generateTrees(n: int) -> List[TreeNode]:
    def helper(left, right):
        if left >= right:
            return [None]
        res = []
        for i in range(left, right):
            leftList = helper(left, i)                  # all valid subtrees on left
            rightList = helper(i+1, right)              # all valid subtrees on right
            for leftNode in leftList:
                for rightNode in rightList:
                    res.append(TreeNode(i, leftNode, rightNode))        # create a [root] and connect with each valid left and right combination,
        return res
    return helper(1, n+1)

# 55.96 Unique Binary Search Trees ==================================================== https://leetcode.com/problems/unique-binary-search-trees/
# Problem: Givne an integer [n], return the number of structually unique binary search trees which contains exactly nodes from 1 to [n].
# Description: Dynamic Programming. Consider the number of structually unique BST of [n] nodes is denoted as G(n). If we pick 1 as the root node,
#              no node in left subtree, and n-1 nodes in right subtree. It produces two sub-problems aka G(0) and G(n-1). If we pick 2 as the
#              root node, 1 node in left subtree G(1) and n-2 nodes in right subtree G(n-2). So on so forth, If we pick n-1 as the root node, there
#              are n-2 nodes in left subtree G(n-2), and 1 node in right subtee G(1). If we pick n as the root node, there aren-1 nodes in left 
#              subtree G(n-1) and 0 node in right subtree G(0). And the number of possiblilty is the cross-product of left and right subtree, thus
#              G(i-1)*G(n-i) the production of left and right is the number of BSTs when pick [i] at the root. Thus it has
#              G(n) = G(0)*G(n-1) + G(1)*G(n-2) + ... + G(n-2)*G(1) + G(n-1)*G(0). Use DP to record from G(0) to G(n), and return G(n) at the end
# Time complexity: O(n^2) build [dp] of size n, each dp[i] need to iterate i times
def numTrees(n: int) -> int:
    dp = [0]*(n+1)
    dp[0], dp[1] = 1, 1                 # G(0) and G(1) both equals to 1, since zero node and 1 node both have one unique structual
    for i in range(2, n+1):                 # iterate [i] to build [dp]
        for j in range(i):                      # [j] represents pick [j] at root
            dp[i] += dp[j]*dp[i-j-1]
    return dp[n]
    
# 56.97 Interleaving String =============================================================== https://leetcode.com/problems/interleaving-string/
# Problem: Given three strings [s1], [s2], [s3], check if [s3] is formed by an interleaving of [s1] and [s2].
#          And interleaving of two string [s] and [t] is a string that consists of character from [s] and [t], where the sequence of characters
#          of [s] and [t] doesn't change
#          Ex: s1 = [abcdef], s2 = {ghijk}, one of the interleaving s3 = {gh}[a]{i}[bcd]{j}[e]{k}[f]
#          where if we remove character of s2 then s3 becomes s1, and if we remove character of s1 then s3 become s2
# Description: Dynamic Programming. Create a 2d boolean list, with len(s1)+1 rows and len(s2)+1 columns. A cell dp[i][j] represents if substring
#              s3[:i+j] is an interleaving that formed from characters of s1[:i] and s2[:j]. To determine value of dp[i][j], it can be formed in
#              two ways. 1) from dp[i-1][j] and pick s1[i-1] as next character, if dp[i-1][j] is True and s1[i-1]==s[i+j-1], meaning dp[i-1][j] 
#              formed an interleaving, and adding s1[i-1] also form an interleaving, then dp[i][j] is True. 2) from dp[i][j-1] and pick s2[j-1] 
#              as next character, if dp[i][j-1] is True and s2[j-1]==s3[i+j-1], then dp[i][j] is Ture. Follow the manner and fullfill every cell
#              in dp, and return dp[-1][-1] as result
# Time complexity: O(m*n), n, m = len(s1), len(s2)
def isInterleave(s1: str, s2: str, s3: str) -> bool:
    r, c, l = len(s1), len(s2), len(s3)
    if r+c!=l:                  # early termination if the size of s1, s2, and s3 doesn't match
        return False
    dp = [[False for _ in range(c+1)] for _ in range(r+1)]      # create [dp]. has len(s1)+1 rows and len(s2)+1 columns
    dp[0][0] = True                                             # when there is not character picked, [0][0] is always true
    for i in range(1, r+1):                                         # the first row represent only pick characters from s2 to form interleaving
        dp[i][0] = dp[i-1][0] and s1[i-1]==s3[i-1]
    for j in range(1, c+1):                                         # the first column represent only pick character from s1 to form interleaving
        dp[0][j] = dp[0][j-1] and s2[j-1]==s3[j-1]
    for i in range(1, r+1):
        for j in range(1, c+1):                                         # iterate thorugh cells in [dp]
            dp[i][j] = (dp[i-1][j] and s1[i-1]==s3[i+j-1]) or (dp[i][j-1] and s2[j-1]==s3[i+j-1])  # fill [i][j] from either [i-1][j] or [i][j-1]
    return dp[-1][-1]               # last element of [dp] holds result

# Description: DP with Space complexity O(len(s2)). Maintain a DP of size len(s2)+1, that represent a row of DP in previous method. Populate initial
#              DP by only picking character of [s2]. Iterate through [s1] and [s2], dp[0] is only picking character of [s1]. 
#              note: DP has len(s1)+1 rows and len(s2)+1 columns
# Time complexity: O(n*m)
# Space complexity: O(m)
def isInterleave(s1: str, s2: str, s3: str) -> bool:
    if len(s1)+len(s2)!=len(s3): return False
    dp = [True for _ in range(len(s2)+1)]
    for j in range(1,len(s2)+1):
        dp[j] = dp[j-1] and s2[j-1] == s3[j-1]              # initialize dp by picking only s2 character
    for i in range(1, len(s1)+1):
        dp[0] = dp[0] and s1[i-1]==s3[i-1]                  # index 0 represent picking only s1 character
        for j in range(1, len(s2)+1):
            dp[j] = (dp[j-1] and s2[j-1] == s3[i+j-1]) or (dp[j] and s1[i-1] == s3[i+j-1])   # (take char from [s2]) or (take char from [s1])
    return dp[-1]

# 57.98 Validate Binary Search Tree =================================================== https://leetcode.com/problems/validate-binary-search-tree/
# Problem: Given the [root] of a binary search tree, determine if the tree is valid. A valid BST has following proerties:
#          1. left subtree of a [node] contains nodes that smaller than [node.val]
#          2. right subtree of a [node] contains nodes that larger than [node.val]
#          3. both left and right subtrees also satisfy there two rules
# Description: DFS. Maintain helper function which track the [floor] and [ceiling] of current [node.val]. For a [node], the node in its left subtree
#              can have a value that is less than [node.val] meaning ceiling = node.val, and the node in its right subtree can have value that is 
#              greater than [node.val] meaning floor = node.val. [floor] and [ceiling] are intially "float(-inf)" and "float('inf)" and passed down to 
#              each tree level
def isValidBST(root: TreeNode) -> bool: 
    return isValidBST_helper(root, float('-inf'), float('inf'))         # traverse from root of tree, give initial value of floor and ceiling

def isValidBST_helper(root, floor, ceiling):
    if not root:                                # no node left, this path is valid
        return True
    if root <= floor or root >= ceiling:        # current val is invalid
        return False
    # traverse to left child, use root.val as ceiling
    # traverse to right child, use root.val as floor
    # return True when both subtrees are valid
    return isValidBST_helper(root.left, floor, root.val) and isValidBST_helper(root.right, root.val, ceiling)
                
# 58.99 Recover Binary Search Tree =================================================== https://leetcode.com/problems/recover-binary-search-tree/
# Problem: Given the [root] of BST, there are exactly two nodes were swapped and out of place. Swap these two node back to place
# Description: Inorder traversal with stack. If traverse BST inorder, the node values are traversed in ascending order. If node are swapped. then
#              there will have node value than [prev.val] > [cur.val]. And there are two cases: 
#               1) swapped nodes are adjacent, the nodes sequence looks like following when traverse inorder, "... nodes < A > B < nodes ...". Node 
#                   "A" and "B" are out of place, need to swap them.
#               2) swapped nodes are not adjacent, the sequence looks like, "... nodes < A > X <  ... < Y > B < nodes ... ". There are two value
#                   decreasings, and "A" and "B" are the nodes that out of place, need to swap them. 
#              In both case, we want to swap "A" and "B", thus maintain a list of tuples [wrong] to record nodes that out of place. 
#              If swapped nodes are adjacent, then [wrong] contain single tupple (A, B), swap wrong[0][0].val and wrong[0][1].val
#              If swapped nodes are not adjacent, then [wrong] contains two tupples (A, X) and (Y, B), swap wrong[0][0].val and wrong[1][1].val 
#              Swap in both case can be simplified as swapping wrong[0][0].val and wrong[-1][1].val
# Time complexity: O(N)
# Space complexity: O(logN)
def recoverTree(root: TreeNode) -> None:
    cur, prev = root, TreeNode(float('-inf'))               # [prev] as previous node of [cur], that [prev.val] < [cur.val]
    stack, wrong = [], []
    while cur or stack:                                      # inorder traversal with [stack]
        if cur:
            stack.append(cur)
            cur = cur.left
        elif stack:
            cur = stack.pop()
            if cur.val<prev.val:                            # detect swapped nodes
                wrong.append((prev, cur))
            prev, cur = cur, cur.right
    wrong[0][0].val, wrong[-1][1].val = wrong[-1][1].val, wrong[0][0].val   # swap nodes back to place

# 59.102 Binary Tree Level Order Traversal ====================================== https://leetcode.com/problems/binary-tree-level-order-traversal/
# Problem: Given the [root] of a binary tree, return the "level-order traversal" of its node values. Return node values in a 2D list, where each 
#          element of the list is a sub-list that contains nodes in same level
# Description: DFS while tracking node level. Create a helper function, that take a [node], its [level], and a 2D list [res] as parameters.
#              Initally, the [node] is [root] and [level] is 0 since [root] is at depth 0. If len(res)<=level, means the current [node] comes from
#              a new [level], and need to create a new sub-list in [res] to hold nodes from new level. Always append current [node] to res[level],
#              and recursive invoke helper function on [node.left] and [node.right], with [level+1]. If [node] is None, means current path reach 
#              end, end current traversal by returning
# Time complexity: O(N)
def levelOrder_DFS(root: TreeNode) -> List[List[int]]:
    res = []
    levelOrder_helper(root, 0, res)
    return res  

def levelOrder_helper(root, level, res):
    if not root:                            # current path reach end
        return
    if len(res)<=level:                         # hits new level, add new sub-list to hold nodes from new level
        res.append([])
    res[level].append(root.val)                     # add current [node] to corresponding level in [res]
    levelOrder_helper(root.left, level+1, res)          # recursive invoke on child nodes
    levelOrder_helper(root.right, level+1, res)   

# Description: BFS while tracking node level
# Time complexity: O(N)
def levelOrder_BFS(root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    res = []
    queue = deque()             # queue for BFS
    queue.append((root, 0))
    while queue:
        node, level = queue.popleft()
        if len(res) <= level:           # new level reached, increase [res] size
            res.append([node.val])
        else:                           # node of existing level
            res[level].append(node.val)
        if node.left:
            queue.append((node.left, level+1))
        if node.right:
            queue.append((node.right, level+1))
    return res         

# 60.103 Binary Tree Zigzag Level Order Traversal ============================ https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
# Problem: Given a [root] of binary tree, return the "zig-zag" level order traversal. Return node values in a 2D list, where each 
#          element of the list is a sub-list that contains nodes in same level
#          Ex:           directions      res
#                3         ->          [[3],[20,9],[15,7]]
#             /   \                     
#            9     20      <-
#                /   \
#               15    7    ->
# Descrpiton: DFS while tracking [level]. Create a helper function, that takes [node], and 2D list [res] as parameter. Initially, the [node] is [root]
#             and [level] is zero. Use "deque" to hold node of each level. If len(res)<=level, means the current [node] comes from a new [level], and 
#             need to create a new "deque" to hold it. For even [level], "zig-zag" from left to right, append [node.val] to end of deque res[level]. 
#             For odd [level], "zig-zag" from right to left, prepend [node.val] to beginning of deque res[level]. 
#             At last, convert deque to list then return [res]
from collections import deque
def zigzagLevelOrder(root: TreeNode) -> List[List[int]]:
    def helper(root, level, res: List[deque]):
        if not root:
            return
        if len(res)<=level:                 # hits new level, expand [res] with a new deque
            res.append(deque())
        if level%2==0:                          # even [level] zig-zag from left to right
            res[level].append(root.val)
        else:                                   # odd [level] zig-zag from right to left
            res[level].appendleft(root.val)
        helper(root.left, level+1, res)
        helper(root.right, level+1, res)
    res = []
    helper(root, 0, res)
    return [list(deq) for deq in res]           # convert deques in [res] to list then return

# 61.105 Construct Binary Tree from Preorder and Inorder Traversal ========= https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
# Problem: Given two lists that prepresent the "preorder" and "inorder" traversal of same binary tree. Construct the binary tree, and return its root.
#          Both [preorder] and [inorder] consist of unique values
# Descrption: The first element of [preorder] is always the "root", and nodes in left-subtree always come before nodes in right-subtree in [preorder].
#             Find index of preorder[0] in [inorder], the elements comes before preorder[0] belong to left-subtree of "root", elements comes after preorder[0] belong 
#             to right-subtree of "root". Leverage this property, each recursive call retrieve preorder[0], create a "TreeNode" with value=preorder[0]. Find [index] of 
#             preoreder[0] in [inorder], since nodes of left-subtree comes first, then preorder[1:index] are nodes in left-subtree, and preorder[index+1:] are nodes
#             in right-subtree. Similarly, inorder[:index] are left-subtree and inorder[index+1:] are right-subtree. Pass partitions of [preorder] and [inorder] to 
#             build children of current [root]
# Time complexity: O(n^2), [index] look up is linear, and [n] level of recursive
def buildTreePre(preorder: List[int], inorder: List[int]) -> TreeNode:        
    if preorder:
        index = inorder.index(preorder[0])              # find index of preorder[0] in [inorder]
        node = TreeNode(val=preorder[0])                # create current [root] with preorder[0]
        # recursively build children of current [root]
        node.left = buildTree(preorder[1:index+1], inorder[:index])         # preorder[1:index+1] and inorder[:index] contains nodes in left-subtree
        node.right = buildTree(preorder[index+1:], inorder[index+1:])       # preorder[index+1:] and inorder[index+1:] contains nodes in right-subtree
        return node
    # when [preorder] is empty, means path hits end, thus return None
    return None

# Descrption: Maintain a Dictionary of inorder elements and their indices "{inorder[i]:i}" for quick preorder[0] index look up. Use [pre_beg], [pre_end],
#             [in_beg], [in_end] to specify the range of [preorder] and [inorder] to be passed to next recursive call. Borrow same idea of previous,
#             The first element of [preorder] is the "root", find index of preorder[0] in [inorder], denote as [ind]. Current "root" is preorder[0].
#             inorder[in_beg:ind] are node of left-subtree, inorder[ind+1:in_end] are nodes in right-subtree. And we know the number of nodes in left-subtree
#             is "[ind]-[in_beg]". Thus, sublist of [inorder] for left subtree is from [in_beg] to [ind], and for right subtree is from [ind+1] to [in_end]. 
#             sublist of [preorder] for left subtree is from [pre_beg]+1 to [pre_beg]+[ind]-[in_beg], sublist for right subtree is fronm [pre_beg]+1+[ind]-[in_beg] 
#             to [pre_end]. 
# Time complexity: O(n), [ind] look up is constant
def buildTreePre_2(preorder: List[int], inorder: List[int]) -> TreeNode:   
    def helper(pre_beg, pre_end, in_beg, in_end):
        if pre_beg<=pre_end:                                     # [pre_beg]>=[pre_end] means 
            ind = dic[preorder[pre_beg]]
            root = TreeNode(preorder[pre_beg])
            root.left = helper(pre_beg+1, pre_beg+ind-in_beg, in_beg, ind-1)        # pass sublist to build left subtree
            root.right = helper(pre_beg+1+ind-in_beg, pre_end, ind+1, in_end)       # pass sublist to build right subtree
            return root
        return None
            
    dic = {num:i for i, num in enumerate(inorder)}          # create an "index" loop up dictionary for [inorder]
    return helper(0, len(preorder)-1, 0, len(inorder)-1)

# 62.106 Construct Binary Tree from Inorder and Postorder Traversal ======= https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
# Problem: Given two list [inorder] and [postorder] representing "inorder" and "postorder" traversal of a binary tree. Construct the binray tree and return its root
# Description: "inorder" traversal is left-root-right, "postorder" traversal is left-right-root. Thus, the "root" is always the last element of "postorder". And in 
#              "inorder", nodes of left-subtree are on the left-side of "root", nodes of right-subtree are on right-side of "root".Each recursive call, we can get
#              "root" from "postorder", and from "inorder" we know the number of nodes in left-subtree and right-subtree. Then for "inorder" and "postorder" lists 
#              we can divide them into two parts that contains left/right-subtree respectively. 
#              Maintain a dictionary that map "postorder" values and indices {postorder[i]:i} for quick index look up. Maintain indices [post_beg], [post_end], 
#              [in_beg] and [in_end] that specify the list range of [postorder] and [inorder] list in current recursive level
# Time complexity: O(n)
def buildTree(inorder: List[int], postorder: List[int]) -> TreeNode:
    def helper(in_beg, in_end, post_beg, post_end):
        if post_beg >= post_end: 
            return None
        ind = dic[postorder[post_end-1]]
        root = TreeNode(postorder[post_end-1])
        root.left = helper(in_beg, ind, post_beg, post_beg+ind-in_beg)
        root.right = helper(ind+1, in_end, post_beg+ind-in_beg, post_end-1)
        return root
    dic = {num:i for i, num in enumerate(inorder)}
    return helper(0, len(inorder), 0, len(postorder))

# 63.107 Binary tree level order traversal II =================================== URL: https://leetcode.com/problems/binary-tree-level-order-traversal-ii/
# Problem: Bottom-up level order traversal, given a root of a binary tree, return a 2D list where nodes of same level are wrapped in same sub-list,
#          and highest level is at the beginning of the list, root is at the end of the list.
# Description: DFS with [level] count. Use a helper function, which tracks [level] as the "depth" of current node. Maintain [res] as 2D result list, where
#              "i"th element contains nodes at "i"th depth. Return reversed [res] at the end as "bottom-up" level order traversal.
#              Initally, helper function takes [root] as starting node, [level] = 0, and [res] is an empty list
#              In helper function. if len(res)<=level, means a new level is hit and need to create a new sub-list in [res] to hold nodes of new level.
#              Add current [node] to its corresponding level res[level].append(node.val). Then invoke helper function on [node.left] and [node.right]
#              with [level+1]. If current [node] is None, then stop current path by returning
# Time Complexity: O(n)
def levelOrderBottom(root: TreeNode) -> List[List[int]]:
    res = []
    levelOrderBottom_helper(root, 0, res)
    return res[::-1]

def levelOrderBottom_helper(root, level, res):
    if not root:                # hit None, end current path
        return
    if len(res)<=level:             # a new level is hit, add new sub-list to hold nodes from new level
        res.append([])
    res[level].append(root.val)         # add current node to corresponding level
    levelOrderBottom_helper(root.left, level+1, res)            # invoke function on child nodes
    levelOrderBottom_helper(root.right, level+1, res)

# 64.109 Convert Sorted List to Binary Search Tree ================================ https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/
# problem: Given the [head] of a linked list, whose nodes are sorted in ascending order. Convert it to a height balanced BST, and return the root.
# Description: Recursive. Use two pointer [slow] and [fast] to find the middle node [mid], [slow] starts from [head] and [fast] starts from [head.next.next]. 
#              Traverse [slow] and [fast] towards tail, [slow] moves a node at a time, [fast] moves two nodes at a time. When [fast] or [fast.next] reach end, 
#              [slow.next] is [mid]. Use [mid] to make [root] of current recursion, and nodes between [head] and [slow] is the sublist that belongs to 
#              left-subtree, and nodes from [mid.next] to end is the sublist that belongs to right-subtree. Break the connect between [slow] and [mid] to 
#              separate left and right sublist.
# Time compexity: O(NlogN), O(N) to build node for a level, there are O(logN) levels
def sortedListToBST(head: ListNode) -> TreeNode:
    if not head:
        return None
    if not head.next:
        return TreeNode(head.val)
    slow, fast = head, head.next.next               # use [slow] and [fast] to search for [mid]
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    mid = slow.next
    slow.next = None                                # break connection between left and right sublist
    root = TreeNode(mid.val)
    root.left = sortedListToBST(head)               # build left subtree with nodes from [head] to [slow]
    root.right = sortedListToBST(mid.next)          # build right subtree with nodes from [mid.next] to end
    return root       

# 65.113 Path Sum II =========================================================================================== https://leetcode.com/problems/path-sum-ii/
# Problem: Given the [root] of a binary tree and an integer [targetSum]. Return all "root-to-leaf" path that the nodes along the path sum up to [targetSum]
#          Return the result as 2D list, where each element consists of nodes in a path
# Description: DFS. Implement a helper function, that maintain [res] as the result 2D list, maintain [temp] that consists nodes along a path, maintain 
#              [targetSum] representing the remaining sum after subtracting node values in [temp], and the current node [root]
#              If current node [root] is a leaf that has not child, and [targetSum]==[root.val], meaning a path is found then append [temp] to [res]. If
#              [root] hit end of path, means this path doesn't qualify, stop current recursion by returning. Each recursion, reduce [targetSum] by [root.val]
#              and append [root.val] to [temp], then search both [root.left] and [root.right] subtrees.
def pathSum2(root: TreeNode, targetSum: int) -> List[List[int]]:
    def helper(root, temp, targetSum, res):
        if not root:                        # DFS hit end, stop current DFS
            return
        if not root.left and not root.right:        # hit leaf
            if root.val == targetSum:                   # qualified path is found. append [temp] to [res]
                temp.append(root.val)
                res.append(temp)
                return
            else:
                return
        helper(root.left, temp+[root.val], targetSum-root.val, res)     # recursion on left child
        helper(root.right, temp+[root.val], targetSum-root.val, res)    # recursion on right child
    res = []
    helper(root, [], targetSum, res)
    return res

# 66.114 Flatten Binary Tree to Linked List ===================================== https://leetcode.com/problems/flatten-binary-tree-to-linked-list/
# Problem: Given [root] of a binary tree, flatten it to a linked list. The linked list use [root] as its "head", and "right" child points to next
#          node, and "left" node is set to None. The linked list follow "pre-order" traversal order. (root-left-right)
# Description: Post-order traversal and a global variable [prev] tracking previous node to connect with. Use "post-order" traversal to construct 
#              the linked list from right to left. [prev] is the head of previously created linked list, and need to be appended to the [right] of current 
#              node. After traverse [left] and [right] of a node, connect [prev] to [right] of current node, set [left] to None and update [prev] to
#              current node, since current node is the new head of previously created linked list
class FlattenBinaryTree:
    def __init__(self):
        self.prev = None                # tracking the head of previously created linked list
    def flatten(self, root: TreeNode) -> None:
        if not root:
            return
        # "post-order" traverse
        self.flatten(root.right)        
        self.flatten(root.left)
        root.right = self.prev          # append previously created linked list to current node
        root.left = None                # set its [left] to None
        self.prev = root                # update [prev] to current node

# 67.116 Populating Next Right Pointers in Each Node ==================== https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
# Problem: Given the [root] of a "perfect binary tree", where all leaves are on the same level. And every non-leaf node has two childen. The node
#          has reference to "left", "right" and "next", as following
#          struct Node {
#               int val;
#               Node *left;
#               Node *right;
#               Node *next; 
#          }
#          Populate each next pointer to point to its next right node. If there is no right node, then points to None
# Description: DFS. For a [node]. [node.left] connects to [node.right], and [node.right] connect to [node.next.left].
#                       node ----------> node.next --> None
#                       /  \              /    \
#                   left -> right ---> left --> right --> None
#              Recursivly call on [node.left] and [node.right] to build next level. If [node] is None, return to stop
def connect(root: 'Node') -> 'Node':
    if not root:
        return
    if root.left:                               # connect [left] to [right]
        root.left.next = root.right
    if root.right:                              # connect [right] to [node.next.left]
        root.right.next = root.next.left if root.next else None
    connect(root.left)
    connect(root.right)
    return root

# 68.117 Populating Next Right Pointers in Each Node II ============= https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/
# Problem: Given the [root] of a bianry tree, where the tree may not be "perfect binary tree". Populate each next pointer to point to its next 
#          node on right. If there is no next-right node, point the next point to None.. The node has references to "left", "right", and "next":
#               struct Node {
#                   int val;
#                   Node *left;
#                   Node *right;
#                   Node *next; 
#               }
# Description: Maintian [prev] as node in previous level that [prev.left] and [prev.right] are nodes in current level. [dummy] refers to the "head"
#              node of current level, where [dummy.next] is the first node on current level. [cur] refers to the current node, that is used to 
#              connect "next-right" node. 
#              Start with [prev] at [root]. And traverse though each level, [dummy] and [cur] start from first node of next level. Assign [prev.left]  
#              and [prev.right] to [cur.next] if they exists. Update [prev] to [prev.next] and update [cur] to [cur.next] to build tree horizontally.
#              When [prev] hits None, means traverse of current level is finished. Start traverse next level, update [prev] with [dummy.next].
#              If [prev] is still None after updating, means the entire traversal is finished. 
def connect(self, root: 'Node') -> 'Node':
    prev = root
    while prev:                         # build completed if outter [prev] is None
        cur = dummy = Node(0)               # [dummy] track first node of current level
        # traverse and build current level
        while prev:
            if prev.left:
                cur.next = prev.left
                cur = cur.next
            if prev.right:
                cur.next = prev.right
                cur = cur.next
            prev = prev.next                # move [prev] horizontally to continue build current level
        prev = dummy.next               # move [prev] to the beginning of next level
    return root

# 69.129 Sum Root to Leaf Numbers ==================================================== https://leetcode.com/problems/sum-root-to-leaf-numbers/
# Problem: Given a [root] of binary tree, where nodes in the tree can only contain value from 0 to 9. Each path from root to leaf represents
#          a number, where value of root is highest digit and value of leaf is the lowest digit. Ex, a path 1 -> 2 -> 3 represents number 123
#          Return the total sum of all root-to-leaf path. A leaf node is a node without any child
# Description: DFS, maintain [cur] as the current sum along the path. Each resursive call, [cur] is multiplied by 10 and added with [root.val],
#              [cur] = [cur]*10+[root.val]. If [root.left] or [root.right] exists, DFS on [root.left] and [root.right] with updated [cur].
#              When [root] is a leaf, aka, [root.left] and [root.right] are None. Add [cur] into [res], that [res] = [cur]*10+[root.val]. Return
#              [res] at the end
def sumNumbers(root: Optional[TreeNode]) -> int:
    def helper(root, cur, res):
        if not root.left and not root.right:            # hit leaf, add [cur] into [res]
            res[0] += (cur*10)+root.val
            return
        # DFS on left and right child, [cur] is updated
        if root.left:                                   
            helper(root.left, cur*10+root.val, res)
        if root.right:
            helper(root.right, cur*10+root.val, res)
    res = [0]               # [res] is a list, that is passed by reference
    helper(root, 0, res)
    return res[0]

# 70.130 Surrounded Regions ================================================================ https://leetcode.com/problems/surrounded-regions/
# Problem: Given a m*n matrix [board] contains "X" and "O", capture regions of "O" that are 4-direction surrounded by "X", and "O" into "X"
#          Surrounded regions should not be on the border, that "O"s on border is not converted to "X"
#          Ex:
#          X X X X          X X X X
#          X O O X   ===>   X X X X 
#          X X O X          X X X X
#          X O X X          X O X X
# Description: If a "O" is no connect with border, it need to be converted. Use DFS to find the "O" on border and convert it to a temporary 
#              character. Then iterate through [board] convert remaining "O"s to "X", since they are not connected to boarder.
# Time complexity: O(n), n = m*n
def solve(board: List[List[str]]) -> None:
    def dfs(row, col):
        # convert "O" to "." if it is connected to border
        if 0<=row<len(board) and 0<=col<len(board[row]) and board[row][col] == "O":
            board[row][col] = "."
            dfs(row+1, col)
            dfs(row-1, col)
            dfs(row, col+1)
            dfs(row, col-1)
            
    for r in [0, len(board)-1]:         # dfs on first and last row
        for c in range(len(board[r])):
            dfs(r, c)
    for r in range(len(board)):         # dfs on first and last column
        for c in [0, len(board[r])-1]:
            dfs(r, c)
    for r in range(len(board)):         # convert remaining "O" to "X" and revert "." to "O"
        for c in range(len(board[r])):
            if board[r][c] == "O":
                board[r][c] = "X"
            elif board[r][c] == ".":
                board[r][c] = "O"

# 71.131 Palindrome Partitioning =================================================== https://leetcode.com/problems/palindrome-partitioning/
# Problem: Given a string [s], find all possible ways to partition the string, so that each substring is a palindrome.
# Description: DFS backtracking. Backtrack through substring of [s], if s[:i] is a palindrome, then add s[:i] to [temp] list and continue
#              backtracking on the rest of substring s[i:]. If not more character left in [s], add [temp] to [res] list. 
#              To check palindrome, use s[:i] == s[:i][::-1]. 
#              And [i] should start from 1 to len(s)+1, since s[:i] should start from s[:1]
# Time complexity: O(2^n) worst case when all characters are same in [s]
def partition(s: str) -> List[List[str]]:
    def helper(s, temp, res):
        if not s:
            res.append(temp)
        for i in range(1,len(s)+1):             # backtracking on every character from i=1 to i=len+1
            if s[:i][::-1] == s[:i]:                    # check palindrome
                helper(s[i:], temp+[s[:i]], res)        # append palindrome to [temp] then backtrack on rest of string
    res =[]
    helper(s, [], res)
    return res

# 72.133 Clone graph ========================================================================= https://leetcode.com/problems/clone-graph/
# Problem: Given a reference of a [node] in a connected graph. Deep copy the entire graph and return to cooresponding node of [node] in 
#          copied graph. Definition of graph node is as follow
#          class GraphNode:
#               def __init__(self, val=0, neighbors=[]):
#                   self.val = val
#                   self.neighbors = []
# Description: DFS. Maintain a dictionary that keys are node of original graph, and values are cloned node in new graph. While DFS through
#              original graph, for every [neigh] of [node.neighbors]. If [neigh] is in dictionary, means [neigh] is already cloned, but
#              still need to be appended to the cloned [node.neighbors]. If [neigh] is not in didctionary, then create a new node with same
#              value and insert it to dictionary, and append it to the cloned [node.neighbors]. 
#              Initally, dictionary contains the "root" [node] and its cloned node, and DFS from there
# Time complexity: O(V+E) iterate through vertices and edges
def cloneGraph(node: 'GraphNode') -> 'Node':
    def helper(node: 'GraphNode', dic: dict):
        for neigh in node.neighbors:
            if neigh not in dic:                    # if [neigh] is not cloned
                dic[neigh] = GraphNode(neigh.val)       # clone [neigh], and save it in dictionary
                helper(neigh, dic)                      # DFS on [neigh]
            dic[node].neighbors.append(dic[neigh])  # always connect cloned [neigh] with cloned [node]
    
    if not node:
        return None
    dic = {node: GraphNode(node.val)}   # initially start with [node]
    helper(node, dic)
    return dic[node]                # return cloned [node]

# 73.134 Gas Station ========================================================================= https://leetcode.com/problems/gas-station/
# Problem: Given two integer arrays [gas] and [cost], where elements in [gas] represent the amount of gas a car can refill at certain index
#          [i] and elements in [cost] represent the amount of gas cost to move from current index [i] to next index [i+1]. Assume indeices 
#          form a circle that the car can move from last index to first index. Find and return the starting index that the car has enough 
#          gas to travel all indecies, return -1 if the car can't travel the circle
# Description: The only way the the car can not travel the cirle is when total amount of gas is less than total cost sum(gas)<sum(cost).
#              Check it and return -1, otherwise there always exists a starting index. 
#              Greedy. traverse through [gas] and [cost] from index 0 to end, calculate the accumulate remaining gas at each index. Find 
#              the index with lowest remaining gas, the index next to the lowest is the answer. 
# Time complexity: O(n)
def canCompleteCircuit(gas, cost):
    if sum(gas)<sum(cost):
        return -1
    remain, index, lowest = 0, 0, float('inf')
    for i in range(len(gas)):           # search for index with lowest [remain]
        remain += gas[i]-cost[i]
        if lowest >= remain:
            lowest, index = remain, i
    return (index+1)%len(gas)           # return the next index of [lowest]

# 74.137 Single Number II ===================================================================== https://leetcode.com/problems/single-number-ii/
# Problem: Given a list of integer [nums], where every element appear three times except for one element only appear once. Find and return the 
#          single number in O(n) time complexity and use O(1) space
# Description: Bitwise solution. If a number appear three times, for each bit of these number, they must sum up to either 3 or 0. Thus, sum up 
#              all numbers in [nums] and modulo 3, the remaining must be the bit of the single number. Do the same from bit 0 to 32, we have
#              the bitwise of the single number. 
#              In case the single number is negative (2's complement), if the [res] is larger than (1<<31), [res] is negative. Use [res]-(1<<32)
#              to get 2's complement of [res]
# Time complexity: O(n)
# Space complexity: O(1)
def singleNumber(nums: List[int]) -> int:
    res = 0
    for i in range(32):
        cnt = 0
        for n in nums:
            if n & (1<<i) == (1<<i):            # increate [cnt] if [i]th bit of [n] is 1
                cnt += 1
        res |= (cnt%3)<<i                       # extract the single bit
    return res if res<(1<<31) else res-(1<<32)      # check negativity before returning

# 75.139 Word break ================================================================================= https://leetcode.com/problems/word-break/
# Problem: Given a string [s] and a list of string [wordDict]. Return True, if [s] can be segmented into one or more words from [wordDict] 
#          without any remaining words, same word in [wordDict] can be used for multiple times. Return False, if [s] contains word that can not
#          be segmented with [wordDict]
# Description: Dynamic Programming. Maintain a boolean list [dp], that dp[i] represent s[0:i] can be segment into [wordDict]. Iterate through 
#              index of [s]. And iterate [word] of [wordDict] in the inner loop. If [word] exists at the end of s[:i], word == s[i-len(word):i],
#              while dp[i-len(word)] is True, meaning everything before occurence of [word] can be segmented. After iterate through [s], dp[-1]
#              contains the result for whether [s] can be segmented or not
# Time complexity: O(s*w) s=len(s) w=len(wordDict)
def wordBreak(s: str, wordDict: List[str]) -> bool:
    dp = [True]+[False]*len(s)          # dp contains len(s)+1 elements, where first element is always True as base case
    for i in range(1, len(s)+1):            # iterate [i] from 1 to len(s)+1
        for word in wordDict:
            if word == s[i-len(word):i] and dp[i-len(word)]:    # [word] appear at end of s[:i], can segment before occurence of [word]
                dp[i] = True
    return dp[-1]

# 76.142 Linked List Cycle II ============================================================== https://leetcode.com/problems/linked-list-cycle-ii/
# Problem: Given [head] if a linked list, may or may not contains a cycle. Find and return the node where the cycle begins, return None if no
#          cycle found.
# Description: Draw the linked list here with a cycle, if traverse the list with [slow] and [fast] pointers, they will meet at node [X]
#                                     ______        
#              head ------------- E /       \       [H] is distance between "head" and "E"
#                                  |        |       [D] is distance between "E" and "X"
#                                  \______ X        [L] is length of cycle
#              When [fast] and [slow] meet at [X], [slow] traveled [H+D]. Since [fast] travels twice faster than [slow], [fast] traveled [2H+2D].
#              [fast] may travel multiple laps in cycle, then [2H+2D = H + D + nL], where [n] is integer prepresent laps of cycle. Manipulate 
#              equation by canceling [H] and [D], we habe [H+D = nL]. Conver it to [H = nL-D], meaning the distacne [E] and [X] is same as [H].
#              If a node start from [head] and a node start from [X] and move at same pace, they will meet at [E], which is the starting point
#              of cycle。
#              First, need to find [X], and if [slow] and [fast] doesn't meet there is not cycle, return None. Once found [X], move two pointers
#              from [head] and [X], and return the node where they meet
# Time complexity: O(n)
def detectCycle(head: ListNode) -> ListNode:
    fast = slow = head
    while fast and fast.next:       # use "while-else" to detact if cycle exist
        fast = fast.next.next
        slow = slow.next
        if fast == slow:                # [X] is found, break loop
            break
    else:                           # fast is None at some point, no cycle detected
        return None
    fast, slow = head, slow         # start from [head] and [X], to find [E]
    while fast != slow:
        fast = fast.next
        slow = slow.next
    return slow

# 77.143 Reorder List ============================================================================== https://leetcode.com/problems/reorder-list/
# Problem: Given [head] of singly linked-list. The nodes of list is l1->l2->l3 ... ->ln-1->ln. Reorder the list to be l1->ln->l2->ln-1->l3-> ... 
#          The reorder should be done in-place
# Description: The reordered list consists of in-ordered first half linked list and reversed second half linked list. Therefore, we can find the
#              middle of linked list with [fast] and [slow] pointers. Then reverse second half with three pointers [prev], [cur] and [next]. Lastly 
#              merge two halves together
# Time complexity: O(n)
def reorderList(head: Optional[ListNode]) -> None:
    # find middle
    fast = slow = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    # reverse second half
    prev, cur = None, slow.next
    while cur:
        nextt = cur.next
        cur.next = prev
        prev = cur
        cur = nextt
    slow.next = None            # first node in second half, become tail
    # merge together
    head1, head2 = head, prev
    while head2:                # [head1] and [head2] swtich the linked list they belong to
        head1.next, head1, head2 = head2, head2, head1.next

# 78.146 LRU Cache =============================================================================== https://leetcode.com/problems/lru-cache/
# Problem: Design a data structure Least Recently Used cache (LRUcache), that 
#          "LRUCache(int capacity)"" initialize LRU cache with positive size [capacity]. 
#          "get(int key)" return the value of [key], if [key] is not present return -1. 
#          "put(int key, int value)", update/insert [key]-[value] pair. If adding [key]-[value] pair and [capacity] is exceeded, evit the
#          least recently used key, then add the new [key]-[value] pair
# Description: OrderedDict, which is a dictionary with items track in order. "dict.move_to_end(key, last=True)" moves key to end of order
#              "dict.popitem(key, last=True)" remove and return the last key in order. Maintain a OrderedDict [data], where least recently 
#              used key is at the beginning of dict. Maintain [remain] track remaining slot in list. When adding an item, add it to end of
#              [data], and reduce [remain]. When getting an item, move the [key] to end of [data]
# Time Complexity: average O(1)   
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity: int):
        self.data = OrderedDict()
        self.remain = capacity
    
    def get(self, key: int) -> int:
        if key in self.data:
            self.data.move_to_end(key, last=True)           # move accessed [key] to end
            return self.data[key]
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.data:                        # update a [key]
            self.data[key]=value
            self.data.move_to_end(key, last=True)
        else:                                       # insert a new [key]
            if self.remain > 0:                         # enough space
                self.data[key]=value
                self.remain -= 1
            else:                                       # no space, remove least recently used [key] then add 
                self.data.popitem(last=False)
                self.data[key]=value

# 79.200 Number of islands ============================================================= https://leetcode.com/problems/number-of-islands/
# Problem: Given a m*n matrix [grid], which represent land ("1") and water ("0"). Return the number of islands. 
#          An island is surrounded by water and lands are connected by 4-way connection
# Description: DFS. Iterate through every cell in [grid], once find a "1" increase island count [cnt] by 1, convert current "1" to "2",
#              and recursively convert its 4 neighbors. 
# Time complexity: O(n*m) every cell is accessed twice
def numIslands(self, grid: List[List[str]]) -> int:
    def flood(grid, i, j):
        if grid[i][j] == "1":
            grid[i][j] = "2"                # covert a land to "2"
            if i>0:                         # and recursively invoke on its 4 neighbors
                flood(grid, i-1, j)
            if i<len(grid)-1:
                flood(grid, i+1, j)
            if j>0:
                flood(grid, i, j-1)
            if j<len(grid[i])-1:
                flood(grid, i, j+1)
    cnt = 0
    for i in range(len(grid)):              # iterate every elements in [grid]
        for j in range(len(grid[i])):
            if grid[i][j] == "1":   
                cnt += 1                        # find a new island
                flood(grid, i, j)               # convert island recursively
    return cnt

# 80.532 K-diff pairs in an array =========================================== https://leetcode.com/problems/k-diff-pairs-in-an-array/
# Problem: Given an integer array [nums] and an integer [k]. Return nunber of unique k-diff pairs in the array.
#          K-diff pair is two numbers nums[i] and nums[i] that abs(nums[i]-nums[j]) = k
# Desciption: For a number nums[i], find if nums[i]+k is also in the array. If k=0, then find numbers that appeared more than once,
#             because same number is difference of zero
# Time complexity: O(n)
from collections import Counter
def findPairs(nums: List[int], k: int) -> int:
    cnt = Counter(nums)
    if k == 0:
        return sum([1 for val in cnt.values() if val>1])    # find same number, if k==0
    res = 0
    for key in cnt:
        if key+k in cnt:           # find if nums[i]+k is in array
            res += 1
    return res

# 81.148 Sort list ==================================================================== https://leetcode.com/problems/sort-list/
# Problem: Given [head] of a single linked list, return the head of list after sorting in ascending order. 
#          Do it in O(NlogN) time and use O(1) space
# Description: Merge sort. Use [slow] and [fast] pointer to find middle of each list to partition, break [left] and [right] parts
#              by letting last node of [left] partition poionts to None. Merge [left] and [right] into a list with [dummy] node 
#              at beginning, and return [dummy.next]
# Time complexisty: O(NlogN)
def sortList(head: Optional[ListNode]) -> Optional[ListNode]:
    def merge(left, right):
        dummy = cur = ListNode(0)
        while left and right:                       # merge [left] and [right]
            if left.val<right.val:
                cur.next = left
                left, cur = left.next, cur.next
            else:
                cur.next = right
                right, cur = right.next, cur.next
        if left or right:                           # connect rest of nodes to result list
            cur.next = left if left else right
        return dummy.next
    
    if not head or not head.next:           # stop when there is only one node left
        return head
    prev, slow, fast = None, head, head               
    while fast and fast.next:
        prev, slow, fast = slow, slow.next, fast.next.next
    prev.next = None                                # [prev] is last node of [left], break [left] and [right] 
    left = sortList(head)           # recursively partition left part
    right = sortList(slow)          # recursively partition right part
    return merge(left, right)

# 82.662 Maximum Width of Binary Tree ============================= https://leetcode.com/problems/maximum-width-of-binary-tree/
# Problem: Given [root] of a binary tree, return the maximum width of the tree among all levels.
#          Width of tree is the number of nodes (inclusive) between left-most and right-most node in a level, where null node
#          are also counted into width calculation
# Descrption: DFS with record. Denote [pos] is the postion of node at its level, where left most node has [pos]=0, and right
#             most node has [pos]=2**(level-1). If a node at [pos], then its left child is at [pos*2] and right child at
#             [pos*2+1] in the next level. Traverse through tree, and track min and max [pos] of each level in [record].
#             Find and return the maximum difference between min and max among all level
# Time complexity: O(N)
def widthOfBinaryTree(root: Optional[TreeNode]) -> int:
    def helper(node, level, pos, record):
        if node:
            if level not in record:
                record[level] = (pos, pos)          # inital min and max at new level
            else:
                record[level] = (min(pos, record[level][0]), max(pos, record[level][1]))        # updating min and max of a level
            helper(node.left, level+1, pos*2, record)           # left child at [pos*2]
            helper(node.right, level+1, pos*2+1, record)        # right child at [pos*2+1]
            
    record = {}
    helper(root, 0, 0, record)
    return max([ele[1]-ele[0] for ele in record.values()])+1    # find max width of all level

# 83.150 Evaluate Reverse Polish Notation =================== https://leetcode.com/problems/evaluate-reverse-polish-notation/
# Problem: Given a array of string [tokens], representing Reverse Polish Notation. Evaluate the array and return the result of
#          the notation. Valid operators are +, -, *, /. And the division should truncate towards zero.
#          Reverse Polish Notation: two operends come before the operator
#          Ex: [4, -13, 5, /, +] => 4+(-13/5) = 4-2 = 2
# Description: Stack. Append operands to stack. When hit an operator, pop two operands, where right operand is popped first and
#              left operand is popped later. 
#              Note, division should truncate toards zero. Thus, division should use int(l/r), since l//r always round down
# Time complexity: O(n)
def evalRPN(tokens: List[str]) -> int:
    stack = []
    for ele in tokens:
        if ele not in "+-*/":
            stack.append(int(ele))              # append operands to stack
        else:
            r, l = stack.pop(), stack.pop()
            if ele == "+":
                stack.append(l+r)
            elif ele == "-":
                stack.append(l-r)
            elif ele == "*":
                stack.append(l*r)
            elif ele == "/":
                stack.append(int(l/r))      # division truncate towards zero      
    return stack.pop()              # last element is result

# 84.138 Copy List with Random Pointer ================================== https://leetcode.com/problems/copy-list-with-random-pointer/
# Problem: Given the [head] of a linked list, where each node in the linked list contains an [random] pointer that points one of the
#          node in the list or points to None. Make a "deep-copy" of original list and return the head of copied list
# Description: Dictionary. Use Dictionary [pair] to maintain relation between original node and its corresponding copied node. First 
#              iteration, copy each node and link copied nodes with [next] pointer, and add "origin: copied" mapping to [pair]. In
#              second iteration, connect "copied" node with [random] pointer. Note, if "origin" node points to None, and [pair] does
#              not contain "None", assign "copied" with None value specifically
# Time complexity: O(n)
def copyRandomList(head: 'Node') -> 'Node':
    pair = {}
    dummy = copied = Node(0)            # [copied] is tail of copied list
    origin = head                       # [origin] is current node being copied
    while origin:                           # link copied with [next] pointer
        pair[origin] = Node(origin.val)
        copied.next = pair[origin]
        copied, origin = pair[origin], origin.next
    for k, v in pair.items():               # link [random] pointer
        v.random = pair[k.random] if k.random is not None else None     # if [origin] points to None, let [copied] points to None
    return dummy.next 

# 85.316 Remove Duplicate Letters ======================================= https://leetcode.com/problems/remove-duplicate-letters/
# Problem: Given a string [s], remove duplicated characters without changing relative order of characters so that each character 
#          appear only once. Return the result is the smallest in lexicographcial order among all possible results.
#          Such as, "bacdcb" will return "acdb".
# Description: Track three things, last index of each character, whether a character is added to result, and result [stack] which
#              maintain character in minimum lexicographical order. Iterate through [s], and check following conditions.
#              If a character [c] is not visited, and it is smaller than top of [stack], and top of [stack] has other occurance
#              in later index. Then remove [top] from [stack] and [visited], and append [c] to top. For character [c] that is
#              un-visited and larger than [top], alway append to [stack] and [visited]. 
# Time Complexity: O(n)
def removeDuplicateLetters(s: str) -> str:
    lastOccur = {}
    for i in range(len(s)):     # record last occurance of each character
        lastOccur[s[i]] = i
    stack, visited = [], set()
    for i in range(len(s)):
        if s[i] not in visited:
            while stack and s[i]<stack[-1] and lastOccur[stack[-1]]>i:      # keep removing larger element
                visited.remove(stack[-1])
                stack.pop()
            visited.add(s[i])                   # append un-visited and smaller character
            stack.append(s[i])
    return "".join(stack)

# 86.946 Validate Stack Sequences ====================================== https://leetcode.com/problems/validate-stack-sequences/
# Problem: Given tow integer lists [pushed] and [popped] of distinct value, where [pushed] and [popped] are permutation of each
#          other. Check if this could have been the result of a sequence of push and pop operations on a stack. 
#          Ex: pushed=[1,2,3,4,5] popped=[4,5,3,2,1]
#              push 1, 2, 3, 4 => pop 4 => push 5 => pop 5, 3, 2, 1
# Description: Perform actual push and pop operations. Iterate through [pushed], push elements to [stack]. And maintain [i] as
#              index of [popped], as the next element need to be popped. Pop element from [stack] when popped[i] matches top
#              of [stack]. At the end, [stack] must be empty, otherwise return False
# Time complexity: O(n)
def validateStackSequences(pushed: List[int], popped: List[int]) -> bool:
    stack = []
    i = 0                   # [i] track the next element need to be popped
    for num in pushed:
        stack.append(num)
        while stack and stack[-1] == popped[i]:     # check if top matches popped[i]
            stack.pop()
            i += 1
    return not stack

# 87.1663 Smallest String With A Given Numeric Value ====== https://leetcode.com/problems/smallest-string-with-a-given-numeric-value/
# Problem: Given two integers [n] and [k], return the lexicographically smallest string with length [n] and numeric value equal to [k].
#          "Numeric value" is the value of a lowercase character, where 'a'=1 and 'z'=26
#          A string [x] is "lexicographically smaller" than [y], that is, either [x] is prefix of [y], or index [i] is the first index
#          that x[i] != y[i] and x[i] comes before y[i] in alphabetic order 
# Descrption: Maintain array [res] of size [n], that represent numberic value of characters. Maintain [gap] as the gap between total
#             numberic value and [k], [gap] is k-n initially. For an item of [res], if [gap] is larger than 25, then it should be
#             set to 'z'. Add 25 to res[i] and decrease [gap] by 25. If [gap] is less than or equals to 25, then [gap] can be closed
#             on current index [i]. Add [gap] to res[i] and return coorespanding string
# Time complexity: O(n)
def getSmallestString(n: int, k: int) -> str:
    res = [1]*n
    gap = k-n
    for i in range(n):
        if gap>25:                  # set [i] to 'z'
            res[i] += 25
            gap -= 25
        else:                       # [gap] can be closed
            res[i] += gap
            break
    return "".join([chr(ele-1+ord('a')) for ele in res[::-1]])      # return character from reversed [res]

# 88.763 Partition Labels ======================================================== https://leetcode.com/problems/partition-labels/
# Problem: Given a string [s]. Partition the string into as many parts as possible, so that each letter only appear in one partition
#          Return a list of integers representing size of each partition from left to right
# Description: Dictionary, maintain last index of each character. Use two pointers [left] and [right] track boundary of a partition.
#              Iterate through [s], update [right] as last index of characters in current partition. If iteration reaches [right],
#              means all letter encountered before, appear in this partition only. Thus, from [left] to [right] is a partition.
#              Calculate the length and save in [res]. Set [left] to next index of [right] to start a new partition.
# Time complexity: O(n)
def partitionLabels(s: str) -> List[int]:
    lastIndex = {c:i for i, c in enumerate(s)}     # track last index of each character
    left = right = 0
    res = []
    for i, c in enumerate(s):
        right = max(right, lastIndex[c])            # Iterate [s] and find right bound of current partition
        if i == right:                      # reached [right] and a partition is found
            res.append(right-left+1)            # size of partition is index difference +1
            left = i+1
    return res

# 89.991 Broken Calculator ======================================================= https://leetcode.com/problems/broken-calculator/
# Problem: There is a broken calculator that has an integer [startValue] on display. There are only two operations can be done:
#          1. multiple current number by 2, or 2. subtract 1 from current number
#          Given two integers [startValue] and [target], use two given operations and bring [startValue] to [target]. Return the
#          minimum number of operations needed
# Description: Bring [target] to [startValue]. Consider the reversed way, and operations becomes divid by 2 and add 1. If [startValue]
#              is greater than [target], we can only subtract 1 and number of operations is [target-startValue]. If [target] is odd, 
#              it can't be divided by 2, the only choice is to add 1. If [target] is even, we should always divid by 2, because divid
#              2 is reduce [target] faster.
# Time complexity: O(log(target))
def brokenCalc(startValue: int, target: int) -> int:
    if startValue >= target:                    # base case, add [startValue-target] time to bring [target] to [startValue]
        return startValue-target
    if target%2==1:                                 # odd [target], add 1 and increase operation number
        return brokenCalc(startValue, target+1)+1       
    return brokenCalc(startValue, target//2)+1          # even [target], divid 2 and increase operation number

# 90.287 Find the Duplicate Number ====================================== https://leetcode.com/problems/find-the-duplicate-number/
# Problem: Given an array with [n+1] item, where its items are integers in the range of [1,n]. There exists a number that is repeated
#          at least once. Find and return the repeated number. Do it in O(n) time with O(1) space
# Description: Slow and fast pointer chasing. It is same as find cycle starting point in a linked list. Consider each item is a node.
#              The value of item represent the index of next node. For example, if nums[0] is 2, then nums[0] points to nums[2].
#              Therefore, the repeated number is the starting node of cycle. Because the repeated number points to same node that
#              is traversed, which create a cylce. 
#              Use Floyd's algorithm to find the starting point. Firstly, user [slow] and [fast] to find the node where they [met],
#              the distance from beginning to [cycle] start is same as distance between [met] and [cycle]. user two pointers to 
#              traverse from beginning and [met], the repeated number is where two pointers meet
# Time complexity: O(n)
def findDuplicate(nums: List[int]) -> int:
    slow = fast = 0         # [slow] and [fast] start from dummy node
    while True:
        slow = nums[slow]           # [slow] move one step
        fast = nums[nums[fast]]     # [fast] move two steps
        if slow == fast:
            break
    res = 0                     # traverse from beginning and [slow]
    while res != slow:              
        res = nums[res]
        slow = nums[slow]
    return res                  # repeated number is located at where they meet

# 91.152 Maximum Product Subarray ========================================= https://leetcode.com/problems/maximum-product-subarray/
# Problem: Given an integer array [nums], find a contigous non-empty subarray, whose elements have the largest product, return
#          the product. 
# Description: Dynamic Programming. Track both max and min product upto an index. We need to track both [cur_min] and [cur_max],
#              because when an element [n] is negative, and if [cur_min] is also negative, [n*cur_min] can produce a larger number
#              than [n*cur_max] when [cur_max] is positive. For each of the index, both [cur_max] and [cur_min] can come from
#              [cur_max*n], [cur_min*n] or [n] (starting a new subarray). 
# In Short: Track min and max, because min is negative and multiple with negative number can produce positive
# Time complexity: O(n)
def maxProduct(nums: List[int]) -> int:
    cur_max = cur_min = 1
    res = float("-inf")
    for n in nums:
        temp = cur_max                      # [temp] store original value of [cur_max], as it is updated before [cur_min]
        cur_max = max(temp*n, cur_min*n, n)     
        cur_min = min(temp*n, cur_min*n, n)
        res = max(cur_max, res)
    return res

# 92.153 Find Minimum in Rotated Sorted Array =================== https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
# Problem: Given a sorted integer list, which has been shifted to right by arbitrary times, each integer of the list is unique.
#          Find and return the minimum element in O(logN) time
# Description: Bianry Search. The minimum element locates at the unsorted portion, and maintain [res] the current min. If [left]<[right]
#              means this portion is sorted, the smallest number is [left], compare [left] with [res]. If [left] to [right] is not sorted
#              find the [mid] and compare with [res]. Then if [mid]>=[left], left portion is sorted, shift [left] to [mid+1] to search
#              right portion. If [mid]<[left], then left portion is not sorted shift [right] to [mid-1] to search left portion
# Time complexity: O(logN)
def findMin(nums: List[int]) -> int:
    res = nums[0]
    left, right = 0, len(nums)-1
    while left<=right:
        if nums[left]<=nums[right]:     # entire sublist is sorted, [left] contains minimum value
           return min(res, nums[left])
        mid = (left+right)//2
        res = min(res, nums[mid])
        if nums[left]<=nums[mid]:       # left is sorted, search right
            left = mid+1
        else:                           # right is sorted, search left
            right = mid-1
    return res

# 93.162 Find Peak Element ====================================================== https://leetcode.com/problems/find-peak-element/
# Problem: Given a list [nums], find and return the index of an element that is a peak element. A peak element is strictly greater
#          than its neighbors. You can assume that there are negative infinity element at the two boundaries of [nums]
# Description: Binary Search. Maintain [left] and [right], and [mid] as middle index. If [mid] is the largest among its neighbors
#              [mid]>[mid-1] and [mid]>[mid+1], then [mid] is peak and return [mid]. If [mid-1]>[mid], then left part is increasing
#              there must exists a peak in left, shift [right] to [mid-1] and search in left part. Else [mid]<[mid+1], then right 
#              part is increasing, must exists a peak in right, shift [left] to [mid+1]
# Time Complexity: O(logN)
def findPeakElement(nums: List[int]) -> int:
    def helper(nums, left, right):
        if right == left:               # corner case, only one element left, return it
            return left
        if right-left == 1:             # corner case, only two elements left, return the larger one
            return left if nums[left]>nums[right] else right
        mid = (left+right)//2
        if nums[mid-1]<nums[mid] and nums[mid]>nums[mid+1]:     # [mid] is largest among neighbors, return it
            return mid
        else:
            if nums[mid-1]>nums[mid]:                   # left part is increasing, search in left
                return helper(nums, left, mid-1)
            elif nums[mid]<nums[mid+1]:                 # right part is increasing, search in right
                return helper(nums, mid+1, right)
    return helper(nums, 0, len(nums)-1)

# 94.538 Convert BST to Greater Tree ================================== https://leetcode.com/problems/convert-bst-to-greater-tree/
# Problem: Given the [root] of a binary search tree, convert it to a Greater Tree, such that value of node is the sum of all node
#          that are greater than it.
#          Ex:               4(30)                              
#                          /       \                            
#                       1(36)       6(21)
#                     /   \         /   \        
#                   0(36)  2(35)  5(26)  7(15)
#                           \              \
#                           3(33)         8(8)
# Description: Post-order traversal. Use a global variable to track the current sum [curSum]. When traveral the tree, keep adding
#              [node.val] to [curSum] and update [node.val]. Return [root] at the end
def convertBST(root: Optional[TreeNode]) -> Optional[TreeNode]:
    curSum = 0
    def helper(node):
        if not node:
            return 
        nonlocal curSum             # indicate [curSum] is declared outside of function 
        # post-order traveral
        helper(node.right)
        val = node.val + curSum     # add [node.val] to [curSum]
        node.val = curSum = val     # update [curSum] and [node.val]
        helper(node.left)
    helper(root)
    return root

# 95.1584 Min Cost to Connect All Points ================================= https://leetcode.com/problems/min-cost-to-connect-all-points
# Problem: Given an array of points represent coordinate of points on 2D plate, where points[i] = [xi, yi]
#          Return the minimum "manhattan distance" to connect all points. 
#          Manhattan distance between two points [x1, y1] and [x2, y2] is |x1-x2| + |y1-y2|
# Description: Prim's Algorithm, which starts from a random node and always pick the smallest edge which connect a visited node to an
#              un-visited node, until all nodes are connected. 
#              Create an adjacency map [adj] where key is index of point [i], and value is a list of tuples (distance, j), where [j] is 
#              index of other point, and [distance] is distance between [i] and [j]
#              Maintain min [heap] and set [visited] to track connected nodes. Initially [heap] contains (0, 0), that we start from point
#              0 and it is 0 distance from itself. Find the point that is not connected and has minimum ditance to connected points. Pop
#              the minimum distance point [i] from [heap], if [i] is not visited then add its [distance] to [res], and mark it as visited
#              Iterate its adjacent point of adj[i], push unvisited point to [heap] with their [distance]. So that to find next un-visited
#              point whose distance is minimum. Return [res] when all points are visited
# Time complexity: O(N^2*logN), each points need to iterate all its neighbors (n^2), heappush in each iteration is logN
import heapq
def minCostConnectPoints(points: List[List[int]]) -> int:
    # create adjacency map
    adj = {i:[] for _ in range(points)}
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = abs(points[i][0]-points[j][0])+abs(points[i][1]-points[j][1])
            adj[i].append((dist, j))        # noted distance between [i] and [j]
            adj[j].append((dist, i))
    visited, res = set(), 0
    heap = [(0, 0)]
    # pick minimum cost edge to connect unconnected points
    while len(visited)<len(points):
        cost, i = heapq.heappop(heap)       # use [heap] to find minimum cost edge
        if i in visited:                    # skip [i] if it is already connceted
            continue
        res += cost                         # [i] is the point being connected, add cost and mark [i] as connected
        visited.add(i)
        for neighborCost, neighbor in adj[i]:       # add [i]'s neighbor to [heap] to find next min cost
            if neighbor not in visited:
                heapq.heappush(heap, (neighborCost, neighbor))
    return res
    
# 96.581 Shortest Unsorted Continuous Subarray ====================== https://leetcode.com/problems/shortest-unsorted-continuous-subarray/
# Problem: Given an integer array [nums], a continuous subarray is not sorted in ascending order. Find the minimum length of continuous
#          array. Such that after sorting the subarray, the entire [nums] is sorted in ascending order
# Description: Find the largest and smallest number is unsorted subarray, and find the inserting index of them, the index difference is
#              the length of unsorted subarray. Maintain a boolean flag [searchFlag],iterate [nums] from left to right. Set [searchFlag]
#              to True, when encounter a descending [i]>[i+1], which means the unsorted array starts. When [searchFlag] is True, keep 
#              looking for the minimum number. Then look for maximum number of unsorted array, reset [searchFlag] to False, and iterate 
#              [nums] from right to left. Set [searchFlag] to True, when encounter a descending [i-1]>[i], which means the unsorted array 
#              ends. Then look for maximum number. Iterate through [nums] find the index to insert minimum and maximum, return difference
#              of index.
# Time complexity: O(n)
def findUnsortedSubarray(nums: List[int]) -> int:
    minimum = float('inf')
    searchFlag = False
    for i in range(len(nums)-1):                    # find minimum in unsorted subarray
        if not searchFlag and nums[i]>nums[i+1]:
            searchFlag = True
        if searchFlag and nums[i]<minimum:
            minimum = nums[i]
    maximum = float('-inf')
    searchFlag = False
    for i in reversed(range(1, len(nums))):         # find maximum in unsorted subarray
        if not searchFlag and nums[i-1]>nums[i]:
            searchFlag = True
        if searchFlag and nums[i]>maximum:
            maximum = nums[i]
    left = right = 0
    for i in range(len(nums)):                      # find insert index of minimum
        if nums[i]>minimum:
            left = i
            break
    for i in reversed(range(len(nums))):            # find insert index of maximum
        if nums[i]<maximum:
            right = i
            break
    return right-left+1 if right-left>0 else 0      # length is [index_diff + 1]

# 97.341 Flatten Nested List Iterator ================================== https://leetcode.com/problems/flatten-nested-list-iterator/
# Problem: Given a nested list of integers [nestedList]. Each element can be an integer or a list that may contains integers of other
#          lists. Implement [NesttedIterator] class with following methods
#          "hasNext()" return True if there exists at least an integer in the nested list, otherwise return False
#          "next()" return the next integer in the nested list, assume "next()" is called after checking existence of next integer
#          "constructor" that takes a [nestedList] as initial value
# Description: Maintain a [stack]. Element of [nestedList] are stored in reverse order, so that first element of [nestedList] is at 
#              the [top]. In constructor, [nestedList] is added to [stack] in reversed order. In "hasNext()" method, if no element 
#              lefts in [stack] return False. Peek the [top], if [top] is an integer, return True. Else, [top] is a list, pop [top]
#              and append its element to [stack] in reverse order. In "next()" method, since "hasNext()" already unwrapped list and
#              [top] must be an integer, pop and return [top]
class NestedInteger:
    def isInteger(self) -> bool:
        pass
    def getInteger(self) -> int:
        pass
    def getList(self):
        pass

class NestedIterator:
    def __init__(self, nestedList):
        self.stack = nestedList[::-1]       # reversed stack, so that first element of [nestedList] come at [top]
    
    def next(self) -> int:
        return self.stack.pop().getInteger()

    def hasNext(self) -> bool:
        if not self.stack:                  # no element in [stack]
            return False
        while self.stack:
            top = self.stack[-1]            # peek [top]
            if top.isInteger():             # [top] is an integer
                return True
            self.stack.pop()                    # [top] is a list
            self.stack += top.getList[::-1]     # unwrap list, pop and append [top] to [stack] in reversed order
        return False  
    
# 98.1658 Minimum Operations to Reduce X to Zero ================ https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero/
# Problem: Given an positive integer array [nums] and an integer [x]. In an operation, you can remove an element from either left-most  
#          or right-most element from array [nums] and subtract it from [x]. Return the minimum operations needed to bring [x] to zero.
#          If not possible return -1
# Description: Convert this problem to find the longest subarray that has sum equals to [sum(nums)-x]. Thus, the remaining elements 
#              must sum up to [x], and the remaining elements need to be removed via "operations".
#              Maintain [left] index of subarray, current sum of subarray [currSum], and the maximum size of subarray [maxSize]. Iterate  
#              through [nums] and keep adding elements to [currSum]. If [currSum] is larger than [target], subtract element [left] from
#              [currSum] and move [left] to right by 1 index. If [currSum] equals to [target], calculate the current size of subarray,
#              which is [right-left+1] update [maxSize] by taking the larger one between [maxSize] and current size. if none of above,
#              keep adding elements to [currSum]. 
#              At the end, if [maxSize] is larger than 0, then there exists such subarray, the number of operation is [len(nums)-maxSize].
#              If [maxSize] is zero or negative, can't reduce [x] to zero
# Time complexity: O(n)
def minOperations(nums: List[int], x: int) -> int:
    target = sum(nums)-x                    # [target] is target sum of subarray
    if target<0:
        return -1
    if target == 0:
        return len(nums)
    left = currSum = maxSize = 0
    for right, num in enumerate(nums):
        currSum += num
        while currSum > target:                 # [currSum] too big, remove element from left
            currSum -= nums[left]
            left += 1
        if currSum == target:
            maxSize = max(maxSize, right-left+1)        # update [maxSize] when a subarray is found
    return len(nums)-maxSize if maxSize > 0 else -1

# 99.820 Short Encoding of Words =============================================== https://leetcode.com/problems/short-encoding-of-words/
# Problem: There's a string encoding method, consists of string [s] and an array of [indices]. Where [s] consists of several words
#          that are separated by "#" and end with "#". Numbers in [indices] represent the starting index of [s], where the words between
#          [index] and "#" is the word being encoded. Ex
#          s = "time#bell#be#", indics = [0, 2, 5, 10], words = ["time", "me", "bell", "be"] => res = 13
#          Given a list of [words], return the minimum length of [s] after encoding [words]
# Description: Set solution. The words added to [s] are not suffix of any other words. "me" is suffix of "time" it is not a word in [s],
#              and "be" is prefix of "bell" but not suffix of any, it is part of [s]. Find and remove words that are suffix of other 
#              word, length of the remaining words combining "#" is the result
# Time complexity: O(NM), N = number of words, M = number of characters in a word
def minimumLengthEncoding(words: List[str]) -> int:
    wordSet = set(words)
    for w in words:
        for i in range(1, len(w)):
            if w[i:] in wordSet:            # find and remove [w] that is suffix of others
                wordSet.remove(w[i:])
    return len('#'.join(wordSet))+1         # combine with "#" and return length

# Desciption: Trie solution. Trie is a tree of character nodes, the node in tree has an boolean attributes indicate whether this node
#             is the end of a words. Thus, the path from root to a "end" node is a word. Each node also maintain a list of children,
#             indicates the character appear after current character. 
#             Load word to [Trie] with characters reversed, and maintain a list of tuple, each tuple contains end node of a word and the 
#             length of that word. After loading words, check node of each tuple. If node has no children, means it is an end character
#             and its word length is part of encoding string. If node has children, means the word ends here is a suffix of other word,
#             the word can be encoded by other word and it should not go into encoded string. For each word in encoding string, add 1
#             to length, as "#" is appended at the end of each word
# Time complexity: O(N), N = total character of all word
class TrieNode:
    def __init__(self):
        self.children = {}      # [childeren] is "character-TrieNode" mapping
class TrieTree:
    def __init__(self):
        self.root = TrieNode()
        self.leaves = []        # elements of [leaves] are tuples that consists of TrieNode of last character and length of current word
    def add(self, word):
        node = self.root
        for c in word:
            if c not in node.children:          # add new character after current node
                node.children[c] = TrieNode()       
            node = node.children[c]             # move to next node
        self.leaves.append((node, len(word)))   # append tuple after iterate through a [word]

def minimumLengthEncoding(words: List[str]) -> int:
    tree = TrieTree()
    for word in set(words):         # use "set" to remove duplicated word
        tree.add(word[::-1])            # load [word] with character reversed
    res = 0
    for node, length in tree.leaves:
        if len(node.children) == 0:           # find "ending" node, and calculate length
            res += length + 1
    return res

# 100.166 Fraction to Recurring Decimal ======================================= https://leetcode.com/problems/fraction-to-recurring-decimal/
# Problem: Given two integers [numerator] and [denominator] of a fraction, return the fraction in string format. If the fraction is repeating, 
#          enclose the repeating part with parentheses. 
#          Ex: numerator = 1, denominator = 2, res = "0.5"
#              numerator = 1, denominator = 6, res = "0.1(6)"
# Description: 1) Get the sign of result. 2) Extract the integer part. 3) Reminder multiplies by 10 to become [numerator]. Record numerator and
#              if same numerator appear twice, it is repeating. 
#              Maintain the [fraction] part as string. Record numerator and its index in [fraction]. When a [numerator] appear twice, the substring 
#              of [fraction] between previous index and current is index is repeating, and we can enclose it with parentheses. If [numerator]
#              becomes 0, the division is over without repeating fraction. Return combination of [sign], [integer] and [fraction]
def fractionToDecimal(numerator: int, denominator: int) -> str:
    sign = "" if numerator*denominator>=0 else "-"                      # extract [sign] of result
    numerator, denominator = abs(numerator), abs(denominator)
    integer = str(numerator//denominator)                               # extract [integer] part
    numerator = (numerator%denominator)*10
    record, i = {}, 0                           # record {numerator: index}
    fraction = ""
    while numerator:
        if numerator in record:                 # repeating detected, repeating part is between record[numerator] to the end of [fraction]
            return sign+integer+"."+fraction[:record[numerator]]+"("+fraction[record[numerator]:]+")"
        record[numerator] = i
        i += 1
        fraction += str(numerator//denominator)             # append current quotient to [fraction]
        numerator = (numerator%denominator)*10
    return sign+integer+"."+fraction if fraction != "" else sign+integer        # no repeating found, return result w/ or w/o [fraction]

# 101.167 Two Sum II - Input Array Is Sorted ========================== https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
# Problem: Given an array of integers [numbers], its elements are sorted in non-decreasing order. Given another integer [targer], find
#          two numbers in [numbers], whose sum is equal to [target]. Return the indices of two number of 1-indexed
# Description: Dictionary. Maintain a [dic], whose key is compensation of each number [target-num], and value is index of [num]. 
#              Iterate through [numbers] and record compensation of [nums] and index. If a [num] is found in [dic] as compensation,
#              return compenstaion's index +1 and current index +1
# Time complexity: O(N)
def twoSum(numbers: List[int], target: int) -> List[int]:
    dic = {}
    for i, num in enumerate(numbers):
        if num not in dic:
            dic[target-num] = i         # record compensation and its index
        else:
            return [dic[num]+1, i+1]        # a [nums] is found as compensation

# 102.172 Factorial Trailing Zeroes ======================================= https://leetcode.com/problems/factorial-trailing-zeroes/
# Problem: Given an integer [n], return the number of tailing zeros of factorial n!
# Description: Zeros comes from the production of 2 and 5, and there are more 2s than 5s in a factorial. Therefore, need to count
#              the number of 5 in the factoral.
#              Keep dividing [n] by 5, each division tells you how many 5s. 
#              Ex: 5! = 1*2*3*4*5 ==> 5/5 = 1 ==> there is a 5 in 5!
#                  25! = 1*...5...10...15...20...25 
#                      = 1*...5...2*5...3*5...4*5...5*5 ==> 25/5 = 5, 5/5 = 1 ==> there are six 5s
# Time complexity: O(log5(N))
def trailingZeroes(n: int) -> int:
    res = 0
    while n>0:
        n //= 5
        res += n
    return res 

# 103.173 Binary Search Tree Iterator ==================================== https://leetcode.com/problems/binary-search-tree-iterator/
# Problem: Implement BTSIterator class that represent over the in-order traversal of a binary search tree.
#          Constructor of BSTIterator takes [root] of a BST 
#          boolean hasNext(), returns True if exists a node in BST that has not been traversed yet, return False otherwise
#          int next(), return the value of current node in the traversal 
#          Implement next() and hasNext() to run in average O(1) time and use O(h) space, where [h] is height of BST
# Description: Stack. Constructor push nodes from left-most path into [stack], where the last element is the smallest node in BST.
#              next() pop element from [stack], traverse left path of right child of poppped node and append to [stack]. The second
#              smallest node is at the end of that path
#              hasNext() depends on whether element left in stack or not
# Time complexity: next() => average O(1), constructor => O(h), hasNext() => O(1)
class BSTIterator:
    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        while root:                     # append left-most path
            self.stack.append(root)
            root = root.left

    def next(self) -> int:
        ret = self.stack.pop()          # pop smallest node
        cur = ret.right                 # append left path of right child of smallest node
        while cur:
            self.stack.append(cur)
            cur = cur.left
        return ret.val

    def hasNext(self) -> bool:
        return len(self.stack)!=0

# 104.377 Combination Sum IV ==================================================== https://leetcode.com/problems/combination-sum-iv/
# Problem: Given an array of distince integers [nums] and an integer [target]. Return the number of combinations that sum up to
#          [target]. Integers of [nums] can be used zero or more times
# Description: Dynamic Programming. Maintain [dp], where dp[i] represents the number of combination that sum up to [i].
#              A sub-target [i] can be achieved from [i-n]. If dp[i-n] has "x" combination, then dp[i] has same. Because every 
#              combination in dp[i-n] can be appended by [n] to sum up to [i]. Therefore, dp[i] = dp[i]+dp[i-n] for every dp[i-n]!=0.
#              Iterate from 1 to [target] to find number of combinations that sum up to [i]. Iterate [n] in [nums] in inner loop, to
#              accumulate combination of dp[i]. Return dp[target] as the number of combination to sum up to [target]
# Time complexity: O(len(nums)*target)
def combinationSum4(nums: List[int], target: int) -> int:
    dp = [1]+[0 for _ in range(target)]          # dp[0] must be 1, otherwise nothing can be added to [dp]
    for curTarget in range(target+1):
        for n in nums:
            if n<=curTarget:
                dp[curTarget] += dp[curTarget-n]        # adding [n] to dp[curTarget-n] sum up to [curTarget].
    return dp[-1]

# 105.823 Binary Trees With Factors ======================================== https://leetcode.com/problems/binary-trees-with-factors/
# Problem: Given an integer array [arr], where each element is strictly greater than 1. Make binary tree with only elements of [arr], 
#          where a non-leaf node is equal to the product of both children. Return the number of binary trees can make. 
#          Ex: [2,4,5,20] => 9 ways
#              [2] = 2
#              [4], [2,2,4] = 4
#              [5] = 5
#              [20], [4,5,20], [5,4,20], [2,2,5,20], [5,2,2,20] = 20
#          The answer may be too large, return the answer modulo 10**9+7
# Description: Dynamic Programming. Maintain [dp], where "keys" are elements [target] from [arr] and "values" are number of ways to 
#              create tree with root of [target]. Sort [arr] in ascending order, and iterate through [arr]. For each iteration, find
#              ways to reach [target]. In the inner loop, iterate from beginning to index [i]. For an element arr[j] and its complement 
#              target//arr[j] also in [arr], then combination of dp[arr[j]]*dp[target//arr[j]] is the combination to mulitply to [target].
#              At the last, sum of [dp] values is the total number of binary trees, mod it by 10**9+7 to get final result
# Time complexity: O(N**2), N= len(arr)
def numFactoredBinaryTrees(arr: List[int]) -> int:
    arr = sorted(arr)
    dp = {n:1 for n in arr}                 # [dp] initally all values are 1
    for i, target in enumerate(arr):            # outer loop, iterate all elements in [arr] to be [target]
        for j in range(i):                          # inner loop, iterate up to [i], since elements after [i] are larger then [target]
            if target%arr[j] == 0 and target//arr[j] in dp:         # find arr[j] and its complement that multiply to [target]
                dp[target] += dp[target//arr[j]]*dp[arr[j]]         # [target] is equal to number of combination
    return sum(dp.values()) % (10**9+7)

# 106.179 Largest Number ================================================================= https://leetcode.com/problems/largest-number/
# Problem: Given a list of non-negative integers [nums]. Arrange them such that they form the largest number and return it in String form
# Description: Sort with customized Comparison. Implement "compare" method to compare two [n1] and [n2], who should come first. Compare
#              [n1+n2] and [n2+n1], the bigger one comes first. Implement sort algorithm leverage "compare" method. In this case, merge
#              sort is used.
# Time complexity: O(nlogn)
class LargestNumber:
    def compare(self, n1, n2):                          # compare method, that compare two numbers, who should come first
        return str(n1)+str(n2)>str(n2)+str(n1)
    
    def partition(self, nums, left, right):             # split [nums] from middle
        if left>right: 
            return
        elif left == right:
            return [nums[left]]
        mid = (right-left)//2+left                      # mid is [left+len//2]
        leftArr = self.partition(nums, left, mid)           
        rightArr = self.partition(nums, mid+1, right)
        return self.merge(leftArr, rightArr)
    
    def merge(self, leftArr, rightArr):
        res, left, right = [], 0, 0
        while left<len(leftArr) and right<len(rightArr):
            if self.compare(leftArr[left], rightArr[right]):        # Leverage "compare" method, larger number comes first
                res.append(leftArr[left])
                left += 1
            else:
                res.append(rightArr[right])
                right += 1
        if left < len(leftArr):
            res += leftArr[left:]
        if right < len(rightArr):
            res += rightArr[right:]
        return res

    def largestNumber(self, nums: List[int]) -> str:
        res = self.partition(nums, 0, len(nums)-1)
        return str(int("".join([str(ele) for ele in res])))     # Remove leading zeros, by converting [res] to int then convert to str

# 107.187 Repeated DNA Sequences ================================================ https://leetcode.com/problems/repeated-dna-sequences/
# Problem: Given a DNA sequence [s] with letters "A", "C", "G" and "T". Find all ten-letter-long sequence(substring) that occur more than
#          once. Answer can be returned in any order
# Description: Slide window with counter. Iterate through [s], look at 10-letter substring at a time, while record occurence of substrings.
#              At the end of iteration, if a substring's count is greater than 1. Then add it to result
# Time complexity: O(N)
def findRepeatedDnaSequences(s: str) -> List[str]:
    record = defaultdict(int)
    for i in range(len(s)-9):
        record[s[i:i+10]] += 1                          # record occurence of substring
    return [seq for seq in record.keys() if record[seq]>1]        # return substring [seq] if it's counter is larger than 1

# 108.622 Design Circular Queue =================================================== https://leetcode.com/problems/design-circular-queue/
# Problem: Implement a circular queue with fixed capacity. Circular queue is a linear data structure performs FIFO. 
#          Implement following methods
#          1. MyCircularQueue(k), k is the capacity of queue
#          2. Front() -> int, Return the front item, if queue is empty return -1
#          3. Rear() -> int. Return the last item, if queue is empty return -1
#          4. enQueue(val) -> bool. Insert val to the end of queue, return True if succeed, return False if failed
#          5. deQueue() -> bool. Delete last element from queue, return True if suceed, return False if failed
#          6. isEmpty() -> bool. Check queue is empty or not
#          7. isFull() -> bool. Check queue is full or not
# Description: List with [head] and [tail] index pointers. Inital a list with [k] elements. [head] is 1 and [tail] is -1, to indicate
#              an empty queue. [head] and [tail] should be set to 0 and -1 when deQueued to empty. The size of queue is calculated from
#              size = (tail+1-head)%capacity.
class MyCircularQueue:
    def __init__(self, k: int):
        self.data = [0] * k
        self.capacity = k
        self.head = 0
        self.tail = -1
        
    def enQueue(self, val: int) -> bool:
        if self.isFull(): return False
        else:
            self.tail = (self.tail+1)%self.capacity     # Move [tail] when enQueue
            self.data[self.tail] = val
        return True
        
    def deQueue(self) -> bool:
        if self.isEmpty(): return False
        if self.head == self.tail:              # reset [head] and [tail], when deleting to empty
            self.head, self.tail = 0, -1
            return True
        self.head = (self.head+1)%self.capacity         # Move [head] when deQueue
        return True

    def Front(self) -> int:
        if self.isEmpty(): return -1
        return self.data[self.head]

    def Rear(self) -> int:
        if self.isEmpty(): return -1
        return self.data[self.tail]

    def isEmpty(self) -> bool:
        return self.tail == -1

    def isFull(self) -> bool:               # when queue is full, size is modulo of capacity. Use isEmpty() to exclude corner case
        return not self.isEmpty() and (self.tail+1-self.head) % self.capacity == 0    

# 109.658 Find K Closest Elements From Sorted Array =========================== https://leetcode.com/problems/find-k-closest-elements/
# Problem: Given a sorted integer array [arr], and two integers [k] and [x]. Find [k] consecutive numbers that are the cloest to 
#          [x] within [arr]. The returned array should also be sorted in ascending order. Always take the smaller number, when there
#          is a tie between two numbers in [arr]
# Desciption: Binary search with Sliding window. Binary search for [x] in [arr], meanwhile maintain a slide window of size [k]. For 
#             each search, consider [mid] is the left-most element of sliding window and [mid+k] is the right-most element.
#             Here are several considtions to shift [left] and [right] after compare [mid], [mid+k] with [x]
#             1. x <= mid: Sliding window doesn't include [x], shift left
#             2. mid+k <= x: Sliding window doesn't include [x], shift right
#             3. mid < x < mid+k: there are two sub condistions
#               3.1: x-mid < [mid+k]-x: [x] is closer to [mid], shift left
#               3.2 x-mid > [mid+k]-x: [x] is closer to [mid+k], shift right
#             When shifting [right] make it [mid+1], when shifting [left] make it [mid]. At the end of loop, [left] will be the left
#             end of sliding window. Return [arr] between [left] and [left+k]
# Time complexity: O(logN)
def findClosestElements(arr: List[int], k: int, x: int) -> List[int]:
    left, right = 0, len(arr)-k
    while left < right:
        mid = (left+right)//2
        if x<=arr[mid]:                     # [x] is to the left of window, shift left
            right = mid
        elif arr[mid+k]<=x:                 # [x] is to the right of window, shift right
            left = mid+1
        elif x-arr[mid] > arr[mid+k]-x:     # [x] is closer to [mid+k], shift right
            left = mid + 1
        else:                               # [x] is closer to [mid], shift left
            right = mid
    return arr[left:left+k]

# 110.1239 Maximum Length of a Concatenated String with Unique Characters === https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters
# Problem: Given an array of string [arr]. A string [s] is formed by concatenate of one or more elements of [arr] that only contains
#          unique characters. Return the maximum length of [s]
# Description: Dynamic programming. Maintain a [dp] where an element of [dp] is a concatenation of unique characters. Iterate through 
#              [arr], for each element [ele], check if [ele] can form new candidate [s] with element of [dp]. To check uniqueness of
#              character, use set() intersection operator. If intersetion is not empty, then character is not unique. Use set() union
#              to concadinate [ele] and [s]. At the end, find the [s] in [dp] with maximum length
# Time complexity:O(2^n), for each [ele] there are two choices, pick or not pick. There are 2^n possible [s]
def maxLength(arr: List[str]) -> int:
    dp = [set()]
    for ele in arr:
        eleSet = set(ele)
        if len(ele) != len(eleSet): continue            # detect duplicates within an [ele]
        for s in dp:
            if s & eleSet: continue                     # duplicates found
            else:
                dp.append(s | eleSet)                   # no duplicates, concate [s] with current [ele]
    return max([len(candidate) for candidate in dp])

# 111.1339 Maximum Product of Splitted Binary Tree ======================= https://leetcode.com/problems/maximum-product-of-splitted-binary-tree/
# Problem: Given the [root] of a binary tree, split the binary tree into two subtrees by breaking one edge, such that the  production of the sums
#          of two subtrees is maximum. Return the maximum sum. if sum overflow, return its modulo of 10**9+7
# Description: Calculate sum of entire tree [total], and calculate sum of each subtree, save in [sums]. An element of [sums] is sum of one part,
#              [total-element] is sum of the other part. Multiple them and find maximum.
# Time complexity: O(N), where N is number of edges
def maxProduct(root: Optional[TreeNode]) -> int:
    sums = set()
    def helper(node):
        if not node: 
            return 0
        subSum = helper(node.left)+helper(node.right)+node.val
        sums.add(subSum)
        return subSum
    total = helper(root)
    return max([total*(total-val) for val in sums]) % 1_000_000_007

# 112.279 Perfect Square ===================================================================== https://leetcode.com/problems/perfect-squares
# Problem: Given an integer [n], return the least number of perfect square number that sum up to [n].
#          Perfect square numbers are is a square of an integer number, such as 1, 4, 9, 16
# Description: Dynamic Programming. Craete a list with [n+1] element [dp], where dp[i] is the least number of perfect square
#              numbers that sum up to [i]. Initially, dp[0] = 0 and dp[i] = i. Because the worst case senario, i = 1*i.
#              For each [i], try to find [squares] that less than [i], and update dp[i] by min(dp[i], 1+dp[i-square]).
#              return dp[-1] at the end, as the least number of perfect square that sum up to [i]
# Time complexity: O(NlogN)
def numSquares(n):
    dp = [i for i in range(n+1)]
    for i in range(n+1):
        j = 0
        while j*j <= i:
            square = j*j                            # find perfect [square] that less than [i]
            dp[i] = min(dp[i], 1+dp[i-square])      # find less number that sum up to [i]
            j += 1
    return dp[-1]

# 113.841 Keys and Rooms ================================================================= https://leetcode.com/problems/keys-and-rooms/
# Problem: There are [n] rooms, all rooms are locked except first room. Each room may or may not contain several keys, each key has a 
#          number on it, that can unlock corresponding rooms. Given an array [rooms], each element of [rooms] is an array that represent
#          [keys] in this room. Return True if can visit all rooms, return False otherwise
# Description: DFS. Maintain a set [visited] denote rooms that are visited. Starting from room 0, recursivly visit keys in current room.
#              Add room number to [visited] if not visited before. If all rooms are visited, [visited] should has same size of [rooms].
# Time complexity: O(N), where N is number of sub-element in [rooms]
def canVisitAllRooms(rooms: List[List[int]]) -> bool:
    visited = set([0])                  # initally only room 0 is visited
    def visit(keys):
        for key in keys:                    # recursivly visit keys in current room
            if key not in visited:
                visited.add(key)
                visit(rooms[key])
    visit(rooms[0])                    # start from room 0, and DFS
    return len(visited)==len(rooms)

# 114.300 Longest Increasing Subsequence ===================================== https://leetcode.com/problems/longest-increasing-subsequence
# Problem: Given an integer array [nums], return the length of longest strictly increasing subsequence(LIS).
#          Subsequence is a sub-array that can be derived from another array by removing some or no elements, without changing order
# Description: Dynamic Programming. Create an array [dp], where each element represent the LIS that starting from [i] to the end of [nums].
#              Construct [dp] from right to left, dp[-1] is always 1, because there only one element to the end. Iterate [nums] towards
#              left, and for each iteration, find all elements to the right of [i] that are greater than nums[i]. nums[i] can be prepended
#              the that subsequence, therefore dp[i] will be the largest among possible subsequence. If no element is greater than nums[i],
#              nums[i] itself will be a subsequence, so that dp[i] = 1. At the end of iteration, find the maximum in [dp]
# Time complexity: O(N**2)
def lengthOfLIS_dp(nums: List[int]) -> int:
    dp = [1]*len(nums)
    for i in reversed(range(len(nums)-1)):          # iterate nums from right to left
        for j in range(i+1, len(nums)):                 # find element from nums[i:] that is greater than nums[i]
            if nums[i]<nums[j]:                         
                dp[i] = max(dp[i], dp[j]+1)                 # find longest among subsequences
    return max(dp)

# Description: Greedy with binary search. Create [res] as result subsequence. Adding element to the end of [res] if greater than any of [res],
#              use binary search to substitue a larger element in [res] if a smaller element is found. It can reduce value of [res] as much
#              as possible, and enlarge buffer to add more elements to [res]
# Time complexity: O(NlogN)
from bisect import bisect_left
def lengthOfLIS_greedy(nums: List[int]) -> int:
    res = []
    for num in nums:
        if len(res)==0 or res[-1]<num:          # append large element to right
            res.append(num)
        else:
            index = bisect_left(res, num)       # substitute larger element in [res] with smaller (with binary search)
            res[index] = num
    return len(res)

# 115.207 Course Schedule ================================================================== https://leetcode.com/problems/course-schedule
# Problem: There are [numCourses] courses labeled from 0 to [numCourses-1]. Given an array of [prerequisites], where each element is
#          prerequisites[i] = [a, b], that b is the prerequisite of a, meaning you must take b before taking a.
#          Return True if all courses in [numCourses] can be finished, otherwise return False
# Description: DFS with vistied memory. Create a [graph], [i]th course's prerequisites is at graph[i]. Create a [visited], elements can 
#              take three values, 0 is not visited, 1 is visited meaning can be finished, -1 is currenting visiting, visiting a course
#              that is being visited indicates a loop, which means a course can't be finished.
#              In DFS, if graph[i] = -1, return False because of loop. If graph[i] = 1, return True because course [i] can be finished.
#              If graph[i] = 0, course [i] is not checked. Then set visited[i] = -1 as being visited, and DFS on course [i]. If any of
#              the prerequisites of course [i] returns False, then course [i] can't be finished return False. If all of its prerequisies
#              return True, set visited[i] = 1 as it can be finished.
#              DFS starting from every course of [nunmCourses], if all of them return True, then all courses can be finished. Otherwise
#              return False
# Time Complexity: O(N)
def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    graph = [[] for i in range(numCourses)]         # graph represents courses and their prerequisites
    visited = [0 for i in range(numCourses)]        # initially all courses are not visited
    for course, prerequisite in prerequisites:      # populate [course]s and their [prerequisite]s
        graph[course].append(prerequisite)

    def dfs(course):
        if visited[course] == -1:       # course is being visited, loop detected
            return False
        elif visited[course] == 1:        # course is checked and can be finished
            return True
        else:                              # course if not check
            visited[course] = -1                # mark [course] as being visited
            for pre in graph[course]:           # check [pre] of course, course can't be finished if any of its [pre] can't be finished
                if not dfs(pre):                    
                    return False
        return True

    for course in range(numCourses):        # check every course
        if not dfs(course):
            return False
    return True

# 116.139 Word Break =============================================================================== https://leetcode.com/problems/word-break/
# Problem: Given a string [s] and a word collection [wordDict], return True if [s] can be segemented into a sequence of one or more words from
#          [wordDict]. The same word from [wordDict] can be reused
# Descritpion: Dynamic Programming. Create a [dp] of Boolean array, dp[i] represent s[:i] can be segemented with [wordDict]. dp[0] is always 
#              True, and dp[i] can be derived by checking if any [word] of [wordDict] is the segement that come after index [i], meaning 
#              [word] == s[i-len(word):i]. While if dp[i-len(word)] is True. Then dp[i] can be set to True.
#              If [s] can be segemented into [wordDict], dp[-1] should be True. Return dp[-1] at the end
# Time complexity: O(N*M), N=len(s), M=len(wordDict) 
def wordBreak(s: str, wordDict: List[str]) -> bool:
    dp = [True]+[False]*len(s)                      # dp[i] represent whether s[0:i] can be segemented or not
    for i in range(1, len(s)+1):
        for word in wordDict:
            if dp[i-len(word)] and s[i-len(word)] == word:
                dp[i] = True
    return dp[-1]

# 117.210 Course Schedule II ============================================================== https://leetcode.com/problems/course-schedule-ii/
# Problem: Given a total of [numCourses] courses that need to be taken, labeled from 0 to [numCourses-1]. Given an array of [prerequisites],
#          each element is prerequisites[i] = [a, b], that must take course "b" before taking course "a". Return the ordering of courses that
#          you should take to finish all given courses. If many possible orders, return any of them. If not possible to finish, return empty
#          list.
# Description: Graph with DFS. Maintin [prerequisites] is an adjacency list, where indecis are number of courses, and elements are prerequisites
#              of each course. Maintain a [visited] set to track if a course can be finished. "0" => not checked, "1" => can be finished, and 
#              "-1" => currently checking. If a "-1" course is visited twice, then a loop is detected the courses in a loop can't be finsihed.
#              Use DFS trace down all courses, keep visiting "unchecked" course. If a course can be finished, add it to [res] list, and return
#              True. If a "-1" course is visited twice, return False to indicate current path can't be finished. If a course is visited first
#              time, and its prerequisites are undetermined, DFS on its prerequisites, and if any prerequisites return False, this course will
#              return False.
# Time Complexity: O(m+n), m: number of courses, n: number of edges
def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    # maintain adjacency list
    graph = [[] for _ in range(numCourses)]
    for cor, pre in prerequisites:
        graph[cor].append(pre)

    visited = {k:0 for k in range(numCourses)}
    res = []
    def dfs(course):
        if visited[course] == -1:         # loop detected when a course is visited twice
            return False
        if visited[course] == 1:          # course is checked, and can be finished
            return True
        visited[course] = -1            # start visiting a unvisited course, check its prerequisites
        for pre in graph[course]:
            if not dfs(pre):                # if any prerequisites can't be finished, current couse return False
                return False
        res.append(course)              # current course can be finished, and add to [res]
        visited[course] = 1             # denote a course as checked
        return True

    for course in range(numCourses):        # check all course 
        if not dfs(course):                     # if any course can't be finished
            return []
    return res

# 118.238 Product of Array Except Self ============================== https://leetcode.com/problems/product-of-array-except-self/description/
# Problem: Given an array [nums], return an array [res] such that res[i] is the production of all elements in [nums] expect nums[i]
#          write an algorithm runs in O(n) and without using division
# Desciption: Maintain two lists, [preList] that contains the accumulated production of [nums] from left to right, and [postList] that contains
#             the accumulated production from right to left. When calculating res[i], it is the production of preList[i] * postList[i].
#             Because, preList[i] is the production of elements to the left of nums[i], and postList[i] is the production of elements to 
#             the right of nums[i].
#             Note: when calculatiing [preList] and [postList], the index are shifted accordingly, so that preList[i] is production of nums[:i]
#             and postList[i] is production of nums[i+1:]
# Time complexity: O(N)
# Space complexity: O(N)
def productExceptSelf(nums: List[int]) -> List[int]:
    preList, postList = [1 for _ in nums], [1 for _ in nums]
    for i in range(1, len(nums)):               
        preList[i] = preList[i-1]*nums[i-1]     # shift index to left
    for i in range(len(nums)-2, -1, -1):
        postList[i] = postList[i+1]*nums[i+1]   # shift index to right
    res = []
    for i in range(len(nums)):
        res.append(preList[i]*postList[i])
    return res

# Description: Maintain a List [res] that contains accumulated production of [nums] from left to right. The index is shifted to right so that 
#              res[i] has accumulated production of nums[:i]. Iterate [nums] from right to left, while tracking the accumulated production [post], 
#              whose initial value is 1. Example
#                               1  2  3  4
#                     res       1  1  2  6  
#               post = 1                 6  
#               post = 1*4            8  6
#               post = 1*4*3       12 8  6
#               post = 1*4*3*2  24 12 8  6
# Time Complexity: O(N)
# Space Complexity: O(1), if output space doesn't count
def productExceptSelf(nums: List[int]) -> List[int]:
    res = [1]
    for i in range(1, len(nums)):           # calculate accumulated from left to right, with index shifted to right by 1
        res.append(res[i-1]*nums[i-1])
    post = 1
    for i in range(len(nums)-1, -1, -1):    # iterate from right to left, and tracking accumulated production
        res[i] *= post
        post *= nums[i]
    return res

# 119.221 Maximal Square  ========================================================== https://leetcode.com/problems/maximal-square/description/
# Problem: Given a m*n [matrix] with "0"s and "1"s. Find the largest square contains only "1"s and return its area
# Description: Dynamic Programming. Maintain a 2D list [dp] with one more row and column than [matrix], where the first row and column represent
#              the initial state, dp[i+1][j+1] represent the maximal area at matrix[i][j]. The value at dp[i+1][j+1] depends on the cell on top 
#              (dp[i-1][j]), to the left (dp[i][j-1]) and up left (dp[i-1][j-1]). If matrix[i][j] == 0, it's not possible to form a square at 
#              this cell, then dp[i+1][j+1] = 0. If matrix[i][j]==1, dp[i+1][j+1] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1. Which means,
#              there are already three squares to left, up and up-left, adding a cell at [i+1][j+1] can form a bigger square.
# Time Complexity: O(m*n)
def maximalSquare(matrix: List[List[str]]) -> int:
    dp = [[0 for _ in range(len(matrix[0])+1)] for _ in range(len(matrix)+1)]       # dp memory
    res = 0
    for i in range(1, len(matrix)+1):
        for j in range(1, len(matrix[0])+1):
            if matrix[i-1][j-1] == "1":
                dp[i][j] = min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1])+1              # form a bigger square
                res = max(res, dp[i][j])                               # track current maximum
    return res**2

# 120.323 Number of Connected components in a undirected graph =============== https://www.youtube.com/watch?v=8f1XPm4WOUc&list=PLot-Xpze53lfOdF3KwpMSFEyfE77zIwiP&index=41
# Problem: Given a graph of [n] nodes by giving an ingteger [n] and an array of [edges], where each edge edges[i] = [ai, bi], indicates there
#          is an edge between node ai and bi. Return the number of connected component in the graph
# Description: UnionFind. Maintain an array [parent] of [n] elements, representing the "root" ancestor of each node. Initially it will be [0, 1, 2, ... , n-1], meaning 
#              each node is the parent of themselves. Maintain an array [rank] of [n] elements, represent the size connected components that the node is part of. Initally,
#              there are all ones in [rank], that each node is the parent of themselves. 
#              Iterate through [edges], for two nodes [n1] and [n2] in the edge. Find the root ancestor of both nodes by seaching [parent] array. If they have same ancetsor
#              they are already connected, no merging is needed. Otherwise, compare the [rank] of both nodes. The node has smaller rank will be merged to the other one. 
#              Update parent of the node with smaller [rank], and update [rank] of new parent node. Maintain the number of connected component [res], intially [n] since all 
#              nodes are disconnected. When a merge is done, decrease [res] by 1 because a disconnected node becomes part of connected component
# Time Complexity: O(v+e), number of vector and edges
def countComponents(n: int, edges: List[List[int]]) -> int:
    parent = [i for i in range(n)]
    rank = [1 for _ in range(n)]
    # find the root ancestor of [node]
    def findParent(node):             
        res = node
        while res != parent[res]:   # when a node is parent of itself. the root is found
            res = parent[res]
        parent[node] = res          # update the ancestor of [node]
        return res
    # union two nodes
    def union(node1, node2):
        parent1, parent2 = findParent(node1), findParent(node2)     # find parent of both node
        if parent1 == parent2:      # two nodes already connected, no merge needed
            return 0
        if rank[parent1]>rank[parent2]:         # [parent1] has higer rank, [parent1] should be parent of [parent2]
            parent[parent2] = parent1
            rank[parent1] += rank[parent2]
        else:                               # [parent2] has higher rank, [parent2] should be parent of [parent1]
            parent[parent1] = parent2
            rank[parent2] += rank[parent1]
        return 1
    
    res = 0
    for node1, node2 in edges:
        res -= union(node1, node2)
    return res

# 121.547 Number of Provinces =========================================================== https://leetcode.com/problems/number-of-provinces/description/
# Problem: Given a n*n matrix [isConnected], where isConnected[i][j]=1 if city "i" is connected to city "j", and isConnected[i][j]=0 if city "i" and city
#          "j" is not connected. Return the total number of "provinces", that a "province" is a group of cities that cities form a connected component 
# Description: UnionFind. Maintain a list [parent] with same size of cities, representing the root parent of each cities. Initially [0,1,2, ... ,n-1], 
#              meaning no cities has been connected and they are parent of themselves. Create a methond "find(city)", that trace back [parent] list to
#              find root parent of [city]. Create a method "union(city1, city2)", that connect [city1] and [city2] if they are connected by an edge.
#              "union" method, looks up root parents of [city1] and [city2] by using "find". If both return same parent then city1 and city2 are already
#              connected. Otherwise, update city2'parent to be city1's parent, as city1 and city2  are connected and should share same root parent.
#              Maintain the nunmber of province [res], intially as number of cities, as none of them are conneced. Union method return 1 if a connected 
#              is established, otherwise return 0. Return value subtract [res] to track the current number of province.
#              Iterate [isConnected], if isConnection[i][j] == 1, try union them by union(i, j)
# Time Complexity: O(v+e), number of vector and edges
def findCircleNum(isConnected: List[List[int]]) -> int:
    parent = [i for i in range(len(isConnected))]           # parent lookup, initally no city is connected, they are parents of themselves
    # loopup root parent of [node]
    def findParent(node):                 
        while node != parent[node]:     
            node = parent[node]
        return node
    # try union [city1] and [city2]
    def union(city1, city2):            
        parent1 = findParent(city1)
        parent2 = findParent(city2)
        if parent1 == parent2:          # city1 and city2 already connected
            return 0
        else:
            parent[parent1] = parent2   # connect city1 and city2, by updating one of the root parent to another root parent
            return 1
    res = len(isConnected)      # track number of province, initally each city is a province
    # iterate through isConnected
    for i in range(len(isConnected)):
        for j in range(1, len(isConnected)):
            if isConnected[i][j] == 1:          # find two connected cities [i][j], and try union
                res -= union(i, j)              # reduce number of province if [i] and [j] is connected
    return res  

# 122.1041 Robot Bounded In Circle ======================================================= https://leetcode.com/problems/robot-bounded-in-circle/description/
# Problem: Given a robot starting at location [0, 0] and fact north. The robot receives one of three instructions at a time.
#          "G" move forward by 1 unit
#          "L" turn 90 degrees to left
#          "R" turn 90 degrees to right
#          The robot perform [instructions] given in order, and repeat the forever. Return True if and only if there exists a circle, such that the robot
#          never leaves the circle
# Desciption: If there exists a circle, the robot must execute the instruction once or twice or four times to create the circle. Because, if the [instructions]
#             create the circle by itself, robot execute it once. If the robot facing south after executing [instructions] once, the robot went through half of
#             the circle, and execute it again will form a complete circle. If the robot facing east or west after executing [instructions] once, the robot went
#             through a quarter of the circle, execute the [instructions] three more time will form a complete circle.
#             Execute [instructions] once, twice, and four time. Return True if any of the executions form a circle
# Time complexity: O(N)
def isRobotBounded(instructions: str) -> bool:
    def execute(instructions, i, j, d):
        directions = [[0,1],[1,0],[0,-1],[-1,0]]        # indicate directions for easy movement
        for ins in instructions:
            if ins == "G":
                i += directions[d][0]
                j += directions[d][1]
            elif ins == "L":
                d = (d-1)%4
            elif ins == "R":
                d = (d+1)%4
        if i == 0 and j == 0:
            return True
        else:
            False
    # executre [instructions] once, twice or four times
    return execute(instructions, 0, 0, 0) or execute(instructions*2, 0, 0, 0) or execute(instructions*4, 0, 0, 0)

# Desciption: Based on previous method, if the robot facing south, east or west after executing [instructions] once. It will form a circle eventually. 
#             Therefore, execute [instruction] once and check if robot return the starting point, or it is facing north. 
# Time complexity: O(N)
def isRobotBounded_2(instructions: str) -> bool:
    directions = [[0,1],[1,0],[0,-1],[-1,0]]
    i = j = d = 0               # [d] tracks direction of robot
    for ins in instructions:
        if ins == "G":
            i += directions[d][0]
            j += directions[d][1]
        elif ins == "L":
            d = (d-1)%4
        elif ins == "R":
            d = (d+1)%4
    if i == 0 and j == 0:       # check robot return to starting point
        return True
    elif d != 0:                # check robot is NOT facing north
        return True
    return False

# 123.417 Pacific Atlantic Water Flow ===================================================== https://leetcode.com/problems/pacific-atlantic-water-flow/
# Problem: Given an island as a m*n matrix, where Pacific Ocean is at its top and left borders, and Atlantic Ocean touches its right and bottom borders.
#          A cell "height[i][j]"" reprents the height above sea leave of an area, and water can flow from high area to low area. Assume the island 
#          receives rain and rain water flow to neighboring cell directly north, south, east or west if the neighboring cell's height is lower or 
#          equals to the current cell's height. When water reaches border, it can flow to ocean.  Return a 2D list of coordinates [results], where
#          results[i] = [ri, ci] denotes that the rain can flow from [ri, ci] to both Pacific Ocean and Altantic Ocean
# Description: Starting from Pacific Ocean and Altantic Ocean, use DFS to search for cells that can reach each Ocean, the intersection of them is [results]
#              In DFS, pass current coordinate, previous height, and a set of cell that are reachable. If current coordiante is visited, or out of boundary 
#              skip it. If current height is higher than previous height, current cell can flow to Ocean. Add it to reachable list, and DFS its four neighbors. 
#              DFS Pacifica Ocean and Atlantic Ocean separately. For cell flow to Pacific Ocean, maintain a set [toPO], DFS starting from first row and first 
#              column. For cell flow to Atlantic Ocean, maintain a set [toAO] and DFS starting from last row and last column. Return intersection of [toPO] and
#              [toAO] 
# In Short: Start from two oceans to find cell can reach each of them and take intersection
# Time Complexity: O(N)
def pacificAtlantic(heights: List[List[int]]) -> List[List[int]]:
    ROWS, COLS = len(heights), len(heights[0])
    def dfs(i, j, preHeight, visited):
        if (i,j) not in visited and 0<=i<ROWS and 0<=j<COLS and heights[i][j]>=preHeight:
            visited.add((i, j))
            dfs(i+1, j, heights[i][j], visited)
            dfs(i-1, j, heights[i][j], visited)
            dfs(i, j-1, heights[i][j], visited)
            dfs(i, j+1, heights[i][j], visited)
        
    toAO, toPO = set(), set()
    for i in range(len(heights)):
        dfs(i, 0, 0, toPO)
        dfs(i, COLS-1, 0, toAO)
    for j in range(len(heights[0])):
        dfs(0, j, 0, toPO)
        dfs(ROWS-1, j, 0, toAO)
    
    return list(toAO & toPO)

# 124.560 Subarray Sum Equals K =================================================================== https://leetcode.com/problems/subarray-sum-equals-k/
# Problem: Given a list of [nums] and a integer [k]. Return total number of subarray that sum up tp [k]
# Description: The sum of a substring from i to j (sum[i:j]) is total sum of 0 to j (sum[0:j]) minues the sum of previous subarray from 0 to i (sum[0:i]). 
#              Itearte through [nums], keep tracking the [preSum] sum[0:i] and its number of occurance. For each iteration, the [curSum] is sum[0:i],
#              check if exists a [preSum] that [curSum]-[preSum]==[k]. Then the number of occurance of [preSum] is the number of subarray that sum up to [k]
# In Short: Track sum of subarry from 0 to i as [preSum] and its occurance, check if exists a [preSum] that [k] = [currentSum] - [previousSum]. 
def subarraySum(nums: List[int], k: int) -> int:
    preSums = defaultdict(int)
    preSums[0] = 1               # preSum=0 for subarray starting from index 0
    res = curSum = 0
    for n in nums:
        curSum += n
        if curSum-k in preSums:             # check if exists a preSum = curSum-k
            res += preSums[curSum-k]
        preSums[curSum] += 1                # add curSum to preSum, as it becomes previous subarray for later iteration
    return res

# 125.1466 Reorder Routes to Make All Paths Lead to the City Zero ===== https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/
# Problem: Given [n] cities from 0 to n-1, and n-1 ways such that there is only one way connects between two cities. Roads are represented by list [connections]
#          where connects[i] = (city_1, city_2) meaning you can travel from city_1 to city_2, but not the other way around. Reorder the dicrection of roads,
#          such that all cities can reach city 0, return the number of roads need to be redirected.
# Description: Maintain a city-neighbors Map, and Depth first search. Iterate through [connections], if a road connects city_1 and city_2, then note in Map 
#              that city_1 is a neighbor of city_2 and city_2 is a neighbor of city_1. 
#              DFS starting from city_0, check neighbor of city. If there exists a road in [connections] that is from city to a un-visited neighbor, then this
#              road need to be redirected. DFS on neighbors of city, and denote city as visited.
# Time Complexity: O(N)
def minReorder(n: int, connections: List[List[int]]) -> int:
    conn = set((i,j) for i,j in connections)
    # create city-neighbors map
    neibors = {city: [] for city in range(n)}
    for i, j in connections:
        neibors[i].append(j)
        neibors[j].append(i)
    res = 0
    visited = set()
    def dfs(city):
        nonlocal res
        if city in visited:
            return
        visited.add(city)       # mark [city] as visited
        for neibor in neibors[city]:
            if neibor not in visited and (city, neibor) in conn:
                res += 1
            dfs(neibor)
    dfs(0)              # start dfs from city_0
    return res

# 126.1963 Minimum Number of Swaps to Make the String Balanced =============== https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced
# Problem: Given a string of lenght [n] consists only opening and closing brackets, and number of opening or closing brackets are both [n/2]. The brackets might
#          not be balanced, that there might not be a opening bracket comes before a closing bracket to pair with each other. You can swap any two indices any 
#          number of times to make entire string balanced. Return the minimum number of swaps.
# Description: Find un-paired closing bracket, and swap with opening bracket from end of string. Maintain s [stack] to track number of opening brackets that
#              are un-paired. Iterate through [s]. If [i] is '[', increase [stack] as new "[" is found. If [i] is "]", try pair with "[" in [stack], 
#                   if [stack] is non-zero, decrease [stack] by 1 to pair a closing with an opening
#                   if [stack] is zero, no "[" can be paired, swap needed. Search from [end] of [s], and swap it with first "[" found. Denote a swap is perfromed,
#                       and increase [stack] by 1, as a "[" is swapped to front. Also denote current location of [end], since next search with start from there
# Time complexity: O(n)
def minSwaps(s: str) -> int:
    stack = 0
    swaps = 0
    end = len(s)-1
    for bracket in s:
        if bracket == "[":      # found "[", need to be paired later
            stack += 1
        else:                   # found "]"
            if stack:               # check if "]" can be paired with a prior "[" 
                stack -= 1
            else:                   # no "[" can be paired, swap
                while s[end] != "[":        # find a "[" from end of [s]
                    end -= 1
                stack += 1                  # a "[" is moved to front, and need to be paired later
                swaps += 1                  # a swap is done
    return swaps

# 127.120 Triangle ==================================================================================================== https://leetcode.com/problems/triangle
# Problem: Given a [triangle] array, return the minimum path sum from top to bottom. For each step, you may move to an adjacent cell of the lower level.
#          A [triangle] is a 2-D array, that an element at index [i] is a sub-list that contains [i+1] element of integers
# Desciption: Dynamic Programming. Move from bottom to top. Maintain a [dp] list, that tracks the minimum path sum at a cell of certain level. and is updated 
#             every level. Initially, [dp] contains elements of bottom level. When moving upwards, the minimum path sum at index [i] at upper level is the 
#             smaller path sum of adjacent cell at lower level, plus the number at index [i] of upper level, dp[i] = min(dp[i], dp[i+1])+n. At the end of 
#             traversal, the minimum path sum is kept at dp[0]
# Time complexty: O(N), N is number of numbers in [triangle]
def minimumTotal(triangle: List[List[int]]) -> int:
    dp = [n for n in triangle[-1]]          # Initially dp can be all zero and same size of triangle[-1]
    for level in reversed(triangle[:-1]):       # traverse all levels, if dp are all zero initially
        for i, n in enumerate(level):
            dp[i] = min(dp[i], dp[i+1])+n       # update min at index [i] for a level
    return dp[0]

# 128.347 Top K Frequent Elements =================================================================== https://leetcode.com/problems/top-k-frequent-elements/
# Problem: Given an array [nums] and an integer [k], return the [k] most frequent elements. Must solve better than O(nlogn) time complexity
# Description: Bucket Sort. Count frequence of each number, store number and its count in an array, where [index] is count/frequence and element is sub-list 
#              that represent the numbers that has that count/frequence. The first [k] elements have biggest indeices are the [k] most frequent.
# Time compleixty: O(N)
def topKFrequent(nums: List[int], k: int) -> List[int]:
    cnt = Counter(nums)
    freq = [[] for _ in range(len(nums)+1)]     # frequence list of size len(nums)+1. Biggest possible frequence is len(nums)+1, where items in [nums] are identical
    for key, val in cnt.items():
        freq[val].append(key)           # load numbers to corresponding frequence
    res = []
    for i in reversed(range(len(freq))):        # iterate from right to left, to find elements have biggest frequence
        if freq[i]:
            res += freq[i]
        if len(res) == k:               # return [res] when [k] elements are found
            return res

# 129.215 Kth Largest Element in an Array ======================================================= https://leetcode.com/problems/kth-largest-element-in-an-array/
# Problem: Given an array [nums] and an integer [k]. Find the [k]th largest element in [nums]. It is the [k]th largest in sorted order, not [k]th distinct largest
#          element. Solve it without sorting
# Description: Heap. Maintain a min heap of size [k]. Iterate through [nums] and replace heap[0] once a larger element is found. At the end, heap will contain
#              [k] largest element, and heap[0] is the [k]th largest.
#              Heap: a binary tree represented by an array, where parent is larger(max heap)/small(min heap) than its children. 
#              heap push: Insert at the end of array and pop up until a valid heap
#              heap pop: remove heap[0] and replace by heap[-1], pop down heap[0] until a valid heap
# Time complexity: O(klogn), need to heapify k times
def findKthLargest(nums: List[int], k: int) -> int:
    heap = nums[:k]
    heapq.heapify(heap)
    for n in nums[k:]:
        if n>heap[0]:
            heapq.heappop(heap)
            heapq.heappush(heap, n)
    return heap[0]

# Description: Quick Select. Partition [nums] that smaller elements are on left side, and larger elements are on right side. Find the element at index [len(nums)-k], 
#              By repeat partitioning, until pivot index is at [len(nums)-k], meaning the nunmber at index [len(nums)-k] is found, and it is the [k]th smallest number.
#              Becase, when nums is in ascending order, which is [len(nums)-k]th smallest and [k]th largest.
# Time Complexity: O(n)
def findKthLargest(nums: List[int], k: int) -> int:
    targetIndex = len(nums)-k       # index of kth largest in ascending order

    def quickSelect(left, right):
        pivot, p = nums[right], left
        for i in range(left, right):                # partition 
            if nums[i]<=pivot:
                nums[i], nums[p] = nums[p], nums[i]
                p += 1
        nums[p], nums[right] = nums[right], nums[p]
        if p == targetIndex:            # [p] index is the kth largest, target found
            return nums[p]
        elif p < targetIndex:           # targetIndex is to the RIGHT-side of pivot, search right partition
            return quickSelect(p+1, right)
        else:                           # targetIndex is to the LEFT-side of pivot, search right partition
            return quickSelect(left, p-1)

    return quickSelect(0, len(nums)-1)

# 139.1888 Minimum Number of Flips to Make the Binary String Alternating ======= https://leetcode.com/problems/minimum-number-of-flips-to-make-the-binary-string-alternating
# Problem: Given a string [s] representing binary number, consisting only "0" and "1". You are allowed to perform two types of opeartions in any sequences
#          Type-1: Remove a character from beginning of [s] and append it to the end
#          Type-2: Pick any index of [s] and flip the bit value
#          Return the minimum number of Type-2 need to perform, such that bits in [s] become alternating, number of Type-2 doesn't count.
# Description: Sliding window. There are two possible alterante results, one start with "0101", the other start with "1010". Create both of the results as [target_1] and
#              [target_2] with length of len(s+s), and use a slide window of length len(s) to mimic Type-1 opeartion. Initially, check number of flips between [s] and 
#              [target]s. Then start sliding by checking s[0] and s[len(s)], and reduce or increase current number of flips based on the comparison of [target]s. Track
#              the minimum flips of all time
# Time compleixty: O(N)
def minFlips(s: str) -> int:
    length = len(s)
    s = s*2             # double [s] for slide
    target_1 = "".join([str(i%2) for i in range(len(s))])           # create two possible [targets]
    target_2 = "".join([str((i+1)%2) for i in range(len(s))])
    curFlip_1 = curFlip_2 = 0           # intial number flips to both [target]s
    for i in range(length):
        if target_1[i] != s[i]:
            curFlip_1 += 1
        if target_2[i] != s[i]:
            curFlip_2 += 1
    res = min(curFlip_1, curFlip_2)
    i, j = 0, length
    while j<len(s):                     # slide window by removing s[i] and append s[j], and track [currFlip]s
        if target_1[i] != s[i]:
            curFlip_1 -= 1
        if target_1[j] != s[j]:
            curFlip_1 += 1
        if target_2[i] != s[i]:
            curFlip_2 -= 1
        if target_2[j] != s[j]:
            curFlip_2 += 1
        i, j = i+1, j+1
        res = min(res, curFlip_1, curFlip_2)        # track minimum flips of all time
    return res

# 140.416 Partition Equal Subset Sum ==================================================================== https://leetcode.com/problems/partition-equal-subset-sum/
# Problem: Given a list of integers [nums], return True if [nums] can be splited into two parts, and the sum of both parts are equal. Return False otherwise.
# Description: 0-1 knapsack that check if any sub-list can sum up to sum(nums)//2. Consider a Boolean 2D list [dp], that dp[i][j] represents whether exists a sub-list
#              in nums[:i+1], that sum up to [j]. Thus, dp[i][j] is True, when 1) dp[i-1][j] is True (already exists a sublist that sum up to [j]), or
#              2) dp[i-1][j-nums[i]] is True (a pre-existing sub-list combining with current nums[i] can sum up to [j])
#              dp[0][0] is True intially, since if don't pick any number it sum up to 0. And dp[i] only depends on dp[i-1], we can update [dp] in-place and only allocate
#              [dp] of size sum(nums)//2
# Time compleixty: O(n*sum). n=len(nums) sum=sum(nums)
def canPartition(nums: List[int]) -> bool:
    s = sum(nums)
    if s%2 == 1:            # odd sum can't be equally partitioned, return False
        return False
    half = s//2
    dp = [True]+[False]*half
    for i in range(len(nums)):
        # use python trick to update [dp] in place, otherwise need to track dp[i-1] and dp[i]
        dp = [dp[j] or (j>=nums[i] and dp[j-nums[i]]) for j in range(half+1)]
        if dp[half]:                # half sum can be partitioned, early return
            return True
    return False

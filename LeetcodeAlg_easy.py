import collections
from typing import *        # enable type hint
class ListNode:
    def __init__(self, val=0, next=None) -> None:
        self.next = next
        self.val = val

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 1.1 Two sum ========================================= URL: https://leetcode.com/problems/two-sum/
# Problem: Given an numeric array, find two number in the array that sum up to a given number
#          Return indices of these two numbers
# Description: Iterate through [nums] array, record into a hash table (key, val) where
#              key is (target-nums[i]), the number required by nums[i]
#              val is index of nums[i] <=> i
# Time complexity: O(n)
#                  Iterate [nums] array check existence of required number of nums[i] in hash or 
#                  if required number not found, add a new required number of nums[i] in to hash
def twoSum(nums, target):
    if len(nums) <= 1:
        return False
    buff_dict = {}
    for i in range(len(nums)):
        if nums[i] in buff_dict:
            return [buff_dict[nums[i]], i]
        else:
            buff_dict[target - nums[i]] = i

# 2.2 Add two integers  ============ URL: https://leetcode.com/problems/add-two-numbers/
# Problem: Given two linked lists of integers which each element represents a single digit, 
#          the digits are stored in reverse order s.t. units -> tens -> hundreds -> thousands
#          Add two linked list together(add two numbers together), return sum as integer type
# Description: Iterate from head of both linkedlist (start from lowest digit), 
#              keep tracking digits of both linkedlist and the carry of last summation
#              if one of them is not None, will do the addition
# Time Complexity: O(n) simply iterate through both linked list and store sum in another linked list
class Node(object):
    def __init__(self, x, nextNode = None):
        self.val = x
        self.next = nextNode

    @staticmethod
    def addTwoNumbers(l1, l2):
        carry = 0
        root = n = Node(0)                              # sum is stored in [n], first digit is root.next
        while l1 or l2 or carry:                        # Do the addition when one of them is not None
            v1 = v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry, val = divmod(v1+v2+carry, 10)      # value = (v1+v2+carry)//10 , carry = (v1+v2+carry)%10
            n.next = Node(val)
            n = n.next
        return root.next

# 3.7 Reverse Integer ============= URL: https://leetcode.com/problems/reverse-integer/
# Problem: Reverse digits of a given integer. If number is positive, reverse directly
#          if number is negative, keep the negative sign and reverse the numeric part
#          return the reversed number if it in the range [-2^31, 2^31-1], return 0 otherwise
# Description: Firstly, store the sign into [s]
#              Second, reverse numeric part
#              Third, combine sign and numeric part by [s]*[r]
#                     and check overflow compare [r] to 2^^31
# Time complexity: O(n) string slicing from end to beginning, [::-1]
def reverse(x):
    s = -1 if x<0 else 1                # compare input with 0, then store sign into [s]
    r = int(str(s*x)[::-1])             # reverse numeric part
    return s*r * (r < 2**31)            # combine sign and numeric part, and check overflow

# 4.9 Palindrome Number ========================================= URL: https://leetcode.com/problems/palindrome-number/
# Problem: Check if an integer is palindrome without converting it to string, 
#          a negative integer is not palindrome, since its first symbol '-' doesn't match with last numeric symbol
# Description: Construct and reversed integer [res], keep grab last digit of x by x%10, and add it into [res].
#              Then swift up [res] by a digit, by [res] *= 10.
#              Doing this until [x] <= [res], meaning [x] has same digits as [res] or [x] has one digit less than [res]
#              If x has even digits, then at end [x] should == [res], if [x] has odd digits then [x] should == [res]//10  
#              Corner case, if [x] is less than 0, return False. Or [x] is not zero but last digit is 0, since [x] can't 
#              have leading, return False
def isPalindrome(x):
    if x<0 or (x!=0 and x%10==0):       # [x] less than 0, or [x]
        return False
    reverse = 0
    while reverse<x:                    # extract number until [reverse] is equal or one digit longer than [x]
        reverse = reverse*10+x%10               # add last digit of [x] to [reverse]
        x//=10                                  # remove last digit from [x]
    return x==reverse or reverse//10==x     # [reverse] can have either same digits as [x] or one more digit than [x]

# 5.13 Roman to integer =================== URL: https://leetcode.com/problems/roman-to-integer/
# Problem: Convert Roman numeral of type string to integer number of base 10, 
#          Symbol       Value
#           I             1
#           V             5
#           X             10
#           L             50
#           C             100
#           D             500
#           M             1000
#        I can be placed before V (5) and X (10) to make 4 and 9. 
#        X can be placed before L (50) and C (100) to make 40 and 90. 
#        C can be placed before D (500) and M (1000) to make 400 and 900.
# Description: Build a dictionary which map from roman numeral to integer number,
#              traverse each character of input string, and map from char to int.
#              If previous char is smaller, such as "IX" where 'I' is less than 'X'
#              do subtraction, where "IX" = -1 + 10 = 9
#              else do addition, such as "XI" = 10 + 1 = 11
# Time complexity: O(n) simply traverse the input string
def romanToInt(s):
    dic = {'I':1, 'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}     # maintain lookup table 
    res = 0
    for i in range(len(s)-1):       # iterate from beginning to second last char
        if dic[s[i]] >= dic[s[i+1]]:            # if a larger char comes before a small char, addition
            res += dic[s[i]]
        else:                                   # if a smaller char comes before a large char, subtract
            res -= dic[s[i]]
    return res+dic[s[-1]]                   # always add the last char to [res]

# 6.14 Longest Common Prefix =================== URL: https://leetcode.com/problems/longest-common-prefix/
# Problem: Given a list of strings, find the longest common prefix of them and return,
#          return empty string if there is no common prefix.
#          Such as, ["flower", "flow", "flight"] = 'fl
# Description: Function: "zip()" function can take zero or more iterator object. 
#              when no arguemnt is passed, it returns an empty iterator object
#              when a single iterator is passed, it returns an iterator of tuples with each tuple has one element
#              Ex: list( zip([1,2,3]) ) => ((1), (2), (3))
#              when multiple iterator is passed, it returns an iterator of tuples with each tuple has element from all iterators
#              Ex: list( zip([1,2,3], ["a", "b", "c"]) ) => ((1, "a"), (2, "b"), (3, "c"))
#              Syntax: "*" sign will "unpack" sequence/collection object.
#              Ex: def sum(a, b):
#                       return a+b
#                   values = [1, 2]
#                   sum( *values )  => 3, [values] is "destructed"
#              Convert each element in the unzipped list into a set, Use property of set where duplicated 
#              elements in a list are regarded as a single element, Ex: set(('a','b','a')) = {'a', 'b'}  
#              if elements in a list are all the same, then set(list) has length of 1
# Time Complexity: O(l) => l is the length of common prefix, since zip(*) and enumerate() take O(1),
#                 and for loop stops when an un-match is detected
def longestCommonPrefix(strs):
    unzipped = list(zip(*strs))             # unpack [strs] that char at same index are grouped together
    res = ""
    for el in unzipped:                     # iterate though each grouped chars
        if len(set(el)) == 1:                   # grouped chars are same letter, add it to [res]
            res += el[0]
        else:                               # found a group has multiple letters, longest common prefix ends here
            break
    return res


# 7.20 Valid parentheses ================== URL: https://leetcode.com/problems/valid-parentheses/
# Problem: Given a string of parentheses including () [] {}, return True if all parentheses are valid
#          Otherwise return False
# Description: Parentheses opened first is closed last, that we use "stack" to track its pair parenthesis
#              Create a [lookup] dictionary where closing parentheses are keys and openging parenthese are values, 
#              and create an empty [stack]. Iterate through string, if current character [c] is a opening parentheis
#              that "c not in lookup.keys()", append the opening parenthisis to [stack]. Elsewise, "c in lookup.keys()" 
#              that c is a  closing parenthesis, check if stack[-1] is same as [c]'s opening pair. If stack[-1] is
#              equals to lookup[c], then pair matches and pop stack, otherwise return False since opening and 
#              closing doesn't match. At the end of loop, return True if [stack] is empty, otherwise return False
# Time Complexity: O(n)
def isValid(s: str) -> bool:
    lookup = {')':'(', ']':'[', '}':'{'}            # maintain a loopup table 
    stack = []
    for c in s:
        if c not in lookup.keys():                  # encounter a opening parenthesis, add it to stack
            stack.append(c) 
        else:                                       # encounter a closing parenthesis, verify its pair in stack
            if stack and stack[-1] == lookup[c]:
                stack.pop()
            else:
                return False
    return len(stack) == 0

# 8.21 Merge Two sorted lists ================================ URL: https://leetcode.com/problems/merge-two-sorted-lists/
# Problem: Given two linked lists of numbers, both of them are sorted in ascending order,
#          Return a combined linked list with elements in both linked lists, where elements are ordered in ascending order
# Description: Recursive solution, if l1.val is smaller, call recursive function on l1.next return l1 later on
#              if l2.val is smaller, call recursive function on l2.next and return l2 later on.
#              When one of the lists is empty(None), return the other list, and the rest nodes of that list are returned automatically,
#              since the rest nodes are already connected.
# Time Complexity: O(min(n,m)) => n, m are number of nodes in each linked list
def mergeTwoLists(l1, l2):
    if not l1 or not l2:            # return the next node of a list if the other list is None, 
        return l1 or l2             # the next node will contain the rest node since they already connected with each other
    if l1.val < l2.val:
        l1.next = mergeTwoLists(l1.next, l2)    # recursively append smaller node
        return l1
    else:
        l2.next = mergeTwoLists(l1, l2.next)    # recursively append smaller node
        return l2
        
# Description: Iteratively solution, iterate through both linked list concurrently, and compare value of nodes from different linked list,
#              append the smaller node to [cur] node, keep doing until one of the linked list is empty, then append the rest nodes to [cur]
# Time Complexity: O(min(n,m)) => n, m are number of nodes in each linked list
def mergeTwoLists1(l1, l2):
    dummy = cur = Node(0)
    while l1 and l2:            # keep compare value of nodes from both linked list and append to [cur] node, until one of the lists is empty
        if l1.val < l2.val:
            cur.next = l1       
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2         # one of the linked list is empty, append the remaining nodes from the other linked list
    return dummy.next

# 9.884 Uncommon words from two sentence ==================== URL: https://leetcode.com/problems/uncommon-words-from-two-sentences/
# Problem: Given two sentences A and B, which only consists of lower case characters and space,
#          A word is uncommon if it only appears once in both sentences, return the collection of this kind of words
#          ex: "this is a great day" "that was a great day" => return ["this", "that", "is", "was"]
# Description: The uncommon words are the words that only appear once in both sentences, then all I need to do is
#              combine two sentences and count occurrence of each word, if count of a word is one, then this word is uncommon
#              Counter() return a dictionary, where [word] is key and [occurrence] is value 
from collections import Counter
def uncommonFromSentences(A, B):
    cnt = Counter((A + " " + B).split())        # combine two sentences and split each word and count their occurrence  
    uncommon = []                               # the cnt is a dictionary where key is [word] and value is [occurrence]
    for word in cnt:
        if cnt[word] == 1:                      # if occurrence is one then this word is uncommon
            uncommon.append(word)
    return uncommon

# 10.26 Remove duplicates from sorted array ===================== URL: https://leetcode.com/problems/remove-duplicates-from-sorted-array/
# Problem: Given a sorted array, remove duplicates in-place such that each element only appear once(space complexity is O(1)),
#          return the length of array which has no duplicates.
# Description: Two pointers. Maintain two pointers [slow] and [fast], that [slow] track the tail index of non-duplicated list, and [fast]
#              look for non-duplicates element to put at the end of non-duplicated list. Both [slow] and [fast] start at the beginning of 
#              [nums], [fast] will skip numbers that are smaller or equals to [slow]. Once [fast] finds a number that is greater than 
#              [slow], meaning a non-duplicated number is found swap [slow+1] and [fast], and increase [slow] by 1 to shift tail to right. 
#              Loop until [fast] hit the end, return [slow+1] as the length of non-duplicated list
# Time Complexity: O(n)
def removeDuplicates(nums: List[int]) -> int:
    slow, fast = 0, 0
    while fast<len(nums):
        if nums[fast] <= nums[slow]:                            # duplicated number are smaller or equals to [slow], skip them
            fast += 1
        else:                                                   # if [fast]>[slow], then [fast] is a non-duplicated number, swap
            slow = slow+1                                           # shift [slow] to next index for inserting new number
            nums[fast], nums[slow] = nums[slow], nums[fast]
    return slow+1

# 11.27 Remove elements =========================== URL: https://leetcode.com/problems/remove-element/
# Problem: Given an array [arr] and a value [val], remove [val] in-place(space complexity O(1)) and return a new length of array
#          without instance of [val], also move all [val] in array to the end of array, meaning arr[:length] are elements that != val
#          and arr[length:] are elements that == val
# Description: Use two pointer one [start] starts from the beginning and another [end] starts from the end. If [start] == [val] then swap [tail] and [end]
#              also make [end] a step forward and [start] stay at same index, even if [end] == [val] swap make no change on array, and [end] is decreased by 1
#              and keep looking for an element that [end] != [val]. If [start] != [val], [start] should be kept at the beginning, so [start] is increased by 1
#              keep looking for an element that [start] == [val] 
# Time Complexity: O(n) worse case is elements in array are all == [val] or are all != [val], then either [start] or [end] will iterate through entire array
def removeElement(nums, val):
    start, end = 0, len(nums) - 1
    while start <= end:
        if nums[start] == val:
            nums[start], nums[end] = nums[end], nums[start]
            end -= 1
        else:
            start +=1
    return start

# 12.28 Implement strStr() =============================================================== URL: https://leetcode.com/problems/implement-strstr/
# Problem: Find the first occurrence index of [needle] in [haystack]. Return -1 if [needle] is not found, return 0 if [needle] is empty.
# Description: Iterate through [haystack] from index 0 to index len(haystack)-len(needle), use string slice and compare substring of [haystack] with
#              [needle]. If [needle] is found => return index where [needle] is found in [haystack]
# Time Complexity: O(n*k) => n is len(haystack), k is len(needle). In worse case, it iterate through entire string, and in each iteration it takes
#                  O(k) to slice the string
def strStr(haystack: str, needle: str) -> int:
    if needle == "":
        return 0
    for i in range(len(haystack) - len(needle)+1):
        if haystack[i:i+len(needle)] == needle:
            return i
    return -1

# 13.35 Search insert position ================= URL: https://leetcode.com/problems/search-insert-position/
# Problem: Given a sorted list of integers and another integer, insert the integer into the right position in the list,
#          and return the index where the integer is inserted
# Description: Binary search. Maintain [left] and [right] starting from 0, and len(nums) respectively. Take [mid] of [left] 
#              and [right], compare nums[mid] with [target]. If equal, return mid. If [mid]>[target], move [right] to [mid]. 
#              If [mid]<[target], move [left] to [mid+1]. Loop when left<right, if [target] doesn't exists in [nums], that
#              no [mid] will be returned in the loop. Thus return [left] outside the loop as its inserting index
# Time complexity: O(logn)
def searchInsert(nums: List[int], target: int) -> int: 
    left, right = 0, len(nums)
    while left<right:
        mid = (left+right)//2
        if nums[mid]<target:
            left = mid+1
        elif nums[mid]>target:
            right = mid
        else:
            return mid
    return left

# 14.38 Count and say ============================== URL: https://leetcode.com/problems/count-and-say/
# Problem: Let the first string be "1" and the next string will "count and say" this string, where "1" is "one 1" => 11
#          and "11" is "two 1s" => 21, so on so forth. The function takes an integer [n] and output the nth string
# Description: Maintain current character [cur], the count of current character [cnt], and the current constructing 
#              count-and-say string [temp]. [cur] initially is first char of [res], [cnt] is zero and [temp] is empty 
#              string. Iterate through current count-and-say string [res], if the char [c] is equals to [cur], increase
#              [cnt] by 1. If [c] is different than [cur], append [cnt] and [cur] to [temp], as consecutive [cur]s are
#              over. Update [cur] to be new [c] and reset [cnt] to 1. After iterating each count-and-say string, always
#              apppend [cnt] and [cur] to [temp] as the last consecutive [c]s need to be recorded. Do this loop "n-1" times
# Time Complexity: O(n^n)? outer loop O(n), and inner loop iterates through each character of previously generated string.
#                  Relationship between n and len(str) is unknown
def countAndSay(n: int) -> str:
    res = "1"                           # [res] initially is "1"
    for _ in range(n-1):                #Iterate n-1 times, since [res] is "1" initially and return "1" directly when n==1
        cur, cnt, temp = res[0], 0, ""
        for c in res:
            if c == cur:                    # current [c] is same as [cur], increase [cnt]
                cnt += 1
            else:                           # [c] is different from [cur], record current [cnt] and [cur], and reset them
                temp += str(cnt)+cur
                cur, cnt = c, 1
        res = temp+str(cnt)+c               # always record last [cur] and [cnt]
    return res

# Description: Useing intertools.groupby() which receive an iterable object, returns a group object of cluster that has same value, 
#              and a the value that the cluster shares. ex: 11335 => [1,1] 1 / [3,3] 3 / [5] 5             
from itertools import groupby
def countAndSay2(n):
    s = '1'
    for _ in range(n - 1):
        comb = ""
        for digit, group in groupby(s):           # itertools.groupby(s) returns: 1) a [group] object (can convert to list) of continuous elements of same kind 
            comb += str(len(list(group)))+str(digit)        #                               2) a single value which indicates the value that the group shares
        s = comb    
    return s

# 15.53 Maximun sum of subarray ======================================= URL: https://leetcode.com/problems/maximum-subarray/
# Problem: Given an integer array, find a contiguous subarray that sum of elements in the subarray has the maximun sum in any possible sub-arrays,
#          Return the maximum sum
# Description: Kadane's Algorithm, compute sum of sub-arrays that end at index i => [temp]. if [temp] < 0, then reset it to zero,
#              because [temp]<0 can only bring down the sum, so get rid of this sub-array and start new sub-array at i+1 index.
#              if [temp] > 0, then it can contributes to next sum, then try adding next element to it. Compare [res] and [temp] 
#              to track the largest sum in [res]
# Time Complexity: O(n)
def maxSubArray(nums: List[int]) -> int:
    res, temp = float("-inf"), 0
    for i in range(len(nums)):
        temp += nums[i]
        res = max(temp, res)        # update [res] if [temp] is larger
        if temp<0:                  # reset [temp] if it is negative
            temp = 0
    return res

# 16.58 Length of last word ============================== URL: https://leetcode.com/problems/length-of-last-word/
# Problem: Given a string of words, where each word is separated by space(s). Return the length of the last word.
#          Note: It is possible that spaces is located at the beginning or end of the string
# Description: Use str.rstrip() to get rid of trailing spaces, then use str.split(" ") to split string into list of words.
#              Return the len() of last element.
# Time Complexity: O(n)
def lengthOfLastWord(s):
    if len(s) == 0:             # corner case empty string
        return 0
    l = s.rstrip().split(" ")    # get rid of prepend/append spaces, and split the string by spaces
    return len(l[-1])

# 17.66 Plus one =========================================== URL: https://leetcode.com/problems/plus-one/
# Problem: Given a list of integer, which the entire list represent a number and each element represent a digit of the number. 
#          index 0 is the most significant digit and index -1 is least significant. Add 1 to the least significant digit and return
#          the sum as a list with the same properties described above.
# Description: Maintain a [carry], and iterate through [digits] from end to beginning. If there is a [carry] then add [carry] to 
#              the digit. If [carry] is zero, stop iteration since no need to modify the rest of the digits. At the end of loop,
#              if [carry] is not zero, then insert "1" before modified [digits]. Otherwise, return [digits]
# Time complexity: O(n)
def plusOne(digits: List[int]) -> List[int]:
    carry, digits[-1] = (digits[-1]+1)//10, (digits[-1]+1)%10               # add 1 to lowest digits
    for i in reversed(range(len(digits)-1)):                                # iterate [digits] from end to beginning
        if carry:                                                               # add [carry] to next digits if it is not zero
            carry, digits[i] = (digits[i]+carry)//10, (digits[i]+carry)%10  
        else:                                                                   # if [carry] is zero, stop loop
            break
    if carry:                               # highest digit produce a [carry], insert it to the beginning of [digits]
        return [1]+digits
    else:
        return digits

# 18.67 Add Binary =========================================== URL: https://leetcode.com/problems/add-binary/
# Problem: Given two binary number of string type [a] and [b], return their sum of binary string
#          Note: both parameters are non-empty and only contains '0' or '1'
# Description: Check length of [a] and [b] prefix "0" to the shorter one. M aintain an integer [carry], 
#              and [res] as the result string. Iterate [a] and [b] from end to beginning. For index i,
#              convert a[i] and b[i] to integer to do additiom, [carry]=(a[i]+b[i])//2, and [res] is 
#              appended with (a[i]+b[i])%2. Return the reversed [res] at end of loop, and prefix "1" to
#              [res] if [carry] is not equal to zero
# Time Complexity: O(n)
def addBinary(a: str, b: str) -> str:
    # prefix the shorter one with "0"
    if len(a)>len(b):
        b = "0"*(len(a)-len(b))+b
    else:
        a = "0"*(len(b)-len(a))+a
    carry, res = 0, ""
    for i in reversed(range(len(a))):       # iterate [a] and [b] from end to beginning
        s = int(a[i])+int(b[i])+carry           # get the sum of a[i] and b[i] 
        res, carry = res+str(s%2), s//2         # append sum%2 to [res], update [carry]=sum//2
    if carry:
        return "1"+res[::-1]
    else:
        return res[::-1]

# 19.69 Sqrt(x) =========================================== URL: https://leetcode.com/problems/sqrtx/
# Problem: Implement function "int sqrt(int x)", it takes an integer [x] as input and return integer part of square root of [x]
# Description: Newton's method of square root approximation. 
#              The equation to find the square root of [num] is x^2 = num, it can written as x^2 - num = 0, which is a polynomial equation.
#              Starting from guess = num, keep finding the tangent line at x=[guess], and find the intersection of tangent line and x-axis, 
#              make the x value of intersection to be the next [guess]. Use equation y=kx+b with k=2x and (x,y)=(i, i^2-num) to find the 
#              equation of tangent line, which should be "y=2*(guess)*x-(guess)^2-num". Make y=0 to equaltion that "y=2*(guess)*x-(guess)^2-num" 
#              and find the intersection, x=guess^2+num)/(2*guess). The x value is [newGuess], check the difference between [guess] and [newGuess]
#              If they are very close(diff<0.01), stop the loop and return [guess]
def mySqrt(x: int) -> int:
    if x == 0:                                  # corner case x=0 will lead to divid by zero
        return 0
    guess, diff = x, float('inf')               # [guess] starts fromg [num]
    while diff > 0.01:                              # stop the loop if different between [guess] and [newGuess] is small enough
        newGuess = (guess*guess+x)/(2*guess)        # Use tangent line to find [newGuess]
        diff = abs(newGuess - guess)
        guess = newGuess
    return int(guess)

# 20.70 Climbing stairs ============================================= URL: https://leetcode.com/problems/climbing-stairs/
# Problem: Climb stairs with n step, each climb can be 1 step or 2 steps, output how many ways that you can climb to the top
# Description: Fibonacci Approach. In order to climb n steps, there are two general ways: 1) climb n-2 steps and climb 2 steps at once.
#              or 2) climb n-1 step and climb one more step. Then the total ways to climb n steps is [n-2]+[n-1], same as fibonacci number
# Time Complexity: O(n)
def climbStairs(n: int) -> int:
    dp1, dp2, dp = 0, 1, 1
    for _ in range(1,n+1):
        dp = dp1 + dp2                  # climb 2 steps from n-2 or climb 1 step from n-1 to reach step n
        dp1, dp2 = dp2, dp              # update value for step n-2 and n-1
    return dp
    
# Description: Recursive Memory Approach. Initall a list to memorize number of way in case i where i<n, recursively calculate M[n]
#              if M[i] is not stored in list then calculate M[n] = M[n-2]+M[n-1], if M[i] is stored in list then directly return M[i]
def climbStairs2(n):
    M = [-1 for _ in range(n+1)]
    if n<=1: return 1           # corner case n<=1 
    else:
        M[1], M[2] = 1, 2       # base case
        return helper(n, M)

def helper(n, M):
    # M[i] presented in list
    if M[n] != -1:
        return M[n]
    # M[i] not presented in list
    else:
        M[n] = helper(n-1,M)+helper(n-2, M)
        return M[n]

# 21.83 Remove duplicates from sorted linked list ======================= URL:https://leetcode.com/problems/remove-duplicates-from-sorted-list/
# Problem: Given a head of sorted linked list with/without duplicates, value of node is [node.val], next node is [node.next]. 
#          Need return a head linked list where duplicated nodes are removed.
# Description: Declare a pointer [cur]. Starting from the head of list iterate whole list, keep compare [cur] and [cur].next, if they have same value, 
#              then skip [cur].next by letting [cur].next = [cur].next.next          
# Time Complexity: O(n)
# Note: Node class is defined in Problem 2
def deleteDuplicates(head):
    cur = head
    while cur:
        while cur.next and cur.next.val == cur.val:
            cur.next = cur.next.next     # skip duplicated node
        cur = cur.next     # not duplicate of current node, move to next node
    return head

# 22.88 Merge sorted array ============================================ URL: https://leetcode.com/problems/merge-sorted-array/
# Problem: Given two arrays [nums1], [nums2] and the number of elements [m], [n] respectively. The size of [nums1] is greater or equal to [m+n]
#          where nums1[m:n] are all zeros. Merge [nums2] into [nums1] in-place, and make [nums1] as a combined sorted array.
# Description: Iterate through both arrays from end to beginning. Keep adding largest element to the end of [nums1]
def merge(nums1, m, nums2, n):
    while m > 0 and n > 0:
        if nums1[m-1] >= nums2[n-1]:
            nums1[m+n-1] = nums1[m-1]
            m -= 1
        else:
            nums1[m+n-1] = nums2[n-1]
            n -= 1
    if n > 0:
        nums1[:n] = nums2[:n]

# 23.970 Powerful integers =============================================== URL: https://leetcode.com/problems/powerful-integers/
# Problem: Given two integers [x], [y] and a upper limit [bound]. Need to find all non-negative powers that satisfies
#          x**i + y**j <= bound, where [i],[j] are non-negative integers. Return a list the sums of possible [x**i + y**i]s
#          without duplicates
# Description: Lets say x**i+y**j <= bound, we need to find the maximum value of of i and j, call them [i_bound], [j_bound].
#              "math.log(bound, x) = i_bound" and "math.log(bound, y) = j_bound". Note, if [x]=1 or [y]=1, "math.log()" throws
#              error. So we need to let [i_bound] or [j_bound] equqls to 1, if [x] or [y] == 1. Then use nested loop to ierate 
#              through all possible combinations of [i_bound] and [j_bound]. If their [sum]<=[bound], store the [sum] is a "set",
#              use set to avoid duplication. After all, convert "set" to "list" and return
# Time Complexity: O(log (bound) of base min(x,y))
# Space Complexity: O(log(bound))
import math
def powerfulIntegers(x, y, bound):
    if bound<=0:                            # corner case, log(0) of any base is undefined
        return []
    res = set()
    i_bound = math.ceil(math.log(bound, x)) if x!=1 else 1      # if x or y == 1, make their bound == 1
    j_bound = math.ceil(math.log(bound, y)) if y!=1 else 1      # because there is only one possible value of 1**n == 1
    for i in range(i_bound):
        for j in range(j_bound):
            sum = x**i+y**j
            if sum<=bound:
                res.add(sum)
    return list(res)

# Description: Dynamic Programming. Find maximum value of i and j, denote as [i_bound] and [j_bound] where "i_bound = math.log(bound,x)"
#              and "j_bound = math.log(bound,y)". Maintain a list [dp] of "j_bound+1" element, each element is a tuple of two elements
#              which represents (x**i, y**j). The first element of [dp] is intially (x**0, y**0) = (1,1). Iterate through [dp], that
#              dp[j] = (dp[j-1][0], dp[j-1][1]*y). The next element is equal to previous element multiply [y] onto second tuple element 
#              dp[j-1][1]*y. Meanwhile, get the [sum] of dp[j][0] and dp[j][1]  if [sum] less than or equal to [bound], add [sum] to 
#              a set [res], which tracking results. 
#              After building the first [dp] where [i]=0, iterate [i] from 1 to [i_bound] to multiply [x], 
#              then dp[j] = (dp[j][0]*x, dp[j][1]). Meanwhile if [sum] of dp[j][0] and dp[j][1] is less than or equal to  bound, add 
#              [sum] to [res]. At the end, convet [res] to list and return
import math
def powerfulIntegers(x: int, y: int, bound: int) -> List[int]:
    if bound<2:
        return []
    res = set()
    i_bound = math.ceil(math.log(bound, x))+1 if x!=1 else 1        # find boundary of i and j
    j_bound = math.ceil(math.log(bound, y))+1 if y!=1 else 1
    dp = [(1,1) for _ in range(j_bound)]                            # create [dp] list
    if 2<=bound:
        res.add(2)
    for j in range(1, j_bound):                         # construct first row of [dp]
        dp[j] = (dp[j-1][0], dp[j-1][1]*y)
        s = dp[j][0]+dp[j][1]
        if s<=bound:
            res.add(s)
    for i in range(1, i_bound):                         # construct next row of [dp] from previous row
        for j in range(j_bound):
            dp[j] = (dp[j][0]*x, dp[j][1])
            s = dp[j][0]+dp[j][1]
            if s <= bound:
                res.add(s)
    return [el for el in res]

# 24.100 Same tree ======================================================================= URL: https://leetcode.com/problems/same-tree/
# Problem: Given roots of two binary trees(each node can has up to two children). Compare two trees, if trees have same structure and 
#          same values in corresponding positions. Return True if two trees are identical, False otherwise
# Description: Recursively iterate both trees at same path. If cur nodes have same value, then keep compare left and right children.
#              If both hit None(end of path) return True, if exactly one of them hit None, meaning not the same structure return False. 
#              If cur nodes have different value, return False.
# Time complexity: O(n)
def isSameTree(p: TreeNode, q: TreeNode) -> bool:
    if p and q:
        if p.val == q.val:          # current nodes are equal, continue comparing left and right children
            return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
    if not p and not q:             # both path hit end, return True
        return True
    return False            # doesn't fall into previous conditions, means an unmatch is found

# 25.101 Symmetric tree ============================================================= URL: https://leetcode.com/problems/symmetric-tree/
# Problem: Given a root of binary of TreeNode class, check if the tree is symmetric by a vertical line through middle, meaning
#          the tree is a mirror of itself. Return True, if it is symmetric, return False otherwise.
# Description: Define a helper recursive function, compare two symmetric sub-roots [left] and [right]. If they matches each other 
#              keep comparing their children, where [left.left] compares with [right.right] and [left.right] compares with [right.left]
#              to maintain symmetric comparison.
# Time Complexity: O(n) recursively bfs 
def isSymmetric(root):
    if root is None:
        return True                 # empty tree is symmetric
    else:
        return isSymmetric_recursive(root.left, root.right)

def isSymmetric_recursive(left, right):
    if left is None and right is None:      # traversed entire tree, and all matches
        return True
    if left is None or right is None:       # unbalanced tree, not symmetric
        return False
    if left.val != right.val:               # value doesn't match, not symmetric
        return False
    else:
        # recursively do symmetric comparison, (left.left <=> right.right) (left.right <=> right.left)
        return isSymmetric_recursive(left.left, right.right) and isSymmetric_recursive(left.right, right.left)

# 26.198 House robber ================================================================= URL: https://leetcode.com/problems/house-robber/
# Problem: You are a robber planning to rob along a street, the amount of money you can rob from each house along this street is
#          given in a list with non-negative integers. The alarm will ring if you robbed adjacent houses, return the maximun amount
#          of money you can rob without triggerring the alarm.
# Description: Dynamic programming, avoid robbing adjacent houses. Maintain [dp1], [dp2], and [dp], where [dp] is the maximum money robbed
#              at current index [i], [dp2] is the maximum money at previous index [i-1], and [dp1] is the maximum money at index [i-2]. 
#              Initially, [dp1] = nums[0], [dp2] = max(nums[0], nums[1]) because we can only choose from nums[0] and nums[1] at index 1,
#              so the larger one is picked, and [dp] don't have initial value. 
#              For index [i], [dp] = max(dp1+nums[i], dp2). We can rob current index and money add to [dp1] or we don't rob index [i], 
#              then the money robbed is equal to [dp2].
# Time Complexity: O(n)
def rob(nums: List[int]) -> int:
    if len(nums)<=2:                # if [nums] has less than 2 numbers, take the larger one
        return max(nums)
    dp1, dp2, dp = nums[0], max(nums[0],nums[1]), 0     # intial value of [dp1] [dp2]
    for i in range(2, len(nums)):       
        dp = max(dp1+nums[i], dp2)              # either rob current [i] + dp1, or don't rob [i], which is dp2
        dp1, dp2 = dp2, dp
    return dp

# 27.205 Isomorphic string ================================================= URL: https://leetcode.com/problems/isomorphic-strings/
# Problem: Given two strings check if they are isomorphic to each other. Two strings are Isomorphic, when characters one string can
#          be mapped one to one to characters in another string. ex: 'paper', 'title'  {p:t, a:i, e:l, r:e}
# Description: Using hash map. Maintain two disctionary [s2t] and [t2s], where [s2t] use character of [s] as "key" and character of
#              [t] as "value". [t2s] is opposite, use character of [t] as "key", and character of [s] as "value". Iterate through 
#              [s] and [t] at same time and populate [s2t] and [t2s]. Also compare the mapping between [s2t] and [t2s], return false 
#              if any dismatch. And return True at the end of loop 
def isIsomorphic(s: str, t: str) -> bool:
    s2t, t2s = {}, {}
    for i in range(len(s)):
        if s[i] in s2t and s2t[s[i]] != t[i]:       # s[i] exists in [s2t] but dismatch with t[i]
            return False
        if t[i] in t2s and t2s[t[i]] != s[i]:       # t[i] exists in [t2s] but dismatch with s[i]
            return False
        s2t[s[i]] = t[i]                # update [s2t] and [t2s] with current s[i] and t[i]
        t2s[t[i]] = s[i]
    return True
# Description: Using zip() & set(). Compare size of set(zip(s,t)), set(s) and set(t). If [s] has a char maps to multiple chars in [t],
#              then set(s) has smaller size. If [t] has a char maps from multiple chars in [s], then set(t) has smaller size. And each char
#              in [s] and [t] should have a map relationship, then should have set(zip(s,t)) == set(t) == set(s)
# Time Complexity: O(n), zip() and set() both take O(n)
def isIsomorphic2(s, t):
    return len(set(zip(s, t))) == len(set(s)) == len(set(t))

# 28.104 Maximum depth of binary tree ============================================== URL: https://leetcode.com/problems/maximum-depth-of-binary-tree/
# Problem: Given a root of a binary tree, return the maximum depth of the tree, where depth is number of nodes along the longest path.
# Description: Recursively trace down the root. Calling the function on current node(initially root), compare the returning depth of left and right child,
#              take the bigger one and add one to it, as the maximum depth from bottom to current node.
# Time Complexity: O(n)
def maxDepth(root):
    if not root: return 0                        # base case: end of branch return 0
    return 1+max(maxDepth(root.left), maxDepth(root.right))  # keep compare the maximum depth of children and add 1 as the depth of current node bottom-up

# 29.107 Binary tree level order traversal II =================================== URL: https://leetcode.com/problems/binary-tree-level-order-traversal-ii/
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

# 30.108 Convert sorted array to binary search tree ====================== URL: https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
# Problem: Given a sorted list of integers, convert it to a height-balanced binary search tree, where the depths of leaves can differ by at most 1,
#          left child < parent < right child. Return the root of the result tree.
# Description: Grab the middle element of list as the root, and divide the list from middle into two sub-lists. Recursively finding the middle, and the
#              middle of left sub-list become left child and right the middle of right sub-list is the right child.
# Time Complexity: O(n)
def sortedArrayToBST(nums: List[int]) -> TreeNode:
    if not nums:
        return None             # No more number left in sub-list, return None to indicate end of sub-tree
    mid = (len(nums)-1) // 2        # find middle number
    root = TreeNode(nums[mid])      # make middle number the parent
    root.left = sortedArrayToBST(nums[:mid])        # recursively get left child from left sub-list
    root.right = sortedArrayToBST(nums[mid+1:])     # recursively get right child from right sub-list
    return root

# 31.46 Permutation ================================================= URL: https://leetcode.com/problems/permutations/
# Problem: Given a list of distinct integers, return all possible permutations as 2D list, where each permutation is a sub-list.
# Description: Backtracking, backtracing function takes three parameters. 1) [nums] originall number list 2) [temp] tracking current
#              permuted elements 3) [pool] un-permuted elements in nums. In each recursive call, iterate through [pool] list, and 
#              adding elements into [temp], when current level of recursion is done, backtrack to upper level and continue iteration
# Time Complexity: O(n!)
def permute(nums):
    res = []
    backtrack(nums, [], res)
    return res

def backtrack(pool, temp, res):
    if not pool:            # base case: no elements left in [pool], append current permute into result
        res.append(temp)        
    # backtrack: take element out of [pool] and add into [temp], 
    # when a iterations is done in certain level -> backtrack to upper level, and continue upper level iteration
    for i in range(len(pool)):
        backtrack(pool[:i]+pool[i+1:], temp+[pool[i]], res)   

# 32.78 Subsets =========================================================== URL: https://leetcode.com/problems/subsets/
# Problem: Given a list of distinct integers, return a list of all subsets of the list(power set), 
#          where each subset if a sub-list in result list
# Description: Backtracking. backtrack function has three parameters: 1) remaining sub-list of input list 
#              2) [temp] current tracking subset 3) [result] list holding subsets. Each recursive call would an element into [temp],
#              append [temp] into [result], then forward track into next level by calling recursive function with sliced [nums] and
#              updated [temp]. When for-loop in a recursive call is over, backtrack to previous recursive call and continue for-loop
# Time Complexity: O(2^n)
def powerSet(nums):
    res = []
    recur(nums,[],res)           # first element is always an empty set
    return res

def recur(nums, temp, result):
    result.append(temp)             # each recursive call, append [temp] of previous call
    for i in range(len(nums)):          # backtrack
        recur(nums[i+1:], temp+[nums[i]], result)

# 33.110 Balanced binary tree ================================================ URL: https://leetcode.com/problems/balanced-binary-tree/
# Problem: Given root of a binary tree, check if the tree is balanced, meaning depth of leaves not differ by more than 1. 
#          Return True when balanced, return False otherwise
# Description: Recursively calculate depth of branches, and compare depth of left and right after each depth calculation. If differs more than 1
#              return -1, return max(depth)+1 if differs less than or equal to 1. Also before the comparsion, check if the return value of previous
#              recursion is -1. If any return value is -1, means there exists unbalanced branch and should return -1 immediately.
# Time complexity: O(n) 
def isBalanced(root):
    return isBalanced_helper(root) != -1          # if balanced, recur return depth, if not balanced, recur return -1

def isBalanced_helper(root):
    if not root:
        return 0
    left = isBalanced_helper(root.left)           # get return value of previous recursion
    right = isBalanced_helper(root.right)         # get return value of previous recursion
    # check previous value, if any -1 means already exists unbalanced, also check if any unbalanced in this level
    if left == -1 or right == -1 or abs(left-right) > 1:        
        return -1
    return 1+max(right,left)                # return depth if is balanced

# 34.111 Minimum depth of binary tree ============================================== URL: https://leetcode.com/problems/minimum-depth-of-binary-tree/
# Problem: Given the root of binary tree, find depth of leaf that is closest to root. Return the depth.
# Description: DFS. If a node have both left and right children, recursively call function on both children and take min(left, right)+1 as the depth
#              If a node have one child, we only consider the path that has child node. Recursively call function on both children and take 
#              max(left, right)+1 as the depth. If a node has no child, it is a leaf and return 1
# Time complexity: O(n)
def minDepth(root: TreeNode) -> int:
    if root == None:            # reach end of branch
        return 0
    if root.left==None or root.right==None:                     # node has one or no child, one of them or both of recursice function will return zero
        return max(minDepth(root.left), minDepth(root.right))+1
    return min(minDepth(root.right), minDepth(root.left))+1     # node has both children  

# 35.112 Path sum ================================================================== URL: https://leetcode.com/problems/path-sum/
# Problem: Given a [root] of binary tree and an number [sum], check if there exists a path from root-to-leaf, that the sum of nodes on the path is equal to [sum].
#          If such path exists return True, else return False. note: leaf is a node has no child  
# Description: DFS. Recursively call function on left and right child of a node and decrease the target [sum] by the value of current node.
#              Base case is when a node is leaf, meaning it doesn't have node.left and node.right. Then if [sum] == node.val, then the path is found return True.
#              If root == None, means it is at the end of branch and [sum] is not satisfied, return False.
#              Call function on both children hasPathSum(left, sum-node.val) and hasPathSum(right,sum-node.val), if any of them return True. Whole function
#              should return True
def hasPathSum(root: TreeNode, targetSum: int) -> bool:
    if not root: return False               # reach end of branch, and [sum] is not satisfied
    if not root.right and not root.left and sum==root.val:      # reach leaf node and [sum] is satisfied
        return True
    else:
        return hasPathSum(root.left, sum-root.val) or hasPathSum(root.right,sum-root.val)     # recursively call function on left and right child

# 36.118 Yanghui's triangle ========================================================== URL: https://leetcode.com/problems/pascals-triangle/
# Problem: Given integer number [numRows] represents number of rows. Return a 2d list of entire Yanghui's triangle from row 1 to row [numRows], where each 
#          sub-list represent each row.
# Description: Using property of Yanghui's triangle, a number of next row is the sum of two adjacent numbers in previous row,
#                  1 3 3 1   =>    1 3 3 1 0
#                 /\/\/\/\   =>  + 0 1 3 3 1
#                1 4 6 4 1       = 1 4 6 4 1   
#              Therefore, append and prepend [0] onto previous row and add them together will give the next row
def generate(numRows):
    res = [[1]]
    for _ in range(1, numRows):
        left, right = res[-1]+[0], [0]+res[-1]      # [left] is previous row append [0], [right] is previous row prepend [0]
        for i in range(len(left)):                # sum up corresponding elements of [left] and [right] to get next row
            left[i]+=right[i]
        res.append(left)
    return res                                                  

# 37.119 Yanghui's triangle II ============================================================== URL: https://leetcode.com/problems/pascals-triangle-ii/
# Problem: Given a row index [rowIndex] of Yonghui's triangle, return entire row of [rowIndex] as the type of list. Note: the first row is index 0
# Description: Using property of Yanghui's triangle, number at Nth row index I is equal to N choose I.
#              for instance: N=3 [3 choose 0, 3 choose 1, 3 choose 2, 3 choose 3] = [1, 3, 3, 1]. And n choose k = n!/(k!*(n-k)!)
import math
def getRow(rowIndex: int) -> List[int]:
    def nCk(k, n):                              # "n Choose k" helper function
        return math.factorial(n)//(math.factorial(k)*math.factorial(n-k))
    return [nCk(i, rowIndex) for i in range(rowIndex+1)]

# 38.121 Best time to buy and sell stock ==================================== URL: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
# Problem: Given a list of number representing a stock price in several continuous days, assume in these day a person can trade once(buy on one day, and sell
#          on a later day). Find and return the maximum profit this person can make in those days.
# Description: Kadane's Alg. Calculate the profit of buying on previous day and selling on next day, Starting with buying stock on day 1 (index i) and sell
#              it on the next day (index i+1). The profit earned is "localMax=prices[i+1]-prices[i]", compare [localMax] with [maxProfit] to update [maxProfit].
#              Iterating through [prices], if [localMax]<0, then reset [localMax] to zero, since sell on current day is not profitable and we should look for 
#              a future day to buy. Keep tracking [maxProfit] along the iteration, return it at the end
# Time Complexity: O(n)
def maxProfit(prices: List[int]) -> int:
    maxProfit = localMax = 0
    for i in range(len(prices)-1):
        localMax += prices[i+1]-prices[i]       # calculat profit made by buying on [i] and sell on [i+1]
        if localMax <= 0:                   # sell on day [i+1] is not profitable, reset [localMax]
            localMax = 0
        else:
            maxProfit = max(maxProfit, localMax)
    return maxProfit

# 39.122 Best Time to Buy and Sell Stock II ========================================= URL: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
# Problem: Given a list of numbers [prices] represent stock price on each day, you are allowed to make as many transactions as you want, try to find maximum
#          profit you can make during those days. You can only make one operation(buy or sell) on a day, and must buy a stock before selling it.
# Description: Iterate through [prices], find a smallest of a consective decreasing to [buy] stock, which is the last element of that consective decreasing 
#              array, then find a largest of a consective increasing to [sell] stock, which is the last element of that consective increasing. Add their
#              differenct [sell-buy] into [profit]. 
#              [buy] and [sell] always come in pairs, If a [buy] is found, then [sell] must exists either at the end of consective increasing or at the end 
#              of array. If [buy] is not found, then iteration will hit the end of array, where [buy] and [sell] will have same value which is prices[-1] 
# Time compleixty: O(N)
def maxProfit(prices):
    buy = sell = profit = 0
    i = 0
    while i<len(prices)-1:
        # find localMin, which is the last element of continuous decreasing
        while i<len(prices)-1 and prices[i]>=prices[i+1]:           # greater and equal sign is needed, otherwise infinit loop when [i] == [i+1]
            i+=1
        buy = prices[i]
        # find localMax, which is the last element of continuous increasing
        while i<len(prices)-1 and prices[i]<prices[i+1]:        
            i+=1
        sell = prices[i]
        profit += sell-buy                # caluclate profit of current [buy]&[sell] 
    return profit

# 40.125 Valid palindrome ======================================================== URL: https://leetcode.com/problems/valid-palindrome/
# Problem: Given a string of sentence including punctuation, alpha letters, numbers and space, check if this string is palindrome when only consider
#          alpha letters and numbers. Return True if it is palindrome, and return False otherwise.
# Description: Two pointers. Maintain two pointers [l] and [r] start from left and right of [s]. Use ".isalnum()" to skip char that is
#              not a alphabet or number. When both s[l] and s[r] are alphabet or number, compare them and return False if they are different,
#              move [l] and [r] towards each other after each comparison. Return True when [l] and [r] meet, since every character in [s] are checked
# Time Complexity: O(n)
def isPalindrome_2(s: str) -> bool:
    l, r = 0, len(s)-1
    while l < r:
        while l < r and not s[l].isalnum():     # find next letter/number for [left]
            l += 1
        while l <r and not s[r].isalnum():      # find next letter/number for [right]
            r -= 1
        if s[l].lower() != s[r].lower():        # compare lowercase of [left] and [right]
            return False
        l +=1
        r -= 1
    return True

# 41.136 Single number ========================================== URL: https://leetcode.com/problems/single-number/
# Problem: Given a list of integers, every element in the list appears in pairs except one of them is unique. Find and return the unique element in 
#          linear run-time and using constant memory space
# Description: Using property of XOR, where if a number [a] is xor by another number [b] twice, the result is still [a], a^b^b=a
#              Also, XOR is commutative, so b^a^c^b^c = a^(b^b)^(c^c) = a^0^0 = a. Therefore, starting with [result]=0 and XOR every number onto 
#              [result], when iteration is done, the unique number = [result]
# Time complexity: O(n)
def singleNumber(nums):
    result = 0
    for num in nums:
        result ^= num
    return result
        
# 42.141 Linked list cycle ======================================= URL: https://leetcode.com/problems/linked-list-cycle/
# Problem: Given a head of linked list, detected if there is a cycle in it. Return True when exists a cycle, return False otherwise
# Description: Use two pointers [fast] and [slow], where [fast] iterates two node at a time, [slow] iterates one node at a time. If [fast] or [fast.next] 
#              is None, means end of list and no cycle detected. If [fast] = [slow] means [fast] catch up with [slow] and cycle is detected
# Time Complexity: O(n), let [x] be number of iterations of while loop and [N] is size of cycle. Then (2*x)%N=x%N, the program must the cycle if exists one 
#                  after iterate while-loop N times
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        fast = fast.next.next       # move [slow] and [fast] before comparision
        slow = slow.next
        if slow == fast:
            return True
    return False

# 43.155 Min stack ========================================= URL: https://leetcode.com/problems/min-stack/
# Problem: Implement an integer minimum stack class, including 1. constructor. 2. push() function, append a given integer at the top of stack. 
#          3. pop() function, remove the element at the top of the stack. 4. top() function, return the element at the top of the stack.
#          5. getMin() function, return the minimum element of the stack in constant time.
# Description: Use list to implement the stack, where each element of the list is a tuple (num, currentMin), first element is the number being added,
#              second element is the current minimum element, tracking min of whole stack 
# Time complexity: O(1) for every function
class MinStack(object):
    def __init__(self):
        self.stack = []          # (num, curMin)

    def push(self, num):
        if len(self.stack) == 0:                # if stack is empty then directly add num 
            self.stack.append((num,num))
        else:
            if self.stack[-1][1] > num:         # compare num to current min, and decide new min 
                self.stack.append((num,num))
            else:
                self.stack.append((num,self.stack[-1][1]))
    
    def pop(self):
        self.stack.pop()[0]

    def top(self):
        return self.stack[-1][0]

    def getMin(self):
        return self.stack[-1][1]

# 44.160 Intersection of two Linked lists ============================= URL: https://leetcode.com/problems/intersection-of-two-linked-lists/
# Problem: Given two heads of two linked list, and at certain node these two linked lists merge together. Such as,
#          1 -> 2 -> 3 \            
#                        6 -> 7     the linked lists [1,2,3,6,7] and [4,5,6,7] merged together at the node 6
#               4 -> 5 /            
#          Need to find and return the node that two linked list merge. Also, it is possible that two given linked lists never merge
# Description: Assume there exists a merge node, letting two pointers traverse through all nodes in two linked list, one pointer starts at one head
#              and the other starts at the other head. Once it traversed one of the lists, it should hit None, then let the pointer jump to the other
#              head, and visit the reset of the nodes. When the pointer visited junction node, it has visited [N] nodes, where [N] is number of nodes
#              in both lists. The other pointer will have the same story, and both pointer traverse at the same time with same pace, meaning when they
#              visited [N] nodes they must meet at the junction node.
#              The junction node must be detected after a single jump, so if there doesn't exist junction, the pointer will jump more than once.
# Time complexity: O(n)
def getIntersectionNode(headA, headB):
    ptA, ptB, jumpToNext = headA, headB, False
    while ptA and ptB:
        if ptA == ptB:
            return ptA
        ptA, ptB = ptA.next, ptB.next
        # reset [ptA] or [ptB] and make sure only reset once
        if not ptA and not jumpToNext:          # reset [ptA] to [headB] when it hits end
            ptA, jumpToNext = headB, True
        if not ptB:                             # reset [ptB] to [headA] when it his end
            ptB = headA
    return None

# 45.167 Two sum of sorted array ============================= URL: https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
# Problem: Given a sorted array, and a integer [target], need to find two number in the [array] that the sum of two number equal to [target].
#          Return the indices of the two numbers (1-indexed) as an integer array answer of size 2
# Description: Use two pointers one starts from left of array and the other starts from the end of array. Compute sum, if [sum] < [target] then
#              increase the sum by moving left pointer to the right. If [sum] > [target] then decrease [sum] by moving right pointer to left
# Time complexity: O(n)
def twoSumII(arr, target):
  l, r = 0, len(arr)-1
  while l<r:
    s = arr[l]+arr[r]
    if s == target: return [l+1, r+1]       # return indices as "1-indexed" 
    elif s>target:      # sum too large, move [r] to left
      r-=1
    else:               # sum too small, move [l] to right
      l+=1
  return None

# 46.168 Convert integer to A-Z column titles =================== URL: https://leetcode.com/problems/excel-sheet-column-title/
# Problem: Given an integer convert and return the form of Excel column titles, ex: A B ... Z AA AB .. AZ BA ... AAA AAB...
# Description: Twenty-six hex, and use "ord()" and "chr()"". Maintain [res] to track reversed result stirng. Every interation, 
#              subtract [columnNumber] by 1  since [columnNumber] starts from 1. Mod [columnNumber] by 26 to get current letter, 
#              divide [columnNumber] by 26 to extract current letter from "pool". Append current letter to the end of [res], 
#              return reversed [res]
def convertToTitle(columnNumber: int) -> str:
    res = ""
    while columnNumber:
        columnNumber -= 1
        res += chr((columnNumber)%26+ord('A'))      # get current letter
        columnNumber = (columnNumber)//26           # extract value of current letter from [columnNumber]
    return res[::-1]
    
# 47.42 Trapping Rain Water ============================= URL: https://leetcode.com/problems/trapping-rain-water/
# Problem: Given a non-negative integer array, where each integer in the array represents the height of a wall and assume each index
#          has width of 1. Need to find how many unit of water can be hold in the array. 
#          ex: [1,0,3] => 1, can hold 1 unit of water at index of 1.
# Description: Use two pointers start from the two ends of array. Track and compare the height of two pointers and move pointer of 
#              lower side, because the water level is decided by lower wall. Update the highest wall of the lower side, and any wall 
#              that is lower than the highest wall means water can be filled in.
# Time Complexity: O(n)
def trap(height: List[int]) -> int:
    left, right = 0, len(height)-1
    leftHeight, rightHeight = max(0, height[left]), max(0, height[right])       # tracking highest of [left] and [right]
    res = 0
    while left<right:
        # lower side is considered to fill water
        if leftHeight<rightHeight:
            left += 1
            res += max(min(leftHeight, rightHeight)-height[left], 0)
        else:
            right -= 1
            res += max(min(leftHeight, rightHeight)-height[right], 0)
        leftHeight, rightHeight = max(leftHeight, height[left]), max(rightHeight, height[right])        # update highest
    return res

# 48.169 Majority Element ============================== URL: https://leetcode.com/problems/majority-element/
# Problem: Given an array of integers, there exists an element that appears more than n//2 times in the array,
#          which is considered as the "majority", return the majority element.
#          You may assume that the array is non-empty and the majority element always exist in the array.
# Description: Moore Voting Algorithm. If a element 'm' is majority, then occurrence of 'm' minus occurrence of other other elements,
#              must end up with a number larger than 0. We will have a [count] and [major] to track the count of an element after cancelling
#              the count of other element, [major] will record the current element which is being tracked.
# Time Complexity: O(n)
def majorityElement(nums: List[int]) -> int:
    cur, cnt = nums[0], 0
    for num in nums:
        if cur == num:
            cnt += 1
        else:
            cnt -= 1
            if cnt == 0:                # reset [cur] and [cnt]
                cur, cnt = num, 1
    return cur

# 49.171 Excel Sheet Column Number ======================== URL: https://leetcode.com/problems/excel-sheet-column-number/
# Problem: Given a string of Excel Sheet Column Number, convert it to integer number. ex A=>1, Z=>26, AB=>28
# Description: Each character represents an integer number of base 26. Split the input string into list, and start from right most of the list
#              calculate the difference between the [char] of list with "A". Since "A"=1 instead of "A"=0, the integer representation is the
#              difference between [char] and "A" plus 1. 
def titleToNumber(columnTitle: str) -> int:
    res, base = 0, 1
    for c in reversed(columnTitle):
        res += (ord(c) - ord('A')+1)*base
        base *= 26
    return res

# 50.172 Factorial Trailing Zeroes ================================= URL: https://leetcode.com/problems/factorial-trailing-zeroes/
# Problem: Given an positive integer n, count how many zeroes are at end of n!.
# Description: Number of zeroes at the end, depends on how many fives between 1 to n. Because any even number time 5 always produces tens.
#              Therefore, counting zeroes is same as counting fives
def trailingZeroes(n: int) -> int:
    res = 0
    while n>0:
        n //= 5
        res += n
    return res

# 51.202 Happy Number ================================================================= URL: https://leetcode.com/problems/happy-number/
# Problem: Happy number is a positive integer that if that the sum of square of each digit call it [sum], and then repeat the same process
#          by replace the number with [sum]. Keep doing this if [sum] == 1 at some point, then the original number is called Happy Number
#          Given a positive integer [n], determine if [n] is a Happy Number
# Description: If a number is not a Happy Number, then repeating the process above, the [sum] will fall into a pattern, where [sum] will 
#              occur in a repeated loop. Therefore, keep tracking the [sum] of each iteration, if a [sum] is repeated, then [n] is not Happy
#              Otherwise, if [sum] hit 1 at some point, [n] is Happy
def isHappy(n: int) -> bool:
    record = set()                      # [record] track occurance of [sum]
    while True:
        s = sum(int(c)**2 for c in str(n))      # get [sum] of current [n]
        # check repeatness of [sum]
        if s in record:                 
            return False
        if s == 1:
            return True
        record.add(s)
        n = s                           # update [n] with last [sum]

# 52.189 Rotate Array ======================================= URL: https://leetcode.com/problems/rotate-array/
# PProblem: Given a array [arr] and an integer [k], modify [arr] in place by rotate the array to the right by k steps.
# Description: If k is positive, then rotate array by k steps is equivalent to move the last k element to the beginning of array.
#              Also, k may be negative number and |k| may greater than length of arr. In both case, let k = k%n.
#              If k is greater than length, k%n can get rid of exceeded part. If k is negative, k%n can convert from rotate left by k
#              rotate right by length-k. If k is negative and |k| > length, then k%n can first get rid of exceeded part and convert from
#              rotate left to right.
def rotate(nums: List[int], k: int) -> None:
    if len(nums)<=1 or k == 0:          # early termination
        return nums
    rotates = k%len(nums)
    nums[:] = nums[len(nums)-rotates:]+nums[:len(nums)-rotates]     # use "nums[:]" to deep copy

# 52.190 Reverse Bits ========================================= URL: https://leetcode.com/problems/reverse-bits/
# Problem: Given a 32 bit unsigned bitwise number, return the bitwise reversed number
# Description: Use bitwise shift. Use "num & 1" to get last bit digits from [n], add last bit digits to [res]. Shift [n] to right to
#              get next bit(n>>=1) and shift [res] to left to reserve space for next bit(res<<=1)
# Time complexity: O(lgN)
def reverseBits(n: int) -> int:
    res = 0
    for _ in range(32):
        res += n&1          # get last bit from [n]
        res <<= 1
        n >>= 1
    return res>>1           # shift [res] to proper digit before return
	
# 53.203 Remove Linked List Element ========================= URL: https://leetcode.com/problems/remove-linked-list-elements/
# Problem: Given the [head] of a linked list(possibly empty), and a value [val]. Return the head of new linked list, with all 
#		   nodes with [val] removed.
# Description: Firstly set a dummy node at beginning. Look at current.next, if cur.next==val, then skip next node by cur.next=cur.next.next
#			   Otherwise, move onto next node, cur=cur.next
# Time Complexity: O(n)
def removeElements(head: Optional[ListNode], val: int) -> Optional[ListNode]:
    dummy = cur = ListNode(0, head)
    while cur and cur.next:
        if cur.next.val == val:
            cur.next = cur.next.next
        else:
            cur = cur.next
    return dummy.next

# 54.204 Count Prime =========================================== URL: https://leetcode.com/problems/count-primes/
# Problem: Count and return number of prime number less than a given non-negative integer [n]
# Description: User sort of reversed "Sieve of Eratosthenes" method. Create a list [primes] with length [n], initially all True.
# 			   Each element represent the index number is prime or not. Iterate [i] from 2 to sqrt(n)+1. Set the upper boundary 
#              to sqrt(n)+1, because number larger than [n] is covered by [product]. If primes[i] is True, all mutiply of [i] 
#              are False, multiple of [i] are "i*(i+k)", where k=0,1,2,3,..., also [product] produce with k<[i] are covered by 
#              other combination. Thus, convert primes[i*i:n:i] to False. Count number of "True"s in primes at the end		   
# Time Complexity: O(N*log(logN))
#                  For number [i], there are [n/i] elements to be marked as False, where [i] is prime number. 
#                  There has n/2+n/3+n/5+n/7+n/13+..., according to "Harmonic Progression of the sum of primes". The sum is 
#                  less than N*(log(logN))
def countPrimes(n):
    if n < 3:
        return 0
    primes = [True] * n
    primes[0] = primes[1] = False
    for i in range(2, int(n ** 0.5) + 1):       # set upper limit to "sqrt(n)+1", because [n] larger than sqrt(n) is coverde
        if primes[i]:
            multi = i
            product = i*multi               # [product] starts from [i*i]
            while product < n:
                primes[product] = False
                multi += 1
                product = i*multi
    return primes.count(True)

# 55.206 Reverse Linked List ====================================== URL: https://leetcode.com/problems/reverse-linked-list/
# Problem: Given the head of a single linked list, return the head of reversed of the linked list.
# Description: Maintain three pointers [prev], [cur] and [next]. [prev] is None initially, that will be the last element in
#              reversed list. [cur] is the node that is reversed in each iteration by [cur.next] = [prev]. [next] is used 
#              to holde refrence of next node that [cur] is moving to, since [cur.next] is reassigned, and oringinal [cur.next]
#              is lost.
# Time Complexity: O(n)
def reverseList(head: Optional[ListNode]) -> Optional[ListNode]:
    prev, cur = None, head
    while cur:
        next = cur.next             # preserve [cur.next] before it is reassigned
        cur.next = prev             # reverse [cur] and [prev]
        prev, cur = cur, next       # [prev] and [cur] move forward
    return prev
# Description: Recursive
# Time Complexity: O(n)
def reverseList_2(head):
	return _rec(head, None)
	
def _rec(head, prev):
	if not head:
		return prev
	else:
		cur = head
		head = head.next
		cur.next = prev
		prev = cur
		return _rec(head, prev)

# 56.217 Contains Duplicate ==================================== URL: https://leetcode.com/problems/contains-duplicate/
# Problem: Given a list of integers, find if the list contains any dulicated element. Return True, if contains duplicate, otherwise return False
# Description: Use "Counter" to count occurence of each number, then iterate though values in [cnt] return Ture if a value is larger than 1.
# Time Complexity: O(n)
from collections import Counter
def containsDuplicate(nums: List[int]) -> bool:
    cnt = Counter(nums)             # count [nums]
    for v in cnt.values():                
        if v>=2:                        # find any value that is larger than 1 
            return True
    return False
# Description: Use python set() function, if set(nums) and nums have same size, meaning no duplicates
# Time Complexity: O(n), convertion from list to set is O(n) alg. (Set is hashtable internally)
def containDuplicate(nums):
	return len(nums)>len(set(nums))

# 57.219 Contains Duplicate II ========================================== URL: https://leetcode.com/problems/contains-duplicate-ii/
# Prbolem: Given a list [nums] of integers and a integer [k], find if there exists two duplicated elements nums[i] and nums[j], and 
#		   there indices i and j are differ at most k, j-i<=k.
# Description: Use dictionary, where elements are keys and indices are values. Adding elements to dict if element is not in dict,
#		       if an element is in dict, than compare the current index and index of last occurance. Alway update [dic] with latest
#              occurance index.
# Time Complexiy: O(n)
def containsNearbyDuplicate(nums, k):
	dic = {}
	for i, v in enumerate(nums):
		if v in dic and i-dic[v]<=k:
			return True
		dic[v] = i                  # record the latest occurance of [v]
	return False

# 58.255 Implement Stack using Queues ===================================== URL: https://leetcode.com/problems/implement-stack-using-queues/
# Problem: Implement a stack using queue, which should have functions including push(x), pop(), top(), empty(). And the queue will only have
#		   following functions pushToEnd(), peek/popFront(), size(), isEmpty()
# Description: Maintain a reversed queue. When push elements, push new element to the end of queue, and rotate the queue len(queueu)-1 times,
#              so that the newly added element is at the beginning of queue. Thus, pushing take O(n). And pop, top takes O(1) time, since
#              the top of stack is at the beginning of queue
# Time Complexity: push()=>O(n) pop()&top()&empty()=>O(1)
class MyStack:

	def __init__(self):
		self._queue = collections.deque()
	def push(self, x: int) -> None:
		self._queue.append(x)
		for _ in range(len(self._queue)-1):
			self._queue.append(self._queue.popleft())
	def pop(self) -> int:
		return self._queue.popleft()
	def top(self) -> int:
		return self._queue[0]
	def empty(self) -> bool:
		return len(self._queue)<=0
        
# 59.226 Invert Binary Tree ============================================== URL: https://leetcode.com/problems/invert-binary-tree/
# Problem: Given a root of binary tree. Invert the tree such that node.left <= node.right. Return the new root.
# Description: DFS recursive traversal, node.left = invert(node.right) and node.right = invert(node.left)
# Time Complexity: O(n)
def invertTree(root):
	if root:	# no need to have else, since function return None by default
		root.right, root.left = invertTree(root.left), invertTree(root.right)
		return root

# 60.231 Power of Two ================================================ URL: https://leetcode.com/problems/power-of-two/
# Problem: Given an integer, determine if the integer is power of 2
# Description: let num starts from 1 and multiple by 2, if num==n, then n is power of 2, if num>n at some point then 
#			   n is not power of 2
# Time Complexity: O(logn)
def isPowerOfTwo(n):
	cur = 1
	while cur <= n:
		if cur == n:
			return True
		cur *= 2
	return False
# Description: Binary operator, numbers that are power of 2, must only has one 1 in its binary base, 1=1, 2=10, 4=100 ...
#			   And if n is power of 2, n-1 must be the complement of n in bianry, 4=100 and 4-1=011, 8=1000 and 8-1=0111
#			   So, if n is power of 2, n&n-1=0 is always true
# Time Complexity; O(1)
def isPowerOfTwo_2(n):
	return n>0 and not n&(n-1)

# 61.232 Implement Queue using Stacks ================================== URL: https://leetcode.com/problems/implement-queue-using-stacks/
# Problem: Implement queue using stack(s), stack only have the functions including pushToTop(), pop/peekTop(), isEmpty(), size()
# Description: Using two stacks, instack stores incoming elements, outstack stores elements that are transferred from instack in reversed order
#			   and return element when pop or peek. 
# Time Complexity: push(x)=>O(1), pop()=>O(n), peek()=>O(n), empty()=>O(1)
class MyQueue:
    def __init__(self):
        self.instack=[]
        self.outstack=[]

    def push(self, x: int) -> None:
        self.instack.append(x)

    def pop(self) -> int:
        if not self.outstack:
            while self.instack:
                self.outstack.append(self.instack.pop())
        return self.outstack.pop()

    def peek(self) -> int:
        if not self.outstack:
            while self.instack:
                self.outstack.append(self.instack.pop())
        return self.outstack[-1]
       
    def empty(self) -> bool:
        return len(self.instack)+len(self.outstack)==0

# 62.234 Palindrome Linked List ====================================== URL: https://leetcode.com/problems/palindrome-linked-list/
# Problem: Check if a linked list is palinddrome, need to be done in O(n) time using O(1) space
# Description: Dividing the linked list into two halves by using slow and fast(2X faster) pointers, then reverse first half and 
#			   compare with second half.
# Time complexity: O(n)
def isPalindrome_3(head):
	rev = None		# end of reversed list
	slow = fast = head
	while fast and fast.next:		# if fast is at the end, slow is at the middle
		fast = fast.next.next
		rev, rev.next, slow = slow, rev, slow.next	# ***reverse list***
	# handle odd number of nodes, make right part starts at one node to the right of middle node
	if fast:
		slow = slow.next		
	while rev and rev.val == slow.val:
		rev, slow = rev.next, slow.next
	return not rev

# 63.235 Lowest Common Ancestor of a Binary Search Tree ====================== URL: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
# Problem: Given a binary search tree and two nodes of the tree, find the lowest common ancestor(LCA)
#		   LCA is “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants 
#          (where we allow a node to be a descendant of itself).”
# Description: Start from root of BST, compare root.val with p.val and q.val. If both p and q.val<root.val, keep search on left branch,
#              if both q and p.val>root.val, keep search on right branch, else current root is LCA.
# Time complexity: O(logn), BST compare to node and ditch half of subtree
def lowestCommonAncestor(root, p, q):
	while root:
		if p.val<root.val>q.val:	# both less than root.val
			root = root.left
		elif p.val>root.val<q.val:	# both greater than root.val
			root = root.right
		else:
			return root
	# no need to return outside loop, since python return None autoly

def lowestCommonAncestor_2(root, p, q):
    while (root.val - p.val) * (root.val - q.val) > 0:		# compare with sigh of difference
        root = (root.left, root.right)[p.val > root.val]	# bracket is index, True is index 1, False is index 0
    return root

# 64.242 Valid Anagram ================================================== URL: https://leetcode.com/problems/valid-anagram/
# Problem: Given two strings s and t, check if they are anagrams(same characters with different orders)
# Description: Covert two strings into two dictionaries where key is character and value is count of each char,
#			   then compare two dictionaries.
# Time complexity: O(n)
def isAnagram(s, t):
	dict_s, dict_t = {}, {}
	for c in s:
		dict_s[c] = dict_s.get(c, 0)+1		# get the value of dict[c], if doesn't exist then return 0(2nd arg)
	for c in t:
		dict_t[c] = dict_t.get(c, 0)+1 
	return dict_s == dict_t
	
# 65.257 Binary Tree Paths ================================================ URL: https://leetcode.com/problems/binary-tree-paths/
# Problem: Given a binary tree, return a list of all the pathes from root to every leaf.
#		   The list should follow format ['root->node1->node2->leaf1', 'root->node1->leaf2']
# Description: Recursively depth first search where the rec function keep tracking current path, and the collection of root-to-leaf path
# Time complexity: O(n)
def binaryTreePaths(root):
	if not root:
		return []
	result = []
	binaryTreePaths_helper(root, result, '')
	return result

def binaryTreePaths_helper(root, collect, curPath):
	if not root.left and not root.right:		# it is a leaf => add leaf to cur path
		collect.append(curPath+str(root.val))
	if root.left:								# existing left child => add this node to path and keep tracking left branch
		binaryTreePaths_helper(root.left, collect, curPath+str(root.val)+"->")
	if root.right:								# existing right child => add this node to path and keep tracking right branch
		binaryTreePaths_helper(root.right, collect, curPath+str(root.val)+"->")
	
# Description: Depth first search with stack
# Time complexity: O(n)
def binaryTreePaths_stack(root):
	if not root:
		return []
	stack, result = [(root, '')], []
	while stack:
		node, curPath = stack.pop()
		if not node.left and not node.right:
			result.append(curPath+str(node.val))
		if node.left:
			stack.append((node.left, curPath+str(node.val)+"->"))
		if node.right:
			stack.append((node.right, curPath+str(node.val)+"->"))
	return result
	
# 66.258 Add Digits ================================================ URL: https://leetcode.com/problems/add-digits/
# Problem: Given a non-negative integer, repeatedly add its digits until the sum has only one digit.
#          Ex: 38, 3+8=11, 1+1=2, Answser is 2
# Description: By number theory, Every integer a is congruent to the sum of its decimal digits modulo 9.
#			   Any number N with digits a[n]a[n-1] ... a[2]a[1]a[0], where a[i] is the number of each digit place. N can be rewirte into
#              N = a[n]*10^n + a[n-1]*10^(n-1) + ... + a[2]*10^2 + a[1]*10^1 + a[0]*10^0
#			     = a[n]*(99..99+1) + a[n-1]*(99..99+1) + ... + a[2]*(99+1) + a[1]*(9+1) + a[0]*1
#              N%9 = (a[n]*99..99 + a[n-1]*99..99 + ... + a[2]*99 + a[1]*9)%9 + (a[n] + a[n-1] + ... + a[2] + a[1] + a[0])%9
#              N%9 = (a[n] + a[n-1] + ... + a[2] + a[1] + a[0])%9, after sum up a[n] to a[0] it gives an integer which can apply the same manner
#              Corner case: multiple of 9. 9%9=0, 18%9=0 not equal to sum of digits.
# Time complexity: O(1)
def addDigits(num):
	return (num-1)%9+1 if num else 0
	
# 67.263 Ugly Number ============================================ URL: https://leetcode.com/problems/ugly-number/
# Problem: Check if a number is ugly number, which is positive number whose prime factors are only 2, 3, 5
# Description: Keep dividing num if it can be divided by 2, 3, 5 in whole. And after all the possible divisions, if num becomes 1
#			   meaning its prime factors are only 2, 3, 5
# Time Complexity: O(n)
def isUgly(num):
	if num<=0: return False
	for x in [5, 3, 2]:
		while num%x == 0:
			num /= x
	return num == 1

# 68.268 Missing Number =============================================== URL: https://leetcode.com/problems/missing-number/
# Problem: Given an array of n distinct number from 0 to n, and one number is missing. Find and return the missing number
# Description: There should be n+1 numbers from 0 to n, and the expected sum of the n+1 number can be calculated.
#			   Subtract the expected sum by the actual sum, the missing number is found
# Time Complexity: O(n), sum(nums) is Linear order
def missingNumber(nums):
	n = len(nums)
	return n*(n+1)//2-sum(nums)
	
# 69.283 Move Zeros ====================================================== URL: https://leetcode.com/problems/move-zeroes/
# Problem: Given an array of intergers, move all zero to the right side of array and maintain the order of non-zero elements.
#		   Must do it in-place, and minimize the total number of operaions.
# Description: Use two pointers [i] [j],  let both of them start at index 0, where [i] tracks the first zero need to be swapped
#		       [j] tracks the head of un-processed array. So, if encounter non-zero, [i] and [j] increase at same pace.
#			   If encounter zero, [i] will track this zero, and [j] goes on to find a non-zero to swap.
# Time Complexity: O(n)
def moveZeros(nums):
	i = 0
	for j in range(len(nums)):		# if no zero is found, [i] and [j] increase at same pace
		if nums[j] != 0:			# if found zero, [i] is not increased, and [j] keep increasing, to find next non-zero
			nums[i], nums[j] = nums[j], nums[i]		# when non-zero if found, then swap [i] [j]
			i += 1					# increase [i] to find next zero

# 70.290 Word Pattern ==================================================== URL: https://leetcode.com/problems/word-pattern/
# Problem: Given a pattern and a string, find if str and pattern are matched up one to one. Meaning a char in pattern matches a 
#		   word in str. Ex pattern="abba", str="car far far car", where "a"="car", "b"="far"
# Description: Use zip(). set(zip(p, s)) return all posible combination between elements at same index of pattern and str. If pattern and str are not bijection,
#			   then set(zip(p, s)) will have more elements than set(p) or set(s)
# Time Complexity: O(n), zip() is O(N*M), N is number of arrays, M is length of array. set() is O(n)
def wordPattern(pattern, str):
	p=pattern
	s=str.split()
	return len(s)==len(p) and len(set(zip(p,s)))==len(set(p))==len(set(s)) 

# Description: Use dictionary, Firstly check length of p and s. Then create two dictionary for both p and s, which stores the index of 
#			   the last occurance of an element in p and s. If any unmatch of index if found means they are not bijection
# Time Complexity: O(n)
def wordPattern_2(pattern, str):
	words = str.split()
	if len(pattern)!=len(words): return False
	p_dict, w_dict= {}, {}
	for i in range(len(pattern)):
		if p_dict.get(pattern[i], -1) != w_dict.get(words[i], -1): 	# check last occurance indices of pattern[i] and words[i]
			return False											# if pattern[i] or words[i] is not presented in dict, use -1 as defalut index
		p_dict[pattern[i]] = w_dict[words[i]] = i		# update the last occurance of pattern[i] and words[i]
	return True

# 71.292 Nim Game ============================================================= URL: https://leetcode.com/problems/nim-game/
# Problem: You paly Nim game with another player, there is a pile of cards with n cards in it, you and your oppponent take turns to move
#	       1-3 cards out of the pile, who move the last cards win the game. Write a function that determines if you can win the game with 
#		   n cards in the pile and you make the first move. Ex: n = 4, you can never win the game because no matter you take 1, 2 or 3 cards the opponent always takes 
#          the last card.
# Description: You can't win when [n] is divisible by 4,
# 			   If n<4, you win at the first pick.
#			   If n=4, you never win as described in problem.
#			   If 4<n<8, you can take out 1-3 cards let 4 cards left in pile, and leave n=4 to opponent.
#              If n==8, can be treated as 2 cases of n=4, and opponent wins both of the cases.
#              If 8<n<12, you can take out 1-3 cards let 8 cards left in pile, and leave n=8 to opponent.
def canWinNim(n):
	return n%4!=0
	
# 72.303 Range Sum Query - Immutable ============================================== URL: https://leetcode.com/problems/range-sum-query-immutable/
# Problem: Create a class that recieves an array of numbers, and return the sum of numbers between two given indices(included) in O(1) time.
# 	       Ex: [2, -1, 0, 3] sumRange(1, 3) = -1+0+3 = 2
# Description: Create a sumArr where sumArr[i] stores the sum of number from index 0 to index i-1. Therefore, sumArr[0] = 0 and 
#			   sumRange(i, j) = sumArr[j+1] - sumArr[i]
class NumArray:
    def __init__(self, nums):
        if nums:
            self.sums = [0]
            for i in range(len(nums)):
                self.sums.append(self.sums[i]+nums[i])
        else:
            self.sums=None

    def sumRange(self, i, j):
        return self.sums[j+1]-self.sums[i]

# 73.443 String Compression
# Problem: Given an array of Character, compress it in-place. Every element of the array should be a character (not int) of length 1. 
#          After you are done modifying the input array in-place, return the new length of the array.
#          Ex: ["a","b","b","b","b","b","b","b","b","b","b","b","b"] => ["a", "b", "1", "2"] return 4.
# Description: Three pointers, [i] denote current char, [j] search for first char that is different than chars[i], [k] tracking tail
#              of result. [j-i] is count of same sequential chars, convert [j-i] to string and iterate throgh it to add count.
#              After each denoting a char and its count, set i=j to seach for the count of new char
def compress(chars):
    i=j=k=0
    while (i<len(chars)):
        if (j<len(chars) and chars[i] == chars[j]):
            j+=1
        else:
            chars[k] = chars[i]
            k+=1
            if (j-i >= 2):
                for c in str(j-i):
                    chars[k] = c
                    k+=1
            i = j
    return k

# 74.447 Number of Boomeranges
# Problem: For any three points (i, j, k), if the distance between i and j equals to distance between i and k, we say (i, j, k) are "boomerang".
#          Order of i, j, k matters, meaning (i, j, k) and (i, k, j) are two different boomerangs.
#          Given n points in 2-d tuple ((x1, y1), (x2, y2), ...), return number of boomerangs.
# Discription: For every point [p] in [points], calculate the distance from [p] to other points [q], and tracking distance and number of points share the distance
#              in a dictionary {distance: count}. EX: for a point [p], the dictionary is {1: 3}, meaning there are 3 points [q] are 1 distance from [p],
#              hence pick any two points from [q]s, can form boomerange. The problem becomes P(3, 2) = 3!/(3-2)! for a single point [p].
#              Repeat the steps for every [p], and add permutations to get the result
# Time complexity: O(n^2) 
# Space complexity: O(n)
def numberOfBoomeranges(points):
    cnt = 0
    for p in points:
        dic = {}                # dictionary {distance: count}
        for q in points:
            d1 = p[0]-q[0]
            d2 = p[1]-q[1]
            d = d1**2 + d2**2               # calculate distance between [p] and [q]
            dic[d] = 1 + dic.get(d, 0)      # tracking distance and number of points share the distance
        for k in dic:
            cnt += dic[k] * (dic[k]-1)      # permutation and add to result
    return cnt

# 75.448 Find All Numbers Disappeared in an Array
# Problem: Given an array of [n] integers, where 1<=arr[i]<=n, some elements appeared twice and some didn't appear
#          Find all elements of [1,n] that didn't appear in the array
# Description: If a number [num] appeared in the array, negate the element arr[num-1], do arr[num-1] = -1*abs(arr[num-1]).
#              Then if a number appeared at least once, the element at its corresponding index (num-1) is turned into negative number
#              After all, search for positive numbers in the array, which are the numbers that didn't appear
def findDisappearedNumbers(nums):
    for i in range(len(nums)):
        index = abs(nums[i])-1                      # get corresponding index of num 
        nums[index] = -1*abs(nums[index])           # negate number at index
    return [i+1 for i in range(len(nums)) if nums[i]>0]

# 76.453 Minimum Moves to Equal Array Elements
# Problem: Given a non-empty integer array of size n, find the minimum number of moves required to make all array elements equal, 
#          where a move is incrementing n - 1 elements by 1.
# Desciprtion: Say before increment, the sum of all numbers is [sum] and minimum element is [min]. After [m] moves, all elements are equal to [x]. 
#              Assume the size of array is [n]. Then {sum+m*(n-1) = x*n}. Also, {x = min+m}, because [min] is added in all move in order to bring
#              it from [min] to [x]. Plus {x = min+m} into {sum+m*(n-1) = x*n} => {sum-min*n = m}, solve for [m]
def minMoves(nums):
    return sum(nums)-min(nums)*len(nums)

# 77.455 Assign Cookies
# Problem: Given two arrays [g] and [s], where every element in [g] represents what size of cookie can satisfy a child.
#          And every element in [s] represents the size of each cookie. If a child is given a cookie that is greater or equal to its satisficaion size,
#          we say this childe is satisfied. Return the maximum number of children can be satisfied
# Description: Sort both array, use two pointers on both arrays. If the cookie is greater or equal to the satisfiction size of the child.
#              assign the cookie to the child, and increase both pointer. Otherwise, increase the pointer of cookie to find a larger one
def findContentChildren(g, s):
    g.sort()
    s.sort()
    i = j = res = 0
    while (i<len(g) and j<len(s)):
        if (g[i] <= s[j]):
            res += 1
            i += 1
            j += 1
        else:
            j += 1
    return res

# 78.459 Repeated Substring Pattern
# Problem: Given a non-empty String [s], check if this string is constructed by multiple copies of its substring. The given string is consist of 
#          lowercase letters only.
# Description: Take substring of length [i] where len(s)%i == 0, because length of substring must divide entire string.
#              Then copy the substring [len(s)//i] times, and check if the constructed string is same as given string
# Time Complexity: O(n*sqrt(n)), there are O(sqrt(n)) possible [i]s, for each [i] construct given string O(n)
def repeatedSubstringPattern(s):
    N = len(s)
    for i in range(1, N//2+1):                  # iterate through all possible length of substring
        if N%i == 0 and s[:i]*(N//i) == s:      # find proper length of substring [i], and check if it construct given string
            return True
    return False
        
# 79.461 Hamming Distance
# Problem: Hamming Distance of two numbers is the number of positions at which the bits of two numbers are different, 
#         x = 1   (0 0 0 1)  => hamming distance = 2
#         y = 4   (0 1 0 0)
#                    ↑   ↑
#         Given two integer numbers [x], [y] (0<=x, y <2^31). Return their Hamming Distance
# Description: Use property of converting decimal number to binary number, mod [x] and [y] to get number of current position,
#              then divide [x] and [y] by 2, to get next posision
# Time Complexity: O(logn)  
def hammingDistance(x, y):
    res = 0
    while x or y:
        if x%2 != y%2:
            res += 1
        x //= 2
        y //= 2
    return res

# 80.463 Island Perimeter =================================================== https://leetcode.com/problems/island-perimeter/
# Problem: You are given a map in form of a two-dimensional integer grid where 1 represents land and 0 represents water.
#          Grid cells are connected horizontally/vertically (not diagonally). 
#          The grid is completely surrounded by water, and there is exactly one island (i.e., one or more connected land cells).
#          The island doesn't have "lakes" (water inside that isn't connected to the water around the island). 
#          One cell is a square with side length 1. The grid is rectangular or square. Determine the perimeter of the island.
# Description: The preimeter is number of island * 4 sides subtract the inner sides, where islands connects.
#              Hence, when we see a island, increase [preimeter] by 4, then check the neighbor of the island. Each neighbor
#              decrease [preimeter] by 1 (inner side)
def islandPerimeter(grid):
    res = 0
    row, col = len(grid), len(grid[0])
    for i in range(row):
        for j in range(col):
            res += grid[i][j]*4             # see an island, increase by 4
            # check neighbors
            if i > 0:               # left neighbor
                res -= grid[i][j]*grid[i-1][j]
            if i < row-1:           # right neighbor
                res -= grid[i][j]*grid[i+1][j]
            if j > 0:               # top neighbor
                res -= grid[i][j]*grid[i][j-1]
            if j < col-1:           # bottom neighbor
                res -= grid[i][j]*grid[i][j+1]
    return res

# 81.475 Heaters
# Problem: Given houses and heaters in two integer arrays separately, where values represents the location of houses and heaters
#          on a horizontal line, find out minimum radius of heaters so that all houses could be covered by those heaters.
# Description: For each house, use binary search on heaters array, find the closes heaters and calculater the distance from
#              house to left and right heaters that are cloest to the house. Take the min, which is the minimum radius needed to 
#              warm the house. Meanwhile, keep tracking the maximum needed radius for each house, and return the max at the end
# Time Complexity: O(mlogn)
#                  m - size of houses
#                  n - size of heaters
import sys
import bisect
def findRadius(houses, heaters):
    heaters.sort()                  # sort heaters for binary search
    radius = -sys.maxsize-1         # result radius, initially max_int
    for h in houses:
        index = bisect.bisect_left(heaters, h)      # binary search on [heaters] find the index, in order to get cloest heaters
        distLeft = distRight = sys.maxsize          # distance to left and right closest heaters

        if index > 0:
            distLeft = h - heaters[index-1]         # distance to left cloest heater
        if index < len(heaters):
            distRight = heaters[index] - h          # distance to right cloest heater
        radius = max(radius, min(distLeft, distRight))      # find min(left, right) and update radius if new radius is larger
    return radius
    
# 82.476 Number Complement
# Problem: Given a positive integer [num], output its complement number. The complement strategy is to flip the bits of its binary representation.
#          EX: 5 = 101, its complement is 010 = 2
#          The given integer [num] is guaranteed to fit within the range of a 32-bit signed integer. [num] >= 1
# Description: Construct an other number [sum], which has same length of [bin(num)] and digits of [sum] are "1"s
#              Use XOR on [sum] and [num] to get the complement
# Time Complexity: O(logn)
def findComplement(num):
    temp, sum = num, 0
    while temp > 0:                 # construct [sum] by shifting bits to left, and adding 1 at the right side
        temp >>= 1
        sum <<= 1 
        sum += 1
    return num^sum                  # XOR on [num] and [sum]

# 83.482 License Key Formatting
# Problem: given a license key represented as a string S which consists only alphanumeric character and dashes.
#          Given a number K, we would want to reformat the strings such that each group contains exactly K characters, 
#          except for the first group which could be shorter than K, but still must contain at least one character. 
#          Furthermore, there must be a dash inserted between two groups and all lowercase letters should be converted to uppercase.
# Description: Replace "-" in the String and convert all characters to uppercase. Check if the length of String is dividable by [K],
#              Extract the first group separately if len(S) is not dividable by [k]. Then extract the reset of String, and 
#              divide each group by "-"
# Time Complexity: O(3n). replace(), upper() takes O(2n) in total. While loop takes another O(n)
def licenseKeyFormatting(S: str, K: int) -> str:
    S = S.replace("-", ""). upper()
    i = K if len(S)%K == 0 else len(S)%K
    res = S[:i]                 # extract first group
    while i < len(S):
        res += "-" + S[i:i+K]
        i += K
    return res

# 84.485 Max Consecutive Ones
# Problem: Given a binary array, find the maximum number of consecutive 1s in this array.
# Description: Use a count variable to count number of consecutive ones. When encount 0, record the current count and compare with 
#              maximun count so far, then reset count to zero. If the last element is 1, will need to compare count and max before returning
# Time Complexity: O(n)

def findMaxConsecutiveOnes(nums: List[int]) -> int:
    res = cur = 0
    for i in nums:
        if i == 1:
            cur += 1
        else:
            res = max(res, cur)
            cur = 0
    return max(res, cur)

# 85.492 Construct the Rectangle
# Problem: Given a integer [area], find a rectangle with length [l] and width [w], that form the given [area].
#          The length [l] is greater or equals to width [w], and the difference between [l] and [w] should be
#          as small as possible
# Description: Let [w] start from sqrt(area), and keep decreasing by 1. Once [area%w] == 0, meaning [w] and [l] are found.
# Time Compliexity: O(sqrt(n))
from math import sqrt
def constructRectangle(area):
    w = int(sqrt(area))
    while area%w != 0:
        w -= 1
    return [area//w, w]

# 86.496 Next Greater Element I
# Problem: Given two number arrays (without duplicates) [nums1] and [nums2], where [nums1] is a subset of [nums2]. Find 
#          all nex greater numbers for [nums1] element in the corresponding places of [nums2]. 
#          The "Next greater number" of a number [x] of [nums1] is the first greater number to its right in [nums2], 
#          if doesn't exist, output -1 for this number
#          Ex: nums1 = [4,2,3] nums2 = [3,2,1,4] output = [-1,4,4]
# Description: For-loop iterate through [nums2], use a stack to hold element [n] of [nums2] and use a dictionary to record
#              [n]'s next greater number. If [n]'s next greater number is found, popLast [n] and record it in dictionary {n:next_greater},
#              and keep check if other [n] in stack can be mapped to this [next_greater], If not found, move on the next [n]. 
#              After exiting for-loop, iterate [nums1] and found their mapping from the dictionary, -1 if not found
# Time Complexity: O(n)
def nextGreaterElement(nums1, nums2):
    stack, map = [], {}
    for n in nums2:
        while stack and stack[-1]<n:            # compare with [n], try to find [next_greater] for last element in stack
            map[stack.pop()] = n                # [next_greater] is found, record it in dictionary
        stack.append(n)
    return [map.get(n, -1) for n in nums1]      # return corresponding [next_greater]  for elements in [nums2]

# 87.500 Keyboard Row
# Problem: Given a List of words, return the words that can be typed using letters of alphabet on only one row's of American keyboard:
#           Q W E R T Y U I O P
#            A S D F G H J K L
#             Z X C V B N M
# Description: Convert each row to a char set. If a [word] can be typed using letter on only one row's, the set([word]) must be a 
#              "subset" of a line
# Time Complexity: O(n)
#                   n: number of characters of all the strings
def findWords(words):
        line1, line2, line3 = set("qwertyuiop"), set("asdfghjkl"), set("zxcvbnm")
        res = []
        for word in words:
            wordSet = set(word.lower())
            # "<=" on sets, means leftOperand is a "subset" of rightOperand
            # "<" on sets, means leftOperand is a "strict subset" of rightOperand 
            if wordSet <= line1 or wordSet <= line2 or wordSet <= line3:
                res.append(word)
        return res

# 88.501 Find Mode in Binary Search Tree
# Problem: "Mode" the most frequently occurred element. Given the root of a binary search tree, where left subtree of a node contains
#          keys less than or equals to the node, and right subtree of a node contains keys greater than or equals to the node.
#          Return a list of keys (possibly multiple modes) that are the modes of the tree
# Description: inorder traversal the tree twice. The first time, find the number of modes and its occurence. Since in BST, node of same key
#              must cluster. Keep tracking the [currVal] and [currCount], once see a different key set [currVal] to new key and set
#              [currCount] to zero. Compare [currCount] with [maxCount]. If [currCount] > [maxCount], update [maxCount] and set [modeCount] 
#              to 1, since a higher count is found, previous modes are no longer modes. Else, increase [modeCount]
#              After, the first traversal, create a list [mode] with the size of [modeCount] to contain the resule. And reset [modeCount] and
#              [currCount] 
#              Second inorder traversal. Keep tracking [curVal] and [currCount], once find a key has [currCount] == [maxCount] then add it into
#              mode[modeCount]. Increase [modeCount] by 1, and keep finding next mode.
# Time Complexity: O(n), traverse tree twice
# Space Complexity: O(1), if doesn't count return lsit [mode]
class FindMode:
    def __init__(self):
        self.currVal=0
        self.currCount=0
        self.maxCount=0
        self.modeCount=0
        self.mode=[]
        
    def findMode(self, root):
        self.inorder(root)
        self.mode = [0]*self.modeCount      # create mode list to store result
        self.modeCount = 0
        self.currCount = 0
        self.inorder(root)
        return self.mode
    
    def handleValue(self,val):
        if val != self.currVal:
            self.currVal = val
            self.currCount = 0
        self.currCount += 1
        # when a higher count is found, previous modes are no longer modes
        if self.currCount > self.maxCount:
            self.maxCount = self.currCount
            self.modeCount = 1
        elif self.currCount == self.maxCount:
            if len(self.mode) != 0:                         # in second round traveral, mode is not empty
                self.mode[self.modeCount] = self.currVal    
            self.modeCount += 1
            
    def inorder(self, root):
        if root is None:
            return
        self.inorder(root.left)
        self.handleValue(root.val)
        self.inorder(root.right)
# Time Complexity: O(n)
# Space Complexity: O(n), using hashmap to track occurence of keys
def findMode(root):
    count = {}
    dfs(root, count)                # traverse though tree and record occurence of keys
    maxCount = max(count.values())  # find maxCount of keys
    return [k for k, v in count.items() if v == maxCount]       # gather keys with maxCount

def dfs(node, count):
    if node:
        count[node.val] = count.get(node.val, 0) + 1    # record occurence of keys
        dfs(node.left, count)
        dfs(node.right, count)

# 89.504 Base 7
# Problem: Given an integer(possiblly negative) return its base 7 String representation
# Description: If the integer [num] is negative, make it positive and record the sign. Module the integer by 7 and append
#              to a String [res], update [num] by dividing 7. Return reversed [res] when [num]==0
def convertTOBase7(num):
    if num == 0:
        return "0"
    res, sign = "", ""
    if num < 0:
        sign = "-"
        num *= -1
    while num != 0:
        res += str(num%7)
        num //= 7
    return sign+res[::-1]

# 90.506 Relative Ranks
# Problem: Given score of N athletes, return a list of their ranks, where top three are "Gold Medal", "Silver Medal" and "Bronze Medal".
#          The reset are their relative ranks starting from "4"
#          Ex: [9, 10, 8, 1, 2] => ["Silver Medal", "Gold Medal", "Bronze Medal", 5, 4]
# Description: Store scores in a 2-D list, where each element is [score, orig_index]. Reverse sort the 2-D list with respect of score. 
#              Iterate through sorted 2-D list, and get their original index at "2dList[i][1]"". The top three are changed as nums[index] = "Medal", and the
#              reset are set to [i+1]
# Time Complexity: O(nlogn)
# Space Complexity: O(n)
def findRelativeRanks(nums):
    rank = []
    for i in range(len(nums)):                  # store into 2D list [score, orig_index]
        rank.append([nums[i], i])
    rank = sorted(rank, key=lambda item: item[0])       # sort list with respect of score
    rank.reverse()
    for i in range(len(rank)):          # iterate through sorted 2d list, get the rank and update nums
        if i == 0:
            nums[rank[i][1]] = "Gold Medal"
        elif i == 1:
            nums[rank[i][1]] = "Silver Medal"
        elif i == 2:
            nums[rank[i][1]] = "Bronze Medal"
        else:
            nums[rank[i][1]] = str(i)  
    return nums


# 91.507 Perfect Number
# Problem: A perfect number is a positive integer(1 ~ inf) that  is equal to the sum of its positive divisors, excluding the number itself.
#          A divisor is an integer that divide evenly (mod = 0)
#          Ex: 28 is a perfect number, as its divisors excluding itself are 1+2+4+7+14 == 28
# Description: Iterate through [2 - sqrt(num)], and find all divisors. Sum up [divisor] and their [complement](complement = num/divisor).
#              [1] is always a divisor, but its complement is [num] itself, thus [1] should be added to the sum separately. However,
#              if [num] is [1], and divisor can't be [num] itself, need to be considered as corner case
# Time complexity: O(sqrt(n))
def checkPerfectNumber(num):
    if num<=1: return False
    res = 0
    for i in range(2, int(math.sqrt(num))+1):           # iterate though 2 to sqrt(num)
        if num%i == 0:
            res += (i+num//i)       # add both [i] and its [complement]
    return res+1 == num         # 1 is always a divisor

# 92.509 Fibonacci Number
# Problem: Given a integer greater or equals to zero, return its fibonacci number
#          fib(0)=0, fib(1)=1, fib(n)=fib(n-1)+fib(n-2)
# Description: Use matrix exponential to get fibonacci number. Construct an initial fibonacci matrix, [[f(2), f(1)], [f(1), f(0)]] 
#              as [[1,1], [1,0]], and [[1,1], [1,0]]^n = [[f(n+1), f(n-1)], [f(n-1), f(n-1)]]. Using this property and divide and conquer, 
#              where U^11 = U^5*U^5*U, U^5 = U^2*U^2*U, U^2 = U*U
# Time Complexity: O(logn)
# Space Complexity: O(logn)
class Fibonacci:
    def fib(self, N: int) -> int:
        if (N <= 1):
            return N

        A = [[1, 1], [1, 0]]
        self.matrix_power(A, N-1)   # because a matrix contains f(n+1), f(n) and f(n-1). [N] should start at 1

        return A[0][0]          # f(n) is located at f[0][0] at the end

    def matrix_power(self, A: list, N: int):
        if (N <= 1):
            return A

        self.matrix_power(A, N//2)          # recursive divide
        self.multiply(A, A)                 # U^(2n) = U^n * U^n

        if (N%2 != 0):                  # if N is odd, U^(2n+1) = U^(2n) * U
            B = [[1, 1], [1, 0]]
            self.multiply(A, B)

    def multiply(self, A: list, B: list):       # perform matrix multiply
        x = A[0][0] * B[0][0] + A[0][1] * B[1][0]
        y = A[0][0] * B[0][1] + A[0][1] * B[1][1]
        z = A[1][0] * B[0][0] + A[1][1] * B[1][0]
        w = A[1][0] * B[0][1] + A[1][1] * B[1][1]

        A[0][0] = x
        A[0][1] = y
        A[1][0] = z 

# 93.520 Detect Capital
# Problem: Given a String detect if it use capitals correctly. There are three cases of correct capital:
#          1. All letters are capitalized, "USA"
#          2. All letters are not caplitalized, "leetcode"
#          3. Only the first letter is capitalized and the rest are not capitalized, "Goolge"
# Description: record the number of uppercase letter and lowercase letter. If there is no lowercase letter, it falls
#              into case 1. If there is no uppercase letter, it falls into case 2. If there is one uppercase letter,
#              and the first letter is capitalized, it falls into case 3. Anyother cases are false
def detectCapitalUse(word):
    lower, upper = 0, 0
    for c in word:
        if c.isupper():
            upper+=1
        elif c.islower():
            lower+=1
    if upper == 1:
        return word[0].isupper()
    return lower==0 or upper==0

# 94.530 Minimum Absolute Difference in BST
# Problem: Given the root of a binary search tree, with non-negative numbers. Find and return the minimum difference between two
#          nodes in the tree
# Description: Use proerty of BST, where left subtree is smaller, right subtree is greater.
#              Call helper() on both left and right children, [fn(left, lo, curNode)] and [fn(right, curNode, hi)]. Because [curNode] 
#              must be the largest node in left subtree, and must be the smallest node in right subtree. When traversal hit leaves,
#              the closest node along the path must be [lo] and [hi]
# Time Complexity: O(n)
def getMinimumDifference(root: TreeNode) -> int:
    return minimummDifferenceBST_helper(root, float("-inf"), float("inf"))
    
def minimummDifferenceBST_helper(node, lo, hi):
    if not node: 
        return hi - lo              # when hit leaves, the [lo] and [hi] must be the cloest node along the path
    left = minimummDifferenceBST_helper(node.left, lo, node.val)         # curNode is largest node in left subtree, thus [hi = curNode.val]
    right = minimummDifferenceBST_helper(node.right, node.val, hi)       # curNode is smallest node in right subtree, thus [lo = curNode.val]
    return min(left, right)         

# 95.541 Reverse String II
# Problem: Given a String [s] and a integer [k], need to reverse the first [k] character for every [2k] characters starting from the beginning
#          of [s]. If there are less than [k] characters, reverse all of them. If there are more than [k] characters but less than
#          [2k] characters, reverse first [k] character and leave the rest unchanged
#          Ex: s="abcdefg", k=2 => "bacdfeg". Reverse [k] character for every [2k], "ab", "ef" are reversed
#              s="abcde", k=3 => "cbade". Length is more than [k] but less than [2k], reverse first [k] and leeave rest unchanged
#              s="abcde", k=7 => "edcba". Length is more than [2k], reverse all
# Description: Python string is immutable, change string directly is slow. Thus, convert string to "list", then reverse subList.
#              Using two pointer [left], [right], indicate the sub-list need to be reversed. For each iteration, increase [left] and 
#              [right] by [2k], until [left] is out of bound. We check [left] instead of [right], because if string length is less than [k],
#              need to reverse all.
def reverseStr(s, k):
        strList = list(s)       # convert string to list
        left, right = 0, k
        while left<len(s):
            strList[left:right] = strList[left:right][::-1]     # if [right] is out of bound, sub-list take list from [left] to the end
            left += 2*k
            right += 2*k
        return "".join(strList)     # join character back to string

# 96.543 Diameter of Binary Tree
# Problem: Given a binary tree, need to find the longest path between two nodes in the tree, aka, Diameter. This path may or may not
#          pass through root.
# Description: For a node, the longest path pass through it, must be its [leftDepth + rightDepth].  Thus, we consider each node, and find
#              its [leftDepth + rightDepth], and maintain the maximum path length.
# Time Compleixty: O(n)
# Space Complexity: O(1)
class DiameterOfBinaryTree:
    def __init__(self):
        self.maximum = 0        # maintain maximum diameter
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.helper(root)
        return self.maximum
    
    def helper(self, root):
        if root:
            left = self.helper(root.left)
            right = self.helper(root.right)
            self.maximum = max(self.maximum, left+right)    # calculate and update maximum diameter
            return max(left, right)+1           # The depth of leaf is 1
        return 0

# 97.551 Student Attendance Record I
# Problem: Given a string representing an anttendance of a student, where "A"="Absent", "P"="Present", "L"="Late".
#          A student can be rewarded if his attendance contain at most One "Absent" or doesn't continuous three "Late"s
#          Return "True" if the student can be rewarded. Otherwise, return False
# Description: Iterate through string, count nunber of "A", and check if the substring contains "LLL".
# Time Complexity: O(n)
# Space Complexity: O(1)
def checkRecord(s):
    countA = 0
    for i in range(len(s)):
        if s[i] == 'A':
            countA += 1
        if s[i:i+3] == "LLL":       # no need to worry about out of bounds
            return False
    return countA <= 1

# 98.557 Reverse Words in a String III
# Problem: Given a string of sentence, reverse each word individually and preserve whitespace and initial order
#          ex: today is a good day => yadot si a doog yad
# Descritpion: "Split" each word and store in a list, reverse each element of list then "join" them into one string
def reverseWords(s):
    l = s.split(" ")
    for i in range(len(l)):
        l[i] = l[i][::-1]
    return " ".join(l)

# 99.559 Maximum Depth of N-ary Tree
# Problem: Given a n-ary tree, find the maximum depth.
#          Maximun depth is the number of node along the longest path from root to a leaf.
# Description: For each node, iterate call the recursive function on each child. Maintain a max depth among all children
#              and return it
# Time Complexity: O(n)
def maxDepth_2(root):
    if root:
        maximum = 0
        for node in root.children:
            maximum = max(maximum, maxDepth(node))
        return maximum+1
    return 0

# 100.561 Array Partition I
# Problem: Given an integer array [nums] with [2n] elements. Arrange all elements in pairs (ai, bi), such that the sum
#          of all min(ai, bi) is maximized. Return the maxmized sum
# Description: Assume in each pair (ai, bi), always exists ai <= bi. 
#              Denote Sm = min(a1 + b1) + ... + min(ai + bi) + ... + min(an + bn), since ai <= bi. 
#                     Sm = a1 + ... + ai + ... + an
#              Denote Sa = a1 + b1 + ... + ai + bi + ... + an + bn, which is a constant for a given array
#              Denote Di = |ai - bi| = bi - ai. Denote Sd = d1 + ... + di + ... + dn
#              There has, (ai + bi) = 2*ai + (bi - ai) = 2*ai + Di, expand it to all "i"s  
#              has, Sa = 2*Sm + Sd => [Sm = (Sa - Sd) / 2].
#              In order to get maximized Sa, need to have smallest Sd, which means
#              [(bi - ai) for all "i"s should be minimized].
#              Adjacent elements in a sorted array, produces minimized difference between pairs of elements
# Time Complexity: O(nlogn)
def arrayPairSum(nums):
    return sum(sorted(nums)[::2])

# 101.563 Binary Tree Tilt
# Problem: Given the root of a BST, return the sum of every tree node's [tilt].
#          [tilt] is the absolute difference between sum of left subtree and right subtree.
#          if a node doesn't have left or right substree, the sum of the empty subtree is zero
# Description: Maintain a variable [ans] which accumulates the absolute difference between left subtree and right subtree.
#              Recursive call function on left and right child, return [root.val + leftDiff + rightDiff], because the upper 
#              node treat current node as a child. 
# Time Complexity: O(n)
class BinaryTreeTilt:
    def __init__(self):
        self.ans = 0
        
    def findTilt(self, root: TreeNode) -> int:
        self.helper(root)
        return self.ans
    
    def helper(self, root):
        if root:
            left = self.helper(root.left)
            right = self.helper(root.right)
            self.ans += abs(left-right)
            return root.val + left + right
        return 0

# 102.566 Reshape the Matrix
# Problem: Given a 2-D list [nums], and two integers [r] and [c]. Reshape the given shape with [r] rows and [c] columns
#          The reshaped matrix need to be filled with all the elements from the original matrix in the same row-traversing
#          order. In case the given [r] and [c] fit the number of elements in original matrix, then return original matrix
#          Otherwise, return the reshaped matrix
# Description: Firstly, check if the original matrix can fit into new size. Use sum(nums, []), flatten 2-d matrix. If doesn't
#              fit, return original matrix immediately. Otherwise, starting traverse original matrix and put element in to 
#              desired dimension
# Time Complexity: O(n)
# Space Complexity: O(n)
def matrixReshape(self, nums: List[List[int]], r: int, c: int) -> List[List[int]]:
    # flatten matrix and check dimension
    flat = sum(nums, [])        # sum(list, []) will flatten matrix
    if len(flat) != r*c:
        return nums
    # iterate every element and put into new dimensional matrix
    res, row = [], []
    for line in nums:
        for ele in line:
            if len(row) == c:
                res.append(row)
                row = []
            row.append(ele)
    if row:
        res.append(row)
    return res

# 103.572 Subtree of Another tree
# Prolem: Given two non-empty binary trees [s] and [t]. Check whether tree [t] is a subtree of [s]
#         A subtree of [s] is a tree [t] that [t]'s root is a node in [s], and all descendants of the node in [t] and [s]
#         matches each other
# Solution 1: DFS treverse down [s]. When find a node is equal to the root of [t], check every descendant of the node
#             if all descendant matches in both [s] and [t] return True. Otherwise, keep looking for node in [s] that is 
#             same as root of [t]. Return False if treversal of [s] reach end
# Time Complexity: O(|S|*|T|), where [S] and [T] are number of nodes in tree [s] and tree [t] 
# Space Complexity: O(|t|), recursive call 
def isSubtree(s: TreeNode, t: TreeNode) -> bool:
    if isMatch(s, t):           
        return True
    if not s:
        return False
    return isSubtree(s.left, t) or isSubtree(s.right, t)        # looking for node in [s] that is equal to root of [t]

def isMatch( s, t):
    if s is None or t is None:
        return s is t
    return s.val == t.val and isMatch(s.left, t.left) and isMatch(s.right, t.right)     # if a match is found, keep looking at their descendants
# Solution 2: Merkle Hashing, for each node in [s] and [t], create a hash to represent its subtree including itself. When comparing subtree
#             we only need to compare the hash code
# Time Complexity: 
class IsSubtree_2:
    def isSubtree(self, s, t):
        from hashlib import sha256
        # Function to generate hash code
        def hash_(x):
            S = sha256()
            S.update(x.encode("utf-8"))     # string need to be encoded before hashing
            return S.hexdigest()

        # Function to generate merkle code of a node and its children
        def merkle(node):
            if not node:
                return '#'
            m_left = merkle(node.left)      # recursive generate hash code for left subtree
            m_right = merkle(node.right)    # recursive generate hash code for left subtree
            node.merkle = hash_(m_left + str(node.val) + m_right)       # combine and stroe hash code for node's descendants
            return node.merkle

        # Treverse down [s] and looking for match
        def dfs(node):
            if not node:
                return False
            return (node.merkle == t.merkle or 
                    dfs(node.left) or dfs(node.right))

        # Intialize hash code for every node in both trees
        merkle(s)
        merkle(t)

        return dfs(s)

# 104.575 Distribute Candies
# Problem: Given an array of integers [candies[]], where each value represents the type of a candy. And there are always even number of candies
#          There are a sister and a brother, need to distribute candies evenly to them, and sister should get as more types as possible
#          Return the maximum types of candie that the sister can get
# Description: The maximum number of types [Max_Div] is the length of set of [candies] array len(set(candies)). And the maximum number of candy [Max_Get] 
#              that sister can get is half of the size of array len(candies)//2. The smaller between these two is the answer. 
#              Since if [Max_Div] > [Max_Get], there are more diversity of candies but sister can only get [Max_Get], so [Max_Get] is the maximum 
#              number of different candies that sister can get. Else [Max_Div] < [Max_Get], the number of candies that sister can get is large enough
#              to cover all diversity of candies.
# Time Complexity: O(n)
# Space Complexity: O(n)
def distributeCandies(candies: List[int]) -> int:
        s = len(candies)//2
        t = len(set(candies))
        return min(s,t)

# 105.589 N-ary tree preorder traversal
# Problem: Given a root of N-ary tree, return the preorder traversal of its nodes' values in a list
#          Children of a node are stored in a list.
#          Preorder: root -> left -> right
# Description: Use a helper function, takes a [result] list as parameter which keeps nodes's values of preorder traversal
#              Recursive append [root.val] to [result] and use a "for-loop" to traverse children nodes
def preorder(root):
    res = []
    preorder_helper(root, res)
    return res
        
def preorder_helper(root, res):
    if root:
        res.append(root.val)
        for child in root.children:
            preorder_helper(child, res)
    return res

# 105.590 N-ary tree postorder traversal
# Problem: Given a root of N-ary tree, return the postorder traversal of its nodes' valies in a list
#          Postorder: left -> right -> root
# Description: Same as Problem 105 (above)
def postorder(root):
    res = []
    postorder_helper(root, res)
    return res

def postorder_helper(root, res):
    if root:
        for child in root.children:
            postorder_helper(child, res)
        res.append(root.val)
    return res

# 105.594 Longest Harmonious Subsequence
# Problem: Given an integer array [nums], find the longest harmonious subsequence, and return its length
#          Harmonious array is an array where the difference between its maximum and minimum value
#          is exactly 1.
#          Subsequence of array is a sequence of element that can be derived from the array by deleting
#          zero or more element without changing the respective order
#          Ex: [1,3,2,2,5,2,3,7] => [3,2,2,2,3] 
#              [1,1,1,1] => []
# Description: Use "Counter" to put value of array and its count into hashmap. Look for values that is differed
#              by 1, sum up count of values that are differed by 1. Maintain the maximum count and return
#              at last
from collections import Counter
def findLHS(nums):
    count = Counter(nums)
    res = 0
    for value in count:
        if value+1 in count:        # look for value and its "adjacent" value
            res = max(res, count[value]+count[value+1])
    return res

# 106.598 Range Addition II ================================================ https://leetcode.com/problems/range-addition-ii/
# Problem: Given an m*n matrix, with all element initialized to 0. Given an 2d array [ops], where each element of [ops]
#          contain two elements [ [a1, b1], [a2, b2], ... ]. Each element represent a operation, which add one to elements
#          M[x][y] that 0<=x<=ai and  0<=y<=bi (meaning elements that are to the left-topper of [ai, bi]). Count and return 
#          the maximum integer in the matrix after operations
# Description: The goal is to find the number of cells that are added most time. Find the minimum X and Y in ops. The cells
#              that to the top and left of [X, Y] are added the most times
def maxCount(m, n, ops):
    if not ops: 
        return m*n
    return min([e[0] for e in ops]) * min([e[1] for e in ops])

# 107.599 Minimum Index Sum of Two Lists
# Problem: Given two arrays consist of restaurant names. Find the restaurants that appear in both lists, and return the 
#          restaurant that has the "least list index sum". There might be multiple restaurant having "list list index sum"
# Description: Maintain a dictionary, which contains the key-value pair as "restaurant_name: index" from [list1]. 
#              Maintain a [res] array and the [minIndex]. Iterate through [list2] when a restaurent is in dictionary, 
#              take the sum of index from both lists, and compare with minIdex. If sum of index is less than [minIndex],
#              clear [res] and add current restaurant_name as candidate. If sum equals to [minIndex], append current
#              restaurant_name to [res]. If sum greater than [minIndex], do nothing
# Time Complexity: O(n+m)
# Space Compleixty: O(n)
def findRestaurant(list1, list2):
    dic = {s:i for i, s in enumerate(list1)}
    res, minIndex = [], float('inf')
    for i, s in enumerate(list2):
        if s in dic:
            if dic[s] + i < minIndex:
                res = [s]               # smaller minIndex is found, clear [res] and add new candidate
            elif dic[s] + i == minIndex:
                res.append(s)           # new candidate is found, append into [res]
    return res

# 108.605 Can Place Flowers
# Problem: Given a array of "0"s and "1"s representing a [flowerbed]. "1" means there is a flower planted at the spot
#          "0" means there is not flower planted at the spot. Flowers can NOT be planted in adjacent spot. Given an 
#          integer [n] represent [n] new flowers. Return true, if [n] flowers can be planted in the [flowerbed], otherwise
#          return false
# Description: Maintain an integer [cnt] represent number of new flowers can be planted. Iterate through [flowerbed],
#              if [i]th spot is empty("0"), check its neighbors. If both neighbors are empty, increase [cnt] and use "1"
#              to occupy the spot. The first and last spot in flowerbed need to be considered separately. Lastly, compare
#              [cnt] and [n], if [cnt]>=[n] then return True, otherwise return False
def canPlaceFlowers(flowerbed: List[int], n: int) -> bool:
    if len(flowerbed) == 1:
        if flowerbed[0] == 0:
            return n<=1
        else:
            return n<=0
    cnt = 0
    if flowerbed[0] + flowerbed[1] == 0:
        flowerbed[0] = 1
        cnt += 1
    for i in range(1, len(flowerbed)-1):
        if flowerbed[i] == 0 and flowerbed[i-1] == 0 and flowerbed[i+1] == 0:
            flowerbed[i] = 1
            cnt += 1
    if flowerbed[-1]+flowerbed[-2] == 0:
        cnt += 1
    return cnt >= n

# 109.606 Construct String from Binary Tree
# Problem: Construct a string consists of parenthesis and integers from a binary tree with "preorder" traversing way.
#          "null" nodes are represented by empty parenthesis
#                       1               
#                      / \              => 1(2()(4))(3)
#                    2    3                "2" has only right child, a empty parenthesis is needed
#                     \                    "3" has no child, no parenthesis is followed by 3
#                      4            
#          If both children of a node are "null" nodes, the parent node should not followed by any empty parenthesis.
#          If only the left child is "null", an empty parenthesis is presented before right node.
#          If only the right child is "null", there is no empty parenthesis after left node
# Description: Recursion. For each node in a recursive call, construct substrnigs for both left and right subtree.
#              If a node has left or right node, construct a parenthesis and recursively call on left-childe node.
#              Because, if the left-child is not "null", the left subtree can be placed in the parenthesis. Even if the
#              left-child is "null", an empty parenthesis is needed since right-child presents
#              If a node has right child, recursively call on the right-child.
#              Combine current node, left substring, right substring and return at the end
def tree2str(t: TreeNode) -> str:
    if not t:
        return ""
    leftSub = rightSub = ""
    if t.left or t.right:
        leftSub = "({})".format(tree2str(t.left))
    if t.right:
        rightSub = "({})".format(tree2str(t.right))
    return "{}{}{}".format(t.val, leftSub, rightSub)

# 110.617 Merge binary tree
# Problem: Given two binary tree [t1] and [t2], merge them together by sum up node value at same position. If only node from one tree
#          present in the position and the node from the other tree is node, use the presented node value as new value.
#          If nodes from both trees are missing, the new node is None
# Description: Recursion. If both [t1] and [t2] are None, return None. Otherwise, return a new node with value of the sum of [t1] and [t2],
#              if any [t1] and [t2] is None, use 0 as its value. Recursive construct the children of the new node. If the current node [t1] 
#              or [t2] is already None meaning it can not have child, thus pass None into recursive call. Otherwise, pass the [t].child into 
#              recursive call. When constructing left child of new node, pass left child of [t1] and [t2]. When constructiing right child 
#              pass right child of [t1] and [t2].
def mergeTrees(t1, t2):
    if not t1 and not t2:
        return None
    root = TreeNode((t1.val if t1 else 0)+(t2.val if t2 else 0))
    root.left = mergeTrees(t1 and t1.left, t2 and t2.left)      # "[a] and [b]" in python, if [a] is truthy return [b], if [a] is falthy return [a]
    root.right = mergeTrees(t1 and t1.right, t2 and t2.right)
    return root

# 111.628 Maximum Product of Three Numbers
# Problem: Given an integer array [nums], find three elements of the array whose product is maximum, return the maximum product
# Descrpiton: There are two possible cases. 1) three positive numbers produce maxmimum product. 2) two negative number and a positive number
#              produce the maximum, the negative number should be the smallest and their absolute value are the largest.
#              Thus, sort the array, compare the product of case 1) nums[-1] [-2] [-3] and the product of case 2) nums[0] [1] [-1]. And return the larger one
# Time Complexity: O(nlogn)
def maximumProduct1(nums: List[int]) -> int:
    nums.sort()
    return max(nums[-1]*nums[-2]*nums[-3], nums[0]*nums[1]*nums[-1])
# Description: Still consider as two cases as above. Use heap queue to get largest 3 and smallest 2， which take O(nlogK), where k = 3
# Time Complexity: O(nlogk) => O(n) since log(k) is constant
import heapq
def maximumProduct2(nums):
        a, b = heapq.nlargest(3, nums), heapq.nsmallest(2, nums)
        return max(a[0] * a[1] * a[2], b[0] * b[1] * a[0])

# 112.637 Average of Levels in Binary Tree ================================================ https://leetcode.com/problems/average-of-levels-in-binary-tree/
# Problem: Given a non-empty binary tree, return the average value of each level in an array
# Description: Use DFS and track depth. Maintain a 2D [sumArr], where sumArr[i][0] is the sum of current level and sumArr[i][1] is the number of node in this level
#              Traverse down the tree, when current [depth] is >= len(sumArr), meaning we hit a new level need to increase size of [sumArr]. If [depth] < len(sumArr),
#              meaning we have encounter a node in this level, just need to increase the sum and node count.
#              At the end, take the average of every level by avg[i] = sumArr[i][0]/sumArr[i][1]
def averageOfLevels(root):
    sumArr = []     # maintain a 2D array, where sumArr[i][0] is the sum of current level, sumArr[i][1] is the number of node in this level
    avg = []        # result array
    averageOfLevels_dfs(root, 0, sumArr)
    for i in range(len(sumArr)):
        avg.append(sumArr[i][0]/sumArr[i][1])       # take average of each level
    return avg

def averageOfLevels_dfs(root, depth, sumArr):
    if root:
        if len(sumArr)<=depth:      # hit a new level, add new element to [sumArr]
            sumArr.append([0, 0])
        sumArr[depth][0] += root.val        # add node val to sum
        sumArr[depth][1] += 1               # add node count
        averageOfLevels_dfs(root.left, depth+1, sumArr)
        averageOfLevels_dfs(root.right, depth+1, sumArr)
    
# 113.643 Maximum Average Subarray I
# Problem: Given an integer array [nums], and an int [k]. Find the contiguous subarray of length [k], that has the maximum average value. 
#          Return the maximum average value
# Description: Slide window. Initialiy the sum of subarray [curSum] is the sum of first [k] element. Iterate through the array from 0 to len(nums)-k,
#              each iteration subtract the first element in subarray from [curSum] and add next element nums[k+i] into [curSum], meaning
#              [curSum] = [curSum] - nums[i] + nums[k+i]. Tracking the maximum sum and return its average at the end
def findMaxAverage(nums, k):
    curSum = maxSum = sum(nums[:k])
    for i in range(len(nums)-k):
        curSum = curSum - nums[i] + nums[i+k]
        maxSum = max(curSum, maxSum)
    return maxSum/k

# 114.645 Set Mismatch
# Problem: Given an integer array [nums] with [n] elements, where the elements are from 1 to [n], except one of the number is duplicated by another number,
#          and a number is missing due to the duplication. Find the duplicated number and the missing number return as a list [dup, miss]
# Description: Assume the duplicated number is [x], the missing number is [y], there has 1) sum(nums)-x+y = 1+2+ ... + n = (1+n)*n//2
#              2) sum(n**2 for n in nums)-x**2+y**2 = 1*1+2*2+ ... + n*n = n*(n+1)*(2n+1)//6 (formula of sum of square) 
#              From 1) x-y = sum(nums) - (1+n)*n//2. From 2) x**2-y**2 = sum(n**2 for n in nums) - n*(n+1)*(2n+1)//6
#              Hence x**2 - y**2 = (x-y)*(x+y) => x+y = (x**2-y**2) / (x-y). We have both (x+y) and (x-y), solve for x and y
# Time Complexity: O(n)
# Space Complexity: O(1)
def findErrorNums(nums):
    n = len(nums)
    d = sum(nums) - (1+n)*n//2      # x-y
    s = (sum([n*n for n in nums]) - n*(n+1)*(2*n+1)//6)//d      # x+y
    return [s+d/2, s-d/2]       # x = [(x-y)+(x+y)]/2   y = [(x+y)-(x-y)]/2

# 115.653 Two Sum in BST
# Problem: Given a numeric [root] of BST and number [k], chech if there exists two nodes that sum up to [k]. Return True is exists, return False otherwise
# Description: Maintain a set with node values. Inorder traverse through the tree to find if current node [i] that (k-i.val) exists in the set.
#              Return True if (k-i.val) is in set. Return False when traverse is end
# Time Complexity: O(n)
# Space Complexity: O(n)
def findTarget(root, k):
    if not root: return False
    bfs, s = [root], set()
    for i in bfs:
        if k - i.val in s: return True
        s.add(i.val)
        if i.left: bfs.append(i.left)
        if i.right: bfs.append(i.right)
    return False

# 116.657 Robot Return to Origin
# Problem: There is a robot start from origin (0,0), it can moves up(U), down(D), left(L) and right(R). Given a series of commands consists of UDLR
#          Check if the robot return back to origin after the series of commands. If it returns back return True, otherwise return False
# Description: If the robot returns back, means the number of [L]s is equal to number of [R]s, the number of [D]s is equal to number of [U]s,
#              Thus use collections.Counter() to count the nunber of each moves, and compare the number
def judgeCircle(moves):
    c = collections.Counter(moves)
    return c["L"] == c["R"] and c["D"] == c["U"]

# 117.661 Image Smoother
# Problem: Given a 2d integer array representing a grey scale image. Design a smoother method, that each cell is the average value of its surrounding 
#          eight cell along with itself. If a cell is at the boundray, use as many neighbor cells as possible
# Description: Make a copy of original matrix as [res], because the new cell value is generated from original value, meaning we can't overwrite new value
#              in the same array. Iterate through each cell, and look at its neighbors, the index of neighbor [x][y] should have 0<=[x]<len(row) and 
#              0<=[y]<len(col). Sum up the neighbor values and count number of neighbor to get average value
def imageSmoother(M):
    res = [[0 for _ in range(len(M[0]))] for _ in range(len(M))]        # make a [res] array with same dimension of given [M]
    for row in range(len(M)):
        for col in range(len(M[row])):
            s, cnt = 0, 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0<=row+i<len(M) and 0<=col+j<len(M[row]):
                        s += M[row+i][col+j]
                        cnt += 1
            res[row][col] = s//cnt              # store new value in [res] instead of overwrite on original [M]
    return res

# 118.665 Non-decreasing Array
# Problem: Given an integer array, modify at most one element in the array to make the array "non-decreasing", meaning nums[i] <= nums[i+1] 
#          for every elements. Return True, if such modification can be done with at most change of one element, otherwise return False
# Description: Iterate through array, compare nums[i] and nums[i+1] and [count] occurence of decrease. If a decrease is found, increase count
#              if there are more than one decrease order(count>1), it's improssible to change it by modifying one number. If only one is found,
#              then try to do modification to make it non-decrease. There are two ways to modify. 1) decrease nums[i], need to make sure 
#              nums[i-1]<=nums[i+1], otherwise decrease nums[i] will make nums[i-1]>nums[i]. 2) increase nums[i+1], need to make sure
#              nums[i]<=nums[i+2], otherwise will make nums[i+1]>nums[i+2]
# Time Complexity: O(n)
def checkPossibility(nums):
    cnt = 0
    for i in range(len(nums)-1):
        if nums[i]>nums[i+1]:
            cnt += 1
            # if cnt > 1, there are more than one decreasing order, no way to change it by modifying one number
            if cnt>1 or ((i>0 and nums[i-1]>nums[i+1]) and (i<len(nums)-2 and nums[i]>nums[i+2])):
                return False
    return True

# 119.669 Trim a Binary Search Tree
# Problem: Give a [root] of BST, and lower bound [low] and higher bound [high]. Delete node that has value lower than [low] or higher than [high].
#          The relative structure of nodes are maintained. Example [low] = 1, [high] = 4
#                       3                           3
#                     /   \                        /  \
#                   0       5      ===>          2     4
#                    \    /  \                  /
#                     2  4    6               1
#                    /
#                  1
# Description: DFS traverse down the tree and reassign [node.left] and [node.right] for each node. If [node.val] < [low], the left subtree should be 
#              ditched since left subtree all smaller than node.val, we call the recursive function on [node.right] to return the right node to replace
#              current node. If [node.val] > [right], right subtree should be ditched, we call the recursive function on [node.left] to return the left
#              node to replace current node. If [low]<=[node.val]<=[high], we call recursive function on [node.left] and assign to [node.left] and 
#              call recursive function on [node.right] and assign to [node.right]. At the end, the current node is returned to upper level
# Time Complexity: O(n)
# Space Complexity: O(h) where "h" the hight of the tree, BST may not be balanced
def trimBST(root, low, high):
    if not root: return root

    if root.val > high: return trimBST(root.left, low, high)        # current node and right subtree are invalid, try to get valid subtree from left node
    if root.val < low: return trimBST(root.right, low, high)        # current node and left subtree are invalid, try to get valid subtree from right node

    # current node is valid, keep traverse down
    root.left = trimBST(root.left, low, high)
    root.right = trimBST(root.right, low, high)
    
    return root

# 120.671 Second Minimum Node In a Binary Tree
# Problem: Given a special binary tree consists nodes of non-negative value. The special binay tree has two properties. 1) each node can have two children
#          or no child. 2) if a node has two children, value of the node is the smaller value of its children, node.val = min(node.left.val, node.right.val)
#          Find and return the "second smallest" value in the tree.
# Description: According to property_2, the smallest value is located at root, we need to traverse through the tree and find a node that is larger than 
#              root value, thus we maintain a [res] to hold the second smallest value, when we find a value that is less than current [res] and larger than
#              root.val, we update [res].
# Time Complexity: O(n)
def findSecondMinimumValue(root):
    res = [float("inf")]
    findSecondMinimumValue_helper(root, root, res)
    return res[0] if res[0] != float("inf") else -1

def findSecondMinimumValue_helper(root, cur, res):
    if not cur:
        return 
    if root.val<cur.val<res[0]:
        res[0] = cur.val
    findSecondMinimumValue_helper(root, cur.left, res)
    findSecondMinimumValue_helper(root, cur.right, res)

# Description: Use "node.val = min(node.left.val, node.right.val)" to reduce recursive calls. The second larger must be a right child, because parent
#              always hold smaller or same number of right child, and any smaller number is passed up to parent from left child. Thus,
#              if [cur.val] == [parent.val], eigher we hit left childe or a right child with same value of parent, we need to traverse both sub-trees
#              otherwise, [cur.val] > [parent.val] must hold, meaning we hit a right child, it must be the second smallest number since it is larger
#              than root.val
# Time Complexity: O(logN)
def findSecondMinimumValue_helper_2(root, parent, cur, res):
    if not cur:
            return 
    if root.val<cur.val<res[0]:
        res[0] = cur.val
    if parent.val == cur.val:       # current node has same value of parent, need to look for both of its children
        findSecondMinimumValue_helper_2(root, cur, cur.left, res)
        findSecondMinimumValue_helper_2(root, cur, cur.right, res)
    else:           # current node must be the second largest node, since it is greater than its parent, and the parent is the smallest node
        return

# 121.674 Longest Continuous Increasing Subsequence
# Problem: Given an unordered array with integers [nums], return the longest length of increasing subarray, where element must be strictly increasing
# Description: Iterate through [nums], maintain [curMax] and [res] as globalMax. Look at previous and current element. If [prev] > [cur], then 
#              increase [curMax] by 1, and updateing [res]. Return [res] at end of iteration
# Time Compleixty: O(n)
# Space Complexity: O(1)
def findLengthOfLCIS(nums):
    if len(nums) <= 1:
        return len(nums)
    res, cur = 1, 1
    for i in range(len(nums)-1):
        if nums[i] < nums[i+1]:
            cur += 1
            res = max(cur, res)
        else:
            cur = 1
    return res 

# 122.680 Valid Palindrome II
# Problem: Given a non-empty String [s], Determine if the string can be a palindrome by deleting at most one char.
# Description: Maintain two pointers [left] and [rigth] start from both end of string. Checking palidrome one char at a time, until they meet eachother.
#              If [left] != [right], consider two conditions 1) skip one at left side, by checking if s[left+1:right+1] is palindrome, or 
#              2) skip one from right side, by checking if s[left:right] is palindrome. We can use str == str[::-1] to check palindrome
# Time Complexity: O(n)
# Space Complexity: O(n)
def validPalindrome(s):
    left, right = 0, len(s)-1
    while left<right:
        if s[left] != s[right]:
            skipLeft, skipRight = s[left+1, right+1], s[left, right]
            return skipLeft[::-1]==skipLeft or skipRight[::-1]==skipRight
        left, right = left+1, right-1
    return True

# 123.682 Baseball Game
# Problem: Given a list of String [ops], consists of integer strings and characters, where integer string represent score of each baseball match, characters
#          represent operations. You start with an empty record, 1) if you see an integer string, record the new score in the list. 2) if you see a 
#          plus sign "+", sum up the previous two record to get the new record. 3) if you see a letter "D", double the previous score and add to 
#          the record. 4) if you see a letter "C", remove the previous score from record. Assume all operations in [ops] are valid
#          At the end, return the sum of scores in the new record
#          ex: ops = ["5", "2", "C", "D", "+"]
#              "5" => add 5 to record: [5]
#              "2" => add 2 to record: [5, 2]
#              "C" => delete previous score: [5]
#              "D" => double the previous score and add to record: [5, 10]
#              "+" => sum up previous two score and add to recrod: [5, 10, 15]
#              at the end return sum([5, 10, 15]) = 30
# Description: Maintain a [res] as new record, iterate through [ops], and base on element value do operations
# Time Complexity: O(n)
def calPoints(ops):
    res = []
    for op in ops:
        if op == 'C':
            res.pop()
        elif op == 'D':
            res.append(res[-1] * 2)
        elif op == '+':
            res.append(res[-1] + res[-2])
        else:
            res.append(int(op))
    return sum(res)

# 124.690 Employee Importance
# Problem: Given a data structure of employees, including "employee id", "importance", and "subordinates" like [[1,5,[2,3]], [2,3,[4]], [3,4,[]], [4,1,[]]]. 
#          This means, there are 4 employees which id 1, 2, 3, 4. Each of them has an importance of 5, 3, 4, 1. Employee_1 has subordinates [2, 3], Employee_2
#          has subordinates [4], Employee_3 and Employee_4 have no subordinates. Given a employee id, need to return the sum of the importance of itself plus
#          the importance of all its direct-subordinates and indirect-subordinates. This case, if the given id is 1, the direct-subordinates are [2, 3], the
#          indirect-subordinates are [4], the of importance is 5+3+4+1 = 13
# Description: First conver the data structure into dictionary, where key is [id], value is a list of [importance, subordinates].
#              Both BFS and DFS can be used, here I used BFS. Maintain a Deque [relation] to track subordinates, every itertaion retreive an Employee from Deque, 
#              append its subordinates to [relation] and add [importance] to result. Keep doing this, until [relation] is empty
# Time Complexity: O(n)
# Space Comlexity: O(n)
from collections import deque
def getImportance(employees, id):
    lookup = {}
    for e in employees:
        lookup[e.id] = [e.importance, e.subordinates]
    res = 0
    relation = deque(lookup[id][1])
    while relation:
        sub = relation.popleft()
        res += lookup[sub][0]
        for sub_2 in lookup[sub][1]:
            relation.append(sub_2)
    return res+lookup[id][0]

# 125.693 Binary Number with Alternating Bits
# Problem: Given a positive interger, check if its binary bits are 0 and 1 alternating. If two adjacent bits are same, return False
# Description: Use bit manipulation. Use "num&1" to get current bit, use "num>>1" to move to next bit. Maintain [prev] as previous bit, and compare with 
#              current bit get from "num&1"
# Time Complexity: O(logN)
# Space Complexity: O(1)
def hasAlternatingBits(n):
    prev, n = n&1, n>>1
    while n>0:
        if prev == n&1:
            return False
        prev, n = n&1, n>>1
    return True

# 126.696 Count Binary Substrings ============================================================ https://leetcode.com/problems/count-binary-substrings/
# Problem: Given a binary string [s] with 0s and 1s. Find the number of substring from [s] where the substrings have same amount of 0s and 1s, also 
#          all 0s in the substring are at one side of the string, and all 1s are at the other side. Return the number of substring that satisfy the conditions
#          Ex: 00110011 => "0011", "01", "10", "1100", "0011", "01". result = 6
# Description: Count the length consecutively substring of 0s and 1s. For example, "00110" will be [2 zeros, 2 ones, 1 zero] => [2, 2, 1]. Then look at
#              adjacent counts, they number of substring they can construct is "min(prev, cur)". For example "00110" will have counts [2, 2, 1], for the 
#              first pair min([2, 2]) = 2, they can construct 2 substrings, which are "0011" and "01". for the second pair min([2,1]) = 1, they can construct
#              a single substring, which is "10". Therefor, sum up the min of adjacent count will be the answer. 
#              Besides, we don't need to maintain the entire count array, since we just track previous and current count.
# Time Complexity: O(n)
# Space Complexity:O(1)
def countBinarySubstrings(s):
    cur, prev, res = 1, 0, 0
    for i in range(1, len(s)):
        if s[i] == s[i-1]:          # encount same char, increase current count
            cur+=1
        else:                       # encount different char
            res += min(prev, cur)
            prev = cur
            cur = 1
    return res + min(prev, cur)

# 127.697 Degree of an Array =============================================================https://leetcode.com/problems/degree-of-an-array/
# Problem: Given a non-empty integer array [nums]. We say the "degree" of the array is the frequency of the number that appeared the most of times. 
#          Find the shorted sub-array of [nums] with the same degree of [nums], return the length of the sub-array
#          ex: nums = [1,2,2,3,1,4,2], degree = 3, subarray = [2,2,3,1,4,2], result = 6
# Description: Miantain a dictionary [first] which track the index of first occurence of the element in [nums], maintain another dictionary [cnt],
#              which track the number of each element in [nums]. Maintain [degree] to track the maximum frequency of element. 
#              Iterate through [nums], if a number is occurred the first time, add it and its idndex into [first]. Update [cnt] of the number,
#              If cnt[n] > [degree], update [degree] = cnt[n], and [res] must be recauclated to be the distance between the index of first 
#              occurrence and current index "i - first[n] + 1". If cnt[n] == [degreee], then calculate the distance of current number, compare
#              with [res] and take the smaller one
# Time Complexity: O(n)
# Space Complexity: O(n)
def findShortestSubArray(nums):
    first, cnt, deg, res = {}, {}, 0, 0
    for n, i in enumerate(nums):
        first.setdefault(n, i)          # record index of first occurrence
        cnt[n] = cnt.get(n, 0)+1        # record count of an element
        # a higher degree is found
        if deg < cnt[n]:            
            deg = cnt[n]              
            res = i-cnt[n]+1            # previous [res] associate with lower [deg], must be overwritten
        # a new element with same degree is found
        elif deg == cnt[n]:
            res = min(res, i-cnt[n]+1)  # compare and find a shorter subarray
    return res

# 128.700 Search in a Binary Search Tree  ============================================== https://leetcode.com/problems/search-in-a-binary-search-tree/
# Problem: Given a [root] of a BST and an integer [val]. Find the node in BST that contains [val], and return the node including its subtree.
#          If the [val] is not presented in BST, return None
# Description: Use DFS comparing node.val with [val] and take the proper path to find the Node, return the node directly since its subtree
#              is linked with node
# Time Complexity: O(logN)
# Space Complexity: O(1)
def searchBST(root, val):
    cur = root
    while cur:
        if cur.val > val:
            cur = cur.left
        elif cur.val < val:
            cur = cur.right
        else:
            return cur

# 129.703 Kth Largest Element in a Stream ============================================== https://leetcode.com/problems/kth-largest-element-in-a-stream/
# Problem: Design a class to find kth largest element in a stream/list. Note it is kth largest element in sorted order, not kth distinct element
#          The class should be initialized with integer [k] representing [k]th largest, and an integer list [nums] represents initial list
#          "add(val)" is a method of the class, which insert [val] into the list, and return kth largest number after insertion. If the [stream] has
#          less than [k] element, return the smallest element
# Description: Use "heapq" module in Python to maintain a min-heap of size [k], where heap[-1] is the smallest element and heap[0] is the kth
#              largest element. Use "heapq.heappushpop(val)" which push(val) and then pop min value to maintain heap size
# Time Compexity: constructor O(n), add O(1)
import heapq
class KthLargest:
    def __init__(self, k, nums):
        self.k = k
        heapq.heapify(nums)                     # convert [nums] as min-heap (linear time)
        self.pool = nums
        while len(self.pool) > k:               # get rid of extra element, and keep [k] elements
            heapq.heappop(self.pool)
    def add(self, val):
        # add val to heap if there is empty space in heap
        if len(self.pool)<self.k:
            heapq.heappush(self.pool, val)
        # if no more empty space, and inserting [val] is larger than minimum value in [heap]
        elif val > self.pool[0]:
            heapq.heappushpop(self.pool, val)              # "heappushpop(val)" firstly push(val), then pop and return min
        return self.pool[0]                     # pool[0] always hold the kth largest number

# 130.705 Design HashSet
# Problem: Implement a HashSet of integers, with following methods
#          1) void add(key), insert "key" into HashSet in O(1)
#          2) boolean contains(key), check if the HashSet contains "key" in O(1)
#          3) void remove(key), Remove the "key" from HashSet in O(1), if "key" is not presented, do nothing
# Description: Use a dynamic list to store keys. If size of array exceeds 2/3 of capacity, expand capacity by 2 times, and reinsert "key"s into new list.
#              The average time complexity of inserting an element is still O(1). 
#              For "add(key)"" method, implement a hash method to find index to insert, where "insertPos = key%capacity". Keys will have "collision", 
#              use "double hashing", that use previous [insertPos] and hash it again with another hash function [insertPos] = (5*insertPos+1)%capacity,
#              keep looking for [insertPos] until a empty spot in list is found.
#              For "remove(key)", use the same "double hashing" mechanic to find the "key". If any "None" element is encountered, return and do nothing,
#              because the "key" is not presented. If the "key" is found, mark it as "deleted" instead of make it "None", because we don't want to stop
#              searching at unexpected spot.
#              For "contains(key)" method, Use the same strategy of "remove(key)", leverage "double hashing" to find the "key" or return False when encount
#              "None"
class MyHashMap:
    def __init__(self) -> None:
        self.capacity = 8
        self.size = 0
        self.data = [None]*self.capacity

    def hash_1(self, key):
        return key % self.capacity

    def hash_2(self, key):
        return (5*key+1) % self.capacity
    
    def add(self, key):
        # expand capacity if exceeds 2/3 of capacity
        if self.size >= 2/3*self.capacity:
            self.capacity *= 2
            newData = [None] * (self.capacity)
            for num in self.data:
                if num and num != "deleted":
                    pos = self.hash_1(num)
                    while newData[pos] is not None:
                        pos = self.hash_2(pos)
                    newData[pos] = num
            self.data = newData
        # find position to insert key 
        pos = self.hash_1(key)
        while self.data[pos]  is not None:
            if self.data[pos] == key:           # "key" is already in the hashset, no need to add it again
                return 
            if self.data[pos] == "deleted":      # "key" is added and deleted, add it at where it was deleted
                break
            pos = self.hash_2(pos)
        # insert key
        self.data[pos] = key    
        self.size += 1
    
    def remove(self, key):
        pos = self.hash_1(key)
        while self.data[pos]  is not None:
            if self.data[pos] == key:
                self.data[pos] = "deleted"
                self.size -= 1
                return
            pos = self.hash_2(pos) 

    def contains(self, key):
        pos = self.hash_1(key)
        while self.data[pos]  is not None:
            if self.data[pos] == key:
                return True
            pos = self.hash_2(pos)
        return False

# 131.706 Design Hashmap ============================================= https://leetcode.com/problems/design-hashmap/
# Problem: Implement a Hashmap class, where key and value are both positive integers. The methods of the class are following
#          1) void put(key, value). Insert (key, value) into hashmap. If "key" is presented in hashmap, update the "value"
#          2) int get(key). Return the "value" if "key" is presented in hashmap. Return -1 if "key" not present
#          3) void remove(key). Remove the (key, value) pair if "key" is presented, otherwise do nothing 
# Description: Use Node chaining. Initial an array with fixed size, and use a simple hash function, [pos = key%size] to insert
#              pair. If the [pos] is already occupied, use chaining to chain pairs as linked list. Therefor, each pair is an 
#              instance of ListNode, with "key", "value" and "next" to refer to next ListNode
class MyHashMap_2:
    def __init__(self) -> None:
        self.capacity = 1000
        self.data = [None]*self.capacity

    def hash(self, key):
        return key%self.capacity

    def put(self, key, value):
        pos = self.hash(key)
        # pos is not occupied, insert new node directly
        if self.data[pos] is None:
            self.data[pos] = ListNode(key, value)
        # pos is occupied
        else:
            cur = self.data[pos]
            while cur is not None:
                # key presents, update its value and return
                if cur.key == key:
                    cur.value = value
                    return
                # hit end of chain, insert new node at the tail
                elif cur.next == None:
                    cur.next = ListNode(key, value)
                    return
                cur = cur.next

    def get(self, key):
        pos = self.hash(key)
        cur = self.data[pos]
        while cur:
            if cur.key == key:
                return cur.value
            cur = cur.next
        return -1     

    def remove(self, key):
        pos = self.hash(key)
        cur = pre = self.data[pos]
        # pos is empty, no key presents
        if cur is None: return
        # first node in pos matches key, remove it
        elif cur.key == key:
            self.data[pos] = cur.next
        # look into chain and find key
        while cur is not None:
            if cur.key == key:
                pre.next = cur.next
            else:
                pre = cur
                cur = cur.next
            
# 132.709 To lower case ================================================== https://leetcode.com/problems/to-lower-case/
# Problem: Implement "toLowerCase(string)" that convert all uppercase character to lowercase, and return the new string
#          in lowercase. 
# Description: use Python function ord(char) to get ascii value of "char" and chr(num) to convert an ascii value to character.
#              The range of uppercase english character is ascii => [65, 90], where ord('A') = 32, ord('Z') = 90
#              and difference between lowercase and uppercase is 32, ord('a') - ord('A') = 32. Thus, iterate through the [str]
#              if a [char] is between 65 and 90, add 32 to it, convert it back to char.
# Time Complexity: O(n)
def toLowerCase(str):
    res = ""
    for c in str:
        if 65<=ord(c)<=90:
            res += chr(ord(c)+32)
        else:
            res += c
    return res

# 132.717 one-bit and two-bits characters ==================================== https://leetcode.com/problems/1-bit-and-2-bit-characters/
# Problem: Assuem a one-bit character is represented by "0", a two-bits character can be represented by "10" or "11". Given a list of 
#          0 and 1 bits, determine if the right-most character can be a one-bit character
#          Example: [1,1,0,1,0] => [(11),(0),(10)] => False
#                   [1,0,1,0,0] => [(10),(10),(0)] => True
# Description: Iterate through the list, if encount "1", it must be paired with the following element. If encounter "0" it must be by
#              itself. Maintain the [index], increase [index] by 2 if encounter "1", increase [index] by 1 if encounter "0". 
#              At the end, if [index] >= len(bits), meaning the last character is two-bits. If [index]<len(bits) and the last element
#              is "0", the last character is one-bit
# Time Complexity: O(n)
def isOneBitCharacter(bits):
    if bits[-1] == 1: return False          # if last element is 1, the last character must be "11" two-bits
    i = 0
    while i < len(bits):
        if bits[i] == 1:                    # pair the following element with "1"
            i += 2
        else:                                 
            i += 1
    return i<len(bits)

# 133.720 Longest Word in Dictionary =========================================== https://leetcode.com/problems/longest-word-in-dictionary/
# Problem: Given a list of strings [words], find the longest word in [words] that can be built one character at a time by other words in 
#          [words]. If there is more than one possible answer, return the smallest lexicographical order
#          EX: [a, b, ap, app, bl, bla, appl, apple, blac, black] => apple
#              both "apple" and "black" can be built from other words by adding one character at a time
#              "apple" has smaller lexicographical order, so "apple" is the answer
# Description: Maintain a set [pool], and sort [words] by word length. Iterate though [words] start from shortest word first, if the current
#              word can be built by adding one character on an existing word in [pool], add current word to [pool]. After iteration, the words 
#              in [pool] describe the "path" to construct to longest word. In order to handle multiple possible caes, sort the [pool] by 
#              lexicographical order, then find the max() of longest length. This returns the longest words with smallest lexicographic
# Time Complexity: O(nlogn)
# Space Complexity: O(n)
def longestWord(words):
    pool = set([""])
    words = sorted(words, key=lambda word: len(word))
    for word in words:
        if word[:-1] in pool:
            pool.add(word)
    return max(sorted(pool), key=lambda word: len(word))

# 134.724 Find Pivot index =========================================================== https://leetcode.com/problems/find-pivot-index/
# Problem: Given an integer list, find a pivot index that the sum of elements to the left of the pivot(not include pivot) is equals to
#          the sum of elements to the right of the pivot(not include pivot). Return the pivot index if exists such pivot. Return -1 
#          if the pivot doesn't exist
# Description: Let the pivot start from index 0, meaning [sumLeft]=0 and [sumRight]=sum(nums). Compare [sumLeft] and [sumRight], if 
#              they are not equal, move pivot to the right. Take away the current element from [rightSum] and add it to [leftSum].
#              Keep moving the pivot until [rightSum] equals to [leftSum]. 
# Time Complexity: O(n)
# Space Complexity: O(1)
def pivotIndex(nums):
    leftSum, rightSum = 0, sum(nums)
    for i, n in enumerate(nums):
        rightSum -= n
        if leftSum == rightSum:
            return i
        leftSum += n
    return -1

# 135.728 Self Dividing Numbers ========================================================= https://leetcode.com/problems/self-dividing-numbers/
# Problem: A self-divid number is an integer that is divisible by every digit of the number, and there is no "zero" in the number
#          Ex: 128 is a self-divid number, 128%1==0, 128%2==0, and 128%8==0.
#          Given a lower and upper bound, return a list of self-divid number within the bounds
# Description: Brute force, try every number between bound. For each number [x], use "x%10" to get last digit. Check if the digit is "zero" or 
#              can divid the original number. If not divisible break the loop, use "x//=10" to get next digit. Check if [x] == 0 after loop,
#              if it x==0, meaning every digit of [x] divid original number, add it to result list
# Time Complexity: O(nm), n=right-left, m=avg(number of digits in a number)
def selfDividingNumbers(left, right):
    res = []
    for x in range(left, right+1):
        y = x
        while y>0:
            if (y%10==0) or (x%(y%10) != 0):        # digit can't be zero, digit shuold divid original number
                break
            y //= 10
        if y==0:                            # y==0 means every digits of y can divid [x]
            res.append(x)
    return res

# 136.733 Flood Fill ===================================================================== https://leetcode.com/problems/flood-fill/
# Problem: Given a 2D list with integers [image], each integer represent a pixel value (from 0 to 65535). Given a coordinate (sr, sc)
#          as the starting point of flood fill, the color at starting point image[sr][sc] is the original color. Use 4-direction 
#          flood fill to fill original color to [newColor]. Return the new iamge
# Description: DFS start at (sr, sc), there are four recursive calls for each (row, col). (row-1,col), (row,col-1), (row+1,col), (row,col+1)
#              Check (row, col) is in bound, check if current pixel is oldColor, and is not newColor, then perform the filling
# Time Complexity: O(n)
def floodFill(image: List[List[int]], sr, sc, newColor) -> List[List[int]]:
    floodFill_rec(image, sr, sc, image[sr][sc], newColor)
    return image
        
def floodFill_rec(image, row, col, oldColor, newColor):
    if row>=0 and row<len(image) and col>=0 and col<len(image[row]):    # check boundary
        if image[row][col]==oldColor and image[row][col]!=newColor:     # check current pixel is oldColor and is not newColor
            image[row][col] = newColor
            # recursively 4-direction fill
            floodFill_rec(image, row-1, col, oldColor, newColor)
            floodFill_rec(image, row, col-1, oldColor, newColor)
            floodFill_rec(image, row+1, col, oldColor, newColor)
            floodFill_rec(image, row, col+1, oldColor, newColor)

# 137.744 Find Smallest Letter Greater Than Target ======================================= https://leetcode.com/problems/find-smallest-letter-greater-than-target/
# Problem: Given a sorted list of characters [letters] with lowercase characters, and Given a lowercase character [target]. Find the first character in [letters]
#          that comes after [target]. 
#          Ex: target = d, letter = [c, f, j]. "f" is the first letter comes after "d" => "f"
#          Characters in [letters] are wrap around, if no character in [letters] comes after [target], then the first character is the answer.
#          Ex: target = z, letter = [a, b, c]. No character comes after 'z', return first character in [letters] => 'a'
# Description: Because character in [letters] are sorted, use "bisect.bisect_right" to find the right insertion position of [target]. If [pos] is at the end of 
#              [letters], meaning [target] comes after every character in [letters], return letters[0]. If [pos] is found inside of [letters] return letters[pos]
# Time Complexity: O(logN)
from bisect import bisect_right
def nextGreatestLetter(letters, target):
    pos = bisect_right(letters, target)
    return letters[0] if pos==len(letters) else letters[pos]

# 138.746 Min cost Climbing stairs =========================================================== https://leetcode.com/problems/min-cost-climbing-stairs/
# Problem: Given a non-negative integer list [cost], each element represent the cost to climb a step. You need to start from index [0] or [1] and climb to
#          the top("top" is the position after the last element). Find and return the minimum cost to climb to the top
# Description: Dynamic programming, the cost to climb to [i] step can be calculated by 1) cost of climbing to [i-1] plus cost of [i] or 2) cost of climing 
#              to [i-2] plus cost of [i]. Keep tracking [i-2] as [first] and [i-1] as [second], calculate cost to climb to current index [cur] and reassign
#              [first] and [second]. Also, need to "zero" to the end of [cost], to represent the top
# Time Complexity: O(n)
# Space Complexity: O(1)
def minCostClimbingStairs(cost: List[int]) -> int:
    if len(cost)<=2:
        return min(cost)
    cost.append(0)          # add 0 to the end of [cost] to represent top
    first = cost[0]
    second = cost[1]
    for i in range(2, len(cost)):
        cur = min(first, second)+cost[i]
        first, second = second, cur
    return second

# 139.747 Largest Number At Least Twice of Others =============================================== https://leetcode.com/problems/largest-number-at-least-twice-of-others/
# Problem: Given an integer list [nums]. If the largest number in [nums] is more than twice larger than any other elements in the list, return the index
#          of the largest number. If no number in [nums] is more than twice larger than any other elements, return -1
# Description: Find the largest number, its index and second largest number in the list. Maintain largest number [fisrt], its index [index] and second 
#              largest number [second]. Iterate through [nums], if a number [n]  is larger than [first], then [second] = [fisrt] and [first] = [n]. 
#              If a number [n] is smaller than [first] but larger than [second], then [second] = [n]. Compare [first] and [second], return accordingly
# Time Complexity:O(n)
# Space Complexity: O(1)
def dominantIndex(nums):
    index = -1
    first = second = float("-inf")
    for i, n in enumerate(nums):
        if n > first:
            second = first
            first = n
            index = i
        elif n>second and n<first:
            second = n
    return index if first>=second*2 else -1

# 140.748 Shortest Completing Word ============================================================= https://leetcode.com/problems/shortest-completing-word/
# Problem: A completing word is a word that contains all letter in [licensePlate], ignore numbers and spaces. Given a string list [words], find the shortest
#          word that case-insensitively contains the completing word. If there are multiple shortest completing words, return the first one that appears
#          Ex: licensePlate="1s3 PSt", completingWord="psst". Look into [words] and find the shortest word that contains all character in completingWord
# Description: Maintain a dictionary [plate] that contains characters and their counts of [licensePlate]. Iterate though every word in [words], construct
#              a dictionary [cnt] with characters and their counts of each word. Compare [plate] and each [cnt], a valid [word] must have a [cnt] that 
#              contains all characters in [plate]. Then use min( key=len) to get the shortest one
# Time Complexity: O(nm), n=len(words), m=length of each word  
def shortestCompletingWord(licensePlate: str, words: List[str]) -> str:
    # construct alphabets count for [licensePlate]
    plate = {}
    for c in licensePlate.lower():      # convert to lowercase for case-insensitive
        if c.isalpha():
            plate[c] = plate.get(c, 0)+1
    res = []
    for word in words:
        # construct alphabets count for every [word]
        cnt, valid = {}, True
        for c in word.lower():          # convert to lowercase for case-insensitive
            if c.isalpha():
                cnt[c] = cnt.get(c, 0)+1
        for k, v in plate.items():      # check if [word] is valid
            if cnt.get(k, 0) < v:
                valid = False
                break
        if valid:
            res.append(word)
    return min(res, key=len)            # return the shortest one

# Shorter version with Counter
# Description: use "Counter()" on [licensePlate] after filter out non-alphabets to get [plate]. Use "Counter()" on every [word], and use "&" on 
#              "Counter(word) & plate" to get "intersection" of them, if the intersection is same as [plate], means [word] contains completing word
from collections import Counter
def shortestCompletingWord_2(licensePlate: str, words: List[str]) -> str:
    plate = Counter([c for c in licensePlate.lower() if c.isalpha()])
    return min([word for word in words if (Counter(word) & plate) == plate], key=len)

# 141.762 Prime Number of Set Bits in Binary Representation ========== https://leetcode.com/problems/prime-number-of-set-bits-in-binary-representation/
# Problem: Given a integer range [L, R]. For number between the range, if there are prime number of "1"s in the binary representation, we say the number
#          has a prime number of "set bits". Find and count the number in range [L, R] that have a prime number of set bits. 
#          Note: both [L] and [R] are in range [1, 10^6]. [R-L] <= 10^4
# Description: The range of [L] and [R] are at most 10^6, the numbe can have at most 20 bits. Thus, a number can have at most 20 "1"s. The prime number
#              less than 20 are {2,3,5,7,11,13,17,19}. Iterate through every number between [L] and [R], convert them into binary string, and count number
#              of "1"s. And compare number of "1"s with the prime number in [prime_set]
# Time Complexity: O(NlogM), N=number of integer within [L] and [R], M=the acutal integer
# Space complexity: O(1)
def countPrimeSetBits(L: int, R: int) -> int:
    primes = {2,3,5,7,11,13,17,19}
    res = 0
    for i in range(L, R+1):
        cnt = bin(i).count('1')
        if cnt in primes:
            res += 1
    return res

# 142.766 Toeplitz Matrix ================================================================== https://leetcode.com/problems/toeplitz-matrix/
# Problem: Given a m by n [matrix], check if the [matrix] is "Teoplitz". "Teoplitz" matrix is a matrix that every diagonal from top-left to
#          button-right has same elements
#          Ex: 1 2 3 4 ==> every diagonal has same elements, 9, 5, 1, 2, 3, 4
#              5 1 2 3
#              9 5 1 2 
# Description: Iterate through elements in matrix, for each element check if the button-left element is equal, matrix[i][j] == matrix[i+1][j+1].
#              If any of elements failed the condition, return False, otherwise return True at the end of iteration
# Time complexity: O(n)
def isToeplitzMatrix(matrix: List[List[int]]) -> bool:
    for i in range(len(matrix)-1):
        for j in range(len(matrix[i])-1):
            if matrix[i][j] == matrix[i-1][j-1]:
                return False
    return False

# 143.771 Jewels and Stones =============================================================== https://leetcode.com/problems/jewels-and-stones/
# Problem: Given a String [jewels], every character in [jewels] is a type of jewel. Given a String [stones], each character in [stones] is 
#          a stone, characters in [jewels] and [stones] are case sensitive. Return the number of jewels in [stones].
# Description: Put [jewels] into a "set()". Then iterate every character in [stones], check if they are in [jewels_set], maintain and
#              count number of jewels
# Time complexity:
def numJewelsInStones(jewels: str, stones: str) -> int:
    jewels_set = set(jewels)
    res = 0
    for c in stones:
        if c in jewels_set:
            res += 1
    return res


# 144.783 Minimum Distance Between BST Nodes ============================ https://leetcode.com/problems/minimum-distance-between-bst-nodes/
# Problem: Given a root of integer BST. return the minimum difference between any two different nodes in the tree
#          The problem is same as "94.530 Minimum Absolute Difference in BST"
# Description: If In-Order traverse a BST, the order is ascending order, so that the closest value can be found by iterate though in-order.
#              Maintain the [res] as minimum difference and [pre] as previous value in ascending order. Take different between current 
#              value and [pre], then compare with [res] to get smaller difference. [pre] starts with "-inf" assuming None node has smallest
#              value. [res] starts with "inf" as we need to find min, and it is updataed when smaller difference is found
# Time complexity: O(n)
# Space complexity: O(h), h=height of tree
class MinDiffInBST:
    pre = float("-inf")
    def solve(self, root):
        if root is None:
            return 
        self.solve(root.left)
        self.res = min(self.res, root.val-self.pre)
        self.pre = root.val
        self.solve(root.right)
        return self.res

# 145.783 Rotated Digits ================================================================= https://leetcode.com/problems/rotated-digits/
# Problem: Assume a digit can be rotate 180 degree to become another digit, where 0,1 and 8 rotate to themselves, 2 and 5 rotate to each
#          other, 6 and 9 rotate to each other, and other number can not be rotated. A number is call a "good number" if every digit can
#          be rotated, and the new number after rotating is different from original number. Given an integer [N], return count of "good
#          number"s between 1 to [N] inclusive.
# Description: If a number contains 2, 5, 6 or 9, and doesn't contain 3, 4 or 7. Then it is a good number. Because, 3, 4, or 7 can't be 
#              rotated, and 2, 5, 6, 9 can be rotated to become other number. If a number only contains number 1, 0 or 8, then it is not
#              good number
# Time complexity: O(NlogN), total [N] numbers, every number take "log(N) of base 10", since extract digits of [N]
def rotatedDigits(N: int) -> int:
    set1 = set([1, 8, 0])
    set2 = set([1, 8, 0, 2, 5, 9, 6])
    res = 0
    for n in range(1, N+1):
        s = set([int(i) for i in str(n)])               # convert digits of [n] to set [s]
        if s.issubset(set2) and not s.issubset(set1):   # [s] must contain {2,5,6,9}, so that is subset of [set2], is not subset of [set1]
            res += 1
    return res

# 146.796 Rotating String =========================================================== https://leetcode.com/problems/rotate-string/
# Problem: A "shift" on string is move every character one position to left, and leftmost character moves to rightmost. "abcd" shift to "bcda"
#          Given two strings [A] and [B], determine if [A] can become [B] after certain numbers of shift.
# Description: Concate [A] by itself to create a new string [A+A]. If [B] is a shift of [A], [B] can be found in [A+A]. 
#              Ex: A = "abcde", B = "cdeba" 
#                  A+A = "abcdeabcde", [B] is found at index 2, because [A] is shifted twice to get [B]
# Time complexity: O(nm), search [B] in [A+A] will take n*m, where "n" and "m" are length of [A] and [B]
#                  By apply Knuth-Morris-Prattern matching alg, can be reduced to O(n)
def rotateString(A: str, B: str) -> bool:
    return len(A) == len(B) and B in A+A

# 147.804 Unique Morse Code Words ================================================== https://leetcode.com/problems/unique-morse-code-words/
# Problem: Different words can be translated into same Morse Code, if space between each character is ignored. For example, "gin" and "zen"
#          can be translated into "--...-.". Given a list of [words], and proivide the morse code for each alphabet from a to z is provided below
#          [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
#          Find how many unique morse codes can be translated from [words]
# Description: Maintain a list of Morse code from a to z. Iterate through [words]. For each [word], iterate each character in [word] and 
#              convert character to ASCII number and minue by "ord('a')" to get its index in [dic] list. Concate morse code of each character
#              to generate morse code for each [word]. Store morse code of words in a "set()", and return the size of "set()" at the end
# Time complexity: O(nm) n=number of [word] in [words], m=length of each [word]
def uniqueMorseRepresentations(self, words: List[str]) -> int:
    dic = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
    res = set()
    for word in words:
        morse = "".join(dic[ord(c)-ord('a')] for c in word)
        res.add(morse)
    return len(res)

# 148.806 Number of Lines To Write String ========================================= https://leetcode.com/problems/number-of-lines-to-write-string/
# Problem: Given a array [widths] with 26 integers, each integer represent the pixel width of lowercase English character from "a" to "z" respectively. 
#          Given a string [s] of lowercase letters, write the string across lines, where each line holds up to 100 pixels. Keep writing letters onto
#          a line, and start a newline when a letter is about to exceed 100 pixels. Return a list [res], where "res[0]" is the number of lines and 
#          "res[1]" is the pixel width written in the last line
# Description: Iterate through [s], width of each letter can be retrieved from [widths] list by "widths[ord(letter)-ord("a")]". Track the total width
#              of current line. When exceeding 100 pixels, increase line count "res[0]", and reset width of current line res[1]. After iteration, 
#              return res[0] and pixel in current line res[1]
# Time complexity: O(n)
def numberOfLines(widths: List[int], s: str) -> List[int]:
    res = [1, 0]
    for c in s:
        w = widths[ord(c)-97]           # ord("a") = 97
        res[1] += w
        if res[1] > 100:                # exceeding 100:
            res[1] = w                      # reset pixel of current line
            res[0] += 1                     # increase line count
    return res
            
# 149.811 Subdomain Visit Count ================================================================= https://leetcode.com/problems/subdomain-visit-count/
# Problem: A website domain like "www.leetcode.com" consists subdomains that are separated by dot ".". At top level, we have "com", next level
#          is "leetcode", and lowest level is "www". When visiting "www.leetcode.com", we also visited "leetcode.com" and "com" subdomains.
#          Given a list of string, where each string contain an integer number and a domain, represent the domain is visitied that number of 
#          times. Return a list of string, shows how many time of each domain is visited
#          Ex: ["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
#              "com" is visited 900+50+1, "org" is visited 5, "mail.com" is visited 900+1, "yahoo.com" is visited 50, etc
#              => ["901 mail.com","50 yahoo.com","900 google.mail.com","5 wiki.org","5 org","1 intel.mail.com","951 com"]
# Description: Maintain a "Counter()" object to count each domain. For each element in [dpdomains], split count and each domain to array [domains]. 
#              Extract integer number as [visited], construct all possoble subdomains by iterating from index 1 to len(domains), where subdomains are 
#              "."join(domains[i:]). Record number of visits to subdomains in Counter(). At the end, convert Counter() object to list and return
# Time complexity: O(n)
from collections import Counter
def subdomainVisits(cpdomains: List[str]) -> List[str]:
    record = Counter()
    for ele in cpdomains:
        domains = ele.replace(".", " ").split()             # split [cpdomains] to list
        visits = int(domains[0])                                # get visits
        for i in range(1, len(domains)):                    # construct subdomains and record in Counter
            record[".".join(domains[i:])] += visits
    res = []
    for k, v in record.items():                         # convert Counter to list
        res.append(str(v)+ " "+k)
    return res

# 150.812 Largest Triangle Area =========================================================== https://leetcode.com/problems/largest-triangle-area/
# Problem: Given a 2D list [points], where each element represent a point points[i] = [xi, yi]. Pick three points from [points] to form a triangle
#          so that the area of triangle is largest. Return the largest area
# Description: Brutal Force. Iterate all possible combination of picking three points from [points], calculate their area and find maximum.
#              For each combination, the area can be calculated as follow. Assume we picked three points, A(xa, ya), B(xb, yb) and C(xc, yc). 
#              And add a point O(xo, yo), where xo=xa, and yo=yb. So that AO and OB are perpenticular. The area of ABC is splited into three
#              parts AOB+AOC+COB. Where AOB = (xb-xa)(ya-yb)/2, AOC = (xa-xc)(ya-yb)/2, COB = (xb-xa)(yb-yc)/2. 
# Time complexity: O(N^3), N=len(points)
def largestTriangleArea(points: List[List[int]]) -> float:
    res = 0
    # iterate all combination of three points
    for a in points:
        for b in points:
            for c in points:
                area = (a[0]-b[0])*(a[1]-c[1])+(c[0]-a[0])*(a[1]-c[1])+(c[0]-a[0])*(c[1]-b[1])      # use formuale to get area of ABC
                res = max(area, res)
    return res/2

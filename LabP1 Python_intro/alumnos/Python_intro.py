
# coding: utf-8

# # A brief tutorial of basic python

# From the wikipedia: "Python is a widely used general-purpose, high-level programming language. Its design philosophy emphasizes code readability, and its syntax allows programmers to express concepts in fewer lines of code than would be possible in languages such as C++ or Java. The language provides constructs intended to enable clear programs on both a small and large scale."
# 
# Through this tutorial, students will learn some basic characteristics of the Python programming language, that will be useful for working with corpuses of text data.

# ## 1. Introduction to Strings

# Among the different native python types, we will focus on strings, since they will be the core type that we will recur to represent text. Essentially, a string is just a concatenation of characters.

# In[ ]:

str1 = '"Hola" is how we say "hello" in Spanish.'
str2 = "Strings can also be defined with quotes; try to be sistematic."


# It is easy to check the type of a variable with the type() command:

# In[ ]:

print str1
print type(str1)
print type(3)
print type(3.)


# The following commands implement some common operations with strings in Python. Have a look at them, and try to deduce what the result of each operation will be. Then, execute the commands and check what are the actual results.

# In[ ]:

print str1[0:5]


# In[ ]:

print str1+str2


# In[ ]:

print str1.lower()


# In[ ]:

print str1.upper()


# In[ ]:

print len(str1)


# In[ ]:

print str1.replace('h','H')


# In[ ]:

str3 = 'This is a question'
str3 = str3.replace('i','o')
str3 = str3.lower()
print str3[0:3]


# It is interesting to notice the difference in the use of commands 'lower' and 'len'. Python is an object-oriented language, and str1 is an instance of the Python class 'string'. Then, str1.lower() invokes the method lower() of the class string to which object str1 belongs, while len(str1) or type(str1) imply the use of external methods, not belonging to the class string. In any case, we will not pay (much) attention to these issues during the session.

# Finally, we remark that there exist special characters that require special consideration. Apart from language-oriented characters or special symbols (e.g., \euro), the following characters are commonly used to denote carriage return and the start of new lines

# In[ ]:

print 'This is just a carriage return symbol.\r Second line will start on top of the first line.'


# In[ ]:

print 'If you wish to start a new line,\r\nthe line feed character should also be used.'


# In[ ]:

print 'But note that most applications are tolerant\nto the use of \'line feed\' only.'


# ## 2. Working with Python lists

# Python lists are containers that hold a number of other objects, in a given order. To create a list, just put different comma-separated values between square brackets

# In[ ]:

list1 = ['student', 'teacher', 1997, 2000]
print list1
list2 = [1, 2, 3, 4, 5 ]
print list2
list3 = ["a", "b", "c", "d"]
print list3


# To check the value of a list element, indicate between brackets the index (or indices) to obtain the value (or values) at that position (positions).
# 
# Run the code fragment below, and try to guess what the output of each command will be.
# 
# Note: Python indexing starts from 0!!!!

# In[ ]:

print list1[0]
print list2[2:4]
print list3[-1]


# To add elements in a list you can use the method append() and to remove them the method remove()

# In[ ]:

list1 = ['student', 'teacher', 1997, 2000]
list1.append(3)
print list1
list1.remove('teacher')
print list1


# Other useful functions are:
#    
#     len(list): Gives the number of elements in a list.    
#     max(list): Returns item from the list with max value.  
#     min(list): Returns item from the list with min value.

# In[ ]:

list2 = [1, 2, 3, 4, 5 ]
print len(list2)
print max(list2)
print min(list2)


# ## 3. Flow control (with 'for' and 'if')

# As in other programming languages, python offers mechanisms to loop through a piece of code several times, or for conditionally executing a code fragment when certain conditions are satisfied. 
# 
# For conditional execution, you can use the _if_, _elif_ and _else_ statements.
# 
# Try to play with the following example:

# In[ ]:

x = int(raw_input("Please enter an integer: "))
if x < 0:
    x = 0
    print 'Negative changed to zero'
elif x == 0:
    print 'Zero'
elif x == 1:
    print 'Single'
else:
    print 'More'


# ### Indentation
# The above fragment, allows us also to discuss some important characteristics of the Python language syntaxis:
# 
# * Unlike other languages, Python does not require to use the 'end' keyword to indicate that a given code fragment finishes. Instead, Python recurs to **indentation**.
# 
# * Indentation in Python is mandatory. A block is composed by statatements indected at same level and if it constains a nested block it is simply indented further to the right.
# * As a convention each indentation consists of 4 spaces (for each level of indentation).
# 
# * The condition lines conclude with ':', which are then followed by the indented blocks that will be executed only when the indicated conditions are satisfied.
#    
# 
# 
# The statement _for_ lets you iterate over the items of any sequence (a list or a string), in the order that they appear in the sequence

# In[ ]:

words = ['cat', 'window', 'open-course']
for w in words:
     print w, len(w)


# In combination with enumerate(), you can iterate over the elementes of the sequence and have a counter over them

# In[ ]:

words = ['cat', 'window', 'open-course']
for (i, w) in enumerate(words):
     print 'element ' + str(i) + ' is ' + w


# ## 4. Variables and assignments

# In python the equal "=" sign in the assignment shouldn't be seen as "is equal to". It should be "read" or interpreted as "is set to". Let's see an example:

# In[ ]:

x = 42
y = x
y = 50
print x 
print y


# The first two lines do not seem problematic. But when y is set to 50, what will happen to the value of x? C programmers will assume that x will be changed to 50 as well, because we said before that y "points" to the location of x. But this is not a C-pointer. Because x and y will not share the same value anymore, y gets his or her own memory location, containing 50 and x sticks to 42.
# 
# If you are not a C programmer, the observable results of the assignments answer our expectations. But it can be problematic, if we copy mutable objects like lists and dictionaries.
# 
# Python creates real copies only if it has to, i.e. if the user, the programmer, **explicitly demands it**. 
# 
# Let's see a couple of examples:

# In[ ]:

colors1 = ["red", "green"]
colors2 = colors1
colors2 = ["rouge", "vert"]
print colors1


# Ok. This is what we expected, _colors1_ is keeping its own values. 
# Let's change one element of _colors2_ now:

# In[ ]:

colors1 = ["red", "green"]
colors2 = colors1
colors2[1] = "blue"
print colors1


# Ouch! That wasn't expected.
# 
# The explanation is that there has been no new assignment to _colors2_, _colors2_ still points to _colors1_. Only one of its elements, and consequently an element of _colors1_ has been changed. 
# 
# 
# It is possible to completely copy shallow list structures with the slice operator without having any of the side effects, which we have described above. But that will be a problem when having nested lists. In that case you should use import the module _copy_.
# 

# In[ ]:

list1 = ['a','b','c','d']
list2 = list1[:]
list2[1] = 'x'
print list2
print list1
['a', 'b', 'c', 'd']
list3 = ['a','b',['ab','ba']]
list4 = list3[:]
list4[0] = 'c'
list4[2][1] = 'd'
print(list3)


# Conclusion: Be very careful with list or dictionaries copying.

# ## 5. Functions, arguments and scopes
# In python a function is defined using the keyword _def_. As usual, they can take aguments and return results. Let's see an example:

# In[ ]:

def my_sqrt(number):
    """Computes the square root of a number."""
    return number ** (0.5) #  In python ** is exponentiation (^ in other languages) 

x = my_sqrt(2)
print x


# As we said, you must define a function using _def_, then the name of the function and in brackets ( ) the list of arguments of the function. The function will not return anything unless you specify it with a _return_ statement. 
# The expresion under the name of the function in triple quotes is a _Documentation string_ or _DOCSTRING_ and, as expected, it is used to document your code. For example, it is printed if you type the help command:

# In[ ]:

help(my_sqrt)


# Another interesting feature of python is that you can give default values to arguments in a function. For example, in the following code, when the second argument is not used during the call its value is 2.

# In[ ]:

def nth_root(base, exp=2):
    """Computes the nth root of a number."""
    return base ** (1.0/exp) #  In python ** is exponentiation (^ in other languages) 

print nth_root(10000)
print nth_root(10000,4)


# One tricky feature in python is how it evaluates the arguments that you pass to a function call. The most common evaluation strategies when passing arguments to a function have been _call-by-value_ and _call-by-reference_. Python uses a mixture of these two, which is known as "Call-by-Object", sometimes also called "Call by Object Reference" or "Call by Sharing". Let's see it with an example:

# In[ ]:

def add_square_to_list(x, my_list, dummy_list):
    x = x ** 2
    my_list.append(x)
    dummy_list = ["I", "am", "not", "a" , "dummy", "list"]
    
x = 5
my_list =[4, 9, 16]
dummy_list = ["I", "am", "a" , "dummy", "list"]

add_square_to_list(x, my_list, dummy_list)
print x
print my_list
print dummy_list


# If you pass **immutable** arguments like integers, strings or tuples to a function, the passing acts like call-by-value. The object reference is passed to the function parameters. They can't be changed within the function, because they can't be changed at all, i.e. they are immutable. 
# 
# It's different, if we pass **mutable** arguments. They are also passed by object reference, but they can be changed in place in the function. If we pass a list to a function, we have to consider two cases: Elements of a list can be changed in place, i.e. the list will be changed even in the caller's scope. If a new list is assigned to the name, the old list will not be affected, i.e. the list in the caller's scope will remain untouched.
# 
# So, be careful when modifying lists inside functions, and its side effects.

# ## 6. File input and output operations

# First of all, you need to open a file with the open() function (if it does not exist, it creates it). 

# In[ ]:

f = open('workfile', 'w')


# The first argument is a string containing the filename. The second argument defines the mode in which the file will be used:
# 
#     'r' : only to be read,
#     'w' : for only writing (an existing file with the same name would be erased),
#     'a' : the file is opened for appending; any data written to the file is automatically appended to the end. 
#     'r+': opens the file for both reading and writing. 
# 
# If the mode argument is not included, 'r' will be assumed.

# Use f.write(string) to write  the contents of a string to the file.  When you are done, do not forget to close the file:

# In[ ]:

f.write('This is a test\n with 2 lines')
f.close()


# To read the content of a file, use the function f.read():

# In[ ]:

f2 = open('workfile', 'r')
text=f2.read()
f2.close()
print text


# You can also read line by line from the file identifier

# In[ ]:

f2 = open('workfile', 'r')
for line in f2:
    print line

f2.close()


# ## 7. Modules import

# Python lets you define modules which are files consisting of Python code. A module can define functions, classes and variables.
# 
# Most Python distributions already include the most popular modules with predefined libraries which make our programmer lifes easier. Some well-known libraries are: time, sys, os, numpy, ...
#     
# There are several ways to import a library:

# 1) Import all the contents of the library: import lib_name
# 
# Note: You have to call these methods as part of the library

# In[ ]:

import time
print time.time()  # returns the current processor time in seconds
time.sleep(2) # suspends execution for the given number of seconds
print time.time() # returns the current processor time in seconds again!!!


# 2) Define a short name to use the library: import lib_name as lib
# 

# In[ ]:

import time as t
print t.time()


# 3) Import only some elements of the library 
# 
# Note: now you have to use the methods directly

# In[ ]:

from time import time, sleep
print time() 


# ## 8. Exercise
# Program a function _primes_ that returns a list containing the _N_ first prime numbers. Then call that function with the value _1000_ and save this list in a _.txt_ file with one number per line. 

# In[ ]:




# In[ ]:




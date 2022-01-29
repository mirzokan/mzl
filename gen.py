'''
General Purpose Tools
'''

import re


def sdir(obj, sunder=False):
    '''
    Modification of the dir function to detail special and object
    specific callables and attributes. Prints out Callables and 
    Attributes.

    Args:
        obj: An object to instpect
        sunder: Boolean, False shows non-undered attributes,
                True includes single-under attributes
    '''

    print(type(obj))
    if sunder:
        unders = [x for x in dir(obj) if re.match("^_(?!_)", x)]
        callables = [x for x in unders if callable(getattr(obj, x))]
        attribs = [x for x in unders if x not in callables]
        print("--Special--")
        print("Callables:")
        print(callables)
        print("Attributes:")
        print(attribs)
    
    regs = [x for x in dir(obj) if re.match("^[^_]", x)]
    callables = [x for x in regs if callable(getattr(obj, x))]
    attribs = [x for x in regs if x not in callables]
    print("")
    print("--Regular--")
    print("Callables:")
    print(callables)
    print("Attributes:")
    print(attribs)


def subl(a, b):
    """Subtract list b from list a.
    
    Args:
        a (list): List from which to subtract
        b (list): List which to subtract
    
    Returns:
        list: List that results from the subtraction
    """
    return [x for x in a if x not in b]

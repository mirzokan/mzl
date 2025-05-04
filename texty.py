'''
Tools for working with text data
'''


def parse_replace(input_string, start_delim="(", end_delim=")",
                  replacement="", level=1, start_alt=None, end_alt=None,
                  strict_eos=False):
    '''
    Parse a string using a set of delimiters and replace said delimiters
    with something else only at a specified level of nesting. Useful 
    when processing semi-structured strings.

    Arguments:
    - input_string: String containing the nested delimiters to process
    - start_delim: String specifying the start delimiter, defaults to "(" 
    - end_delim: String specifying the end delimiter, defaults to ")"
    - replacement: String that replaces both delimiters, defaults to ""
    - level: Integer specifying the level of nesting at which replacements
             occur, defaults to 1
    - start_alt: String specifying an alternative version of the start
                 delimiter that will trigger a level change
                 but will not be replaced, must be shorter than the
                 start delimiter
    - end_alt: String specifying an alternative version of the end 
               delimiter that will trigger a level change but will not
               be replaced, must be shorter than the end delimiter
    - strict_eos: Boolean, if false will allow replacement of delimiter
                  set that contains the alternative end delimiter, but
                  only when it is located at the end of the string or
                  followed only by the replacement string

    Returns:
    String where the delimiters are replaced with the replacement
    '''
    
    if start_delim == end_delim:
        raise Exception("Delimeters must be different.")
       
    if ((start_alt is not None and len(start_alt) > len(start_delim)) or
       (end_alt is not None and len(end_alt) > len(end_delim))): 
        raise Exception("Alternative delimiter must not "
                        "be longer than the delimiter.")
        
    if start_delim not in input_string:
        return input_string
        
    common_len = 2 if start_delim[0] == end_delim[-1] else 1
    current_level = 0
    chunk_list = []
    rep_len = len(replacement)
    chunk_start = None
    chunk_end = None

    i = 0
    while i <= len(input_string):
    #print(f"{i}. {current_level}|{input_string[i:i+len(start_delim)]}|{input_string[i:i+len(start_alt)]}")

        start_del_hit = input_string[i:i+len(start_delim)] == start_delim
        start_alt_hit = input_string[i:i+len(start_alt)] == start_alt
        
        if start_alt_hit and not start_del_hit:
            current_level += 1
        if start_del_hit:
            current_level += 1
            if current_level == level:
                chunk_start = i
            i = i + len(start_delim)
            continue
                    
        end_del_hit = input_string[i:i+len(end_delim)] == end_delim
        end_alt_hit = input_string[i:i+len(end_alt)] == end_alt
        if strict_eos:
            eos_alt_hit = False
        else:
            eos_alt_hit = (input_string[i:i+len(end_alt)] == end_alt and
                           len(input_string[i:].replace(replacement, ""))
                           <= len(end_alt))
        
        if end_alt_hit and not (end_del_hit or eos_alt_hit):
            current_level -= 1
            if current_level <= 0 and chunk_start is not None:
                current_level = 0
                chunk_start = None
                
        if end_del_hit or eos_alt_hit:
            current_level -= 1
            if current_level < 0 and chunk_start is not None:
                current_level = 0
                chunk_start = None
            if current_level+1 == level:            
                chunk_end = i
                if chunk_start is not None and chunk_end is not None:
                    chunk_list.append((chunk_start, chunk_end))
                    chunk_start = None
                    chunk_end = None
            i = i + len(end_delim)-common_len
        i += 1

#     if len(chunk_list) > 0:
#         print(chunk_list)
    output_string = input_string
    for i, chunk in enumerate(chunk_list):
        adjusted_start = chunk[0] + ((rep_len)*2
                                     - len(start_delim)
                                     - len(end_delim))*i
        adjusted_end = chunk[1] + ((rep_len)*2 
                                   - len(start_delim)
                                   - len(end_delim))*i
        
        output_string = (output_string[:adjusted_start] 
                         + replacement 
                         + output_string[adjusted_start
                         + len(start_delim):adjusted_end] 
                         + replacement
                         + output_string[adjusted_end 
                         + len(end_delim):])

    return output_string

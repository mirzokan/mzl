'''
General Purpose Tools
'''

import re


def sdir(obj, sunder=False):
    '''
    Special dir function
    Modification of the dir function to detail special and object
    specific callables and attributes
    Arguments: 
    * obj: Object, any object to instpect
    * sunder: Boolean, False shows non-undered attributes,
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


# Needs Cleanup.
# def ftpUpload(ftpServer, ftpUser, ftpPass, ftpRemote, payload_path):
#     '''
#     Arguments:
#     * ftpServer: string, address of the ftp server, e.g. ftp.mirzo.net
#     * ftpUser: string, username
#     * ftpPass: string, password
#     * ftpRemote: string, remote starting path
#     * payload_path: filepath to upload file
#     '''
#     import ftplib

#     try:
#         ftp = ftplib.FTP(ftpServer)
#         ftp.login(ftpUser, ftpPass)
#         ftpUpload(ftp, ftpRemote, payload_path)
#     except:
#         status = "Upload Failed"
#     finally:
#         ftp.quit()

#     ext = os.path.splitext(file)[1]
#     filename = os.path.basename(file)
#     # if ext in (".txt", ".htm", ".html"):
#     #     ftp.storlines("STOR " + filename, open(file))
#     # else:
#     #     ftp.storbinary("STOR " + filename, open(file, "rb"), 1024)
#     ftp.storbinary("STOR " + filename, open(file, "rb"), 1024)

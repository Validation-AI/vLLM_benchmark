import os
import subprocess
#import requests
import argparse
import datetime

#error_tup = (requests.exceptions.ProxyError) # urllib3.exceptions.MaxRetryError
allfiles = []

def getallfiles(p):
    files = os.listdir(p)
    for fi in files:
        fi_d = os.path.join(p, fi)
        if os.path.isdir(fi_d):
            getallfiles(fi_d)
        else:
            allfiles.append(fi_d)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="target folder")
    parser.add_argument("--path", type=str, help="target path")
    parser.add_argument("--file", type=str, help="target file")
    parser.add_argument("--cred", type=str, help="user credential for upload")
    #parser.add_argument("--retention", type=str, default="2800", help="retension property which is needed for every file on artifcatory")
    return parser.parse_args()


#os.environ["http_proxy"] = "http://child-prc.intel.com:913"
#os.environ["https_proxy"] = "http://child-prc.intel.com:913"
#os.environ["ftp_proxy"] = "http://child-prc.intel.com:913"

args = parse_arguments()
baseurl = "https://ubit-artifactory-ba.intel.com/artifactory/aipc_releases-ba-local/"
#test/add.py;retention.days=100"

basecmd = "curl -u" + args.cred + " -T "
if not args.path:
    print("Warning: we will upload your folders or contents to the top level of the server")
    upath = baseurl
else:
    upath = baseurl + args.path + "/"
if args.folder:
    folder = args.folder
    getallfiles(folder)
    lindex = len(folder)
    print(lindex)
elif args.file:
    ufile = args.file
    allfiles.append(ufile)
    lindex = ufile.rfind("/")
else:
    print("please make sure you use folder argument or file argument")


#print(allfiles)
for eachfile in allfiles:
    if lindex == -1:
        url = upath + eachfile + ";retention.days=2800"
        print(url)
    else:
        url = upath + eachfile[lindex + 1:] + ";retention.days=2800"
        print(url)
    cmd = basecmd + eachfile + " " + "\"" + url + "\""
    print(cmd)
    result = os.system(cmd)
    print(result)
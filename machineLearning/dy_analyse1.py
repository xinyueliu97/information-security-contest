import os
import sys
import re
import json

DROIDBOX_PATH = '/home/ubuntu/DroidBox/'
CURRENT_PATH = os.getcwd()

def dy_demo(filename,packagename=''):
    #os.chdir(DROIDBOX_PATH)
    #print "analysing %s..."%filename
    #data = os.system('./droidbox1.sh %s 60'%filename).read()
    #os.system('adb uninstall %s'%packagename)
    #os.chdir(CURRENT_PATH)
    outpath = '/home/ubuntu/DroidBox/output/out1.json'
    data=[json.loads(line) for line in open(outpath)]
    print data[0]
    return data[0]

def run(filename,packagename=''):
    data = dy_demo(filename,packagename)
    result = ''
    if data:
        match = re.findall(ur'{.*}', data)
        if len(match):
            result = eval(match[0])
    return result

if __name__ == '__main__':
    result = run(sys.argv[1],sys.argv[2])
    print `result`


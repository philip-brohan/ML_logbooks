#!/usr/bin/env python

# Retrieve the oldWeather1 logbook images from MASS

import os
import subprocess
import tarfile

moose_dir=("moose://adhoc/users/philip.brohan/logbook_images/NA_WW1")

local_dir="%s/logbook_images/NA_WW1" % os.getenv('SCRATCH')
if not os.path.isdir(local_dir):
    os.makedirs(localdir)

def list_logs():
    lst = subprocess.check_output("moo ls %s" % moose_dir,shell=True)
    lst = lst.split() # split on newline
    lst = [os.path.basename(x)[:-4].decode('utf-8') for x in lst]
    return lst

def check_done(log):
    ldir = "%s/%s" % (local_dir,log)
    if not os.path.isdir(ldir):
        return False
    nfiles=os.listdir(ldir)
    if len(nfiles)>5:
        return True
    else:
        return False

def unarchive(log):
    proc = subprocess.call("moo get %s/%s.tgz %s" % (moose_dir,log,local_dir),shell=True)
    tar_file="%s/%s.tgz" % (local_dir,log)
    tf = tarfile.open(tar_file)
    tf.extractall(path=local_dir)
    # Update the extracted file times
    #  To stop SCRATCH deleting them as too old
    nfiles=os.listdir("%s/%s" % (local_dir,log))
    for nfile in nfiles:
        os.utime("%s/%s/%s" % (local_dir,log,nfile))
    os.remove(tar_file)
 
for log in list_logs():
    if check_done(log):
        continue
    print(log)
    unarchive(log)

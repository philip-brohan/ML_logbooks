# Make a tf.data.Dataset from the logbook image tensors

import os
import tensorflow as tf
import numpy

# Load an image tensor from a file
def load_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, numpy.float32)
    imt = tf.reshape(imt, [1024, 1024, 3])
    return imt


# Get a logbook tensors dataset - for 'training' or 'test'.
#  Optionally specify how many images to use.
def getLogbooksDataset(purpose="training", nImages=None):

    # Get a list of filenames containing image tensors
    dirs = os.listdir("%s/ML_logbooks/images/%s" % (os.getenv("SCRATCH"), purpose))
    inFiles = []
    for dirn in dirs:
        files = os.listdir(
            "%s/ML_logbooks/images/%s/%s" % (os.getenv("SCRATCH"), purpose, dirn)
        )
        for filen in files:
            inFiles.append(
                "%s/ML_logbooks/images/%s/%s/%s"
                % (os.getenv("SCRATCH"), purpose, dirn, filen)
            )
    if nImages is not None:
        if len(inFiles) >= nImages:
            inFiles = inFiles[0:nImages]
        else:
            raise ValueError(
                "Only %d images available, can't provide %d" % (len(inFiles), nImages)
            )

    # Create TensorFlow Dataset object from the file namelist
    itList = tf.constant(inFiles)
    tr_data = tf.data.Dataset.from_tensor_slices(itList).repeat()

    # Convert the Dataset from file names to file contents
    tr_data = tr_data.map(load_tensor)

    return tr_data

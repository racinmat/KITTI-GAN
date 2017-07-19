from smop.core import *
from lxml import etree
from io import open
from bs4 import BeautifulSoup
import pickle
from devkit.python.extractPoses import extractPoses
import os

def readTracklets(filename=None):
    if os.path.isfile(filename + '.cache'):
        file = open(filename + '.cache', 'rb')
        tracklets = pickle.load(file)
        return tracklets

    # READTRACKLETS reads annotations from xml-files

    ## read Document Object Model (DOM)
    with open(filename) as fp:
        root = BeautifulSoup(fp, "lxml")

    ## Extract tracklets
        trackletsElement = root.tracklets

    objIdx = 0

    count = int(trackletsElement.find('count', recursive=False).text)
    tracklets = []

    for element in trackletsElement.find_all('item', recursive=False):
        tracklet = {}
        tracklet['objectType'] = str(element.objecttype.text)
        tracklet['h'] = float(element.h.text)
        tracklet['w'] = float(element.w.text)
        tracklet['l'] = float(element.l.text)
        tracklet['first_frame'] = int(element.first_frame.text)
        tracklet['poses'] = extractPoses(element.poses)
        tracklet['finished'] = bool(element.finished.text)
        tracklets.append(tracklet)
        objIdx += 1

    # plausibility check
    if count != objIdx:
        fprintf(2, 'number of tracklets (%d) does not match count (%d)!', objIdx, count)

    file = open(filename + '.cache', 'wb')
    pickle.dump(tracklets, file)
    file.close()

    return tracklets


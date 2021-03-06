from io import open
import pickle
from devkit.python.extractPoses import extractPoses
import os
import sys
from functools import lru_cache


@lru_cache(maxsize=32)
def read_tracklets(filename):
    version = '.'.join([str(i) for i in sys.version_info[0:3]])
    if os.path.isfile(filename + '.' + version + '.cache'):
        file = open(filename + '.' + version + '.cache', 'rb')
        try:
            tracklets = pickle.load(file)
            file.close()
            return tracklets
        except UnicodeDecodeError:
            pass

    # READTRACKLETS reads annotations from xml-files
    from bs4 import BeautifulSoup

    ## read Document Object Model (DOM)
    with open(filename) as fp:
        root = BeautifulSoup(fp, "lxml")

    ## Extract tracklets
    trackletsElement = root.tracklets

    objIdx = 0

    count = int(trackletsElement.find('count', recursive=False).text)
    tracklets = []

    for element in trackletsElement.find_all('item', recursive=False):
        posesVec, posesDict = extractPoses(element.poses)
        tracklet = {
            'objectType': str(element.objecttype.text),
            'h': float(element.h.text),
            'w': float(element.w.text),
            'l': float(element.l.text),
            'first_frame': int(element.first_frame.text),
            'poses': posesVec,
            'poses_dict': posesDict,
            'finished': bool(element.finished.text)
        }
        tracklets.append(tracklet)
        objIdx += 1

    # plausibility check
    if count != objIdx:
        print('number of tracklets {:d} does not match count {:d}!'.format(objIdx, count))

    file = open(filename + '.' + version + '.cache', 'wb')
    pickle.dump(tracklets, file)
    file.close()

    return tracklets

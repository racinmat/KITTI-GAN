from smop.core import *
from lxml import etree
from io import open
from bs4 import BeautifulSoup

from devkit.python.extractPoses import extractPoses


def readTracklets(filename=None):
    # READTRACKLETS reads annotations from xml-files

    ## read Document Object Model (DOM)
    with open(filename) as fp:
        root = BeautifulSoup(fp, "lxml")

    ## Extract tracklets
        trackletsElement = root.tracklets

    objIdx = 0
    count = 0
    tracklets = []

    for element in trackletsElement.children:
        if element == "\n":
            continue

        # fprintf('No attributes\n')
        # else
        # fprintf('found: #s\n',char(element.getTagName))
        if element.name == 'count':
            count = int(element.text)
            tracklets = []
        # fprintf( '#d\n', count );
        # print out version info
        # elseif strcmp(element.getTagName,'item_version') # meta-data
        # fprintf( '#d\n', str2double(element.getFirstChild.getData) );
        elif element.name == 'item':
                tracklet = {}
                tracklet.objectType = str(element.objecttype.text)
                tracklet.h = float(element.h.text)
                tracklet.w = float(element.w.text)
                tracklet.l = float(element.l.text)
                tracklet.first_frame = int(element.first_frame.text)
                poses = element.getElementsByTagName('poses').item(0)
                tracklet.poses = extractPoses(poses)
                tracklet.finished = bool(element.finished.text)
                tracklets.append(tracklet)
                objIdx += 1

    # plausibility check
    if count != objIdx - 1:
        fprintf(2, 'number of tracklets (%d) does not match count (%d)!', objIdx, count)

    return tracklets


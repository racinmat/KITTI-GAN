from smop.core import *
from xml.dom.minidom import parse


@function
def readTracklets(filename=None):
    # READTRACKLETS reads annotations from xml-files

    ## read Document Object Model (DOM)
    try:
        dom = parse(filename)
        # dom=xmlread(filename)
    finally:
        pass

    ## Extract tracklets
    allTracklets = dom.getElementsByTagName('tracklets')
    trackletsElement = allTracklets.item(0)

    objIdx = 0

    for element in trackletsElement.childNodes:
        attributes = element.attributes
        if attributes:
            # fprintf('No attributes\n')
            # else
            # fprintf('found: #s\n',char(element.getTagName))
            if element.tagName is 'count':
                count = float(element.firstChild.getData)
                tracklets = []
            # fprintf( '#d\n', count );
            # print out version info
            # elseif strcmp(element.getTagName,'item_version') # meta-data
            # fprintf( '#d\n', str2double(element.getFirstChild.getData) );
            else:
                if element.tagName is 'item':
                    tracklet = {}
                    tracklet.objectType = char(element.getElementsByTagName('objectType').item(0).getTextContent)
                    # ./../matlab/readTracklets.m:39
                    tracklet.h = copy(str2double(element.getElementsByTagName('h').item(0).getTextContent))
                    # ./../matlab/readTracklets.m:42
                    tracklet.w = copy(str2double(element.getElementsByTagName('w').item(0).getTextContent))
                    # ./../matlab/readTracklets.m:43
                    tracklet.l = copy(str2double(element.getElementsByTagName('l').item(0).getTextContent))
                    # ./../matlab/readTracklets.m:44
                    tracklet.first_frame = copy(
                        str2double(element.getElementsByTagName('first_frame').item(0).getTextContent))
                    # ./../matlab/readTracklets.m:46
                    poses = element.getElementsByTagName('poses').item(0)
                    # ./../matlab/readTracklets.m:49
                    tracklet.poses = copy(extractPoses(poses))
                    # ./../matlab/readTracklets.m:50
                    tracklet.finished = copy(
                        str2double(element.getElementsByTagName('finished').item(0).getTextContent))
                    # ./../matlab/readTracklets.m:52
                    tracklets.append(tracklet)
                    # ./../matlab/readTracklets.m:57
                    objIdx += 1
                # ./../matlab/readTracklets.m:59

    # plausibility check
    if count != objIdx - 1:
        fprintf(2, 'number of tracklets (%d) does not match count (%d)!', objIdx, count)

    return tracklets


if __name__ == '__main__':
    pass

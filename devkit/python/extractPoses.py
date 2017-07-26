import numpy as np


def extractPoses(poses):
    # EXTRACTPOSES extracts poses from subtree of labels.xml

    nPoses = int(poses.count.text)
    posesVec = np.zeros((15, nPoses), dtype=float)
    posesDict = np.empty(nPoses, dtype=dict)
    poseIdx = 0
    for pose in poses.find_all('item', recursive=False):
        posesVec[0, poseIdx] = float(pose.tx.text)
        posesVec[1, poseIdx] = float(pose.ty.text)
        posesVec[2, poseIdx] = float(pose.tz.text)
        posesVec[3, poseIdx] = float(pose.rx.text)
        posesVec[4, poseIdx] = float(pose.ry.text)
        posesVec[5, poseIdx] = float(pose.rz.text)
        posesVec[6, poseIdx] = float(pose.state.text)
        posesVec[7, poseIdx] = int(pose.occlusion.text)
        posesVec[8, poseIdx] = float(pose.occlusion_kf.text)
        posesVec[9, poseIdx] = float(pose.truncation.text)
        posesVec[10, poseIdx] = float(pose.amt_occlusion.text)
        posesVec[11, poseIdx] = float(pose.amt_occlusion_kf.text)
        posesVec[12, poseIdx] = float(pose.amt_border_l.text)
        posesVec[13, poseIdx] = float(pose.amt_border_r.text)
        posesVec[14, poseIdx] = float(pose.amt_border_kf.text)

        posesDict[poseIdx] = {
            'tx': float(pose.tx.text),
            'ty': float(pose.ty.text),
            'tz': float(pose.tz.text),
            'rx': float(pose.rx.text),
            'ry': float(pose.ry.text),
            'rz': float(pose.rz.text),
            'state': float(pose.state.text),
            'occlusion': int(pose.occlusion.text),
            'occlusion_kf': int(pose.occlusion_kf.text),
            'truncation': int(pose.truncation.text),
            'amt_occlusion': int(pose.amt_occlusion.text),
            'amt_occlusion_kf': int(pose.amt_occlusion_kf.text),
            'amt_border_l': int(pose.amt_border_l.text),
            'amt_border_r': int(pose.amt_border_r.text),
            'amt_border_kf': int(pose.amt_border_kf.text)
        }
        poseIdx += 1

        # plausibility check
    if nPoses != poseIdx:
        print('number of poses {:d} does not match count {:d}!'.format(nPoses, poseIdx))

    return posesVec, posesDict

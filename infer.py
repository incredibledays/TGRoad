import numpy as np
import cv2
import os
import math
import pickle
import scipy
import scipy.ndimage.morphology as morphology
import scipy.ndimage.filters as filters
from PIL import Image
import torch
from torch.autograd import Variable as V
import matplotlib.pyplot as plt

from extractor import Extractor
from network import TGDLinkNet34
from utils import simpilfyGraph


def distance(p1, p2):
    a =p1[0] - p2[0]
    b = p1[1] - p2[1]
    return np.sqrt(a*a + b*b)


def cosine_similarity(k1, k2, k3):
    vec1 = distance_norm(k2, k1)
    vec2 = distance_norm(k3, k1)
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]


def distance_norm(k1, k2):
    l = distance(k1, k2)
    a = k1[0] - k2[0]
    b = k1[1] - k2[1]
    return a/l, b/l


def graph_refine(graph, isolated_thr=150, spurs_thr=30):
    neighbors = graph

    gid = 0
    grouping = {}

    for k, v in neighbors.items():
        if k not in grouping:
            # start a search
            queue = [k]

            while len(queue) > 0:
                n = queue.pop(0)

                if n not in grouping:
                    grouping[n] = gid
                    for nei in neighbors[n]:
                        queue.append(nei)

            gid += 1

    group_count = {}

    for k, v in grouping.items():
        if v not in group_count:
            group_count[v] = (1, 0)
        else:
            group_count[v] = (group_count[v][0] + 1, group_count[v][1])

        for nei in neighbors[k]:
            a = k[0] - nei[0]
            b = k[1] - nei[1]

            d = np.sqrt(a * a + b * b)

            group_count[v] = (group_count[v][0], group_count[v][1] + d / 2)

    # short spurs
    remove_list = []
    for k, v in neighbors.items():
        if len(v) == 1:
            if len(neighbors[v[0]]) >= 3:
                a = k[0] - v[0][0]
                b = k[1] - v[0][1]

                d = np.sqrt(a * a + b * b)

                if d < spurs_thr:
                    remove_list.append(k)

    remove_list2 = []
    remove_counter = 0
    new_neighbors = {}

    def isRemoved(k):
        gid = grouping[k]
        if group_count[gid][0] <= 1:
            return True
        elif group_count[gid][1] <= isolated_thr:
            return True
        elif k in remove_list:
            return True
        elif k in remove_list2:
            return True
        else:
            return False

    for k, v in neighbors.items():
        if isRemoved(k):
            remove_counter += 1
            pass
        else:
            new_nei = []
            for nei in v:
                if isRemoved(nei):
                    pass
                else:
                    new_nei.append(nei)

            new_neighbors[k] = list(new_nei)

    return new_neighbors


def graph_shave(graph, spurs_thr=50):
    neighbors = graph

    # short spurs
    remove_list = []
    for k, v in neighbors.items():
        if len(v) == 1:
            d = distance(k, v[0])
            cur = v[0]
            l = [k]
            while True:
                if len(neighbors[cur]) >= 3:
                    break
                elif len(neighbors[cur]) == 1:
                    l.append(cur)
                    break

                else:

                    if neighbors[cur][0] == l[-1]:
                        next_node = neighbors[cur][1]
                    else:
                        next_node = neighbors[cur][0]

                    d += distance(cur, next_node)
                    l.append(cur)

                    cur = next_node

            if d < spurs_thr:
                for n in l:
                    if n not in remove_list:
                        remove_list.append(n)

    def isRemoved(k):
        if k in remove_list:
            return True
        else:
            return False

    new_neighbors = {}
    remove_counter = 0

    for k, v in neighbors.items():
        if isRemoved(k):
            remove_counter += 1
            pass
        else:
            new_nei = []
            for nei in v:
                if isRemoved(nei):
                    pass
                else:
                    new_nei.append(nei)

            new_neighbors[k] = list(new_nei)

    # print("shave", len(new_neighbors), "remove", remove_counter, "nodes")

    return new_neighbors


def graph_refine_deloop(neighbors, max_step=10, max_length=200, max_diff=5):
    removed = []
    impact = []

    remove_edge = []
    new_edge = []

    for k, v in neighbors.items():
        if k in removed:
            continue

        if k in impact:
            continue

        if len(v) < 2:
            continue

        for nei1 in v:
            if nei1 in impact:
                continue

            if k in impact:
                continue

            for nei2 in v:
                if nei2 in impact:
                    continue
                if nei1 == nei2:
                    continue

                if cosine_similarity(k, nei1, nei2) > 0.984:
                    l1 = distance(k, nei1)
                    l2 = distance(k, nei2)

                    # print("candidate!", l1,l2,neighbors_cos(neighbors, k, nei1, nei2))

                    if l2 < l1:
                        nei1, nei2 = nei2, nei1

                    remove_edge.append((k, nei2))
                    remove_edge.append((nei2, k))

                    new_edge.append((nei1, nei2))

                    impact.append(k)
                    impact.append(nei1)
                    impact.append(nei2)

                    break

    new_neighbors = {}

    def isRemoved(k):
        if k in removed:
            return True
        else:
            return False

    for k, v in neighbors.items():
        if isRemoved(k):
            pass
        else:
            new_nei = []
            for nei in v:
                if isRemoved(nei):
                    pass
                elif (nei, k) in remove_edge:
                    pass
                else:
                    new_nei.append(nei)

            new_neighbors[k] = list(new_nei)

    for new_e in new_edge:
        nk1 = new_e[0]
        nk2 = new_e[1]

        if nk2 not in new_neighbors[nk1]:
            new_neighbors[nk1].append(nk2)
        if nk1 not in new_neighbors[nk2]:
            new_neighbors[nk2].append(nk1)

    # print("remove %d edges" % len(remove_edge))

    return new_neighbors, len(remove_edge)


def detect_local_minima(arr, mask, threshold=0.5):
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    local_min = (filters.minimum_filter(arr, footprint=neighborhood) == arr)
    background = (arr == 0)
    eroded_background = morphology.binary_erosion(background, structure=neighborhood, border_value=1)
    detected_minima = local_min ^ eroded_background
    # cv2.imwrite('./test.jpg', detected_minima * 255)
    return np.where((detected_minima & (mask > threshold)))


def line_pooling(seg, x0, y0, x, y, mask=None):
    step = int(max(abs(x - x0), abs(y - y0)))
    sample = np.linspace(np.array([x0, y0]), np.array([x, y]), step, dtype=int)
    if mask is not None:
        valid_num = np.sum(mask[sample[2: -2, 0], sample[2: -2, 1]])
        if valid_num > 0:
            mean = np.sum(seg[sample[2: -2, 0], sample[2: -2, 1]] * mask[sample[2: -2, 0], sample[2: -2, 1]]) / valid_num
        else:
            mean = 0
        return mean
    else:
        mean = np.mean(seg[sample[2: -2, 0], sample[2: -2, 1]])
        std = np.std(seg[sample[2: -2, 0], sample[2: -2, 1]])
        return mean, std


def detect_keypoints(ach, v_thr):
    kp = np.copy(ach)
    smooth_kp = scipy.ndimage.filters.gaussian_filter(np.copy(kp), 1)
    smooth_kp = smooth_kp / max(np.amax(smooth_kp), 0.001)
    keypoints = detect_local_minima(-smooth_kp, smooth_kp, v_thr)
    return keypoints


def DecodeRoadGraph(seg, ach, ori, rad):
    seg = np.clip(seg + (ori > -0.5), a_min=0, a_max=1)
    anchors = detect_keypoints(ach[:, :, 0] * (seg > 0.5), 0.05)
    junctions = detect_keypoints(ach[:, :, 1] * (seg > 0.5), 0.5)
    keypoints = np.concatenate((anchors, junctions), 1)

    neighbors = {}
    for j in range(len(keypoints[0])):
        x0 = keypoints[0][j]
        y0 = keypoints[1][j]
        if j < len(anchors[0]):
            z0 = 0
        else:
            z0 = 1
        proposal = []
        for k in range(len(keypoints[0])):
            x = keypoints[0][k]
            y = keypoints[1][k]
            if x == x0 and y == y0:
                continue
            if abs(x - x0) > rad or abs(y - y0) > rad:
                continue
            if abs(x - x0) < 5 and abs(y - y0) < 5:
                if (x0, y0) in neighbors:
                    if (x, y) in neighbors[(x0, y0)]:
                        pass
                    else:
                        neighbors[(x0, y0)].append((x, y))
                else:
                    neighbors[(x0, y0)] = [(x, y)]

                if (x, y) in neighbors:
                    if (x0, y0) in neighbors[(x, y)]:
                        pass
                    else:
                        neighbors[(x, y)].append((x0, y0))
                else:
                    neighbors[(x, y)] = [(x0, y0)]
            else:
                line_mean, line_std = line_pooling(seg, x0, y0, x, y)
                # angle_mean = line_pooling(ori, x0, y0, x, y, ori > 0)
                dist = distance((x0, y0), (x, y))
                angle = math.atan2(y - y0, x - x0)
                # angle_gt = angle / math.pi
                # if angle_gt <= 0:
                #     angle_gt += 1
                if (line_mean > 0.5) and (line_std < 0.3):
                    proposal.append([x, y, dist, angle, line_mean, line_std])
                    for point in proposal[:-1]:
                        if abs(angle - point[3]) < 0.01:
                            if dist < point[2]:
                                proposal.remove(point)
                            else:
                                proposal.remove([x, y, dist, angle, line_mean, line_std])
                            break

        proposal.sort(key=lambda t: t[4], reverse=True)
        if z0:
            for point in proposal[:8]:
                if (x0, y0) in neighbors:
                    if (point[0], point[1]) in neighbors[(x0, y0)]:
                        pass
                    else:
                        neighbors[(x0, y0)].append((point[0], point[1]))
                else:
                    neighbors[(x0, y0)] = [(point[0], point[1])]

                if (point[0], point[1]) in neighbors:
                    if (x0, y0) in neighbors[(point[0], point[1])]:
                        pass
                    else:
                        neighbors[(point[0], point[1])].append((x0, y0))
                else:
                    neighbors[(point[0], point[1])] = [(x0, y0)]
        else:
            for point in proposal[:2]:
                if (x0, y0) in neighbors:
                    if (point[0], point[1]) in neighbors[(x0, y0)]:
                        pass
                    else:
                        neighbors[(x0, y0)].append((point[0], point[1]))
                else:
                    neighbors[(x0, y0)] = [(point[0], point[1])]

                if (point[0], point[1]) in neighbors:
                    if (x0, y0) in neighbors[(point[0], point[1])]:
                        pass
                    else:
                        neighbors[(point[0], point[1])].append((x0, y0))
                else:
                    neighbors[(point[0], point[1])] = [(x0, y0)]

    spurs_thr = 50
    isolated_thr = 200
    graph = graph_refine(neighbors, isolated_thr=isolated_thr, spurs_thr=spurs_thr)
    rc = 100
    while rc > 0:
        graph, rc = graph_refine_deloop(graph_refine(graph, isolated_thr=isolated_thr, spurs_thr=spurs_thr))
    graph = graph_shave(graph, spurs_thr=spurs_thr)
    # graph = neighbors
    return graph


def DecodeRoadGraphBaseline(seg, ach, rad):
    keypoints = detect_keypoints(ach, 0.05)

    neighbors = {}
    for j in range(len(keypoints[0])):
        x0 = keypoints[0][j]
        y0 = keypoints[1][j]
        proposal = []
        for k in range(len(keypoints[0])):
            x = keypoints[0][k]
            y = keypoints[1][k]
            if x == x0 and y == y0:
                continue
            if abs(x - x0) > rad or abs(y - y0) > rad:
                continue
            if abs(x - x0) < 5 and abs(y - y0) < 5:
                if (x0, y0) in neighbors:
                    if (x, y) in neighbors[(x0, y0)]:
                        pass
                    else:
                        neighbors[(x0, y0)].append((x, y))
                else:
                    neighbors[(x0, y0)] = [(x, y)]

                if (x, y) in neighbors:
                    if (x0, y0) in neighbors[(x, y)]:
                        pass
                    else:
                        neighbors[(x, y)].append((x0, y0))
                else:
                    neighbors[(x, y)] = [(x0, y0)]
            else:
                line_mean, line_std = line_pooling(seg, x0, y0, x, y)
                dist = distance((x0, y0), (x, y))
                angle = math.atan2(y - y0, x - x0)
                if (line_mean > 0.5) and (line_std < 0.3):
                    proposal.append([x, y, dist, angle, line_mean, line_std])
                    for point in proposal[:-1]:
                        if abs(angle - point[3]) < 0.01:
                            if dist < point[2]:
                                proposal.remove(point)
                            else:
                                proposal.remove([x, y, dist, angle, line_mean, line_std])
                            break

        for point in proposal[:2]:
            if (x0, y0) in neighbors:
                if (point[0], point[1]) in neighbors[(x0, y0)]:
                    pass
                else:
                    neighbors[(x0, y0)].append((point[0], point[1]))
            else:
                neighbors[(x0, y0)] = [(point[0], point[1])]

            if (point[0], point[1]) in neighbors:
                if (x0, y0) in neighbors[(point[0], point[1])]:
                    pass
                else:
                    neighbors[(point[0], point[1])].append((x0, y0))
            else:
                neighbors[(point[0], point[1])] = [(x0, y0)]

    spurs_thr = 50
    isolated_thr = 200
    graph = graph_refine(neighbors, isolated_thr=isolated_thr, spurs_thr=spurs_thr)
    rc = 100
    while rc > 0:
        graph, rc = graph_refine_deloop(graph_refine(graph, isolated_thr=isolated_thr, spurs_thr=spurs_thr))
    graph = graph_shave(graph, spurs_thr=spurs_thr)
    return graph


def DecodeRoadGraphPoint(seg, ach, rad):
    anchors = detect_keypoints(ach[:, :, 0], 0.05)
    junctions = detect_keypoints(ach[:, :, 1], 0.5)
    keypoints = np.concatenate((anchors, junctions), 1)

    neighbors = {}
    for j in range(len(keypoints[0])):
        x0 = keypoints[0][j]
        y0 = keypoints[1][j]
        if j < len(anchors[0]):
            z0 = 0
        else:
            z0 = 1
        proposal = []
        for k in range(len(keypoints[0])):
            x = keypoints[0][k]
            y = keypoints[1][k]
            if x == x0 and y == y0:
                continue
            if abs(x - x0) > rad or abs(y - y0) > rad:
                continue
            if abs(x - x0) < 5 and abs(y - y0) < 5:
                if (x0, y0) in neighbors:
                    if (x, y) in neighbors[(x0, y0)]:
                        pass
                    else:
                        neighbors[(x0, y0)].append((x, y))
                else:
                    neighbors[(x0, y0)] = [(x, y)]

                if (x, y) in neighbors:
                    if (x0, y0) in neighbors[(x, y)]:
                        pass
                    else:
                        neighbors[(x, y)].append((x0, y0))
                else:
                    neighbors[(x, y)] = [(x0, y0)]
            else:
                line_mean, line_std = line_pooling(seg, x0, y0, x, y)
                dist = distance((x0, y0), (x, y))
                angle = math.atan2(y - y0, x - x0)
                if (line_mean > 0.5) and (line_std < 0.3):
                    proposal.append([x, y, dist, angle, line_mean, line_std])
                    for point in proposal[:-1]:
                        if abs(angle - point[3]) < 0.01:
                            if dist < point[2]:
                                proposal.remove(point)
                            else:
                                proposal.remove([x, y, dist, angle, line_mean, line_std])
                            break

        proposal.sort(key=lambda t: t[4], reverse=True)
        if z0:
            for point in proposal[:8]:
                if (x0, y0) in neighbors:
                    if (point[0], point[1]) in neighbors[(x0, y0)]:
                        pass
                    else:
                        neighbors[(x0, y0)].append((point[0], point[1]))
                else:
                    neighbors[(x0, y0)] = [(point[0], point[1])]

                if (point[0], point[1]) in neighbors:
                    if (x0, y0) in neighbors[(point[0], point[1])]:
                        pass
                    else:
                        neighbors[(point[0], point[1])].append((x0, y0))
                else:
                    neighbors[(point[0], point[1])] = [(x0, y0)]
        else:
            for point in proposal[:2]:
                if (x0, y0) in neighbors:
                    if (point[0], point[1]) in neighbors[(x0, y0)]:
                        pass
                    else:
                        neighbors[(x0, y0)].append((point[0], point[1]))
                else:
                    neighbors[(x0, y0)] = [(point[0], point[1])]

                if (point[0], point[1]) in neighbors:
                    if (x0, y0) in neighbors[(point[0], point[1])]:
                        pass
                    else:
                        neighbors[(point[0], point[1])].append((x0, y0))
                else:
                    neighbors[(point[0], point[1])] = [(x0, y0)]

    spurs_thr = 50
    isolated_thr = 200
    graph = graph_refine(neighbors, isolated_thr=isolated_thr, spurs_thr=spurs_thr)
    rc = 100
    while rc > 0:
        graph, rc = graph_refine_deloop(graph_refine(graph, isolated_thr=isolated_thr, spurs_thr=spurs_thr))
    graph = graph_shave(graph, spurs_thr=spurs_thr)
    # graph = neighbors
    return graph


def infer_exp1():
    """DLASeg as backbone, including segmentation branch, anchor point branch."""

    from network import TGDLABaseline

    input_dir = './datasets/cityscale/test/'
    output_dir = './results/exp1/'
    weight_dir = './checkpoints/exp1/best.th'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = Extractor(TGDLABaseline, eval_mode=True)
    model.load(weight_dir)

    for i in range(180):
        if i % 10 < 8 or i % 20 == 18:
            continue

        sat = cv2.imread(input_dir + "region_%d_sat.png" % i)
        sat_img = np.array(sat, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre = pre['seg'].squeeze().cpu().data.numpy()
        ach_pre = pre['ach'].squeeze().cpu().data.numpy()

        sat_img = np.flip(sat, 0)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_v = pre['seg'].squeeze().cpu().data.numpy()

        sat_img = np.flip(sat, 1)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_h = pre['seg'].squeeze().cpu().data.numpy()

        sat_img = np.rot90(sat)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_r = pre['seg'].squeeze().cpu().data.numpy()

        seg_pre = np.clip(seg_pre + np.flip(seg_pre_v, 0) + np.flip(seg_pre_h, 1) + np.rot90(seg_pre_r, k=-1), a_min=0, a_max=1)

        graph = DecodeRoadGraphBaseline(seg_pre, ach_pre, 50)
        pickle.dump(graph, open(output_dir + "region_%d_graph.p" % i, "wb"))
        graph = simpilfyGraph(graph)
        for u, v in graph.items():
            n1 = u
            for n2 in v:
                cv2.line(sat, (int(n1[1]), int(n1[0])), (int(n2[1]), int(n2[0])), (255, 255, 0), 3)
        for u, v in graph.items():
            n1 = u
            cv2.circle(sat, (int(n1[1]), int(n1[0])), 3, (255, 0, 0), -1)
        Image.fromarray(sat).save(output_dir + "region_%d_vis.png" % i)


def infer_exp2():
    """DLASeg as backbone, including segmentation branch, junction and anchor point branch."""

    from network import TGDLAPoint

    input_dir = './datasets/cityscale/test/'
    output_dir = './results/exp2/'
    weight_dir = './checkpoints/exp2/best.th'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = Extractor(TGDLAPoint, eval_mode=True)
    model.load(weight_dir)

    for i in range(180):
        if i % 10 < 8 or i % 20 == 18:
            continue

        sat = cv2.imread(input_dir + "region_%d_sat.png" % i)
        sat_img = np.array(sat, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre = pre['seg'].squeeze().cpu().data.numpy()
        ach_pre = pre['ach'].squeeze().cpu().data.numpy().transpose((1, 2, 0))

        sat_img = np.flip(sat, 0)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_v = pre['seg'].squeeze().cpu().data.numpy()

        sat_img = np.flip(sat, 1)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_h = pre['seg'].squeeze().cpu().data.numpy()

        sat_img = np.rot90(sat)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_r = pre['seg'].squeeze().cpu().data.numpy()

        seg_pre = np.clip(seg_pre + np.flip(seg_pre_v, 0) + np.flip(seg_pre_h, 1) + np.rot90(seg_pre_r, k=-1), a_min=0, a_max=1)

        graph = DecodeRoadGraphPoint(seg_pre, ach_pre, 50)
        pickle.dump(graph, open(output_dir + "region_%d_graph.p" % i, "wb"))
        graph = simpilfyGraph(graph)
        for u, v in graph.items():
            n1 = u
            for n2 in v:
                cv2.line(sat, (int(n1[1]), int(n1[0])), (int(n2[1]), int(n2[0])), (255, 255, 0), 3)
        for u, v in graph.items():
            n1 = u
            cv2.circle(sat, (int(n1[1]), int(n1[0])), 3, (255, 0, 0), -1)
        Image.fromarray(sat).save(output_dir + "region_%d_vis.png" % i)


def infer_exp3():
    """DLASeg as backbone, including segmentation branch, junction and anchor point branch and orientation branch."""

    from network import TGDLA

    input_dir = './datasets/cityscale/test/'
    output_dir = './results/exp3/'
    weight_dir = './checkpoints/exp3/best.th'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = Extractor(TGDLA, eval_mode=True)
    model.load(weight_dir)

    for i in range(180):
        if i % 10 < 8 or i % 20 == 18:
            continue

        sat = cv2.imread(input_dir + "region_%d_sat.png" % i)
        sat_img = np.array(sat, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre = pre['seg'].squeeze().cpu().data.numpy()
        ach_pre = pre['ach'].squeeze().cpu().data.numpy().transpose((1, 2, 0))
        ori_pre = pre['ori'].squeeze().cpu().data.numpy().transpose((1, 2, 0))
        # ori_pre = pre['ori'].squeeze().cpu().data.numpy()

        xx, yy = np.meshgrid(range(2048), range(2048))
        afx = np.sign(ori_pre[:, :, 0]) * np.exp(- np.abs(ori_pre[:, :, 0])) * 2048 + xx
        afy = np.sign(ori_pre[:, :, 1]) * np.exp(- np.abs(ori_pre[:, :, 1])) * 2048 + yy
        # plt.plot(afx, afy, 'r.', markersize=0.5)
        plt.imshow(ori_pre[:, :, 0] * ori_pre[:, :, 0] + ori_pre[:, :, 1] * ori_pre[:, :, 1])
        plt.show()
        break

        sat_img = np.flip(sat, 0)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_v = pre['seg'].squeeze().cpu().data.numpy()

        sat_img = np.flip(sat, 1)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_h = pre['seg'].squeeze().cpu().data.numpy()

        sat_img = np.rot90(sat)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_r = pre['seg'].squeeze().cpu().data.numpy()

        seg_pre = np.clip(seg_pre + np.flip(seg_pre_v, 0) + np.flip(seg_pre_h, 1) + np.rot90(seg_pre_r, k=-1), a_min=0, a_max=1)

        graph = DecodeRoadGraphPoint(seg_pre, ach_pre, 50)
        pickle.dump(graph, open(output_dir + "region_%d_graph.p" % i, "wb"))
        graph = simpilfyGraph(graph)
        for u, v in graph.items():
            n1 = u
            for n2 in v:
                cv2.line(sat, (int(n1[1]), int(n1[0])), (int(n2[1]), int(n2[0])), (255, 255, 0), 3)
        for u, v in graph.items():
            n1 = u
            cv2.circle(sat, (int(n1[1]), int(n1[0])), 3, (255, 0, 0), -1)
        Image.fromarray(sat).save(output_dir + "region_%d_vis.png" % i)


def infer_exp9():
    """DLinkNet34 as backbone, including segmentation branch, junction and anchor point branch, orientation branch."""

    input_dir = './datasets/cityscale/test/'
    output_dir = './results/exp3/'
    weight_dir = './checkpoints/exp2/best.th'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = Extractor(TGDLinkNet34, eval_mode=True)
    model.load(weight_dir)

    for i in range(180):
        if i % 10 < 8 or i % 20 == 18:
            continue

        sat = cv2.imread(input_dir + "region_%d_sat.png" % i)
        sat_img = np.array(sat, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre = pre['seg'].squeeze().cpu().data.numpy()
        ach_pre = pre['ach'].squeeze().cpu().data.numpy().transpose((1, 2, 0))
        ori_pre = pre['ori'].squeeze().cpu().data.numpy()

        sat_img = np.flip(sat, 0)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_v = pre['seg'].squeeze().cpu().data.numpy()
        ach_pre_v = pre['ach'].squeeze().cpu().data.numpy().transpose((1, 2, 0))

        sat_img = np.flip(sat, 1)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_h = pre['seg'].squeeze().cpu().data.numpy()
        ach_pre_h = pre['ach'].squeeze().cpu().data.numpy().transpose((1, 2, 0))

        sat_img = np.rot90(sat)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_r = pre['seg'].squeeze().cpu().data.numpy()
        ach_pre_r = pre['ach'].squeeze().cpu().data.numpy().transpose((1, 2, 0))

        seg_pre = np.clip(seg_pre + np.flip(seg_pre_v, 0) + np.flip(seg_pre_h, 1) + np.rot90(seg_pre_r, k=-1), a_min=0, a_max=1)
        ach_pre = np.clip(ach_pre + np.flip(ach_pre_v, 0) + np.flip(ach_pre_h, 1) + np.rot90(ach_pre_r, k=-1), a_min=0, a_max=1)

        graph = DecodeRoadGraph(seg_pre, ach_pre, ori_pre, 50)
        pickle.dump(graph, open(output_dir + "region_%d_graph.p" % i, "wb"))
        graph = simpilfyGraph(graph)
        for u, v in graph.items():
            n1 = u
            for n2 in v:
                cv2.line(sat, (int(n1[1]), int(n1[0])), (int(n2[1]), int(n2[0])), (255, 255, 0), 3)
        for u, v in graph.items():
            n1 = u
            cv2.circle(sat, (int(n1[1]), int(n1[0])), 3, (255, 0, 0), -1)
        Image.fromarray(sat).save(output_dir + "region_%d_vis.png" % i)


if __name__ == '__main__':
    infer_exp3()

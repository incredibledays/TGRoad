import numpy as np
import cv2
import math
import pickle
import scipy
import scipy.ndimage.morphology as morphology
import scipy.ndimage.filters as filters
from PIL import Image
import torch
from torch.autograd import Variable as V

from extractor import Extractor
from network import TGDLinkNet34
from utils import simpilfyGraph


def distance(A, B):
    a = A[0]-B[0]
    b = A[1]-B[1]
    return np.sqrt(a*a + b*b)


def neighbors_cos(neighbors, k1, k2, k3):
    vec1 = neighbors_norm(neighbors, k2, k1)
    vec2 = neighbors_norm(neighbors, k3, k1)
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]


def neighbors_dist(neighbors, k1, k2):
    a = k1[0] - k2[0]
    b = k1[1] - k2[1]
    return math.sqrt(a*a+b*b)


def neighbors_norm(neighbors, k1, k2):
    l = neighbors_dist(neighbors, k1, k2)
    a = k1[0] - k2[0]
    b = k1[1] - k2[1]
    return a/l, b/l


def graph_refine(graph, isolated_thr=150, spurs_thr=30, three_edge_loop_thr=70):
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

    # print(len(new_neighbors), "remove", remove_counter, "nodes")

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

                if neighbors_cos(neighbors, k, nei1, nei2) > 0.984:
                    l1 = neighbors_dist(neighbors, k, nei1)
                    l2 = neighbors_dist(neighbors, k, nei2)

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
    return np.where((detected_minima & (mask > threshold)))


def line_pooling(seg, x0, y0, x, y):
    step = int(max(abs(x - x0), abs(y - y0)))
    sample = np.linspace(np.array([x0, y0]), np.array([x, y]), step, dtype=int)
    mean = np.mean(seg[sample[2: -2, 0], sample[2: -2, 1]])
    std = np.std(seg[sample[2: -2, 0], sample[2: -2, 1]])
    # print(step)
    return mean, std


def DecodeRoadGraph(seg, ach, ori, v_thr, e_thr, s_thr, a_thr, rad):
    kp = np.copy(ach)
    smooth_kp = scipy.ndimage.filters.gaussian_filter(np.copy(kp), 1)
    smooth_kp = smooth_kp / max(np.amax(smooth_kp), 0.001)
    keypoints = detect_local_minima(-smooth_kp, smooth_kp, v_thr)
    neighbors = {}
    spurs_thr = 50
    isolated_thr = 200
    for j in range(len(keypoints[0])):
        x0 = keypoints[0][j]
        y0 = keypoints[1][j]
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
                if line_mean > e_thr and line_std < s_thr:
                    angle = abs(math.atan2(x - x0, y0 - y))
                    mx = (x0 + x) // 2
                    my = (y0 + y) // 2
                    angle_pre = seg[mx, my] * ori[mx, my] * math.pi
                    if abs(math.cos(angle_pre - angle)) > a_thr:
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

    graph = graph_refine(neighbors, isolated_thr=isolated_thr, spurs_thr=spurs_thr)
    rc = 100
    while rc > 0:
        graph, rc = graph_refine_deloop(graph_refine(graph, isolated_thr=isolated_thr, spurs_thr=spurs_thr))
    # graph = graph_shave(graph, spurs_thr=spurs_thr)
    # graph = neighbors
    return graph


# v_thr = 0.05
# e_thr = 0.3
# s_thr = 0.3
# a_thr = 0.9
# radius = 50

v_thr = 0.05
e_thr = 0.3
s_thr = 0.3
a_thr = 0.0
radius = 50

input_dir = './datasets/test/'
output_dir = './results/exp2/'
weight_dir = './checkpoints/exp2/best.th'

model = Extractor(TGDLinkNet34, eval_mode=True)
model.load(weight_dir)

for i in range(180):
    if i % 10 < 8 or i % 20 == 18:
        continue

    sat = cv2.imread(input_dir + "region_%d_sat.png" % i)
    sat_img = np.array(sat, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
    sat_img = np.expand_dims(sat_img, axis=0)
    sat_img = torch.Tensor(sat_img)
    sat_img = V(sat_img.cuda())
    seg_pre, ach_pre, ori_pre = model.predict(sat_img)
    seg_pre = seg_pre.squeeze().cpu().data.numpy()
    ach_pre = ach_pre.squeeze().cpu().data.numpy()
    ori_pre = ori_pre.squeeze().cpu().data.numpy()

    sat_img = np.array(np.flip(sat, 0), np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
    sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)))
    seg_pre_0, ach_pre_0, ori_pre_0 = model.predict(sat_img)
    seg_pre_0 = seg_pre_0.squeeze().cpu().data.numpy()
    ach_pre_0 = ach_pre_0.squeeze().cpu().data.numpy()
    ori_pre_0 = ori_pre_0.squeeze().cpu().data.numpy()

    sat_img = np.array(np.flip(sat, 1), np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
    sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)))
    seg_pre_1, ach_pre_1, ori_pre_1 = model.predict(sat_img)
    seg_pre_1 = seg_pre_1.squeeze().cpu().data.numpy()
    ach_pre_1 = ach_pre_1.squeeze().cpu().data.numpy()
    ori_pre_1 = ori_pre_1.squeeze().cpu().data.numpy()

    sat_img = np.array(np.rot90(sat), np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
    sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)))
    seg_pre_2, ach_pre_2, ori_pre_2 = model.predict(sat_img)
    seg_pre_2 = seg_pre_2.squeeze().cpu().data.numpy()
    ach_pre_2 = ach_pre_2.squeeze().cpu().data.numpy()
    ori_pre_2 = ori_pre_2.squeeze().cpu().data.numpy()

    seg_pre = (seg_pre + np.flip(seg_pre_0, 0) + np.flip(seg_pre_1, 1) + np.rot90(seg_pre_2, k=-1))
    seg_pre[seg_pre > 1] = 1
    ach_pre = (ach_pre + np.flip(ach_pre_0, 0) + np.flip(ach_pre_1, 1) + np.rot90(ach_pre_2, k=-1))
    ach_pre[ach_pre > 1] = 1
    ori_pre_2[ori_pre_2 > 0.5] -= 0.5
    ori_pre_2[ori_pre_2 <= 0.5] += 0.5
    ori_pre = (ori_pre + np.flip(1 - ori_pre_0, 0) + np.flip(1 - ori_pre_1, 1) + np.rot90(ori_pre_2, k=-1)) / 4

    graph = DecodeRoadGraph(seg_pre, ach_pre, ori_pre, v_thr, e_thr, s_thr, a_thr, radius)
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

from math import log
import random
import copy
import os


# label表示当前节点询问属性/最终结果（叶子结点）（类型：非叶子节点：询问属性/叶子结点：YES/NO）
# son表示孩子节点（类型：Node列表）
# son_label表示询问属性答案（类型：属性值列表）
class Node:
    def __init__(self, label, son, son_label):
        self.label = label
        self.son = son
        self.son_label = son_label


train_set = [["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 1],
             ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 1],
             ["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 1],
             ["青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 1],
             ["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘", 1],
             ["青绿", "硬挺", "清脆", "清晰", "平坦", "软粘", 0],
             ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", 0],
             ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0],
             ["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑", 0],
             ["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑", 0]]
test_set = [["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 1],
            ["浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 1],
            ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑", 1],
            ["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑", 0],
            ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑", 0],
            ["浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘", 0],
            ["青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑", 0]]
labels = [["青绿", "乌黑", "浅白", "色泽"],
          ["蜷缩", "稍蜷", "硬挺", "根蒂"],
          ["浊响", "沉闷", "清脆", "敲声"],
          ["清晰", "稍糊", "模糊", "纹理"],
          ["凹陷", "稍凹", "平坦", "脐部"],
          ["硬滑", "软粘", "触感"]]
labels_name = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]


def is_same_class(data):
    clas = data[0][-1]
    for i in range(len(data)):
        if clas != data[i][-1]:
            return False
    return True


def is_same_attribute(data):
    for j in range(len(data[0]) - 1):
        attr = data[0][j]
        for i in range(len(data)):
            if attr != data[i][j]:
                return False
    return True


def split_dataset(old_data, axis, value):
    data = copy.deepcopy(old_data)  # 深拷贝，否则实参依然会改变！
    new_data = []
    for i in data:
        if i[axis] == value:
            i.pop(axis)
            new_data.append(i)
    return new_data


def split_attribute(attribute, best_attribute):
    attribute_copy = copy.deepcopy(attribute)
    attribute_copy.pop(best_attribute)
    return attribute_copy


def max_class(data):
    tot = [0, 0]
    tot[0] = len(split_dataset(data, -1, 0))
    tot[1] = len(split_dataset(data, -1, 1))
    if tot[0] > tot[1]:
        return 0
    elif tot[0] < tot[1]:
        return 1
    else:
        return random.randint(0, 1)


def get_ent(num0, num1, num):
    if num0 == 0 and num1 == 0:
        return 0
    elif num0 == 0:
        return -num1 / num * log(num1 / num, 2)
    elif num1 == 0:
        return -num0 / num * log(num0 / num, 2)
    else:
        return -(num0 / num * log(num0 / num, 2) + num1 / num * log(num1 / num, 2))


def choose(data, attribute):
    tot0 = len(split_dataset(data, -1, 0))
    tot1 = len(split_dataset(data, -1, 1))
    tot_num = tot0 + tot1
    ent = get_ent(tot0, tot1, tot_num)
    attr_ent = []
    max_ent = -1
    for i in range(len(attribute)):
        attr_ent.append(ent)
        for attr in range(len(attribute[i]) - 1):
            class_num = len(split_dataset(data, i, attribute[i][attr]))
            num0 = len(split_dataset(split_dataset(data, i, attribute[i][attr]), -1, 0))
            num1 = len(split_dataset(split_dataset(data, i, attribute[i][attr]), -1, 1))
            class_cnt = get_ent(num0, num1, class_num)
            attr_ent[-1] -= class_num / tot_num * class_cnt
        if attr_ent[-1] > max_ent:
            axis = i
            name = attribute[i][-1]
            max_ent = attr_ent[-1]
    return axis, name


def build(data, attribute):
    node = Node(None, [], [])
    if is_same_class(data):
        node.label = "YES" if data[0][-1] == 1 else "NO"
        return node
    if attribute == [] or is_same_attribute(data):
        node.label = "YES" if max_class(data) == 1 else "NO"
        return node
    axis, name = choose(data, attribute)
    for value in range(len(attribute[axis]) - 1):
        new_data = split_dataset(data, axis, attribute[axis][value])
        new_attribute = split_attribute(attribute, axis)
        if not new_data:
            son_node = Node("YES" if max_class(data) == 1 else "NO", [], [])
        else:
            son_node = build(new_data, new_attribute)
        node.label = name
        node.son.append(son_node)
        node.son_label.append(attribute[axis][value])
    return node


def dfs(now, deep):
    for i in range(len(now.son)):
        nex = now.son[i]
        label = now.son_label[i]
        print(deep, now.label, nex.label, label)
        dfs(nex, deep + 1)


def get_class(data, attribute, now):
    if now.label == "YES":
        return 1
    elif now.label == "NO":
        return 0
    data_attr = data[attribute.index(now.label)]
    for i in range(len(now.son)):
        if now.son_label[i] == data_attr:
            return get_class(data, attribute, now.son[i])


def test(data, attribute, rt):
    tmp = 0
    for i in data:
        clas = get_class(i, attribute, rt)
        if clas == i[-1]:
            tmp += 1
    return tmp / len(data)


def split(data, attribute):
    new_data = []
    for i in data:
        if attribute in i:
            new_data.append(i)
    return new_data


def post_pruning(data, attribute, rt):
    q = [[rt, data]]
    cnt = 0
    while cnt < len(q):
        now = q[cnt][0]
        for i in range(len(now.son)):
            nex = now.son[i]
            if nex.son:
                q.append([nex, split(q[cnt][1], now.son_label[i])])
        cnt += 1
    q.reverse()
    for i in range(len(q)):
        now = q[i][0]
        now_data = q[i][1]
        pre_acc = test(test_set, attribute, rt)
        pre_label = now.label
        pre_son = now.son
        pre_son_label = now.son_label
        now.label = "YES" if max_class(now_data) == 1 else "NO"
        now.son = now.son_label = []
        cut_acc = test(test_set, attribute, rt)
        if pre_acc >= cut_acc:
            now.label = pre_label
            now.son = pre_son
            now.son_label = pre_son_label


def dot_dfs(now, now_cnt, cnt):
    for i in range(len(now.son)):
        cnt += 1
        nex = now.son[i]
        label = now.son_label[i]
        with open("decision_tree.gv", "a", encoding="UTF-8") as dt:
            dt.write("%d[label=\"%s=？\"];\n" % (cnt, nex.label))
            dt.write("%d:s->%d[label=%s];\n" % (now_cnt, cnt, label))
        cnt = dot_dfs(nex, cnt, cnt)
    return cnt


def dot_decision_tree(rt):
    with open("decision_tree.gv", "w", encoding="UTF-8") as dt:
        dt.write("digraph decision_tree {\n")
        dt.write("fontname=\"Microsoft YaHei\"; labelloc=t; labeljust=l; rankdir=TB;\n")
        dt.write("node[fontname=\"Microsoft YaHei\",color=darkgreen,shape=ellipse];\n")
        dt.write("edge[fontname=\"Microsoft YaHei\",color=darkgreen,style=solid,arrowsize=0.7];\n")
        dt.write("0[label=\"%s=？\"];\n" % rt.label)
    cnt = 0
    dot_dfs(rt, 0, cnt)
    with open("decision_tree.gv", "a", encoding="UTF-8") as dt:
        dt.write("}\n")


root = build(train_set, labels)
print("剪枝前：")
dfs(root, 0)
ans = test(test_set, labels_name, root)
print(ans)
dot_decision_tree(root)
os.system("dot.exe -Tpng decision_tree.gv -o decision_tree1.png")
post_pruning(train_set, labels_name, root)
print("剪枝后：")
dfs(root, 0)
ans = test(test_set, labels_name, root)
print(ans)
dot_decision_tree(root)
os.system("dot.exe -Tpng decision_tree.gv -o decision_tree2.png")

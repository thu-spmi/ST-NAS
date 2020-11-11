'''
Copyright 2020 Tsinghua University
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)
'''

from graphviz import Digraph
import sys
from itertools import cycle

COLOR = ['skyblue', 'forestgreen','blueviolet', 'palegoldenrod', 'orange', 'yellow', 
        'lightcyan', 'aliceblue', 'lightgray', 'beige']
MultiArgOP = ['add', 'concat']

def rm_unused(connections):
    def _rm_duplicate(nodes:list)->list:
        return list(set(nodes))

    nodes_tail = []
    nodes = []
    for item in connections:
        i, j = item.split('@')[0].split('-')
        i, j = int(i), int(j)
        nodes_tail.append(i)
        nodes.append(i)
        nodes.append(j)
    nodes = _rm_duplicate(nodes)
    nodes_tail = _rm_duplicate(nodes_tail)
    nodes = sorted(nodes)

    rm_nodes = []
    for i in nodes[:-1]:
        if i not in nodes_tail:
            rm_nodes.append(i)
    # print(rm_nodes)
    if len(rm_nodes) == 0:
        return connections
    
    new_connections = []
    for item in connections:
        i, j = item.split('@')[0].split('-')
        i, j = int(i), int(j)
        if j in rm_nodes:
            continue
        new_connections.append(item)
    
    return rm_unused(new_connections)

def draw(connections, filename='tmp'):
    dot = Digraph(comment='Neural Network', filename=filename, format="png")
    dot.attr(rankdir='LR', dpi='250', concentrate='true', fontsize='8')
    
    connections = rm_unused(connections)
    nodes = []
    operations = []
    color_stack = {}
    cycle_color = cycle(iter(COLOR))
    for element in connections:
        link, op = element.split("@")
        link = link.split('-')
        if link[0] not in nodes:
            nodes.append(link[0])
            dot.attr('node', shape='circle', style='filled', fillcolor='cornsilk',width='0.4',fixedsize='true')
            dot.node(link[0], label="<<i>x</i><sub>{}</sub>>".format(link[0]))
        if link[1] not in nodes:
            nodes.append(link[1])
            dot.attr('node', shape='circle', style='filled', fillcolor='cornsilk',width='0.4',fixedsize='true')
            dot.node(link[1], label="<<i>x</i><sub>{}</sub>>".format(link[1]))

        if op == "Identity":
            dot.edge(link[0], link[1])
            continue
        if op == "Zero":
            continue

        if op in color_stack.keys():
            color = color_stack[op]
        else:
            color = next(cycle_color)
            color_stack[op] = color

        dot.attr('node', shape="box", style='filled', fillcolor=color, fixedsize='false')

        if op in MultiArgOP:
            element = element.split('-')[-1]
        dot.node(element, label=op)
        dot.edge(link[0], element, arrowhead='none')

        dot.edge(element, link[1])
    
    dot.render()
    # dot.view()

if __name__ == "__main__":
    connections = [
        '0-1@TDNN-1-1',
        '0-2@add',
        '1-2@add', 
        '1-2@TDNN-1-1',
        '2-3@TDNN-2-2', 
        '3-4@TDNN\nSubsample',
        '4-5@TDNN-1-1',
        '5-6@TDNN-1-1', 
        '6-7@TDNN-2-2',
        '3-7@Identity',
        '1-5@concat',
        '3-5@concat',
        '7-8@Linear']

    if sys.argv[1:] == []:
        draw(connections)
    else:
        draw(connections, sys.argv[1])
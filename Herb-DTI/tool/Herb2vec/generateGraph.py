## 生成向量图类 ##
import networkx as nx
from container import Container
# 括号编码替换
def split_sentence(sentence):
    new_sentence=""
    for d in sentence:
        if d=="(":
            new_sentence+="( "
        elif d==")":
            new_sentence+=" )"
        else:
            new_sentence+=d
    return new_sentence
def judge_or_and_inside(data):
    if ("and" in data) or ("or" in data):
        return True
    else:
        return False
def judge_restrictions(data):
    restrctions = ["exactly","min","max","some","only"]
    for d in data:
        if d in restrctions:
            return True
    return False
def judge_condition(container):
    if container.is_empty()==False:
        if container.peak() !="(":
            return True
        else:
            return False
    else:
        return False
def convert_triple(data):
    restrictions = ["some","only","exactly","min","max"]
    # <http://purl.obolibrary.org/obo/GO_0090141> EquivalentTo <http://purl.obolibrary.org/obo/GO_0065007>
    # and ( <http://purl.obolibrary.org/obo/RO_0002213> some <http://purl.obolibrary.org/obo/GO_0000266> )
    result = []
    first_entity = data[0]   # <http://purl.obolibrary.org/obo/GO_0090141>
    # 作者定义的存储数据容器
    container = Container()
    tag=False   # 标志
    for entity in data[2:]:  # )
        if entity ==")":
            tag=True
            temp_data =[]
            while container.peak() !="(":
                temp_data.append(container.pop())
            container.pop()
            if judge_restrictions(temp_data):
                new_relation = temp_data[-1]
                tail_node = temp_data[0]
                new_node = new_relation+" "+tail_node
                container.push(new_node)
            else:
                if (not container.is_empty()):
                    if container.peak() =="(" or container.peak() =="and" or container.peak()=="or":
                        for da in temp_data:
                            container.push(da)
                    else:
                        new_temp_data=[]
                        while judge_condition(container):
                            new_temp_data.append(container.pop())
                        if new_temp_data!=[]:
                            new_relation = new_temp_data[-1]
                            for da in temp_data:
                                if da !="and" and da !="or":
                                    new_element = new_relation+" "+da
                                    container.push(new_element)
                            # for da in new_temp_data:
                            #     if da !="and" and da !="or":
                            #         new_element = new_relation+" "+da
                            #         container.push(new_element)
                        else:

                            for da in temp_data:
                                container.push(da)
        else:
            container.push(entity)
    if tag:
        final_axioms = []
        while(container.is_empty()==False):
            final_axioms.append(container.pop())

        for element in final_axioms:
            if (element in restrictions):
                end_node = final_axioms[0]
                new_relation = final_axioms[-1]
                axiom = new_relation+" "+end_node
                final_axioms=[axiom]
                break
        for axiom in final_axioms:
            if axiom!="and" and axiom !="or":
                axiom=first_entity+" "+axiom
                axiom=axiom.split(" ")
                result.append(axiom)
        return result
    else:
        final_axioms = []
        while(container.is_empty()==False):
            axiom =container.pop()
            if axiom !="and" and axiom!="or":
                final_axioms.append(axiom)
        end_node = final_axioms[0]
        new_relation=final_axioms[-1]
        axiom = new_relation+" "+end_node
        axiom=first_entity+" "+axiom
        axiom=axiom.split(" ")
        result.append(axiom)
        return result
def convert_graph(data):
    sentence = split_sentence(data)
    # <http://purl.obolibrary.org/obo/GO_0090141> SubClassOf <http://purl.obolibrary.org/obo/GO_0065007>
    sentence = sentence.split(" ")
    if len(sentence) <3:
        pass
    elif len(sentence)==3:
        result =[[sentence[0], sentence[1], sentence[2]]]
        new_result=[]
        for da in result:
            # <http://purl.obolibrary.org/obo/GO_0090141> SubClassOf <http://purl.obolibrary.org/obo/GO_0065007>
            new_result.append([da[0]," ".join(da[1:-1]),da[-1]])
        return new_result
    else:
        result = convert_triple(sentence)
        new_result=[]
        for da in result:
            new_result.append([da[0]," ".join(da[1:-1]),da[-1]])
        return new_result
def generate_graph(annotation, axiom_file):
    G = nx.Graph()
    # the restriction are min,max,exactly,some,only there are conjunction or disjunction
    with open(axiom_file, "r") as f:
        for line in f.readlines():
            result = convert_graph(line.strip())
            for entities in result:
                # [<http://purl.obolibrary.org/obo/GO_0090141>,SubClassOf,<http://purl.obolibrary.org/obo/GO_0065007>]
                G.add_edge(entities[0].strip(), entities[2].strip())
                G.edges[entities[0].strip(), entities[2].strip()]["type"] = entities[1].strip()
                G.nodes[entities[0].strip()]["val"] = False
                G.nodes[entities[2].strip()]["val"] = False
    with open(annotation, "r") as f:
        for line in f.readlines():
            entities = line.split()
            # print(entities)
            G.add_edge(entities[0].strip(), entities[1].strip())
            G.edges[entities[0].strip(), entities[1].strip()]["type"] = "HasAssociation"
            G.nodes[entities[0].strip()]["val"] = False
            G.nodes[entities[1].strip()]["val"] = False
    return G















































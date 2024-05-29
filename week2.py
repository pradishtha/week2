import streamlit as st
import numpy as np

class Node:
    def __init__(self, attribute=None, value=None, result=None):
        self.attribute = attribute  
        self.value = value          
        self.result = result        
        self.children = {}          

def entropy(class_probabilities):
    entropy = 0
    for probability in class_probabilities:
        if probability != 0:
            entropy -= probability * np.log2(probability)
    return entropy

def calculate_information_gain(data, attribute_index, class_index):
    total_entropy = entropy(np.bincount(data[:, class_index]) / len(data))
    values, counts = np.unique(data[:, attribute_index], return_counts=True)
    weighted_entropy = 0
    for value, count in zip(values, counts):
        subset = data[data[:, attribute_index] == value]
        subset_entropy = entropy(np.bincount(subset[:, class_index]) / len(subset))
        weighted_entropy += (count / len(data)) * subset_entropy
    return total_entropy - weighted_entropy

def build_tree(data, attributes, class_index):
    if len(np.unique(data[:, class_index])) == 1:
        return Node(result=np.unique(data[:, class_index])[0])

    if len(attributes) == 0:
        majority_class = np.argmax(np.bincount(data[:, class_index]))
        return Node(result=majority_class)

    information_gains = [calculate_information_gain(data, i, class_index) for i in range(len(attributes))]
    best_attribute_index = np.argmax(information_gains)
    best_attribute = attributes[best_attribute_index]

    node = Node(attribute=best_attribute)

    remaining_attributes = [attr for i, attr in enumerate(attributes) if i != best_attribute_index]

    values = np.unique(data[:, best_attribute_index])
    for value in values:
        subset = data[data[:, best_attribute_index] == value]
        if len(subset) == 0:
            majority_class = np.argmax(np.bincount(data[:, class_index]))
            node.children[value] = Node(result=majority_class)
        else:
            node.children[value] = build_tree(subset, remaining_attributes, class_index)
    return node

def classify(tree, sample):
    if tree.result is not None:
        return tree.result
    value = sample[tree.attribute]
    if value not in tree.children:
        return tree.result
    return classify(tree.children[value], sample)

def main():
    st.title("ID3 Decision Tree Classifier")

    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        st.sidebar.header("Attributes")
        attributes = st.sidebar.text_input("Enter attributes separated by commas (e.g., Attribute1,Attribute2)")

        if attributes:
            attributes = [attr.strip() for attr in attributes.split(",")]
            class_label = st.sidebar.selectbox("Select class label", attributes)

            if st.sidebar.button("Build Decision Tree"):
                data = np.genfromtxt(uploaded_file, delimiter=",", dtype=str)
                data_numeric = np.zeros_like(data)
                for i in range(len(data[0])):
                    unique_values, mapping = np.unique(data[:, i], return_inverse=True)
                    data_numeric[:, i] = mapping
                data_numeric = data_numeric.astype(int)

                class_index = attributes.index(class_label)
                attributes.remove(class_label)

                tree = build_tree(data_numeric, attributes, class_index)

                st.write("Decision Tree Built!")

                st.sidebar.header("Classify New Sample")
                new_sample = {}
                for attribute in attributes:
                    value = st.sidebar.text_input(attribute)
                    if value:
                        new_sample[attribute] = int(value)

                if st.sidebar.button("Classify"):
                    classification = classify(tree, new_sample)
                    st.write("Classification result:", classification)

if __name__ == "__main__":
    main()

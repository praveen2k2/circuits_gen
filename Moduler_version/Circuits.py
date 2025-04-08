import os 
import pandas as pd
import numpy as np

class Circuits:
    vocab = None
    vocab_to_index = None
    index_to_vocab = None
    graphs = []
    component_lists = []

    def __init__(self):
        self.graphs = []
        self.component_lists = []
        self.load_dataset_files()
        self.create_vocabulary()
        self.convert_components_to_indices()

    



    def load_dataset_files(self):

        start = 1
        end = 3501

        for i in range (start, end+1):
            number = str(i)
            # Define file names
            graph_file = '../Dataset/' + number + '/Graph' + number + '.csv'
            if not os.path.isfile(graph_file):
                # If it doesn't exist, skip the rest of this loop iteration
                continue
            else :
                print()
            # Load the adjacency matrix from the CSV file
            adjacency_matrix = pd.read_csv(graph_file, index_col=0)

            # Convert the DataFrame to a NumPy array
            component_list = adjacency_matrix.columns.tolist()
            matrix = adjacency_matrix.to_numpy()
            self.graphs+= [matrix]
            self.component_lists += [component_list]

            print("Graph " + number ,end='')
            print(component_list,end='')
        return self.component_lists, self.graphs

    def create_vocabulary(self):
        # Create a vocabulary of unique components from the component lists
        flattened_components = [item for sublist in self.component_lists for item in sublist]
        self.vocab = set(flattened_components)

        self.vocab_to_index = {component: idx for idx, component in enumerate(self.vocab)}
        self.index_to_vocab = {idx: component for component, idx in self.vocab_to_index.items()}

    def convert_components_to_indices(self):
        # Ensure vocabulary is created
        if not hasattr(self, 'vocab_to_index') or not self.vocab_to_index:
            self.create_vocabulary()

        # Convert each component list to a list of indices
        self.component_indices = [
            [self.vocab_to_index[component] for component in component_list]
            for component_list in self.component_lists
        ]
        return self.component_indices
    
    def get_component(self, index):
        # Check if the index is valid
        if index < 0 or index >= len(self.index_to_vocab):
            raise IndexError("Index out of range")
        return self.index_to_vocab[index]
    
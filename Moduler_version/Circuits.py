import os 
import pandas as pd
import numpy as np
import torch

class Circuits:
    vocab = None
    vocab_to_index = None
    index_to_vocab = None
    graphs = []
    component_lists = []

    def __init__(self):
        self.graphs = []
        self.pdfiles=[]
        self.component_lists = []
        self.load_dataset_files()
        self.create_vocabulary()
        self.convert_components_to_indices()
        
    def load_dataset_files(self):

        start = 1
        end = 3501
        print("Loading dataset files...")
        for i in range (start, end+1):
            number = str(i)
            # Define file names
            graph_file = '../Dataset/' + number + '/Graph' + number + '.csv'
            if not os.path.isfile(graph_file):
                # If it doesn't exist, skip the rest of this loop iteration
                continue
            # Load the adjacency matrix from the CSV file
            adjacency_matrix = pd.read_csv(graph_file, index_col=0)
            self.pdfiles+=[(i,adjacency_matrix)]
            # Convert the DataFrame to a NumPy array
            component_list = adjacency_matrix.columns.tolist()
            matrix = adjacency_matrix.to_numpy()
            self.graphs+= [matrix]
            self.component_lists += [component_list]
        print("Loaded dataset files successfully.")
        return self.component_lists, self.graphs

    def create_vocabulary(self):
        # Create a vocabulary of unique components from the component lists
        flattened_components = [item for sublist in self.component_lists for item in sublist]
        self.vocab = set(flattened_components)

        self.vocab_to_index = {component: idx for idx, component in enumerate(self.vocab)}
        self.index_to_vocab = {idx: component for component, idx in self.vocab_to_index.items()}
        self.index_to_vocab[len(self.index_to_vocab)]="end"

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
            raise IndexError(f"Index is not in the vocabulary input={index} max={len(self.index_to_vocab)}")
        return self.index_to_vocab[index]
    
    def get_component_fromlist(self,indexlist):
        output=[]
        for index in indexlist:
            index=int(index)
            output+=[self.get_component(index)]
        return output

    def data_lodder(self):
        """
        Returns:List: A sequence of All data
        """
        graph_dataset =  self.graphs
        text_dataset = self.component_indices

        # Convert graph_dataset and text_dataset to numpy arrays
        graph_dataset = np.array(graph_dataset, dtype=object)
        text_dataset = np.array(text_dataset, dtype=object)

        # Convert numpy arrays to PyTorch tensors
        graph_dataset = [torch.tensor(graph, dtype=torch.float32) for graph in graph_dataset]
        graph_dataset = [torch.cat((torch.nn.functional.pad(graph, (0, 310 - graph.size(0))),torch.tensor([[9]*310]))) for graph in graph_dataset]
        text_dataset = [torch.tensor(text+[892], dtype=torch.int) for text in text_dataset]

        # Combine graph_dataset and text_dataset into a single list of tuples
        combined_dataset = list(zip(graph_dataset, text_dataset))

        # Shuffle the combined dataset
        np.random.shuffle(combined_dataset)
    

        # Unzip the shuffled dataset back into graph_dataset and text_dataset
        graph_dataset, text_dataset = zip(*combined_dataset)

        # Convert back to the original data types
        return list(graph_dataset) ,list(text_dataset)
    
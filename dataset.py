import pandas as pd
import numpy as np
import os

def read_netlist(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    netlist = [line.strip().replace('(', '').replace(')', '').split() for line in lines]
    return netlist

def read_ports(filename):
    with open(filename, 'r') as file:
        ports = file.readline().strip().split()
    return ports

def build_connection_matrix(netlist, ports):
    if not netlist or not ports:
        print("Empty netlist or ports")
        return None, None
    
    forbidden_components = ['INVERTER', 'XOR', 'PFD', 'nmos4', 'pmos4']
    if any(component[-1].upper() in forbidden_components for component in netlist):
        print("Skipping due to forbidden component")
        return None, None
    
    if len(netlist) > 5:
        print("Skipping due to netlist size")
        return None, None
    
    component_mapping = {'resistor': 1, 'capacitor': 2, 'inductor': 3, 'diode': 4, 'npn': 7, 'pnp': 8}
    component_list = [component_mapping[comp[-1].lower()] for comp in netlist if comp[-1].lower() in component_mapping]
    
    nodes = ports[:]
    counters = {key: 0 for key in component_mapping.keys()}
    node_types = {
        'npn': lambda i: [f'NPN{i}', f'NPN{i}_C', f'NPN{i}_B', f'NPN{i}_E'],
        'pnp': lambda i: [f'PNP{i}', f'PNP{i}_C', f'PNP{i}_B', f'PNP{i}_E'],
        'resistor': lambda i: [f'R{i}', f'R{i}_P', f'R{i}_N'],
        'capacitor': lambda i: [f'C{i}', f'C{i}_P', f'C{i}_N'],
        'inductor': lambda i: [f'L{i}', f'L{i}_P', f'L{i}_N'],
        'diode': lambda i: [f'DIO{i}', f'DIO{i}_P', f'DIO{i}_N']
    }
    
    for component in netlist:
        component_type = component[-1].lower()
        if component_type in node_types:
            index = counters[component_type]
            counters[component_type] += 1
            nodes.extend(node_types[component_type](index))
    
    matrix = pd.DataFrame(0, index=nodes, columns=nodes)
    return matrix, component_list

circuit_number = "1"  # Change this to process a different circuit
netlist_file = f'Dataset/{circuit_number}/{circuit_number}.cir'
port_file = f'Dataset/{circuit_number}/Port{circuit_number}.txt'

if not os.path.isfile(netlist_file):
    print(f"Netlist file '{netlist_file}' does not exist.")
elif not os.path.isfile(port_file):
    print(f"Port file '{port_file}' does not exist.")
else:
    try:
        netlist = read_netlist(netlist_file)
        ports = read_ports(port_file)
        connection_matrix, component_list = build_connection_matrix(netlist, ports)
        if connection_matrix is not None:
            np.save(f'matrix_1/{circuit_number}.npy', connection_matrix.to_numpy())
            np.save(f'component_lists/{circuit_number}.npy', np.array(component_list))
            print(f"Circuit {circuit_number} processed successfully.")
        else:
            print(f"Circuit {circuit_number} was skipped.")
    except Exception as e:
        print(f"Error processing circuit {circuit_number}: {e}")

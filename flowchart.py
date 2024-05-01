import matplotlib.pyplot as plt
import networkx as nx

def create_flowchart():
    # Create a directed graph
    G = nx.DiGraph()

    # Nodes with their descriptions
    nodes = {
        'Input': 'Source Objects\nSpatial & Lensing Parameters',
        'Form Initial': 'Form Initial Guesses',
        'Optimize Guesses': 'Optimize Guesses',
        'Filter Lenses': 'Filter Lenses',
        'Select Lenses': 'Select Number of Lenses',
        'Merge Lenses': 'Merge Near Lenses',
        'Optimize Strengths': 'Optimize Strengths',
        'Output': 'Candidate Lenses\nPositions, Strengths, Chi^2 Value'
    }

    # Add nodes with labels
    for node, label in nodes.items():
        G.add_node(node, label=label)

    # Define edges between the nodes
    edges = [
        ('Input', 'Form Initial'),
        ('Form Initial', 'Optimize Guesses'),
        ('Optimize Guesses', 'Filter Lenses'),
        ('Filter Lenses', 'Select Lenses'),
        ('Select Lenses', 'Merge Lenses'),
        ('Merge Lenses', 'Optimize Strengths'),
        ('Optimize Strengths', 'Output')
    ]

    G.add_edges_from(edges)

    # Position nodes using the dot layout
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

    # Draw nodes
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
            node_size=3000, node_color='skyblue', font_size=10, font_color='black', font_weight='bold')

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)

    plt.title('Gravitational Lensing Analysis Pipeline')
    plt.axis('off')  # Turn off the axis
    plt.show()

if __name__ == '__main__':
    create_flowchart()

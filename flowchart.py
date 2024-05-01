from graphviz import Digraph

def create_flowchart():
    dot = Digraph(comment='Gravitational Lensing Analysis Pipeline')

    # Adding nodes
    dot.node('A', 'Input: Source Objects\nSpatial & Lensing Parameters')
    dot.node('B', 'Step 1: Form Initial Guesses')
    dot.node('C', 'Step 2: Optimize Guesses')
    dot.node('D', 'Step 3: Filter Lenses')
    dot.node('E', 'Step 4: Select Number of Lenses')
    dot.node('F', 'Step 5: Merge Near Lenses')
    dot.node('G', 'Step 6: Optimize Strengths')
    dot.node('H', 'Output: Candidate Lenses\nPositions, Strengths, Chi^2 Value')

    # Adding edges
    dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH'])

    # Render the graph to a file
    dot.render('flowchart.gv', view=True)

if __name__ == '__main__':
    create_flowchart()

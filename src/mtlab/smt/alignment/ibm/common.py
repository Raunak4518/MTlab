from mtlab.smt.alignment.structures import IBMAlignment

def extract_cepts( alignment:IBMAlignment ) -> dict[int , list[int]]:
    cepts = { i:[] for i in range(alignment.source_length+1)}
    for target_position , source_position in enumerate( alignment.target_to_source):
        cepts[source_position].append(target_position)
    return cepts

def calculate_fertilities(cepts:dict[int,list[int]])-> dict[int , int]:
    return { source_position:len(target_positions) for source_position , target_positions in cepts.items()}

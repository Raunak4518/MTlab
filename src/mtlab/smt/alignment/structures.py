from dataclasses import dataclass
from typing import TypeAlias
WordLinks : TypeAlias = set[tuple[int , int]]
@dataclass
class SentencePair:
    sentence_id:int |str
    source_tokens : list[str]
    target_tokens : list[str]
    @property
    def source_length(self) ->int :
        return len(self.source_tokens)
    @property
    def target_length(self)->int:
        return len(self.target_tokens)
    
    
@dataclass
class IBMAlignment:
    source_length :int
    target_to_source :list[int]
    
    def __post_init__(self):
        pass
    @property
    def target_length(self)->int:
        return len(self.target_to_source)
    def to_links(self):
        return {
            (source_position-1,target_position) for target_position , source_position in enumerate(self.target_to_source) if source_position !=0
        }
    
    @classmethod
    def from_links( cls,links:WordLinks,source_length:int , target_length:int):
        assignments =[0]*target_length
        for source_position , target_position in links:
            if not 0<= source_position<source_length:
                raise ValueError("invalid english position")
            assignments[target_position] = source_position+1
        return cls(source_length,assignments)
def transpose_alignment(links:WordLinks) ->WordLinks:
    return {
        (target_position , source_position) for source_position , target_position in links
    }
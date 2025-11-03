import pytest
import torch
import sys
sys.path.append('../')

from src.data import meshed_domain

data = [
    ((2), 2), ((3, 1), )
]

@pytest.mark.parametrize('point,expect', data)
def 

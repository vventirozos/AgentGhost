import pytest
import sys
from unittest.mock import patch, MagicMock
from ghost_agent.main import parse_args

def test_parse_args_coding_nodes():
    # Simulate command line arguments
    test_args = ["main.py", "--coding-nodes", "http://node1:8000,http://node2:8000"]
    
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        
    assert args.coding_nodes == "http://node1:8000,http://node2:8000"
    
def test_parse_args_coding_nodes_missing():
    # Simulate command line arguments without coding nodes
    test_args = ["main.py"]
    
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        
    assert args.coding_nodes is None

def test_parse_args_image_gen_nodes():
    # Simulate command line arguments
    test_args = ["main.py", "--image-gen-nodes", "http://image_node:8000|lcm-model"]
    
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        
    assert args.image_gen_nodes == "http://image_node:8000|lcm-model"
    assert len(args.image_gen_nodes_parsed) == 1
    assert args.image_gen_nodes_parsed[0]["url"] == "http://image_node:8000"
    assert args.image_gen_nodes_parsed[0]["model"] == "lcm-model"
    
def test_parse_args_image_gen_nodes_missing():
    # Simulate command line arguments without image gen nodes
    test_args = ["main.py"]
    
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        
    assert args.image_gen_nodes is None
    assert getattr(args, "image_gen_nodes_parsed", []) == []

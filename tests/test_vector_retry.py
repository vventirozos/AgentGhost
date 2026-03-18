import pytest
import os
import sys
from unittest.mock import MagicMock, patch, call
from pathlib import Path

# Add the src directory to Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture
def mock_chroma():
    with patch("ghost_agent.memory.vector.chromadb") as mock:
        yield mock

@pytest.fixture
def mock_logging():
    with patch("ghost_agent.memory.vector.logger") as mock:
        yield mock

@patch("os.environ")
@patch("time.sleep")
@patch("sys.exit")
def test_vector_retry_success(mock_sys_exit, mock_sleep, mock_environ, mock_chroma, mock_logging, tmp_path):
    from ghost_agent.memory.vector import VectorMemory
    
    # Mock SentenceTransformerEmbeddingFunction to fail on first attempt, succeed on second
    mock_stef = MagicMock()
    
    call_count = [0]
    def stef_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("Hugging Face download timeout")
        return MagicMock() # Succeed on 2nd attempt
        
    # We patch it where it is used
    with patch("chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction", side_effect=stef_side_effect):
        
        # Initialize VectorMemory without Tor Proxy
        memory_dir = tmp_path / "memory"
        
        vm = VectorMemory(memory_dir=memory_dir, upstream_url="UNUSED", tor_proxy="socks5://127.0.0.1:9050")
        
        # Verify it retried 
        assert call_count[0] == 2
        
        # Verify sleep was called after failure
        mock_sleep.assert_called_once_with(6)
        
        # Verify sys.exit was NOT called
        mock_sys_exit.assert_not_called()

@patch("os.environ")
@patch("time.sleep")
@patch("sys.exit")
def test_vector_retry_exhaustion(mock_sys_exit, mock_sleep, mock_environ, mock_chroma, mock_logging, tmp_path):
    from ghost_agent.memory.vector import VectorMemory
    
    # Mock SentenceTransformerEmbeddingFunction to always fail
    def stef_side_effect(*args, **kwargs):
        raise Exception("Hugging Face download permanent failure")
        
    with patch("chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction", side_effect=stef_side_effect):
        
        memory_dir = tmp_path / "memory"
        
        # It should exit after exhausting 3 attempts
        vm = VectorMemory(memory_dir=memory_dir, upstream_url="UNUSED", tor_proxy="socks5://127.0.0.1:9050")
        
        # Sleep should be called twice (after attempt 1 and 2)
        assert mock_sleep.call_count == 2
        
        # sys.exit should be called with 1
        mock_sys_exit.assert_called_once_with(1)

@patch("time.sleep")
@patch("sys.exit")
def test_vector_tor_proxy_ignored_for_hf(mock_sys_exit, mock_sleep, mock_chroma, mock_logging, tmp_path):
    from ghost_agent.memory.vector import VectorMemory
    
    with patch("chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction") as mock_stef:
        with patch.dict(os.environ, clear=True):
            memory_dir = tmp_path / "memory"
            tor_proxy = "socks5://127.0.0.1:9050"
            
            # This should not set HTTP_PROXY or HTTPS_PROXY anymore
            vm = VectorMemory(memory_dir=memory_dir, upstream_url="UNUSED", tor_proxy=tor_proxy)
            
            assert "HTTP_PROXY" not in os.environ
            assert "HTTPS_PROXY" not in os.environ

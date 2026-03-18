import pytest
import pickle
import sys
import os
from pathlib import Path
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

# Ensure src is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ghost_agent.main import proactive_runner

def test_proactive_runner_is_picklable():
    """Verify that proactive_runner can be pickled."""
    try:
        pickled = pickle.dumps(proactive_runner)
        unpickled = pickle.loads(pickled)
        assert unpickled.__name__ == "proactive_runner"
    except Exception as e:
        pytest.fail(f"proactive_runner is not picklable: {e}")

@pytest.mark.asyncio
async def test_scheduler_job_store_integration(tmp_path):
    """Verify that proactive_runner can be added to a persistent job store."""
    db_path = tmp_path / "test_scheduler.db"
    db_url = f"sqlite:///{db_path}"
    
    jobstores = {'default': SQLAlchemyJobStore(url=db_url)}
    scheduler = AsyncIOScheduler(jobstores=jobstores)
    
    try:
        scheduler.start()
        
        # Attempt to add the job
        job = scheduler.add_job(
            proactive_runner, 
            'interval', 
            seconds=60, 
            args=["test_task", "test_prompt"], 
            id="test_job_1",
            replace_existing=True
        )
        
        assert job.id == "test_job_1"
        
        # Verify it persisted without error (implicit in add_job not raising)
        
    except Exception as e:
        pytest.fail(f"Failed to add job to scheduler with persistent store: {e}")
    finally:
        scheduler.shutdown()

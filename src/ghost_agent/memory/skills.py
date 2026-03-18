import json
import logging
import threading
import os
from pathlib import Path
from datetime import datetime
from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

class SkillMemory:
    def __init__(self, memory_dir: Path):
        self.file_path = memory_dir / "skills_playbook.json"
        self._lock = threading.RLock()
        if not self.file_path.exists():
            self.save_playbook([])

    def save_playbook(self, playbook):
        # Helper for atomic save
        with self._lock:
            temp_path = self.file_path.with_suffix('.tmp')
            temp_path.write_text(json.dumps(playbook, indent=2))
            os.replace(temp_path, self.file_path)

    def learn_lesson(self, task: str, mistake: str, solution: str, memory_system=None):
        try:
            with self._lock:
                try:
                    content = self.file_path.read_text()
                    playbook = json.loads(content) if content else []
                except:
                    playbook = []
            
            new_lesson = {
                "timestamp": datetime.now().isoformat(),
                "task": task,
                "mistake": mistake,
                "solution": solution
            }
            # Keep only the last 50 high-value lessons in the JSON backup
            playbook = [new_lesson] + playbook[:49]
            
            self.save_playbook(playbook)
            
            # Index in Vector Memory for Semantic Retrieval
            if memory_system:
                lesson_text = f"SITUATION: {task}\nMISTAKE: {mistake}\nSOLUTION: {solution}"
                memory_system.add(lesson_text, {"type": "skill", "timestamp": new_lesson["timestamp"]})
            
            pretty_log("SKILL ACQUIRED", f"Lesson learned: {task[:30]}...", icon="🎓")
        except Exception as e:
            logger.error(f"Failed to save skill: {e}")

    def get_playbook_context(self, query: str = None, memory_system = None) -> str:
        try:
            if memory_system and query:
                # use semantic search (this part doesn't touch the file, so no lock needed purely for this block)
                # But strict instruction says "Wrap... get_playbook_context within a with self._lock:"
                # However, semantic search is external. Locking around it might be unnecessary blocking.
                # But if we follow instructions:
                
                # The instruction: "Wrap the file read/write operations in... get_playbook_context() within a with self._lock: context manager."
                # It says "Wrap the file read/write operations", not "Wrap the entire function".
                # The semantic path does NOT read/write the file. 
                # The fallback path DOES.
                
                # I will lock only the file reading part in the fallback.
                
                results = memory_system.collection.query(
                    query_texts=[query],
                    n_results=5,
                    where={"type": "skill"}
                )
                
                if results['documents'] and results['documents'][0]:
                    valid_lessons = []
                    # Threshold: 0.65 (Only highly relevant lessons)
                    for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
                        if dist < 0.65:
                            valid_lessons.append(doc)
                    
                    if valid_lessons:
                        context = "## RELEVANT LESSONS LEARNED (Follow these to avoid repeats):\n"
                        for i, doc in enumerate(valid_lessons):
                            context += f"{i+1}. {doc}\n"
                        return context
                    else:
                        return ""

            # Fallback to recent lessons if no vector search or no results
            with self._lock:
                try:
                    playbook = json.loads(self.file_path.read_text())
                except: playbook = []
                
            if not playbook: return "No lessons learned yet."
            
            context = "## RECENT LESSONS LEARNED (Follow these to avoid repeats):\n"
            for i, p in enumerate(playbook[:5]): # Only inject top 5 for efficiency
                context += f"{i+1}. SITUATION: {p['task']}\n   PREVIOUS MISTAKE: {p['mistake']}\n   THE FIX: {p['solution']}\n"
            return context
        except: return ""

    def get_recent_failures(self, limit: int = 5) -> str:
        """Fetch the most recent mistakes and their tasks to generate targeted self-play scenarios."""
        try:
            with self._lock:
                try:
                    content = self.file_path.read_text()
                    playbook = json.loads(content) if content else []
                except:
                    playbook = []
            
            if not playbook:
                return "No recent failures recorded."
                
            recent_lessons = playbook[:limit]
            context = "## RECENT MISTAKES:\n"
            for p in recent_lessons:
                context += f"- TASK: {p.get('task')}\n  ERROR/MISTAKE: {p.get('mistake')}\n\n"
            return context.strip()
        except:
            return "Failed to load recent failures."
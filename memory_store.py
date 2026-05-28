class MemoryStore:
    def __init__(self):
        self._facts = {}
        self._episodic = {}

    def get_facts(self, user_id: str) -> list[str]:
        return list(self._facts.get(user_id, []))
    
    def add_fact(self, user_id: str, fact: str):
        if user_id not in self._facts:
            self._facts[user_id] = []
        self._facts[user_id].append(fact)

    def get_episodic_summary(self, user_id: str) -> str:
        return self._episodic.get(user_id, "")
    
    def update_episodic_summary(self, user_id: str, summary: str):
        self._episodic[user_id] = summary
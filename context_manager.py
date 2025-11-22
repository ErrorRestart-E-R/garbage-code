from collections import deque
import datetime

class ConversationManager:
    def __init__(self, max_history=20):
        self.participants = set()
        self.history = deque(maxlen=max_history)
        self.last_speaker = None
        self.last_interaction_time = datetime.datetime.now()

    def add_participant(self, user_name):
        self.participants.add(user_name)

    def remove_participant(self, user_name):
        if user_name in self.participants:
            self.participants.remove(user_name)

    def add_message(self, user_name, text):
        timestamp = datetime.datetime.now()
        self.history.append({
            "user": user_name,
            "text": text,
            "timestamp": timestamp
        })
        self.last_speaker = user_name
        self.last_interaction_time = timestamp

    def get_active_participants_count(self):
        return len(self.participants)

    def analyze_situation(self):
        count = self.get_active_participants_count()
        if count <= 1:
            return "1:1"
        elif count <= 4:
            return "Small Group"
        else:
            return "Large Crowd"

    def get_system_context(self):
        situation = self.analyze_situation()
        participants_list = ", ".join(self.participants) if self.participants else "No one"
        
        context = f"Current Situation: {situation} conversation.\n"
        context += f"Participants: {participants_list}.\n"
        
        if self.history:
            context += "\nRecent Conversation History:\n"
            for msg in self.history:
                time_str = msg['timestamp'].strftime("%H:%M:%S")
                context += f"[{time_str}] {msg['user']}: {msg['text']}\n"
            
            last_msg = self.history[-1]
            time_diff = (datetime.datetime.now() - last_msg['timestamp']).total_seconds()
            if time_diff > 60:
                context += "\nIt has been a while since the last message.\n"
        
        return context

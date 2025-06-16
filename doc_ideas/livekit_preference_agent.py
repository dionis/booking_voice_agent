import asyncio
import logging
from typing import List, Dict, Optional
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero
import sqlite3
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreferenceDatabase:
    """Mock database class for demonstration"""
    
    def __init__(self):
        # Initialize in-memory SQLite database
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self._setup_database()
    
    def _setup_database(self):
        """Setup sample database with preferences and items"""
        cursor = self.conn.cursor()
        
        # Create preferences table
        cursor.execute('''
            CREATE TABLE preferences (
                id INTEGER PRIMARY KEY,
                category TEXT,
                name TEXT,
                description TEXT
            )
        ''')
        
        # Create items table
        cursor.execute('''
            CREATE TABLE items (
                id INTEGER PRIMARY KEY,
                name TEXT,
                category TEXT,
                preference_id INTEGER,
                description TEXT,
                FOREIGN KEY (preference_id) REFERENCES preferences (id)
            )
        ''')
        
        # Insert sample preferences
        preferences = [
            (1, 'cuisine', 'Italian', 'Italian cuisine preferences'),
            (2, 'cuisine', 'Asian', 'Asian cuisine preferences'),
            (3, 'cuisine', 'Mexican', 'Mexican cuisine preferences'),
            (4, 'activity', 'Indoor', 'Indoor activities'),
            (5, 'activity', 'Outdoor', 'Outdoor activities'),
            (6, 'music', 'Rock', 'Rock music genre'),
            (7, 'music', 'Jazz', 'Jazz music genre'),
            (8, 'music', 'Classical', 'Classical music genre')
        ]
        
        cursor.executemany(
            'INSERT INTO preferences (id, category, name, description) VALUES (?, ?, ?, ?)',
            preferences
        )
        
        # Insert sample items
        items = [
            (1, 'Pizza Margherita', 'cuisine', 1, 'Classic Italian pizza'),
            (2, 'Pasta Carbonara', 'cuisine', 1, 'Traditional Roman pasta'),
            (3, 'Sushi Roll', 'cuisine', 2, 'Japanese sushi'),
            (4, 'Pad Thai', 'cuisine', 2, 'Thai noodle dish'),
            (5, 'Tacos', 'cuisine', 3, 'Mexican street food'),
            (6, 'Chess', 'activity', 4, 'Strategic board game'),
            (7, 'Reading', 'activity', 4, 'Indoor leisure activity'),
            (8, 'Hiking', 'activity', 5, 'Outdoor adventure'),
            (9, 'Led Zeppelin Songs', 'music', 6, 'Classic rock band'),
            (10, 'Miles Davis Albums', 'music', 7, 'Jazz legend recordings')
        ]
        
        cursor.executemany(
            'INSERT INTO items (id, name, category, preference_id, description) VALUES (?, ?, ?, ?, ?)',
            items
        )
        
        self.conn.commit()
    
    def get_preference_categories(self) -> List[str]:
        """Get all available preference categories"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT DISTINCT category FROM preferences')
        return [row[0] for row in cursor.fetchall()]
    
    def get_preferences_by_category(self, category: str) -> List[Dict]:
        """Get preferences for a specific category"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT id, name, description FROM preferences WHERE category = ?',
            (category,)
        )
        return [
            {'id': row[0], 'name': row[1], 'description': row[2]}
            for row in cursor.fetchall()
        ]
    
    def search_items_by_preference(self, preference_id: int) -> List[Dict]:
        """Search items based on selected preference"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT name, description FROM items WHERE preference_id = ?',
            (preference_id,)
        )
        return [
            {'name': row[0], 'description': row[1]}
            for row in cursor.fetchall()
        ]

class PreferenceAgent:
    """Agent for handling user preference selection and database search"""
    
    def __init__(self):
        self.db = PreferenceDatabase()
        self.current_state = "initial"  # initial, category_selection, preference_selection, results
        self.selected_category = None
        self.available_preferences = []
    
    async def get_preference_options(self, category: Optional[str] = None) -> str:
        """Get available preference options"""
        if category:
            preferences = self.db.get_preferences_by_category(category)
            self.available_preferences = preferences
            self.current_state = "preference_selection"
            
            options_text = f"Here are the available {category} preferences:\n"
            for i, pref in enumerate(preferences, 1):
                options_text += f"{i}. {pref['name']} - {pref['description']}\n"
            options_text += "\nPlease tell me which option you'd like to select by saying the number or name."
            
            return options_text
        else:
            categories = self.db.get_preference_categories()
            self.current_state = "category_selection"
            
            options_text = "I can help you find items based on your preferences. Here are the available categories:\n"
            for i, cat in enumerate(categories, 1):
                options_text += f"{i}. {cat.title()}\n"
            options_text += "\nWhich category interests you? You can say the number or category name."
            
            return options_text
    
    async def handle_selection(self, user_input: str) -> str:
        """Handle user selection based on current state"""
        user_input = user_input.lower().strip()
        
        if self.current_state == "category_selection":
            categories = self.db.get_preference_categories()
            
            # Try to match by number
            try:
                selection_num = int(user_input)
                if 1 <= selection_num <= len(categories):
                    self.selected_category = categories[selection_num - 1]
                    return await self.get_preference_options(self.selected_category)
            except ValueError:
                pass
            
            # Try to match by name
            for category in categories:
                if category.lower() in user_input:
                    self.selected_category = category
                    return await self.get_preference_options(self.selected_category)
            
            return "I didn't understand your selection. Please choose a category by number or name."
        
        elif self.current_state == "preference_selection":
            # Try to match by number
            try:
                selection_num = int(user_input)
                if 1 <= selection_num <= len(self.available_preferences):
                    selected_pref = self.available_preferences[selection_num - 1]
                    return await self.search_database(selected_pref['id'])
            except ValueError:
                pass
            
            # Try to match by name
            for pref in self.available_preferences:
                if pref['name'].lower() in user_input:
                    return await self.search_database(pref['id'])
            
            return "I didn't understand your selection. Please choose a preference by number or name."
        
        return "I'm not sure what you're trying to select. Let's start over."
    
    async def search_database(self, preference_id: int) -> str:
        """Search database based on selected preference"""
        items = self.db.search_items_by_preference(preference_id)
        self.current_state = "results"
        
        if not items:
            return "I couldn't find any items matching that preference. Would you like to try another selection?"
        
        result_text = f"Great! I found {len(items)} items based on your preference:\n\n"
        for item in items:
            result_text += f"• {item['name']}: {item['description']}\n"
        
        result_text += "\nWould you like to search for something else or explore different preferences?"
        
        return result_text
    
    async def reset_conversation(self):
        """Reset the conversation state"""
        self.current_state = "initial"
        self.selected_category = None
        self.available_preferences = []

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent"""
    logger.info("Starting preference selection agent")
    
    # Initialize the preference agent
    pref_agent = PreferenceAgent()
    
    # Setup voice assistant with OpenAI
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4"),
        tts=openai.TTS(),
        chat_ctx=llm.ChatContext().append(
            role="system",
            text=(
                "You are a helpful assistant that helps users select preferences and search a database. "
                "When users want to start, call the get_preference_options function. "
                "When they make a selection, call the handle_selection function with their input. "
                "Be conversational and helpful. Keep responses concise but friendly."
            )
        ),
        fnc_ctx=llm.FunctionContext().with_functions(
            get_preference_options=llm.FunctionInfo(
                description="Get available preference options, optionally for a specific category",
                parameters={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Optional category to filter preferences"
                        }
                    }
                },
                handler=pref_agent.get_preference_options
            ),
            handle_selection=llm.FunctionInfo(
                description="Handle user selection input",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_input": {
                            "type": "string",
                            "description": "The user's selection input"
                        }
                    },
                    "required": ["user_input"]
                },
                handler=pref_agent.handle_selection
            ),
            reset_conversation=llm.FunctionInfo(
                description="Reset the conversation to start over",
                parameters={"type": "object", "properties": {}},
                handler=pref_agent.reset_conversation
            )
        )
    )
    
    # Connect to the room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Start the assistant
    assistant.start(ctx.room)
    
    # Wait for the first participant to join
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")
    
    # Greet the user
    await assistant.say(
        "Hello! I'm here to help you find items based on your preferences. "
        "Would you like to explore our available options?"
    )
    
    # Keep the agent running
    await asyncio.Future()  # Run forever

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=None
        )
    )

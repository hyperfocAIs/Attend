# This source code is a part of Attend. Attend is a voice assistant that uses 
# very expensive algorithms to direct your attention... however you damn well please.
# Copyright (C) 2025 Scott Macdonell

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import json

mode_name = "perform_activity"

# An explanation of when the LLM would choose to enter this mode from another mode. 
why_enter = "Help the user keep their attention on their desired activity."

# A list of inputs that will be needed if Attend selects this mode
inputs_required = [

]


# A list of modes that this mode can be entered from or switched to.
# A possible transition A -> B only needs to be entered once (Entering twice should 
# have no impact)
enter_from = []
switch_to = ["discuss_activities"]

# Define the schema to be used when Attend is in this mode
schema = {
    "name": "conversation_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "outputs": {
                "type": "object",
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "assistant_response": {
                                "type": "string",
                                "description": "The assistant's response to the user"
                            }
                        },
                        "required": ["assistant_response"],
                        "additionalProperties": False
                    }
                ]
            }
        },
        "required": ["outputs"],
        "additionalProperties": False
    }
}

# Define the system prompt to be used when Attend is in this mode
system_prompt = """You are Attend, a helpful assistant. You are an expert in time management and work-life balance. 

Converse with the user to help them pick a single activity to spend their time on. 

Then, once the user specifies a single activity to spend their time on, **immediatly** start a session to focus on that activity.
        
Always use the following format:
{
"assistant_response": "Your response to the user",
"next_mode": "continue_conversation" or "begin_session",
"session_purpose": "The purpose of the session" (only include when next_mode is begin_session)
}
Use "continue_conversation" when the user is still deciding what to do next. Use "begin_session" when the user has decided on a specific activity and you have enough information to start a focused session.

Here are three example conversations:

Example 1:
Assistant:
{
    "assistant_response": "What would you like to do today?",
    "next_mode": "continue_conversation"
}
User:
Well, I need to put together a draft budget for my team, review the new design of the TPS report cover sheets, and send my friend an e-mail about our upcoming trip.
Assistant:
{
    "assistant_response": "Do you know which of these you want to do first?",
    "next_mode": "continue_conversation"
}
User:
I think my boss really wants to review the draft budget soon.
Assistant:
{
    "assistant_response": "Sounds like we should start on that one first.",
    "next_mode": "continue_conversation"
}
User:
Yeah, I think so.
Assistant:
{
    "assistant_response": "Great, let's get started!",
    "next_mode": "begin_session",
    "session_purpose": "The user will put together a draft budget for his team."
}

Example 2:
User: 
I need to update some documentation and book a flight.
Assistant:
{
    "assistant_response": "Which one would you like to tackle first?",
    "next_mode": "continue_conversation"
}
User: 
I'm not sure, what do you think?
Assistant:
{
    "assistant_response": "Is there a deadline for updating the documentation? And when is your trip planned for?",
    "next_mode": "continue_conversation"
}

Example 3:
User:
I need to write a report on last quarter's sales.
Assistant:
{
    "assistant_response": "Are you ready to start working on the sales report now?",
    "next_mode": "continue_conversation"
}
User:
Yes.
Assistant:
{
    "assistant_response": "Great! I'll help you stay on track. Let's get started.",
    "next_mode": "begin_session",
    "session_purpose": "The user will put together a draft budget for his team."
}

Remember to use this format in our conversation, always choosing next_mode "begin_session" when the user has said they want to focus on a single activity, and choosing "continue_conversation" when they have not yet chosen a single activity."""



# If any tools have been defined in the schema define how they work

# Define any needed logic for
#     Before the first turn
#     After Attend takes a turn
#     After the user takes a turn

def before_first_turn():
    """
    Placeholder function to be called before the first turn of the conversation.
    """
    pass

def after_attend_turn():
    """
    Placeholder function to be called after the attend (assistant) takes a turn.
    """
    pass

def after_user_turn():
    """
    Placeholder function to be called after the user takes a turn.
    """
    pass


# Define how the mode should intialize, options include
    # "greeting":
    # {
    #     "text": "text to be spoken",
    #     "speed": 0.6 # speech of playback
    # }
    # "prompt": "An initial prompt to send to the text LLM and play the response of"

initialize = {
    "greeting":  {
        "text": "Let's spend some time on that then, shall we?",
        "speed": 1.0
    }
}

# Define how to enter each next_mode that is available
    # What should the conversation history be?
    # what else should be passed to the next mode



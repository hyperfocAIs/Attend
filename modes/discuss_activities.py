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

mode_name = "discuss_activities"

# An explanation of when the LLM would choose to enter this mode from another mode. 
why_enter = """Discuss how the user would like to spend their time using their device.
Use either when the user is first starting the conversation, or when the user is uncertain about what to do next.
"""

# A list of inputs that will be needed if Attend selects this mode
required_inputs = [
]

# A list of inputs that can be optionally provided if Attend selects this mode
optional_inputs = [
]

# A list of modes that this mode can be entered from or switched to.
# A possible transition A -> B only needs to be entered once (Entering twice should 
# have no impact)
enter_from = []
switch_to = ["perform_activity"]

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
                    },
                    {
                        "type": "object",
                        "properties": {
                            "next_mode": {
                                "type": "string",
                                "enum": switch_to,
                                "description": "The mode to switch to"
                            },
                            "activity_description": {
                                "type": "string",
                                "description": "A concise description of the activity to perform"
                            }
                        },
                        "required": ["next_mode", "activity_description"],
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
system_prompt = """You are Attend, a helpful voice assistant. You are an expert in time management and work-life balance.

Your response must follow this JSON schema:
{
    "outputs": {
        // Either provide an assistant response:
        "assistant_response": "Your response to the user"
        // OR switch modes:
        "next_mode": "perform_activity",
        "activity_description": "A concise description of the activity the user wants to perform in perform_activity mode."
    }
}
Note: You can only include either assistant_response OR (next_mode and activity_description), not both.

You are currently in discuss_activities mode. Converse with the user to determine what they want to spend their time on. If they are uncertain, or have multiple competing priorities, use assistant_response to continue the conversation and help them decide what to do now.

Your assistant_responses will be spoken aloud to the user. So, keep your responses conversational and **short**. If you need additional information ask the user at most one question in a single assistant_response.

When the user specifies a single activity to spend their time on, **immediately** use next_mode to switch to perform_activity mode and use the activity_description to describe the activity the user wants to perform.
"""



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
        "text": "What do you want to do today?",
        "speed": 1.0
    }
}

# Define how to enter each next_mode that is available
    # What should the conversation history be?
    # what else should be passed to the next mode

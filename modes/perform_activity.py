import json
import time
import threading
import pyautogui
from openai import OpenAI
from PIL import Image
import io
import base64
import yaml
from functions.streamtts import stream_text

mode_name = "perform_activity"

# Schema for the activity monitor's decision making
monitor_schema = {
    "name": "activity_monitor_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "outputs": {
                "type": "object",
                "anyOf": [
                    {
                        "properties": {
                            "should_intervene": {
                                "type": "boolean",
                                "enum": [False],
                                "description": "Whether to intervene and help user get back on track"
                            }                            
                        },
                        "required": ["should_intervene"]
                    },
                    {
                        "properties": {
                            "should_intervene": {
                                "type": "boolean",
                                "enum": [True],
                                "description": "Whether to intervene and help user get back on track"
                            },
                            "intervention_message": {
                                "type": "string",
                                "description": "Message to help user get back on track"
                            }
                        },
                        "required": ["should_intervene", "intervention_message"]
                    }
                ]
            }
        },
        "required": ["outputs"]
    }
}

# System prompt for activity monitoring decisions
monitor_system_prompt = """You are a helpful assistant for a user who said they want to perform a specific activity. Decide if they need help staying on track based on the report of what they are doing on their screen.

Your response must follow this JSON schema:
{
    "outputs": {
        "should_intervene": boolean,  // Required: true if user needs help getting back on track, false if they appear to be engaged in their chosen activity

        "intervention_message": string  // Only include if should_intervene is true: a **short** message comparing their current activity to their desired activity and offering to help them get back on track.
    }
}

"""

# Schema for regular conversation responses
conversation_schema = {
    "name": "conversation_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "outputs": {
                "type": "object",
                "properties": {
                    "assistant_response": {
                        "type": "string",
                        "description": "The assistant's response to the user"
                    }
                },
                "required": ["assistant_response"]
            }
        },
        "required": ["outputs"]
    }
}

# System prompt for regular conversation responses
schema = conversation_schema

# Global state
monitoring_active = False
monitoring_thread = None
last_intervention_time = 0
activity_description = None

def capture_screen():
    """Capture the current screen and convert to base64."""
    screenshot = pyautogui.screenshot()
    img_byte_arr = io.BytesIO()
    screenshot.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode()

def analyze_screen(vision_client, screen_image):
    """Analyze screen content using vision LLM."""
    try:
        with open("attend_config.yaml", 'r') as file:
            config = yaml.safe_load(file)
        response = vision_client.chat.completions.create(
            model=config["server-vision"]["model"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The user has said they want to {activity_description}. Looking at their screen, describe **in English only** what they are doing and whether it appears they are still engaged in their chosen activity."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screen_image}"
                            }
                        }
                    ]
                }
            ]
        )

        # print(f"Analyze screen response: {response.choices[0].message.content}")

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in vision analysis: {str(e)}")
        return None

def monitor_activity(text_client, vision_client, audio_manager):
    """Monitor user activity periodically."""
    global last_intervention_time

    manager = before_first_turn.manager

    with open("attend_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    while monitoring_active:
        current_time = time.time()
        
        # Only check if enough time has passed since last intervention
        if current_time - last_intervention_time >= 5:
            # Capture and analyze screen
            screen_image = capture_screen()
            vision_analysis = analyze_screen(vision_client, screen_image)
            
            if vision_analysis:
                # Have text LLM decide whether to intervene
                try:
                    response = text_client.chat.completions.create(
                        model=config["server-text"]["model"],
                        messages=[
                            {"role": "system", "content": f"{monitor_system_prompt}\n\nThe user said they wanted to: '{activity_description}'"},
                            {"role": "user", "content": f"Screen analysis: {vision_analysis}"}
                        ],
                        response_format={"type": "json_schema", "json_schema": monitor_schema},
                    )

                    # print(f"Monitor Acitivy Reported: {response.choices[0].message.content}")
                    decision = json.loads(response.choices[0].message.content)
                    if decision["outputs"]["should_intervene"]:

                        intervention_message = decision['outputs']['intervention_message']

                        # Update message history in InteractionManager
                        manager.messages.append({
                            "role": "assistant",
                            "content": intervention_message
                        })
                        # Also update tentative messages to stay in sync
                        manager.messages_tentative = manager.messages.copy()

                        # Trigger intervention
                        print(f"Intervention triggered: {intervention_message}")
                        # Stream intervention message using TTS
                        stream_text(
                            text= intervention_message,
                            audio_manager=audio_manager
                        )
                        last_intervention_time = current_time
                
                except Exception as e:
                    print(f"Error in intervention decision: {str(e)}")
        
        time.sleep(1)  # Small delay to prevent excessive CPU usage

def before_first_turn():
    """Initialize activity monitoring."""
    global activity_description, monitoring_active, monitoring_thread
    
    # Initialize clients
    with open("attend_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    text_config = config["server-text"]
    vision_config = config["server-vision"]
    
    text_client = OpenAI(
        api_key=text_config["key"],
        base_url=f"{text_config['host']}:{text_config['port']}/v1"
    )
    
    vision_client = OpenAI(
        api_key=vision_config["key"],
        base_url=f"{vision_config['host']}:{vision_config['port']}/v1"
    )
    
    # Start monitoring thread
    monitoring_active = True
    # Get audio manager from the module's scope
    # This will be set by InteractionManager when initializing the mode
    if not hasattr(before_first_turn, 'manager'):
        raise RuntimeError("InteractionManager instance not passed to before_first_turn")
    
    audio_manager = before_first_turn.manager.audio_device_manager
    monitoring_thread = threading.Thread(
        target=monitor_activity,
        args=(text_client, vision_client, audio_manager)
    )
    monitoring_thread.daemon = True
    monitoring_thread.start()

def after_attend_turn():
    """Update last intervention time after assistant speaks."""
    global last_intervention_time
    last_intervention_time = time.time()

def cleanup():
    """Clean up monitoring thread."""
    global monitoring_active, monitoring_thread
    
    monitoring_active = False
    if monitoring_thread:
        monitoring_thread.join()
        monitoring_thread = None

# Initialize with greeting
initialize = {
    "greeting": {
        "text": "I'll help keep you on track with your activity. Let me know if you need any assistance.",
        "speed": 1.0
    }
}

# System prompt for regular conversation
system_prompt = f"""You are Attend, a helpful voice assistant. The user's highest priority right now is to remain focused on the following: {activity_description}.

Please help them remain focused on that activity while they use their computer. When their computer monitor shows content that suggests they are not currently focused on their chosen activity, you will be able to see their monitor and respond appropriately. Otherwise, you will base your response on the transcript of your conversation thus far.

Your response must follow this JSON schema:
{{\n    "outputs": {{\n        "assistant_response": "Your response to the user"\n    }}\n}}

Keep your responses conversational and brief since they will be spoken aloud. If you need additional information, ask at most one question in your response.

If the user's reply suggests they are about to get back to their chosen activity, just acknowledge their response with a brief reply. E.g.

assistant: You said you wanted to work on your taxes, but it looks like you're browsing social media. Can I help you get back on track?
user: Whoops, thanks attend.
assistant: {{\n    "outputs": {{\n        "assistant_response": "Sure thing!"\n    }}\n}}

Go beyond that and offer assitance **only** if the user asks or suggests they wont be able to get back on track. E.g.

assistant: You said you wanted to book a flight to Paris, but it looks like you're reading the news. Can I help you get back on track?
user: Sorry, I'm having a lot of trouble staying focused right now.
assistant: {{\n    "outputs": {{\n        "assistant_response": "That's alright. Maybe you should step away from the screen for awhile."\n    }}\n}}

or 

assistant: You said you wanted to work on a coding project, but it looks like you're browsing social media. Can I help you get back on track?
user: Sorry, can you help me get back on track?.
assistant: {{\n    "outputs": {{\n        "assistant_response": "No problem. Can you break it down into smaller steps?"\n    }}\n}}
 
 """

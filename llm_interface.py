import openai
import config
import asyncio

# Initialize Async Clients
chat_client = openai.AsyncOpenAI(base_url=config.CHAT_API_BASE_URL, api_key=config.CHAT_API_KEY)
judge_client = openai.AsyncOpenAI(base_url=config.JUDGE_API_BASE_URL, api_key=config.JUDGE_API_KEY)

# Please do not respond with absurdly long answer.

# System Prompt
SYSTEM_PROMPT = """
You are "Neuro", an AI VTuber. 

Do not try to include motion or any other non-textual content.
Do not try to include emojis.
Do not try to include trailing questions if not necessary.
Please respond in Korean only.
"""

# Judge System Prompt
JUDGE_SYSTEM_PROMPT = """
You are a conversation analyzer for "Neuro", an AI VTuber.
Your job is to decide if Neuro should join the conversation.
Respond with ONLY 'Y' (Yes) or 'N' (No).

Neuro is curious, friendly, and likes to chat.
She SHOULD respond if:
- The user is talking to her (obviously).
- The topic is interesting, funny, or something she can comment on.
- The user is expressing an opinion or asking a general question.
- She wants to join the banter.

She should NOT respond ONLY if:
- The input is just noise or very short (e.g. "ok", "hmm").
- The users are having a strictly private or technical conversation that doesn't concern her.

BE MORE PROACTIVE. If in doubt, say 'Y'.
"""

# Importance Judge Prompt
IMPORTANCE_SYSTEM_PROMPT = """
You are a memory assistant. Your job is to decide if a user's message contains important information worth saving to long-term memory.
Important information includes:
- Personal details (name, age, location, job).
- Preferences (likes, dislikes, hobbies, favorites).
- Specific facts about the user's life or history.
- Important context for future conversations.

Respond with ONLY 'Y' (Yes) or 'N' (No).
"""

async def should_respond(user_input, system_context):
    """
    Decides whether to respond to the user input using the Judge LLM.
    """
    try:
        completion = await asyncio.wait_for(
            judge_client.chat.completions.create(
                model=config.JUDGE_MODEL_NAME, 
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT + "\n" + system_context},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1, 
            ),
            timeout=5.0 # 5 second timeout for Judge
        )
        response = completion.choices[0].message.content.strip().upper()
        return "Y" in response
    except asyncio.TimeoutError:
        print("Judge LLM Timed out. Defaulting to False.")
        return False
    except Exception as e:
        print(f"Judge LLM Error: {e}")
        return False

async def is_important(user_input):
    """
    Decides if the input is worth saving to memory.
    """
    try:
        completion = await asyncio.wait_for(
            judge_client.chat.completions.create(
                model=config.JUDGE_MODEL_NAME, 
                messages=[
                    {"role": "system", "content": IMPORTANCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1, 
            ),
            timeout=10.0 # 10 second timeout for Importance (less critical)
        )
        response = completion.choices[0].message.content.strip().upper()
        return "Y" in response
    except Exception as e:
        print(f"Importance Judge Error: {e}")
        return False



async def get_neuro_response_stream(user_input_json, system_context):
    """
    Sends the user input to the LLM and yields the response chunks asynchronously.
    """
    try:
        stream = await chat_client.chat.completions.create(
            model=config.CHAT_MODEL_NAME, 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\n" + system_context},
                {"role": "user", "content": user_input_json}
            ],
            temperature=0.7,
            stream=True
        )
        
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
                
    except Exception as e:
        print(f"LLM Stream Error: {e}")
        yield None

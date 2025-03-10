import uuid
import os
import boto3
import json
from flask import Flask, request, jsonify, session

app = Flask(__name__)

# AWS & Claude Configuration
AWS_REGION = "eu-west-3" 
KNOWLEDGE_BASE_ID = "Q4EBOW8NKM"
BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"  
os.environ["AWS_ACCESS_KEY_ID"] = "ASIATWBJZ4G6AV3F6NVX"
os.environ["AWS_SECRET_ACCESS_KEY"] = "gNaVq82CevdSJuMXOMlFBGDx4VXijVX7MqXXSSbe"

# Initialize AWS SDK clients
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# In-memory store for user threads
user_threads = {}


def retrieve_legal_knowledge(query):
    """Fetch relevant legal information from AWS Knowledge Base"""
    response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": query},
    )
    documents = response.get("retrievalResults", [])
    knowledge_texts = [doc["content"]["text"] for doc in documents]
    
    return "\n".join(knowledge_texts) if knowledge_texts else "No relevant legal information found."


def chat_with_claude(user_id, user_message):
    """Handle chat session with Claude via AWS Bedrock"""
    
    if user_id not in user_threads:
        user_threads[user_id] = []

    # Retrieve legal knowledge
    legal_info = retrieve_legal_knowledge(user_message)

    # Build chat history
    user_threads[user_id].append({"role": "user", "content": user_message})
    
    prompt = f"""
    You are a knowledgeable Tunisian lawyer. Answer in a professional tone based on Tunisian laws.
    User asked: {user_message}
    
    Relevant legal information from AWS Knowledge Base:
    {legal_info}
    """

    # Call Claude on AWS Bedrock
    bedrock_response = bedrock_runtime.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps({
            "messages": [
                {"role": "system", "content": "You are a Tunisian legal expert."},
                *user_threads[user_id],
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 500,
            "temperature": 0.3,
        })
    )

    # Extract response
    response_body = json.loads(bedrock_response["body"].read().decode("utf-8"))
    assistant_reply = response_body["content"][0]["text"]
    
    user_threads[user_id].append({"role": "assistant", "content": assistant_reply})

    return assistant_reply


@app.route("/chat", methods=["POST"])
def chat():
    """Chat endpoint where users send messages"""
    data = request.json
    user_id = data.get("user_id") or str(uuid.uuid4())  # Generate unique user_id if not provided
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    # Generate response
    response_text = chat_with_claude(user_id, user_message)

    return jsonify({"user_id": user_id, "response": response_text})


if __name__ == "__main__":
    app.run(debug=True)

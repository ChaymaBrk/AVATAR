import os

# AWS Configuration
AWS_CONFIG = {
    'AWS_ACCESS_KEY_ID': 'AKIATWBJZ4G6IN7JILVW',
    'AWS_SECRET_ACCESS_KEY': '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',
    'AWS_DEFAULT_REGION': 'eu-central-1',
    'BUCKET_NAME': 'kb-pdf'
}

# Model Configuration
MODEL_CONFIG = {
    'EMBEDDING_MODEL': "amazon.titan-embed-text-v2:0",
    'LLM_MODEL': "anthropic.claude-3-5-sonnet-20240620-v1:0",
    'TEMPERATURE': 0.3
}

# App Configuration
APP_CONFIG = {
    'FOLDER_PATH': "local_data/",
    'MAX_WORKERS': 5,
    'SEARCH_K': 3
}

# Initialize environment variables
for key, value in AWS_CONFIG.items():
    os.environ[key] = value
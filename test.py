from tensorzero import TensorZeroGateway
import os
from dotenv import load_dotenv
#load_dotenv()
load_dotenv(dotenv_path="C:/Users/user/Documents/Nig_Project/AI_Dataset Analysis_Tool/.env")
load_dotenv(dotenv_path="C:/Users/user/Documents/Nig Project/AI_Dataset_Analysis_Tool/config/.env")
load_dotenv(dotenv_path="C:/Users/user/Documents/Nig_Project/config/.env")
#print("OPENAI:", os.getenv("OPENAI_API_KEY"))
print("GEMINI:", os.getenv("GOOGLE_AI_STUDIO_GEMINI_API_KEY"))
#os.environ["GOOGLE_AI_STUDIO_GEMINI_API_KEY"] = os.getenv("GOOGLE_AI_STUDIO_GEMINI_API_KEY")
with TensorZeroGateway.build_http(
    #clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero",
    #config_file="config/tensorzero.toml"
    gateway_url="http://localhost:3000"
) as client:
    response = client.inference(
        function_name="extract_data",
        input={
            "system": "You are a helpful assistant.",
            "messages": [
                {
                    "role": "user",
                    "content": "Write a short  essay on Japan.",
                }
            ]
        },
    )

print(response)

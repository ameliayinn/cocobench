from openai import AzureOpenAI

client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint="https://opencsg-us.openai.azure.com/",
    api_key="af7aabe2e77b41b1a89452ce694658b5"
)

# 列出可用的模型
models = client.models.list()
for model in models:
    print(model.id)
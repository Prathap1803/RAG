import langchain
import os

langchain_path = os.path.dirname(langchain.__file__)
print(f"LangChain is installed at: {langchain_path}")
print("\nContents of the langchain directory:\n")

for item in os.listdir(langchain_path):
    print(item)

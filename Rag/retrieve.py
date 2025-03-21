from flask import request, jsonify, Blueprint
import os
from langchain_mistralai.chat_models import ChatMistralAI
import chromadb

os.environ["MISTRAL_API_KEY"] = "236mWUjffs24Rg2pkQNfQiJNxg9EUxNO"

client = chromadb.PersistentClient(path='db')
collection = client.get_or_create_collection('collection')

llm = ChatMistralAI(model="mistral-small", temperature=0.7, mistral_api_key=os.getenv("MISTRAL_API_KEY"))
history = []

template = """
You are an experienced financial consultant who helps clients with in their investment and financial planning.
you have to follow the following guidelines:

1.Only answer finance-related questions.
2.polite and professional in your responses.
3.Keep the conversation focused on the client's needs.

"""


finance_keywords = [    
    'stock', 'market', 'investment', 'bond', 'interest rate', 
    'economy', 'financial', 'bank', 'currency', 'inflation','Economic growth',
    'Economic development', 'Economic planning', 'Economic policy', 'Economic reform',
    'Economic system', 'Economic model', 'Economic theory', 'Economic indicator',
    'Economic geography', 'Economic history', 'Economic sociology', 'Economic anthropology','banking',
    'financial planning', 'financial advisor', 'financial consultant', 'financial management',
    'financial services', 'financial analyst', 'financial institution', 'financial market',
    'financial economics', 'financial crisis', 'financial risk', 'financial instrument',
    'Equities','mutual funds','commodities','derivatives','forex','cryptocurrency',
    'investment banking', 'investment management', 'investment fund', 'investment trust','debt',
    'interest rate','interest rate risk','interest rate swap','interest rate cap','interest rate floor',
    'credit','credit risk','credit rating','credit default swap','credit spread','credit crunch',
    'tax','salary','details', 'income tax', 'tax rate', 'tax bracket', 'tax deduction', 'tax credit',
    'taxable income', 'tax return', 'tax planning', 'tax evasion', 'tax avoidance', 'tax haven',
    'State','Government','Public finance','stock market','stock exchange','stock price','stock index',
    'ctc','insurance','insurance policy','insurance premium','insurance claim','insurance company','types',
    'explain','explaination','explain me','explain to me','what is','what are','how does','how do','why','tell me',
    'elaborate','elaborate on','elaborate for me','elaborate to me','clarify','clarify for me','clarify to me','recommend',
    'recommend me','recommend to me','suggest','suggest me','suggest to me','advise','advise me','advise to me','guide',
    'give',
]





def is_finance_related(question):
    for keyword in finance_keywords:
        if keyword in question.lower():
            return True
    return False
   

def generate_response(prompt):
   
    if not is_finance_related(prompt):
        return "I'm sorry, I can only answer finance-related questions." 
    history.append({"role": "user", "content": prompt})
    context = "\n".join([f"{turn['role']}: {turn['content']}" for turn in history])

  
    response = llm.invoke(context)
    history.append({"role": "assistant", "content": response})
    response_str = str(response)
    return response_str






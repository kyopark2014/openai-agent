import utils
import info
import boto3
import traceback
import uuid
import logging
import sys
import re
import base64
import PyPDF2
import csv
import os
import json

from botocore.config import Config
from urllib import parse
from dataclasses import dataclass, field
from io import BytesIO
from PIL import Image
from botocore.exceptions import ClientError

import bedrock_responses
from openai import BedrockOpenAI
from aws_bedrock_token_generator import provide_token

@dataclass
class TextChunk:
    page_content: str
    metadata: dict = field(default_factory=dict)


# Simple memory (no LangChain)
class SimpleMemory:
    def __init__(self, k=5):
        self.k = k
        self.chat_memory = SimpleChatMemory()
    
    def load_memory_variables(self, inputs):
        return {"chat_history": self.chat_memory.messages[-self.k:] if len(self.chat_memory.messages) > self.k else self.chat_memory.messages}

class SimpleChatMemory:
    def __init__(self):
        self.messages = []
    
    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_ai_message(self, message):
        self.messages.append({"role": "assistant", "content": message})
    
    def clear(self):
        self.messages = []

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("chat")

os.environ["BYPASS_TOOL_CONSENT"] = "true"

workingDir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(workingDir, "config.json")

config = utils.load_config()

bedrock_region = config.get("region", "us-west-2")
projectName = config.get("projectName", "openai-agent")
accountId = config.get("accountId", None)
knowledge_base_id = config.get('knowledge_base_id', None)
account_id = config.get("accountId", None)
user_id = 'agent'

if accountId is None:
    raise Exception ("No accountId")
region = config["region"] if "region" in config else "us-west-2"
logger.info(f"region: {region}")

s3_prefix = 'docs'
s3_image_prefix = 'images'
doc_prefix = s3_prefix+'/'

model_name = "OpenAI GPT 5.5"
models = info.get_model_info(model_name)
_profile = models[0] if models else {}
model_type = _profile.get("model_type", "openai")
model_id = _profile.get("model_id", "openai.gpt-5.5")
bedrock_region = _profile.get("bedrock_region", "us-east-2")
debug_mode = "Enable"
reasoning_mode = 'Disable'
skill_mode = 'Disable'

# Memory related variables
MSG_LENGTH = 100
map_chain = dict()
memory_chain = None

aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
aws_session_token = os.environ.get('AWS_SESSION_TOKEN')
aws_region = os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')

def get_responses_region() -> str:
    return bedrock_responses.get_responses_region(model_type, bedrock_region, model_id)


def get_responses_model_id() -> str:
    return bedrock_responses.get_responses_model_id(model_id, model_type)

def update(modelName, reasoningMode, debugMode, skillMode):    
    global model_name, model_id, model_type, bedrock_region, models
    global reasoning_mode, debug_mode, skill_mode

    # load mcp.env    
    mcp_env = utils.load_mcp_env()
    
    if model_name != modelName:
        model_name = modelName
        models = info.get_model_info(model_name)
        if models:
            profile = models[0]
            model_id = profile["model_id"]
            model_type = profile["model_type"]
            bedrock_region = profile["bedrock_region"]
        logger.info(
            f"model_name: {model_name}, model_id: {model_id}, region: {bedrock_region}"
        )

    if reasoningMode != reasoning_mode:
        reasoning_mode = reasoningMode
        logger.info(f"reasoning_mode: {reasoning_mode}")

    if debugMode != debug_mode:
        debug_mode = debugMode
        logger.info(f"debug_mode: {debug_mode}")        

    if skillMode != skill_mode:
        skill_mode = skillMode
        logger.info(f"skill_mode: {skill_mode}")
        mcp_env['skill_mode'] = skill_mode

    # update mcp.env    
    mcp_env['user_id'] = user_id
    utils.save_mcp_env(mcp_env)
    logger.info(f"mcp.env updated: {mcp_env}")

def create_object(key, body):
    """
    Create an object in S3 and return the URL. If the file already exists, append the new content.
    """
    
    # Content-Type based on file extension
    content_type = 'application/octet-stream'  # default value
    if key.endswith('.html'):
        content_type = 'text/html'
    elif key.endswith('.md'):
        content_type = 'text/markdown'
    
    if aws_access_key and aws_secret_key:
        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
        )
    else:
        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region,
        )
        
    s3_client.put_object(
        Bucket=s3_bucket,
        Key=key,
        Body=body,
        ContentType=content_type
    )  

def updata_object(key, body, direction):
    """
    Create an object in S3 and return the URL. If the file already exists, append the new content.
    """
    if aws_access_key and aws_secret_key:
        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
        )
    else:
        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region,
        )

    try:
        # Check if file exists
        try:
            response = s3_client.get_object(Bucket=s3_bucket, Key=key)
            existing_body = response['Body'].read().decode('utf-8')
            # Append new content to existing content

            if direction == 'append':
                updated_body = existing_body + '\n' + body
            else: # prepend
                updated_body = body + '\n' + existing_body
        except s3_client.exceptions.NoSuchKey:
            # File doesn't exist, use new body as is
            updated_body = body
            
        # Content-Type based on file extension
        content_type = 'application/octet-stream'  # default value
        if key.endswith('.html'):
            content_type = 'text/html'
        elif key.endswith('.md'):
            content_type = 'text/markdown'
            
        # Upload the updated content
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=key,
            Body=updated_body,
            ContentType=content_type
        )
        
    except Exception as e:
        logger.error(f"Error updating object in S3: {str(e)}")
        raise e
    
def traslation(_unused, text, input_language, output_language):
    instructions = (
        f"Translate {input_language} to {output_language}. "
        "Put the translation in <result> tags."
    )
    user_input = f"<article>{text}</article>"
    try:
        msg = bedrock_responses.complete_text(
            instructions=instructions,
            user_input=user_input,
            model_id=get_responses_model_id(),
            region=get_responses_region(),
        )
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")
        raise Exception("Not able to request to LLM")

    if "<result>" in msg:
        return msg[msg.find("<result>") + 8 : msg.find("</result>")]
    return msg

def initiate():
    global memory_chain, map_chain, user_id

    user_id = uuid.uuid4().hex
    
    # general conversation memory
    if user_id in map_chain:  
        logger.info(f"memory exist. reuse it!")
        memory_chain = map_chain[user_id]
    else: 
        logger.info(f"memory not exist. create new memory!")
        memory_chain = SimpleMemory(k=5)
        map_chain[user_id] = memory_chain

def clear_chat_history():
    global memory_chain
    # Initialize memory_chain if it doesn't exist
    if memory_chain is None:
        initiate()
    
    if memory_chain and hasattr(memory_chain, 'chat_memory'):
        memory_chain.chat_memory.clear()
    else:
        memory_chain = SimpleMemory(k=5)
    map_chain[user_id] = memory_chain

def save_chat_history(text, msg):
    global memory_chain
    # Initialize memory_chain if it doesn't exist
    if memory_chain is None:
        initiate()
    
    if memory_chain and hasattr(memory_chain, 'chat_memory'):
        memory_chain.chat_memory.add_user_message(text)
        if len(msg) > MSG_LENGTH:
            memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
        else:
            memory_chain.chat_memory.add_ai_message(msg)

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))

    if word_kor and word_kor != 'None':
        return True
    else:
        return False
    
def get_summary(docs):
    text = ""
    for doc in docs:
        text = text + doc

    if isKorean(text):
        instructions = "다음 <article> 안의 문장을 500자 이내로 요약하세요."
    else:
        instructions = (
            "Summarize the article in <article> tags within 500 characters."
        )

    user_input = f"<article>{text}</article>"
    effort = "medium" if reasoning_mode == "Enable" else None

    try:
        summary = bedrock_responses.complete_text(
            instructions=instructions,
            user_input=user_input,
            model_id=get_responses_model_id(),
            region=get_responses_region(),
            reasoning_effort=effort,
        )
        logger.info(f"result of summarization: {summary}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")
        raise Exception("Not able to request to LLM")

    return summary

# load documents from s3 for pdf and txt
def load_document(file_type, s3_file_name):
    # Set AWS credentials
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_session_token = os.environ.get('AWS_SESSION_TOKEN')
    
    if aws_access_key and aws_secret_key:
        s3r = boto3.resource(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token
        )
    else:
        s3r = boto3.resource("s3")
        
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    logger.info(f"s3_bucket: {s3_bucket}, s3_prefix: {s3_prefix}, s3_file_name: {s3_file_name}")
    
    contents = ""
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
    elif file_type == 'txt' or file_type == 'md':        
        contents = doc.get()['Body'].read().decode('utf-8')
        
    logger.info(f"contents: {contents}")
    new_contents = str(contents).replace("\n"," ") 
    logger.info(f"length: {len(new_contents)}")

    texts = bedrock_responses.split_text(
        new_contents,
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    ) 
    if texts:
        logger.info(f"exts[0]: {texts[0]}")
    
    return texts

fileId = uuid.uuid4().hex

def get_summary_of_uploaded_file(file_name, st):
    file_type = file_name[file_name.rfind('.')+1:len(file_name)]            
    logger.info(f"file_type: {file_type}")

    if file_type == 'csv':
        docs = load_csv_document(file_name)
        contexts = []
        for doc in docs:
            contexts.append(doc.page_content)
        logger.info(f"contexts: {contexts}")
    
        msg = get_summary(contexts)
    
    if file_type == 'pdf' or file_type == 'txt' or file_type == 'md' or file_type == 'pptx' or file_type == 'docx':
        texts = load_document(file_type, file_name)

        if len(texts):
            docs = []
            for i in range(len(texts)):
                docs.append(
                    TextChunk(
                        page_content=texts[i],
                        metadata={
                            "name": file_name,
                            "url": path + "/" + doc_prefix + parse.quote(file_name),
                        },
                    )
                )
            logger.info(f"docs[0]: {docs[0]}") 
            logger.info(f"docs size: {len(docs)}")

            contexts = []
            for doc in docs:
                contexts.append(doc.page_content)
            logger.info(f"contexts: {contexts}")

            msg = get_summary(contexts)
        else:
            msg = "문서 로딩에 실패하였습니다."        

    global fileId
    fileId = uuid.uuid4().hex

    return msg

# load csv documents from s3
def load_csv_document(s3_file_name):
    # Set AWS credentials
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_session_token = os.environ.get('AWS_SESSION_TOKEN')
    
    if aws_access_key and aws_secret_key:
        s3r = boto3.resource(
            service_name='s3',
            region_name=bedrock_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token
        )
    else:
        s3r = boto3.resource(
            service_name='s3',
            region_name=bedrock_region
        )
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)

    lines = doc.get()['Body'].read().decode('utf-8').split('\n')   # read csv per line
    logger.info(f"lins: {len(lines)}")
        
    columns = lines[0].split(',')  # get columns
    #columns = ["Category", "Information"]  
    #columns_to_metadata = ["type","Source"]
    logger.info(f"columns: {columns}")
    
    docs = []
    n = 0
    for row in csv.DictReader(lines, delimiter=',',quotechar='"'):
        #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values = {k: row[k] for k in columns if k in row}
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values.items())
        doc = TextChunk(
            page_content=content,
            metadata={
                "name": s3_file_name,
                "row": n + 1,
            },
        )
        docs.append(doc)
        n = n+1
    logger.info(f"docs[0]: {docs[0]}")

    return docs

config = utils.load_config()

bedrock_region = config["region"] if "region" in config else "us-west-2"
projectName = config["projectName"] if "projectName" in config else "mcp-rag"
accountId = config["accountId"] if "accountId" in config else None

s3_prefix = 'docs'
s3_image_prefix = 'images'

s3_bucket = config["s3_bucket"] if "s3_bucket" in config else None

path = config["sharing_url"] if "sharing_url" in config else None

def upload_to_s3(file_bytes, file_name):
    """
    Upload a file to S3 and return the URL
    """
    try:
        # Set AWS credentials
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        aws_session_token = os.environ.get('AWS_SESSION_TOKEN')
        
        if aws_access_key and aws_secret_key:
            s3_client = boto3.client(
                service_name='s3',
                region_name=bedrock_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                aws_session_token=aws_session_token
            )
        else:
            s3_client = boto3.client(
                service_name='s3',
                region_name=bedrock_region
            )
        # Generate a unique file name to avoid collisions
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #unique_id = str(uuid.uuid4())[:8]
        #s3_key = f"uploaded_images/{timestamp}_{unique_id}_{file_name}"

        content_type = utils.get_contents_type(file_name)       
        logger.info(f"content_type: {content_type}") 

        if content_type == "image/jpeg" or content_type == "image/png":
            s3_key = f"{s3_image_prefix}/{file_name}"
        else:
            s3_key = f"{s3_prefix}/{file_name}"
        
        user_meta = {  # user-defined metadata
            "content_type": content_type,
            "model_name": model_name
        }
        
        response = s3_client.put_object(
            Bucket=s3_bucket, 
            Key=s3_key, 
            ContentType=content_type,
            Metadata = user_meta,
            Body=file_bytes            
        )
        logger.info(f"upload response: {response}")

        if content_type == "image/jpeg" or content_type == "image/png":
            url = path + "/" + s3_image_prefix + "/" + parse.quote(file_name)
        else:
            url = path + "/" + s3_prefix + "/" + parse.quote(file_name)
        return url
    
    except Exception as e:
        err_msg = f"Error uploading to S3: {str(e)}"
        logger.info(f"{err_msg}")
        return None

def upload_to_s3_artifacts(file_bytes, file_name):
    """
    Upload a file to S3 and return the URL
    """
    try:
        if aws_access_key and aws_secret_key:
            s3_client = boto3.client(
                service_name='s3',
                region_name=bedrock_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                aws_session_token=aws_session_token
            )
        else:
            s3_client = boto3.client(
                service_name='s3',
                region_name=bedrock_region
        )

        content_type = utils.get_contents_type(file_name)       
        logger.info(f"content_type: {content_type}") 

        s3_key = f"artifacts/{file_name}"
        
        user_meta = {  # user-defined metadata
            "content_type": content_type,
            "model_name": model_name
        }
        
        response = s3_client.put_object(
            Bucket=s3_bucket, 
            Key=s3_key, 
            ContentType=content_type,
            Metadata = user_meta,
            Body=file_bytes            
        )
        logger.info(f"upload response: {response}")

        url = path+'/artifacts/'+parse.quote(file_name)
        return url
    
    except Exception as e:
        err_msg = f"Error uploading to S3: {str(e)}"
        logger.info(f"{err_msg}")
        return None

def add_notification(notification_queue, message):
    if notification_queue is not None:
        notification_queue.notify(message)

def update_streaming_result(notification_queue, message):
    if notification_queue is not None:
        notification_queue.stream(message)

def update_tool_notification(notification_queue, tool_use_id, message):
    if notification_queue is not None:
        notification_queue.tool_update(tool_use_id, message)

def update_rag_result(notification_queue, message):
    if notification_queue is not None:
        notification_queue.stream(message)

####################### BedrockOpenAI #######################
# General Conversation
#########################################################
def general_conversation(query):
    global memory_chain

    if memory_chain is None:
        initiate()  # Initialize memory_chain

    system_prompt = (
        "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )

    # Process conversation history
    messages = []
    if memory_chain and hasattr(memory_chain, 'load_memory_variables'):
        history = memory_chain.load_memory_variables({})["chat_history"]
        for msg in history:
            if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        if messages and messages[0]["role"] == "assistant":
            messages = messages[1:]

    messages.append({"role": "user", "content": f"Question: {query}"})
    response_input = messages if len(messages) > 1 else messages[0]["content"]

    responses_model_id = get_responses_model_id()
    responses_region = get_responses_region()
    logger.info(
        f"general_conversation model: {responses_model_id} (region: {responses_region})"
    )

    def stream_generator():
        try:
            client = BedrockOpenAI(
                aws_region=responses_region,
                bedrock_token_provider=lambda: provide_token(region=responses_region),
            )

            stream = client.responses.create(
                model=responses_model_id,
                instructions=system_prompt,
                input=response_input,
                stream=True,
            )

            full_content = ""
            for event in stream:
                if event.type == "response.output_text.delta":
                    text = event.delta or ""
                    full_content += text
                    yield text

            logger.info(f"full_content: {full_content}")

        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")
            raise Exception("Not able to request to LLM: " + err_msg)

    return stream_generator()

number_of_results = 4

def retrieve(query):
    global knowledge_base_id

    bedrock_agent_runtime_client = boto3.client(
        "bedrock-agent-runtime",
        region_name=bedrock_region
    )
    
    try:
        response = bedrock_agent_runtime_client.retrieve(
            retrievalQuery={"text": query},
            knowledgeBaseId=knowledge_base_id,
                retrievalConfiguration={
                    "vectorSearchConfiguration": {"numberOfResults": number_of_results},
                },
            )
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        
        # Update knowledge_base_id only when ResourceNotFoundException occurs
        if error_code == "ResourceNotFoundException":
            logger.warning(f"ResourceNotFoundException occurred: {e}")
            logger.info("Attempting to update knowledge_base_id...")
            
            bedrock_agent_client = boto3.client("bedrock-agent", region_name=bedrock_region)
            knowledge_base_list = bedrock_agent_client.list_knowledge_bases()
            
            updated = False
            for knowledge_base in knowledge_base_list.get("knowledgeBaseSummaries", []):
                if knowledge_base["name"] == projectName:
                    new_knowledge_base_id = knowledge_base["knowledgeBaseId"]
                    knowledge_base_id = new_knowledge_base_id

                    config['knowledge_base_id'] = new_knowledge_base_id
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(config, f, ensure_ascii=False, indent=4)
                    
                    logger.info(f"Updated knowledge_base_id to: {new_knowledge_base_id}")
                    updated = True
                    break
            
            if updated:
                # Retry after updating knowledge_base_id
                try:
                    response = bedrock_agent_runtime_client.retrieve(
                        retrievalQuery={"text": query},
                        knowledgeBaseId=knowledge_base_id,
                        retrievalConfiguration={
                            "vectorSearchConfiguration": {"numberOfResults": number_of_results},
                        },
                    )
                    logger.info("Retry successful after updating knowledge_base_id")
                except Exception as retry_error:
                    logger.error(f"Retry failed after updating knowledge_base_id: {retry_error}")
                    raise
            else:
                logger.error(f"Could not find knowledge base with name: {projectName}")
                raise
        else:
            # Re-raise other errors that are not ResourceNotFoundException
            logger.error(f"Error retrieving: {e}")
            raise
    except Exception as e:
        # Re-raise other exceptions that are not ClientError
        logger.error(f"Unexpected error retrieving: {e}")
        raise
    
    # logger.info(f"response: {response}")
    retrieval_results = response.get("retrievalResults", [])
    # logger.info(f"retrieval_results: {retrieval_results}")

    json_docs = []
    for result in retrieval_results:
        text = url = name = None
        if "content" in result:
            content = result["content"]
            if "text" in content:
                text = content["text"]

        if "location" in result:
            location = result["location"]
            if "s3Location" in location:
                uri = location["s3Location"]["uri"] if location["s3Location"]["uri"] is not None else ""
                
                name = uri.split("/")[-1]
                encoded_name = parse.quote(name)                
                url = f"{path}/{doc_prefix}{encoded_name}"
                
            elif "webLocation" in location:
                url = location["webLocation"]["url"] if location["webLocation"]["url"] is not None else ""
                name = "WEB"

        page = None
        raw_page = (result.get("metadata") or {}).get("x-amz-bedrock-kb-document-page-number")
        if raw_page is not None:
            try:
                # Bedrock KB uses 0-based page numbers; convert to 1-based for display
                page = int(raw_page) + 1
            except (TypeError, ValueError):
                page = raw_page

        reference = {
            "url": url,
            "title": name,
            "from": "RAG",
        }
        if page is not None:
            reference["page"] = page

        json_docs.append({
            "contents": text,
            "reference": reference,
        })
    logger.info(f"json_docs: {json_docs}")

    return json.dumps(json_docs, ensure_ascii=False)

def run_rag_with_knowledge_base(query, st):
    global reference_docs, contentList
    reference_docs = []
    contentList = []

    # retrieve
    if debug_mode == "Enable":
        st.info(f"RAG 검색을 수행합니다. 검색어: {query}")  

    json_docs = retrieve(query)    
    logger.info(f"json_docs: {json_docs}")

    relevant_docs = json.loads(json_docs)

    relevant_context = ""
    for doc in relevant_docs:
        relevant_context += f"{doc['contents']}\n\n"

    # change format to document
    st.info(f"{len(relevant_docs)}개의 관련된 문서를 얻었습니다.")

    if isKorean(query):
        instructions = (
            "다음 컨텍스트로 질문에 답하세요. 정보가 없으면 모른다고 하세요. "
            "답변은 <result> 태그 안에 작성하세요."
        )
    else:
        instructions = (
            "Answer using the context. If unknown, say so. "
            "Put your answer in <result> tags."
        )
    user_message = f"Question: {query}\n\nContext:\n{relevant_context}"

    try:
        msg = bedrock_responses.complete_text(
            instructions=instructions,
            user_input=user_message,
            model_id=get_responses_model_id(),
            region=get_responses_region(),
            reasoning_effort="medium" if reasoning_mode == "Enable" else None,
        )
        logger.info(f"result: {msg}")
        if "<result>" in msg:
            msg = msg[msg.find("<result>") + 8 : msg.find("</result>")]
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")
        raise Exception("Not able to request to LLM")
    
    if relevant_docs:
        ref = "\n\n### Reference\n"
        for i, doc in enumerate(relevant_docs):
            page_content = doc["contents"][:100].replace("\n", "")
            ref += f"{i+1}. [{doc["reference"]['title']}]({doc["reference"]['url']}), {page_content}...\n"    
        logger.info(f"ref: {ref}")
        msg += ref
    
    return msg

def run_rag_using_retrieve_and_generate(query, notification_queue):
    msg = None

    global reference_docs, contentList
    reference_docs = []
    contentList = []

    # retrieve
    if debug_mode == "Enable":
        add_notification(notification_queue, f"RAG 검색을 수행합니다. 검색어: {query}")  

    bedrock_agent_runtime_client = boto3.client(
        "bedrock-agent-runtime",
        region_name=bedrock_region
    )

    model_arn = f"arn:aws:bedrock:{get_responses_region()}::foundation-model/{get_responses_model_id()}"
    
    retrieve_response = bedrock_agent_runtime_client.retrieve_and_generate_stream(
        input={"text": query},
        retrieveAndGenerateConfiguration={
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": knowledge_base_id,
                "modelArn": model_arn,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": number_of_results
                    }
                }
            },
            "type": "KNOWLEDGE_BASE"
        }        
    )
    logger.info(f"retrieve_response type: {type(retrieve_response)}")

    msg = ""
    for event in retrieve_response['stream']:
        if "output" in event:
            text = event['output']['text']
            logger.info(f"text: {text}")
            msg += text

            update_rag_result(notification_queue, msg)

        if "citation" in event:
            citation = event['citation']
            logger.info(f"citation: {citation}")

            retrieved_references = citation.get('citation', {}).get('retrievedReferences', []) or citation.get('retrievedReferences', [])
                        
            for ref in retrieved_references:
                content_text = url = name = ""

                if "content" in ref:
                    content_text = ref["content"]["text"]

                if "location" in ref:
                    location = ref["location"]
                    if "s3Location" in location:
                        uri = location["s3Location"]["uri"] if location["s3Location"]["uri"] is not None else ""
                        
                        name = uri.split("/")[-1]
                        encoded_name = parse.quote(name)                
                        url = f"{path}/{doc_prefix}{encoded_name}"
                    
                    if "webLocation" in location:
                        url = location["webLocation"]["url"] if location["webLocation"]["url"] is not None else ""
                        name = "WEB"

                reference_doc = {
                    "contents": content_text,              
                    "reference": {
                        "url": url,                   
                        "title": name,
                        "from": "RAG"
                    }
                }
                                
                # duplicate check and add to reference_docs
                if reference_doc not in reference_docs:
                    reference_docs.append(reference_doc)
                    # add_notification(notification_queue, f"{content_text}\n\n{url}")

    if reference_docs:
        ref = "\n\n### Reference\n"
        for i, doc in enumerate(reference_docs):
            page_content = doc["contents"][:100].replace("\n", "")
            ref += f"{i+1}. [{doc["reference"]['title']}]({doc["reference"]['url']}), {page_content}...\n"    
        logger.info(f"ref: {ref}")
        msg += ref

    update_rag_result(notification_queue, msg)

    return msg

sharing_url = config["sharing_url"] if "sharing_url" in config else None
s3_prefix = "docs"
capture_prefix = "captures"


def _build_tool_reference(ref_item: dict) -> dict:
    """Build a display reference from a RAG doc item."""
    reference = ref_item.get("reference") or {}
    contents = ref_item.get("contents") or ""
    content_text = contents[:100] + "..." if len(contents) > 100 else contents
    result = {
        "url": reference.get("url"),
        "title": reference.get("title"),
        "content": content_text,
    }
    if reference.get("page") is not None:
        result["page"] = reference["page"]
    return result


def get_tool_info(tool_name, tool_content):
    tool_references = []    
    urls = []
    content = ""

    # tavily
    if isinstance(tool_content, str) and "Title:" in tool_content and "URL:" in tool_content and "Content:" in tool_content:
        logger.info("Tavily parsing...")
        items = tool_content.split("\n\n")
        for i, item in enumerate(items):
            # logger.info(f"item[{i}]: {item}")
            if "Title:" in item and "URL:" in item and "Content:" in item:
                try:
                    title_part = item.split("Title:")[1].split("URL:")[0].strip()
                    url_part = item.split("URL:")[1].split("Content:")[0].strip()
                    content_part = item.split("Content:")[1].strip().replace("\n", "")
                    
                    logger.info(f"title_part: {title_part}")
                    logger.info(f"url_part: {url_part}")
                    logger.info(f"content_part: {content_part}")

                    content += f"{content_part}\n\n"
                    
                    tool_references.append({
                        "url": url_part,
                        "title": title_part,
                        "content": content_part[:100] + "..." if len(content_part) > 100 else content_part
                    })
                except Exception as e:
                    logger.info(f"Parsing error: {str(e)}")
                    continue                

    # OpenSearch
    elif tool_name == "SearchIndexTool": 
        if ":" in tool_content:
            extracted_json_data = tool_content.split(":", 1)[1].strip()
            try:
                json_data = json.loads(extracted_json_data)
                # logger.info(f"extracted_json_data: {extracted_json_data[:200]}")
            except json.JSONDecodeError:
                logger.info("JSON parsing error")
                json_data = {}
        else:
            json_data = {}
        
        if "hits" in json_data:
            hits = json_data["hits"]["hits"]
            if hits:
                logger.info(f"hits[0]: {hits[0]}")

            for hit in hits:
                text = hit["_source"]["text"]
                metadata = hit["_source"]["metadata"]
                
                content += f"{text}\n\n"

                filename = metadata["name"].split("/")[-1]
                # logger.info(f"filename: {filename}")
                
                content_part = text.replace("\n", "")
                tool_references.append({
                    "url": metadata["url"], 
                    "title": filename,
                    "content": content_part[:100] + "..." if len(content_part) > 100 else content_part
                })
                
        logger.info(f"content: {content}")
        
    # Knowledge Base
    elif tool_name == "QueryKnowledgeBases": 
        try:
            # Handle case where tool_content contains multiple JSON objects
            if tool_content.strip().startswith('{'):
                # Parse each JSON object individually
                json_objects = []
                current_pos = 0
                brace_count = 0
                start_pos = -1
                
                for i, char in enumerate(tool_content):
                    if char == '{':
                        if brace_count == 0:
                            start_pos = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and start_pos != -1:
                            try:
                                json_obj = json.loads(tool_content[start_pos:i+1])
                                # logger.info(f"json_obj: {json_obj}")
                                json_objects.append(json_obj)
                            except json.JSONDecodeError:
                                logger.info(f"JSON parsing error: {tool_content[start_pos:i+1][:100]}")
                            start_pos = -1
                
                json_data = json_objects
            else:
                # Try original method
                json_data = json.loads(tool_content)                
            # logger.info(f"json_data: {json_data}")

            # Build content
            if isinstance(json_data, list):
                for item in json_data:
                    if isinstance(item, dict) and "content" in item:
                        content_text = item["content"].get("text", "")
                        content += content_text + "\n\n"

                        uri = "" 
                        if "location" in item:
                            if "s3Location" in item["location"]:
                                uri = item["location"]["s3Location"]["uri"]
                                # logger.info(f"uri (list): {uri}")
                                ext = uri.split(".")[-1]

                                # if ext is an image 
                                url = sharing_url + "/" + s3_prefix + "/" + uri.split("/")[-1]
                                if ext in ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "ico", "webp"]:
                                    url = sharing_url + "/" + capture_prefix + "/" + uri.split("/")[-1]
                                logger.info(f"url: {url}")
                                
                                tool_references.append({
                                    "url": url, 
                                    "title": uri.split("/")[-1],
                                    "content": content_text[:100] + "..." if len(content_text) > 100 else content_text
                                })          
                
        except json.JSONDecodeError as e:
            logger.info(f"JSON parsing error: {e}")
            json_data = {}
            content = tool_content  # Use original content if parsing fails

        logger.info(f"content: {content}")
        logger.info(f"tool_references: {tool_references}")

    # aws document
    elif tool_name == "search_documentation":
        try:
            # Handle case where tool_content is a list (e.g., [{'type': 'text', 'text': '...'}])
            if isinstance(tool_content, list):
                # Extract text field from the first item in the list
                if len(tool_content) > 0 and isinstance(tool_content[0], dict) and 'text' in tool_content[0]:
                    tool_content = tool_content[0]['text']
                else:
                    logger.info(f"Unexpected list format: {tool_content}")
                    return content, urls, tool_references
            
            # Parse JSON if tool_content is a string
            if isinstance(tool_content, str):
                json_data = json.loads(tool_content)
            elif isinstance(tool_content, dict):
                json_data = tool_content
            else:
                logger.info(f"Unexpected tool_content type: {type(tool_content)}")
                return content, urls, tool_references
            
            # Extract results from search_results array
            search_results = json_data.get('search_results', [])
            if not search_results:
                # If search_results is not found, json_data itself may be an array
                if isinstance(json_data, list):
                    search_results = json_data
                else:
                    logger.info(f"No search_results found in JSON data")
                    return content, urls, tool_references
            
            for item in search_results:
                logger.info(f"item: {item}")
                
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except json.JSONDecodeError:
                        logger.info(f"Failed to parse item as JSON: {item}")
                        continue
                
                if isinstance(item, dict) and 'url' in item and 'title' in item:
                    url = item['url']
                    title = item['title']
                    content_text = item.get('context', '')[:100] + "..." if len(item.get('context', '')) > 100 else item.get('context', '')
                    tool_references.append({
                        "url": url,
                        "title": title,
                        "content": content_text
                    })
                else:
                    logger.info(f"Invalid item format: {item}")
                    
        except json.JSONDecodeError as e:
            logger.info(f"JSON parsing error: {e}, tool_content: {tool_content}")
            pass
        except Exception as e:
            logger.info(f"Unexpected error in search_documentation: {e}, tool_content type: {type(tool_content)}")
            pass

        logger.info(f"content: {content}")
        logger.info(f"tool_references: {tool_references}")
            
    # ArXiv
    elif tool_name == "search_papers" and "papers" in tool_content:
        try:
            json_data = json.loads(tool_content)

            papers = json_data['papers']
            for paper in papers:
                url = paper['url']
                title = paper['title']
                abstract = paper['abstract'].replace("\n", "")
                content_text = abstract[:100] + "..." if len(abstract) > 100 else abstract
                content += f"{content_text}\n\n"
                logger.info(f"url: {url}, title: {title}, content: {content_text}")

                tool_references.append({
                    "url": url,
                    "title": title,
                    "content": content_text
                })
        except json.JSONDecodeError:
            logger.info(f"JSON parsing error: {tool_content}")
            pass

        logger.info(f"content: {content}")
        logger.info(f"tool_references: {tool_references}")

    # aws-knowledge
    elif tool_name == "aws___read_documentation":
        logger.info(f"#### {tool_name} ####")
        if isinstance(tool_content, dict):
            json_data = tool_content
        elif isinstance(tool_content, list):
            json_data = tool_content
        else:
            json_data = json.loads(tool_content)
        
        logger.info(f"json_data: {json_data}")

        if "content" in json_data:
            content = json_data["content"]
            logger.info(f"content: {content}")
            if "result" in content:
                result = content["result"]
                logger.info(f"result: {result}")
                
        payload = {}
        if "response" in json_data:
            payload = json_data["response"]["payload"]
        elif "content" in json_data:
            payload = json_data

        if "content" in payload:
            payload_content = payload["content"]
            if "result" in payload_content:
                result = payload_content["result"]
                logger.info(f"result: {result}")
                if isinstance(result, str) and "AWS Documentation from" in result:
                    logger.info(f"Processing AWS Documentation format: {result}")
                    try:
                        # Extract URL from "AWS Documentation from https://..."
                        url_start = result.find("https://")
                        if url_start != -1:
                            # Find the colon after the URL (not inside the URL)
                            url_end = result.find(":", url_start)
                            if url_end != -1:
                                # Check if the colon is part of the URL or the separator
                                url_part = result[url_start:url_end]
                                # If the colon is immediately after the URL, use it as separator
                                if result[url_end:url_end+2] == ":\n":
                                    url = url_part
                                    content_start = url_end + 2  # Skip the colon and newline
                                else:
                                    # Try to find the actual URL end by looking for space or newline
                                    space_pos = result.find(" ", url_start)
                                    newline_pos = result.find("\n", url_start)
                                    if space_pos != -1 and newline_pos != -1:
                                        url_end = min(space_pos, newline_pos)
                                    elif space_pos != -1:
                                        url_end = space_pos
                                    elif newline_pos != -1:
                                        url_end = newline_pos
                                    else:
                                        url_end = len(result)
                                    
                                    url = result[url_start:url_end]
                                    content_start = url_end + 1
                                
                                # Remove trailing colon from URL if present
                                if url.endswith(":"):
                                    url = url[:-1]
                                
                                # Extract content after the URL
                                if content_start < len(result):
                                    content_text = result[content_start:].strip()
                                    # Truncate content for display
                                    display_content = content_text[:100] + "..." if len(content_text) > 100 else content_text
                                    display_content = display_content.replace("\n", "")
                                    
                                    tool_references.append({
                                        "url": url,
                                        "title": "AWS Documentation",
                                        "content": display_content
                                    })
                                    content += content_text + "\n\n"
                                    logger.info(f"Extracted URL: {url}")
                                    logger.info(f"Extracted content length: {len(content_text)}")
                    except Exception as e:
                        logger.error(f"Error parsing AWS Documentation format: {e}")
        logger.info(f"content: {content}")
        logger.info(f"tool_references: {tool_references}")

    else:        
        try:
            if isinstance(tool_content, dict):
                json_data = tool_content
            elif isinstance(tool_content, list):
                json_data = tool_content
            else:
                json_data = json.loads(tool_content)
            
            logger.info(f"json_data: {json_data}")
            if isinstance(json_data, dict) and "path" in json_data:  # path
                path = json_data["path"]
                if isinstance(path, list):
                    for url in path:
                        urls.append(url)
                else:
                    urls.append(path)            

            if isinstance(json_data, dict):
                for item in json_data:
                    logger.info(f"item: {item}")
                    if "reference" in item and "contents" in item:
                        tool_references.append(_build_tool_reference(item))
            else:
                logger.info(f"json_data is not a dict: {json_data}")

                for item in json_data:
                    if "reference" in item and "contents" in item:
                        tool_references.append(_build_tool_reference(item))
                
            logger.info(f"tool_references: {tool_references}")

        except json.JSONDecodeError:
            pass

    return content, urls, tool_references


def _resize_and_encode(image_content):
    """Resize image if needed and return base64-encoded string."""
    img = Image.open(BytesIO(image_content))
    width, height = img.size
    logger.info(f"width: {width}, height: {height}, size: {width*height}")

    isResized = False
    max_size = 5 * 1024 * 1024  # 5MB

    while width * height > 2000000:
        width = int(width / 2)
        height = int(height / 2)
        isResized = True

    if isResized:
        img = img.resize((width, height))

    max_attempts = 5
    img_base64 = ""
    base64_size = 0
    for attempt in range(max_attempts):
        buffer = BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        base64_size = len(img_base64.encode('utf-8'))
        logger.info(f"attempt {attempt + 1}: base64_size = {base64_size} bytes")

        if base64_size <= max_size:
            break
        width = int(width * 0.8)
        height = int(height * 0.8)
        img = img.resize((width, height))
        logger.info(f"resizing to {width}x{height} due to size limit")

    if base64_size > max_size:
        raise Exception("이미지 크기가 너무 큽니다. 5MB 이하의 이미지를 사용해주세요.")

    return img_base64


def extract_text(img_base64):
    """Extract text from an image using Bedrock OpenAI Responses API."""
    instructions = "이미지에서 텍스트를 추출해 markdown으로 변환하세요. <result> 태그로 감싸세요."
    query = "텍스트를 추출해 주세요."

    extracted_text = ""
    for attempt in range(5):
        logger.info(f"extract_text attempt: {attempt}")
        try:
            extracted_text = bedrock_responses.complete_multimodal(
                instructions=instructions,
                text=query,
                image_base64=img_base64,
                model_id=get_responses_model_id(),
                region=get_responses_region(),
            )
            break
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")

    logger.info(f"extracted_text: {extracted_text}")
    if len(extracted_text) < 10:
        extracted_text = "텍스트를 추출하지 못하였습니다."

    return extracted_text


def summary_image(img_base64, instruction):
    """Summarize an image using Bedrock OpenAI Responses API."""
    instructions = "이미지 내용을 분석해 한국어 markdown으로 답변하세요. <result> 태그로 감싸세요."

    if instruction:
        logger.info(f"instruction: {instruction}")
        query = f"{instruction}. <result> tag를 붙여주세요."
    else:
        query = "이미지가 의미하는 내용을 자세히 설명해 주세요."

    for attempt in range(5):
        logger.info(f"summary_image attempt: {attempt}")
        try:
            return bedrock_responses.complete_multimodal(
                instructions=instructions,
                text=query,
                image_base64=img_base64,
                model_id=get_responses_model_id(),
                region=get_responses_region(),
            )
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")
            if attempt == 4:
                raise Exception("Not able to request to LLM") from None

    return ""


def summarize_image(image_content, prompt, st):
    """Full image analysis: resize, extract text, summarize."""
    img_base64 = _resize_and_encode(image_content)

    if debug_mode == "Enable":
        status = "이미지에서 텍스트를 추출합니다."
        logger.info(f"status: {status}")
        st.info(status)

    text = extract_text(img_base64)

    if text.find('<result>') != -1:
        extracted_text = text[text.find('<result>') + 8:text.find('</result>')]
    else:
        extracted_text = text

    if debug_mode == "Enable":
        status = f"### 추출된 텍스트\n\n{extracted_text}"
        logger.info(f"status: {status}")
        st.info(status)

    if debug_mode == "Enable":
        status = "이미지의 내용을 분석합니다."
        logger.info(f"status: {status}")
        st.info(status)

    image_summary = summary_image(img_base64, prompt)

    if image_summary.find('<result>') != -1:
        image_summary = image_summary[image_summary.find('<result>') + 8:image_summary.find('</result>')]
    logger.info(f"image summary: {image_summary}")

    contents = f"## 이미지 분석\n\n{image_summary}"
    logger.info(f"image contents: {contents}")

    return contents

from dotenv import load_dotenv

load_dotenv() 
from websockets.sync.server import serve
from pymilvus import MilvusClient, model
import json
import os
from websocket import create_connection

embedding_fn = model.dense.SentenceTransformerEmbeddingFunction(
    model_name='BAAI/bge-m3', # Specify the model name
    device=os.environ["DEVICE"] # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)
client = MilvusClient(os.environ["MILVUS_URL"])

bge_rf = model.reranker.BGERerankFunction(
    model_name="BAAI/bge-reranker-v2-m3",  # Specify the model name. Defaults to `BAAI/bge-reranker-v2-m3`.
    device=os.environ["DEVICE"] # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)


def rerank(query, answers, guides):
    rank = bge_rf(query, answers + [x[1] for x in guides])
    ranked_answers = list(map(lambda x: f"{round(x.score*1000)/10}%: {x.text if x.index < len(answers) else guides[x.index - len(answers)][1]}", [x for x in rank if x.score >= 0.005]))
    ranked_ids = list(map(lambda x: (x.index, x.score), [x for x in rank if x.score >= 0.005 and x.index < len(answers)]))
    ranked_guides = list(map(lambda x: {"guide": guides[x.index - len(answers)][0], "score": x.score}, [x for x in rank if x.score >= 0.005 and x.index >= len(answers)]))
    return ranked_answers, ranked_ids, ranked_guides

def get_answers(query):
    query_vectors = embedding_fn.encode_queries([query])

    res = client.search(
        collection_name="demo_collection",  # target collection
        data=query_vectors,  # query vectors
        limit=10,  # number of returned entities
        output_fields=["text", "answer"],  # specifies fields to be returned
    )

    # print("\n".join(map(lambda x: f"{x['distance']}: {x['entity']['answer']}", res[0])))
    answers = list(map(lambda x: x['entity']['answer'], res[0]))
    qa = list(map(lambda x: {"q": x['entity']['text'], "a": x['entity']['answer']}, res[0]))
    return answers, qa

def get_guides(query):
    query_vectors = embedding_fn.encode_queries([query])
    client.load_collection("guides_collection")
    res = client.search(
        collection_name="guides_collection",  # target collection
        data=query_vectors,  # query vectors
        limit=10,  # number of returned entities
        output_fields=["text", "file", "page"],  # specifies fields to be returned
    )

    return list(map(lambda x: (f"{x['entity']['file']}, страница {x['entity']['page']}", x['entity']["text"]), res[0]))

def get_answers_formatted(query):
    answers, qa = get_answers(query)
    guides = get_guides(query)
    ranked_answers, ranked_ids, ranked_guides = rerank(query, answers, guides)
    count = len(ranked_answers)
    format_answers = f"Вероятные ответы:\n- " + '\n- '.join(ranked_answers[:3])
    return format_answers, count, list(map(lambda i: qa[i[0]] | {"score": i[1]}, ranked_ids)), ranked_guides


prompt = \
"### System:\n"\
"Диалог между сотрудником компании РосАтом и автоматизированным ботом технической поддержки в системе 1C.\n"\
"Правила:\n"\
"Ассистент должен пытаться отвечать на вопросы пользователя ТОЛЬКО, если ответ можно найти в вероятных ответах, "\
"и ТОЛЬКО, если вопрос относится к системе 1С, иначе НЕОБХОДИМО предложить пользователю перейти к диалогу с человеком. "\
"Если подходящих ответов нет или вопрос содержит просьбу, необходимо сказать пользователю, что стоит обратиться к человеку.\n"\
"Не нужно передавать ответ из истории в явном виде. Желательно сократить и перефразировать его. Отвечать можно ТОЛЬКО на русском языке.\n"\
# "Если ответ содержит ссылку на инструкцию, необходимо сказать о ней пользователю.\n"\

# print(prompt)


def handler(websocket):
    ws = create_connection(os.environ["PETALS_URL"])

    ws.send('{"type": "open_inference_session", "max_length": 4096, "model":"petals-team/StableBeluga2"}')
    print(ws.recv())
    need_system = True
    for message in websocket:
        # i = input("Вопрос: ")
        i = message
        if i == "":
            break
        format_answers, count, qa, ranked_guides = get_answers_formatted(i)
        websocket.send(json.dumps({"type": "starting", "similar": qa, "guides": ranked_guides}, ensure_ascii=False))
        if count < 1:
            websocket.send(json.dumps({"type":"stop", "msg": "Не смогли найти ответ на Ваш вопрос. Переключаю на оператора..."}, ensure_ascii=False))
            break
        q = ""
        if need_system:
            need_system = False
            q += prompt
        print(format_answers)
        q += "\n### User:\n" + i + f"\n### System:\n{format_answers}\n\n### Assistant:\n"
        print(q, end="")
        ws.send(json.dumps({"type": "generate", "inputs": q, "stop_sequence": "</s>", "extra_stop_sequences": ["<s>"], "max_new_tokens": 3}))
        print("sent to llm")
        while True:
            r = ws.recv()
            try:
                recv = json.loads(r)
            except json.JSONDecodeError:
                print(r)
                break
            if not recv["ok"]:
                print(recv)
                break
            websocket.send(json.dumps({"type":"token", "msg":recv["outputs"].replace("</s>", "")}, ensure_ascii=False))
            if recv["stop"]: # or "</s>" in recv["outputs"]:
                # print(ws.recv())
                websocket.send(json.dumps({"type": "done"}))
                break
        print()
        
    # print()
    ws.close()
    # async for message in websocket:
    #     await websocket.send(message)

def main():
    with serve(handler,"0.0.0.0", 8765) as server:
        print("Waiting for connections")
        server.serve_forever()

main()
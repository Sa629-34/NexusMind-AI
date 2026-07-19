import ollama


def ask_llama(context, question):

    prompt = f"""
You are NexusMind AI.

Answer ONLY from the given context.

Context:
{context}

Question:
{question}

If the answer is not present in the context, reply exactly:
"I couldn't find this information in the uploaded document."
"""

    try:

        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:

        print("LLM Error:", e)

        return "⚠️ Unable to generate answer. Please make sure Ollama and Llama 3 are running."
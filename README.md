# 🖊 EditGPT 📑

## About

EditGPT is a project built on [`LangChain`](https://github.com/langchain-ai/langchain) and [`PrivateGPT`](https://github.com/zylon-ai/private-gpt), offering the ability not only to inquire about the contents of local documents but also to edit them using various LLMs. Almost all the main components of EditGPT (chat models, vectorstores, embeddings) are taken from [`LangChain Integrations`](https://python.langchain.com/docs/integrations/platforms/) and can be easily replaced with alternatives that suit your needs. The number of components integrated from LangChain into EditGPT will continue to grow.

Currently, the document editing feature is based on the [`ReAct agent`](https://python.langchain.com/docs/modules/agents/agent_types/react/). The agent can be given an editing request, access to various documents and the internet, and it will try to edit the documents as needed.

**Important!** This project is at a super early stage of development and testing, so there are many bugs and broken features. It is also important to understand that the quality of the agent directly depends on the LLM used.

## Demo
https://github.com/pashaprokaz/editGPT/assets/46126524/5a8811ba-a618-4276-8b51-2abbdb4af124



## Installation

Clone the repository:
```
git clone https://github.com/pashaprokaz/editGPT
cd editGPT
```

Install Python (if you don't have it yet) version 3.10 or higher.

Install [`Poetry`](https://python-poetry.org/docs/#installing-with-the-official-installer) for dependency management.

Download the necessary packages for EditGPT:
```
poetry install
```

## Project Setup

### Local LLM launch:

**OLLAMA - recommended**
- Install [`Ollama`](https://ollama.com/)
- Install the required [`LLM`](https://ollama.com/library). The default settings are configured for llama3-instruct (~4GB)
```bash
ollama pull llama3
```
Configure model use in the [`settings.yaml`](settings.yaml) configuration file:
```yaml
chat_model:
  provider: ollama
  model: llama3
```

Run `ollama desktop` or execute in the console:
```bash
ollama serve
```
Run EditGPT with one of the commands:
```bash
make run
poetry run python -m edit_gpt
```

**LlamaCPP**
- Install the necessary dependencies for [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python)
- Install llama-cpp-python in EditGPT with
```bash
poetry install -E llamacpp
```
- Set the environment variable `EGPT_PROFILES=llamacpp`, or create a `.env` file in the project directory and add `EGPT_PROFILES=llamacpp`
- Install any compatible model and configure its use in the [`settings-llamacpp.yaml`](settings-llamacpp.yaml) configuration file:
```yaml
chat_model:
  provider: llamacpp
  model: solar-10.7b-instruct-v1.0-uncensored.Q5_K_M.gguf # path to your LLM model
```
- Run EditGPT with one of the commands:
```bash
make run
poetry run python -m edit_gpt
```

### Running through API services

At the moment, the following chat model components from langchain are integrated into EditGPT: google, openai, gigachat, anthropic, huggingface.

You can use them by specifying the necessary API token in the `.env` file or by setting environment variables.

Next, you can either directly modify `settings.yaml`, using the new LLM model settings:

```yaml
chat_model:
  provider: fireworks
  model: accounts/fireworks/models/nous-hermes-2-mixtral-8x7b-dpo-fp8
  temperature: 0.2
  top_p: 0.5
```

Or create a new `settings-{provider_name}.yaml` file and specify it in the `EGPT_PROFILES={provider_name}` environment variable.
The list of providers will be updated and available [here](edit_gpt\chat_models\init_chat_model.py).

After setting up the provider's API token, simply run the project using one of the commands:

```bash
make run
poetry run python -m edit_gpt
```

Important: When first running editGPT, even if using API keys, small models for text vectorization will be downloaded. This is necessary for RAG, as at the moment in editGPT there is no possibility to use the API for text vectorization.


# How to use EditGPT?

After launching the project, you can navigate to http://127.0.0.1:7860/, where a gradio application will be deployed.

Upload data using the Upload button, or specify the necessary file paths or entire folders in `settings.yaml`:

```yaml
filepaths:
  - example/path_file.py
  - dir/example/dir_name
```

**I recommend uploading only a few files because the file explorer may not be very convenient at the moment. Also, this will improve the quality of RAG.**

If you want to use information from documents, use the RAG (Retrieval Augmented Generation) option (set by default). You can configure RAG parameters in `settings.yaml` as well:

```yaml
rag:
  similarity_top_k: 5 # The number of documents retrieved from user-provided files. The higher the value, the more context the LLM will understand, but keep in mind that not all language models can accept a large number of words at the input.
```

You can also use data from the internet according to your query. At the moment, several web APIs are supported: the free [`DuckDuckGo`](https://python.langchain.com/docs/integrations/tools/ddg/) and the conditionally free [`TavilySearch`](https://python.langchain.com/docs/integrations/tools/tavily_search/). Although DuckDuckGo is free, I noticed that my queries to it are often blocked, so I got the [Tavily API](https://tavily.com/) and use its free plan. You can change the number of web pages used for the answer in `settings.yaml` as well:
```yaml
web_search:
  provider: tavily # can be tavily or duckduckgo
  k: 5
```

The main feature of the project is the Agent Chat mode. In it, you can view the contents of uploaded files and edit them using EditGPT. Just enter a query related to the content of a particular file, and EditGPT will edit it.

If the file was successfully edited, the content viewing window will show how the file was changed. You can regenerate the result, roll back the changes, or save the file with new content.

Important! When uploading files via the gradio upload button, they are saved in the gradio cache, not where they were uploaded from. To edit a file in its location, specify the paths in `settings.yaml`.

The Agent uses RAG tools to get information about the contents of files, as well as web search, if it was set in the settings.

You can also ask to edit several files in the necessary way, and if the RAG context and LLM power allow, then EditGPT will do it.


## Tracing
In EditGPT, LLM LangSmith call tracing is enabled. This is a convenient tool for tracking why the LLM gave a particular answer. To work with LangSmith, you need to get an API key, set them in the environment variables, and also enable LangSmith in `settings.yaml`:
```yaml
langsmith:
  use_langsmith: False
```

Here’s an updated README with the requested model changes plus a short, sourced blurb on what Granite 4 brings.

---

# Granite Retrieval and Image Research Agents

> **Models:** Retrieval Agent → `ibm/granite4:latest` • Image Researcher → `ibm/granite4:tiny-h`

# 📚 Agents Overview

| Feature                 | Description                                                          | Models Used                                | Code Link                                                                  | Tutorial Link                                                                                                                                                                         |
| ----------------------- | -------------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Granite Retrieval Agent | General Agentic RAG for document and web retrieval using Autogen/AG2 | **Granite 4 (ibm/granite4:latest)**        | [granite_autogen_rag.py](./granite_autogen_rag.py)                         | [Build a multi-agent RAG system with Granite locally](https://developer.ibm.com/tutorials/awb-build-agentic-rag-system-granite/)                                                      |
| Image Research Agent    | Image-based multi-agent research using CrewAI with Granite Vision    | **Granite 4 Tiny-H (ibm/granite4:tiny-h)** | [image_researcher_granite_crewai.py](./image_researcher_granite_crewai.py) | [Build an AI research agent for image analysis with Granite 3.2 Reasoning and Vision models](https://developer.ibm.com/tutorials/awb-build-ai-research-agent-image-analysis-granite/) |

---

## Why Granite 4 for these agents?

Granite 4 introduces a **hybrid Mamba-2/Transformer** architecture (with MoE variants) that targets **lower memory use and faster inference**, making it a strong fit for agentic RAG and function-calling workflows. It uses **>70% lower memory** and **~2× faster inference** vs. comparable models, which helps these agents run locally or on modest GPUs with lower cost and latency. Models are **Apache-2.0 licensed**, **ISO 42001 certified**, and cryptographically signed for governance and security. 

**Tiny-H (7B total / ~1B active)** is optimized for **low-latency, small-footprint deployments**—ideal for the Image Researcher’s quick tool calls and orchestration steps. The family emphasizes **instruction following, tool calling, RAG, JSON output, multilingual dialog, and code (incl. FIM)**, aligning with both agents’ needs. ([Hugging Face][2])

---

## **Granite Retrieval Agent**

The **Granite Retrieval Agent** is an **Agentic RAG (Retrieval Augmented Generation) system** designed for querying both local documents and web retrieval sources. It uses multi-agent task planning, adaptive execution, and tool calling via **Granite 4 (`ibm/granite4:latest`)**.

### 🔹 Key Features:

* General agentic RAG for document and web retrieval using **Autogen/AG2**.
* Uses **Granite 4 (ibm/granite4:latest)** as the primary language model.
* Integrates with [Open WebUI Functions](https://docs.openwebui.com/features/plugin/functions/) for interaction via a chat UI.
* **Optimized for local execution** (e.g., tested on MacBook Pro M3 Max with 64 GB RAM).

### **Retrieval Agent in Action:**

![The Agent in action](docs/images/GraniteAgentDemo.gif)

### **Architecture:**

![alt text](docs/images/agent_arch.png)

---

## **Image Research Agent**

The **Image Research Agent** analyzes images and performs multi-agent research on image components using **Granite 4 Tiny-H (`ibm/granite4:tiny-h`)** with the **CrewAI** framework.

### 🔹 Key Features:

* **Image-based multi-agent research** using CrewAI.
* **Granite 4 Tiny-H** powers low-latency orchestration and tool calls; pair with a vision backend of your choice.
* Identifies objects, retrieves related research articles, and provides historical backgrounds.
* Demonstrates a **different agentic workflow** from the Retrieval Agent.

### **Image Researcher in Action:**

![alt-text](docs/images/image_explainer_example_1.png)

### **Architecture:**

![alt text](docs/images/image_explainer_agent.png)

---

# 🔑 Key Highlights

* **Common Installation Instructions**: The setup for **Ollama** and **Open WebUI** remains the same for both agents.
* **Flexible Web Search**: Agents use the Open WebUI search API, integrating with **SearXNG** or other search engines. [Configuration guide](https://docs.openwebui.com/category/-web-search).

---

# 🛠 Getting Started

## **1. Setup Ollama**

Go to [ollama.com](https://ollama.com/) and hit Download!

Once installed, pull the Granite 4 Micro model for the Granite Retrieval Agent
```
ollama pull ibm/granite4:latest
```

Pull the Granite 4 Tiny model for the Image Researcher
```
ollama pull ibm/granite4:tiny-h
```

If you would like to use the vision capabilities in these agents, pull the Granite Vision model
```
ollama pull granite3.2-vision:2b
```

## **2. Install Open WebUI**

```bash
pip install open-webui
open-webui serve
```

## **3. Optional: Set Up SearXNG for Web Search**

Although **SearXNG is optional**, the agents can integrate it via Open WebUI’s search API.

```bash
docker run -d --name searxng -p 8888:8080 -v ./searxng:/etc/searxng --restart always searxng/searxng:latest
```

Configuration details: [Open WebUI documentation](https://docs.openwebui.com/category/-web-search).

## **4. Import the Agent Python Script into Open WebUI**

1. Open `http://localhost:8080/` and log into Open WebUI.
2. Admin panel → **Functions** → **+** to add.
3. Name it (e.g., “Granite RAG Agent” or “Image Research Agent”).
4. Paste the relevant Python script:

   * `granite_autogen_rag.py` (Retrieval Agent)
   * `image_researcher_granite_crewai.py` (Image Research Agent)
5. Save and enable the function.
6. Adjust settings (inference endpoint, search API, **model ID**) via the gear icon.

⚠️ If you see OpenTelemetry errors while importing `image_researcher_granite_crewai.py`, see [this issue](https://github.com/ibm-granite-community/granite-retrieval-agent/issues/25).

## **5. Load Documents into Open WebUI**

1. In Open WebUI, navigate to `Workspace` → `Knowledge`.
2. Click `+` to create a new collection.
3. Upload documents for the **Granite Retrieval Agent** to query.

## **6. Configure Web Search in Open WebUI**

To set up a search provider (e.g., SearXNG), follow [this guide](https://docs.openwebui.com/tutorials/web-search/searxng#4-gui-configuration).

---

# ⚙️ Configuration Parameters

## **Granite Retrieval Agent**

| Parameter         | Description                               | Default Value               |
| ----------------- | ----------------------------------------- | --------------------------- |
| task_model_id     | Primary model for task execution          | `ibm/granite4:latest`       |
| vision_model_id   | Vision model for image analysis           | *(set as needed)*           |
| openai_api_url    | API endpoint for OpenAI-style model calls | `http://localhost:11434`    |
| openai_api_key    | API key for authentication                | `ollama`                    |
| vision_api_url    | Endpoint for vision-related tasks         | `http://localhost:11434/v1` |
| model_temperature | Controls response randomness              | `0`                         |
| max_plan_steps    | Maximum steps in agent planning           | `6`                         |


## **Image Research Agent**

| Parameter                | Description                               | Default Value            |
| ------------------------ | ----------------------------------------- | ------------------------ |
| task_model_id            | Primary model for task execution          | `ibm/granite4:tiny-h`    |
| vision_model_id          | Vision model for image analysis           | *(set as needed)*        |
| openai_api_url           | API endpoint for OpenAI-style model calls | `http://localhost:11434` |
| openai_api_key           | API key for authentication                | `ollama`                 |
| vision_api_url           | Endpoint for vision-related tasks         | `http://localhost:11434` |
| model_temperature        | Controls response randomness              | `0`                      |
| max_research_categories  | Number of categories to research          | `4`                      |
| max_research_iterations  | Iterations for refining research results  | `6`                      |
| include_knowledge_search | Option to include knowledge base search   | `False`                  |
| run_parallel_tasks       | Run tasks concurrently                    | `False`                  |

---

# 🚀 Usage

## **Image Research Agent**

* Upload an image to initiate research.
* Prompt with specifics to refine focus.

**Examples**

```text
Analyze this image and find related research articles about the devices shown.
```

```text
Break down the image into components and provide a historical background for each object.
```

## **Granite Retrieval Agent (AG2-based RAG)**

* Queries **local documents** and **web sources**.
* Performs **multi-agent task planning** and **adaptive execution**.

**Examples**

```text
Study my meeting notes to figure out the technological capabilities of the projects I’m involved in. Then, search the internet for other open-source projects with similar features.
```

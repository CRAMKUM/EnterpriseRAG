# Enterprise RAG System
### Multimodal Knowledge Graph with Google Cloud Platform

> **A production-ready RAG system that builds multimodal knowledge graphs from enterprise documents using Gemini, Spanner Graph, and proven MegaRAG methodology.**

## Overview

Enterprise RAG extracts entities and relationships from unstructured documents (PDFs, DOCX, PPTX) and builds a **Multimodal Knowledge Graph (MMKG)** in Google Spanner Graph. Users can ask complex questions using Graph RAG with visual trace explanations.

### **Key Features**

✅ **Multimodal Processing** - Handles text, images, tables, and charts
✅ **Cost-Optimized** - 60-70% savings via intelligent tool routing
✅ **MegaRAG Methodology** - Proven entity extraction approach
✅ **Interactive Graph RAG** - Users can edit and refine the knowledge graph
✅ **Hot-Reload Configs** - Update prompts/models without redeployment
✅ **Scalable** - Cloud Run auto-scaling, parallel processing

## 🏗️ Architecture

### ⚙️ **Dual-Mode Operation**

While MMKG builds, users can interact with:
- ✨ **OpenCV** - Deblur images (FREE)
- 📄 **Tesseract** - Extract text (FREE)
- 🖼️ **Gemini 2 27B** - Interpret images (LOW cost)
- 📊 **Unstructured.io** - Extract tables (LOW cost)

## 💰 Cost Optimization

| Task | Tool | Cost Tier | Savings |
|---|---|---|---|
| Text extraction | Tesseract OCR | **FREE** | vs Gemini Flash |
| Image deblurring | OpenCV | **FREE** | vs Gemini Flash |
| Image interpretation | Gemini 2 27B | **LOW** | vs Gemini 1.5 Flash |
| Table extraction | Unstructured.io | **LOW** | vs Gemini Flash |
| Entity extraction | Gemini 1.5 Flash | MEDIUM | Necessary for quality |
| Graph RAG | Gemini + GQL | HIGH | Only for complex queries |

**Result:** ~60-70% cost reduction

## ⚙️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit | User interface |
| **Hosting** | Google Cloud Run | Serverless deployment |
| **Orchestrator** | Gemini 1.5 Flash | Main LLM, routing |
| **Vision LLM** | Gemini 2 27B | Cost-effective image tasks |
| **OCR** | Tesseract | Pure text extraction |
| **CV** | OpenCV | Image deblurring |
| **Parser** | Unstructured.io | Document structure |
| **Knowledge Graph** | Spanner Graph | MMKG storage |
| **Query Language** | GQL | Graph queries |
| **Storage** | GCS | File & config storage |

## 🏗️ Project Structure

```
Enterprise_RAG/
├── agents/
│   └── orchestrator.py # Orchestrator with Gemini
├── tools/
│   ├── opencv_tool.py # Image deblurring (FREE)
│   ├── tesseract_tool.py # OCR (FREE)
│   ├── unstructured_tool.py # Document parsing
│   └── gemma_tool.py # Image interpretation (LOW cost)
├── pipeline/ # MMKG building pipeline
│   ├── document_processor.py # Parse -> Deblur -> Interpret
│   └── mmkg_builder.py # Entity/relationship extraction
├── graph/ # Spanner Graph interface
│   ├── spanner_graph.py # MMKG operations
│   ├── gql_generator.py # Natural language -> GQL
│   └── hybrid_search.py # Vector + Graph search
├── config/ # Hot-reload configuration
│   ├── config_loader.py # GCS + TTL caching
│   └── samples/ # Config templates
├── utils/ # Shared utilities
│   ├── logger.py
│   ├── gcs_helpers.py
│   └── exceptions.py
├── app/ # Streamlit frontend
│   └── components/ # UI components
├── Dockerfile # Cloud Run container
├── requirements.txt # Python dependencies
├── cloudbuild.yaml # Cloud Build config
├── build_and_deploy.sh # macOS/Linux deployment
└── cloud_build_deploy.ps1 # Cloud Build (recommended for 8GB)
```

## Quick Start

### **Prerequisites**

- Google Cloud SDK
- Python 3.10+
- GCP Project with billing
- (Optional) Docker Desktop

### **1. Configure Project**

```json
// Edit config/samples/graph_config.json
{
  "project_id": "your-gcp-project-id",
  "location": "your-region",
  "bucket_name": "your-bucket-name",
  "instance_id": "enterprise-rag-instance",
  "database_id": "mmkg-database"
}
```

### **2. Deploy to Cloud Run**

**Windows - Use Cloud Build:**
```powershell
.\\cloud_build_deploy.ps1
```

**macOS/Linux:**
```bash
chmod +x build_and_deploy.sh
./build_and_deploy.sh
```

### **3. Access the App**

The deployment will output your service URL:
```
https://enterprise-rag-xxxxx-uc.a.run.app
```

## 🧠 MegaRAG Methodology

This system uses **MegaRAG's proven approach** for entity extraction:
1. **Multimodal Entity Recognition** - Charts, tables, and diagrams become graph nodes.
2. **Relationship Strength Scoring** - Numeric scores (1-10) for weighted graph traversal.
3. **Cross-Modal Linking** - Links text entities to visual elements.
4. **Justified Chunking** - 1200 tokens per chunk with 100 token overlap.

# Netflix RAG System

Detta projekt är en RAG-applikation byggd i Python med LangChain.  
Systemet använder ett Netflix-dataset och gör det möjligt att ställa frågor om datan med hjälp av retrieval, embeddings och en LLM.

## Syfte

Syftet med projektet är att visa hur egen data kan användas som kontext till en språkmodell för att skapa mer relevanta och faktabaserade svar.

## Dataset

Vi använder ett Netflix-dataset från Kaggle. Datasetet innehåller bland annat:

- Title
- Category
- Type / Genre
- Director
- Cast
- Country
- Release_Date
- Rating
- Duration
- Description

Både originaldata och städad data finns i projektets `data`-mapp.

## Projektstruktur

```text
rag-project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   └── 02_rag_application.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

import os
import json
import argparse
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


def load_data(input_file):
    """
    Args:
        input_file (str): JSON 파일 경로
        
    Returns:
        list: JSON 데이터의 리스트
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_vector_db(data, embeddings_model_name, chroma_path, device="cpu"):
    """
    문서와 메타데이터로 Chroma 벡터 DB를 구축
    
    Args:
        data (list): 데이터 리스트
        embeddings_model_name (str): HuggingFace Embedding 모델명
        chroma_path (str): Chroma DB 저장 디렉터리 경로
        device (str): "cpu" 또는 "cuda"
    """
    # HuggingFace Embeddings 모델 초기화
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model_name,
        model_kwargs={"device": device}
    )

    # 데이터 준비
    print("Preparing documents and metadata for Chroma DB...")
    documents = []
    metadatas = []

    for item in data:
        documents.append(item.get("paragraph", ""))  # 'paragraph' 필드를 문서로 사용
        metadatas.append({
            "title": item.get("title", ""),
            "session": item.get("session", ""),
            "committee": item.get("committee", ""),
            "field": item.get("field", ""),
            "terminology": item.get("terminology", ""),
            "disposal": item.get("disposal", ""),
            "enactment": item.get("enactment", ""),
            "amendment": item.get("amendment", ""),
            "date": item.get("date", ""),
            "terminology_en": item.get("terminology_en", ""),
            "paragraph": item.get("paragraph", "")
        })

    # Chroma DB 구축
    print("Building Chroma Vector DB...")
    db = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=chroma_path
    )

    # Chroma DB 저장
    db.persist()
    print(f"Chroma DB 구축 완료, 위치: {chroma_path}")


def main(input_file, chroma_path):
    """
    Args:
        input_file (str): 최종 전처리된 JSON 파일 경로
        chroma_path (str): 벡터 DB 저장 디렉터리 경로
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    embeddings_model_name = "jhgan/ko-sroberta-multitask"

    # 데이터 로드
    print("Loading input data...")
    data = load_data(input_file)
    
    #벡터 DB 구축
    build_vector_db(data, embeddings_model_name, chroma_path, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chroma Vector DB 구축 스크립트")
    parser.add_argument('--input_file', type=str, required=True, help='최종 전처리된 JSON 파일 경로')
    parser.add_argument('--chroma_path', type=str, required=True, help='Chroma DB 저장 경로')

    args = parser.parse_args()
    
    main(
        input_file=args.input_file, 
        chroma_path=args.chroma_path
    )
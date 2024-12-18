import argparse
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


def test_vector_db(chroma_path, embeddings_model_name, query, k=3):
    """
    Chroma 벡터 DB에 쿼리를 수행하여 유사한 문서를 검색
    
    Args:
        chroma_path (str): Chroma DB 저장 경로
        embeddings_model_name (str): HuggingFace Embedding 모델명
        query (str): 검색할 쿼리 문장
        k (int): 반환할 유사 문서의 수 (기본값: 3)
    """
    print("Loading Chroma Vector DB...")
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

    print(f"Performing similarity search for query: '{query}'")
    results = db.similarity_search(query, k=k)

    print(f"Found {len(results)} similar documents")
    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print("문서 내용:", result.page_content)
        print("메타데이터:", result.metadata)
        print("-" * 50)


def main(chroma_path, query, k=3):
    """
    Args:
        chroma_path (str): Chroma DB 저장 경로
        query (str): 검색할 쿼리 문장
        k (int, optional): 반환할 유사 문서의 수 (기본값: 3)
    """
    embeddings_model_name = "jhgan/ko-sroberta-multitask"
    test_vector_db(chroma_path, embeddings_model_name, query, k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chroma Vector DB 쿼리 테스트 스크립트")
    parser.add_argument('--chroma_path', type=str, required=True, help='Chroma DB 저장 경로')
    parser.add_argument('--query', type=str, required=True, help='테스트 쿼리 문장')
    parser.add_argument('--k', type=int, default=3, help='유사 문서 검색에서 반환할 문서 개수 (기본값: 3)')

    args = parser.parse_args()
    
    main(
        chroma_path=args.chroma_path, 
        query=args.query, 
        k=args.k
    )
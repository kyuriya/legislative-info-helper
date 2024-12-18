from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# HuggingFace Embeddings 모델 설정
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={"device": "cuda"}
)

def query_rag(chroma_path, query_text, k=2):
    """
    Chroma DB에서 유사한 문서를 검색

    Args:
        chroma_path (str): Chroma DB 저장 경로
        query_text (str): 검색할 질의 문장
        k (int): 반환할 문서 개수

    Returns:
        tuple: 검색된 문서의 컨텍스트와 메타데이터
    """
    # Chroma DB 로드
    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

    # 유사한 문서 검색
    results = db.similarity_search_with_score(query_text, k=k)
    if not results:
        raise ValueError("No relevant context found.")

    # 컨텍스트와 메타데이터 추출
    context = " ".join([doc.page_content for doc, _ in results])
    metadata = results[0][0].metadata  # 가장 유사한 문서의 메타데이터
    return context, metadata
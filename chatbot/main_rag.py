import argparse
from chroma_query_rag import query_rag
from answer_generator import generate_answer

def main(chroma_path, query_text, k):
    """
    Args:
        chroma_path (str): Chroma DB 저장 경로
        query_text (str): 사용자 질의
        k (int): 검색할 문서 개수
    """
    print(f"사용자 질의: {query_text}")
    print(f"Chroma DB 경로: {chroma_path}")

    # Chroma DB에서 문서 검색
    try:
        context, metadata = query_rag(chroma_path, query_text, k=k)
        print("\n 검색된 문서 컨텍스트 및 메타데이터:")
        print(context)
        print(metadata)
    except ValueError as e:
        print(e)
        return

    # 답변 생성
    print("\n답변 생성 중...")
    answer = generate_answer(query_text, context, metadata)
    print("\n최종 답변:")
    print(answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chroma RAG 기반 답변 생성 스크립트")
    parser.add_argument('--chroma_path', type=str, required=True, help='Chroma DB 저장 경로')
    parser.add_argument('--query', type=str, required=True, help='사용자 질의 문장')
    parser.add_argument('--k', type=int, default=2, help='검색할 문서 개수 (기본값: 2)')

    args = parser.parse_args()
    main(
        chroma_path=args.chroma_path,
        query_text=args.query,
        k=args.k
    )
import streamlit as st
import openai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
import pandas as pd
import torch
import argparse

# HuggingFace Embeddings 모델 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={"device": device}
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

def generate_answer(api_key,query_text, context, metadata):
    """
    OpenAI GPT 모델을 사용해 답변을 생성
    
    Args:
        api_key (str): OpenAI API 키
        query_text (str): 사용자 질의
        context (str): 검색된 문서 컨텍스트
        metadata (dict): 관련 메타데이터

    Returns:
        str: 생성된 답변
    """
    # Title과 발의자 정보 처리
    openai.api_key = api_key
    title = metadata.get('title', 'N/A')
    proposer_info = ""
    if "(" in title and ")" in title:
        proposer_info = title[title.find("(")+1:title.find(")")]
        title_without_proposer = title[:title.find("(")].strip()
    else:
        title_without_proposer = title

    # 회기 정보 처리
    session = metadata.get('session', 'N/A')
    session_info = f"{session}대 국회" if session != "N/A" else "국회 회기 정보 없음"

    # Terminology와 Terminology_en 키워드 선택
    terminology = metadata.get('terminology', '').split(", ")
    terminology_en = metadata.get('terminology_en', '').split(", ")
    selected_korean_term = terminology[0] if terminology else "N/A"
    selected_english_term = terminology_en[0] if terminology_en else "N/A"

    # 첫 문장 구성
    first_sentence = f"의안정보시스템에 {metadata.get('date', 'N/A')}에 게시된 법률안 검토 보고서에 따르면, '{title_without_proposer}'은 "
    if proposer_info:
        first_sentence += f"{proposer_info}이 발의하였으며, "
    first_sentence += f"{metadata.get('committee', 'N/A')}에서 소관하는 법률안으로 {session_info}에서 공개되었습니다."

    # 검색 사이트
    search_site = 'https://likms.assembly.go.kr/bill/main.do'

    # 프롬프트 설계 
    prompt = f"""
    Metadata:
    Title: {metadata.get('title', 'N/A')}
    Session: {metadata.get('session', 'N/A')}
    Committee: {metadata.get('committee', 'N/A')}
    Date: {metadata.get('date', 'N/A')}
    Amendment: {metadata.get('amendment', 'N/A')}
    Enactment: {metadata.get('enactment', 'N/A')}
    Terminology: {metadata.get('terminology', 'N/A')}
    Terminology_en: {metadata.get('terminology_en', 'N/A')}

    Instruction:
    1. 아래 첫 문장을 기반으로 예시와 같이 답변을 시작하십시오:
       "{first_sentence}"
    - 예시: "의안정보시스템에 2020년 5월 29일 게시된 법률안 검토 보고서에 따르면, '성폭력범죄의 처벌 등에 관한 특례법 일부개정법률안'은 고용진 의원 등 10인이 발의하였으며, 법제사법위원회에서 소관하는 법률안으로 21대 국회에서 공개되었습니다."
    2. 질문에 충실히 답변하며, 질문에 대한 답변은 Context의 핵심 내용을 간결히 참고하여 제공합니다.
    3. 질문에 대한 답변 이후에, Amendment와 Enactment 값을 활용하여, 이 법안이 개정되었다면, 해당 발의로 개정되었음만 언급합니다. 마찬가지로 제정되었으면, 해당 발의로 제정되었음만 언급합니다. 개정되지 않거나 제정되지 않았다면 생략합니다. 
    - 예시: "최종적으로 이 법안은 해당 발의로 개정되었습니다."
    4. 마지막으로 Terminology와 Terminology_en에서 관련 키워드를 하나씩 선택하여, 추가 검색을 유도하는 문장을 작성하십시오.
       - 예: "이 법안에 대해 더 자세히 알아보시려면, 의안정보시스템에서  '{selected_korean_term}'과(와) '{selected_english_term}'를 검색해 보시기 바랍니다. ► 의안정보시스템 바로가기 : {search_site}"

    Context:
    {context}

    Question:
    {query_text}

    Answer:
    """

    # 모델 호출
    try:
        completion = openai.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "메타데이터와 컨텍스트를 기반으로 사용자가 이해하기 쉬운 답변을 작성하십시오. 최종적으로 논리적인 흐름을 가진 답변을 제공하십시오 "},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.7  
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""
    

# Streamlit 메인 함수
def main(csv_path, chroma_path, api_key):
    # CSV 파일 로드
    data = pd.read_csv(csv_path)

    # Streamlit 애플리케이션 시작
    st.set_page_config(page_title=" 나를 위한 법이 궁금해", page_icon="⚖️")
    st.title(" 나를 위한 법!이 궁금해 👀🔎")

    # 안내 문구 표시
    st.markdown(
        """
        저는 일상과 관련된 주요 법률안의 제정과 개정 과정을 설명하고, 법률안에 대한 최신 정보를 제공하는 챗봇이에요. 
        궁금한 점이 있다면 편하게 질문해주세요! 🙌
        """
    )
    st.markdown(
        """
        **특히 20대(2016-2020) 그리고 21대 국회(2020-2024)동안 다뤄진 법률안들에 대해 답변드릴 수 있어요!😊**
        """
    )
    st.markdown("---")
    st.markdown(
        """
        먼저 제가 답변을 제공할 수 있는 법률안 목록을 먼저 보여드릴게요!
        """
    )
    # st.markdown("---")
    # **위원회 선택**
    selected_committee = st.selectbox(
        "✔️ 원하는 위원회를 선택하세요",
        options=data["committee"].unique()
    )

    # **법 종류 필터링**
    filtered_data = data[data["committee"] == selected_committee]
    selected_field = st.selectbox(
        "✔️ 법 종류를 선택하세요",
        options=filtered_data["field"].unique()
    )
    # 선택 후 메시지 추가
    if selected_committee and selected_field:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #ccc; margin-top: 10px;">
            <strong>💡 {selected_committee}가 소관하는 <span style="color: #007BFF;">{selected_field}</span>에 대해 선택하셨네요!</strong><br>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.write("아래는 이 법에 대한 주요 법률안 목록이에요.")
    
    # 국회 회기(session)별로 그룹화
    field_data = filtered_data[filtered_data["field"] == selected_field].copy()


    # 보고서 게시일을 기준으로 내림차순 정렬
    field_data["date"] = pd.to_datetime(field_data["date"])  # date 열을 datetime 형식으로 변환
    field_data = field_data.sort_values(by="date", ascending=False)  # 내림차순 정렬

    # 국회 회기 목록 가져오기
    sessions = field_data["session"].unique()

    st.write(f"### 📋 {selected_field} 관련 목록")

    # 국회 회기별로 법안 표시
    for session in sorted(sessions):
        with st.expander(f"🏛️   {session}대 국회"):
            session_data = field_data[field_data["session"] == session]
            for _, row in session_data.iterrows():
                st.markdown(
                    f"""
                    **⚪️ 법률안 제목:** {row['title']}  
                    **소관위원회:** {row['committee']}  
                    **보고서 게시일:** {row['date'].strftime('%Y-%m-%d')}  
                    """
                )




    st.markdown("---")
    st.write("### 🙋‍♀️ 해당 법률안에 대해 더 궁금한 점이 있으신가요?")

    # Streamlit 입력 폼
    with st.form("query_form"):
        user_input = st.text_area("**질문을 입력해주세요**")
        submitted = st.form_submit_button("Submit")

        if submitted:
            try:
                # RAG로 컨텍스트와 메타데이터 검색
                context, metadata = query_rag(chroma_path,user_input, k=3)
                answer = generate_answer(user_input, context, metadata)

                # 답변 표시
                # st.write("### A ")
                st.success(answer)

                # Retrieved Context를 토글 안에 숨기기
                with st.expander("제공한 답변에 참고한 내용을 확인하고 싶다면 열어주세요 🧐"):
                    st.markdown(
                        f"""
                        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #ccc;">
                            {context}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    # st.write(context)

                # Metadata를 보기 좋게 표시
                with st.expander("제공한 답변에 대한 출처를 알려드릴게요 🤗"):
                    st.markdown(
                        f"""
                        <div style="background-color: #f1f3f4; padding: 10px; border-radius: 5px; border: 1px solid #ccc;">
                            <strong>법률안 검토보고서 제목:</strong> {metadata.get('title', 'N/A')}<br>
                            <strong>소관위원회:</strong> {metadata.get('committee', 'N/A')}<br>
                            <strong>보고서 게시일:</strong> {metadata.get('date', 'N/A')}<br>
                            <strong>제정여부:</strong>{metadata.get('enactment', 'N/A')}<br>
                            <strong>개정여부:</strong>{metadata.get('amendment', 'N/A')}<br>
                            <strong>의안 처리결과:</strong> {metadata.get('disposal', 'N/A')}<br>
                            <strong>법률안 관련 전문용어:</strong> {metadata.get('terminology', 'N/A')}<br>
                            <strong>영어로 번역된 법률안 관련 전문용어:</strong> {metadata.get('terminology_en', 'N/A')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                
            except ValueError as e:
                st.warning(f"Error: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlit 법률 정보 애플리케이션")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV 파일 경로")
    parser.add_argument("--chroma_path", type=str, required=True, help="ChromaDB 경로")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API 키")

    args = parser.parse_args()
    main(args.csv_path, args.chroma_path, args.api_key)
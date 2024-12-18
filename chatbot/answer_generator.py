import openai

# OpenAI API 키 설정
openai.api_key = ""  

def generate_answer(query_text, context, metadata):
    """
    OpenAI GPT 모델을 사용해 질문에 대한 답변을 생성

    Args:
        query_text (str): 사용자 질의
        context (str): 검색된 문서의 컨텍스트
        metadata (dict): 관련 메타데이터

    Returns:
        str: GPT 모델이 생성한 답변
    """
    # Title과 발의자 정보 처리
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
            model="gpt-4o-mini",  # GPT 모델 선택
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
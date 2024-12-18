import pandas as pd
import json
import argparse

def generate_metadata(input_json, output_csv):
    """
    JSON 데이터를 읽어 Streamlit용 CSV 파일로 변환

    Args:
        input_json (str): 최종 전처리된 JSON 데이터 경로
        output_csv (str): 생성될 CSV 파일 경로
    """
    try:
        with open(input_json, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        
        # JSON 데이터를 DataFrame으로 변환
        data = pd.DataFrame([
            {
                "committee": item.get("committee", "N/A"),
                "session": item.get("session", "N/A"),
                "field": item.get("field", "N/A"),
                "title": item.get("title", "N/A"),
                "date": item.get("date", "N/A")
            }
            for item in json_data
        ])
        
        # CSV 파일 저장
        data.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"CSV 파일이 성공적으로 생성되었습니다: {output_csv}")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    # 입력 인자 설정
    parser = argparse.ArgumentParser(description="JSON 데이터를 Streamlit용 CSV 파일로 변환")
    parser.add_argument("--input_json", type=str, required=True, help="최종 전처리된 JSON 데이터 경로")
    parser.add_argument("--output_csv", type=str, required=True, help="출력될 CSV 파일 경로")

    args = parser.parse_args()
    generate_metadata(args.input_json, args.output_csv)
import os
import json
import argparse


def load_json_from_folder(input_folder):
    """
    폴더 내의 모든 JSON 파일을 로드하여 하나의 리스트로 반환
    
    Args:
        input_folder (str): JSON 파일이 저장된 폴더 경로
        
    Returns:
        list: 모든 JSON 파일의 데이터를 통합한 리스트
    """
    data_list = []
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):  # 파일의 데이터가 리스트인 경우
                            data_list.extend(data)
                        elif isinstance(data, dict):  # 파일의 데이터가 단일 객체인 경우
                            data_list.append(data)
                        else:
                            print(f"Skipping file {file_name}: Unexpected JSON structure.")
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {file_path}: {e}")
    return data_list


def process_data(data):
    """
    단일 JSON 데이터의 구조를 정리
    
    Args:
        data (dict): 단일 JSON 데이터 객체
        
    Returns:
        dict: 정리된 데이터 객체
    """
    if isinstance(data, dict):
        return {
            "id": data.get("bill_id", ""),
            "session": data.get("session", ""),
            "title": data.get("title", ""),
            "committee": data.get("committee", ""),
            "field": data.get("field", ""),
            "paragraph": data.get("gen_summary", ""),  # gen_summary를 paragraph로 대체
            "enactment": data.get("enactment", ""),
            "amendment": data.get("amendment", ""),
            "terminology": data.get("terminology", ""),
            "disposal": data.get("disposal", ""),
            "date": data.get("date", "")
        }
    return None


def process_all_data(data_list):
    """
    데이터 리스트 내 모든 데이터를 처리하여 반환
    
    Args:
        data_list (list): JSON 데이터의 리스트
        
    Returns:
        list: 모든 JSON 데이터를 처리한 결과 리스트
    """
    processed_data = []
    for data in data_list:
        processed_item = process_data(data)
        if processed_item:
            processed_data.append(processed_item)
    return processed_data


def save_to_json(data, output_file):
    """
    Args:
        data (list or dict): 저장할 데이터
        output_file (str): 저장할 파일 경로
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data successfully saved to {output_file}")


def filter_data(data, selected_committees, selected_sessions):
    """
    위원회 및 회기를 기준으로 데이터를 필터링
    
    Args:
        data (list): 필터링할 데이터 리스트
        selected_committees (list): 필터링할 위원회 목록
        selected_sessions (list): 필터링할 회기 목록
        
    Returns:
        list: 필터링된 데이터 리스트
    """
    filtered_data = [
        item for item in data
        if item.get("committee") in selected_committees and item.get("session") in selected_sessions
    ]
    print(f"Total records after filtering: {len(filtered_data)}")
    return filtered_data


def main(input_folder, merged_output_file, final_output_file):
    """
    Args:
        input_folder (str): 원본 데이터의 폴더 경로
        merged_output_file (str): 중간 병합 데이터 파일 경로
        final_output_file (str): 최종 전처리된 데이터 파일 경로
    """
    #모든 하위 폴더 내의 모든 파일을 하나로 병합
    print("Merging all JSON files from input folder...")
    merged_data = load_json_from_folder(input_folder)
    save_to_json(merged_data, merged_output_file)
    
    #데이터 전처리
    print("Processing merged data...")
    processed_data = process_all_data(merged_data)
    
    #데이터 필터링
    print("Filtering processed data...")
    selected_committees = [
        "민생경제안정특별위원회", "법제사법위원회", "정무위원회", "보건복지위원회",
        "환경노동위원회", "국토교통위원회", "행정안전위원회", "교육문화체육관광위원회",
        "여성가족위원회", "기획재정위원회",
        "농림축산식품해양수산위원회", "예산결산특별위원회",
        "아동·여성대상성폭력대책특별위원회"
    ]
    selected_sessions = ["20", "21"]
    
    filtered_data = filter_data(processed_data, selected_committees, selected_sessions)
    save_to_json(filtered_data, final_output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the folder containing input JSON files')
    parser.add_argument('--merged_output_file', type=str, required=True, help='Path to save merged JSON file')
    parser.add_argument('--final_output_file', type=str, required=True, help='Path to save final preprocessed JSON file')

    args = parser.parse_args()
    
    main(
        input_folder=args.input_folder, 
        merged_output_file=args.merged_output_file, 
        final_output_file=args.final_output_file
    )
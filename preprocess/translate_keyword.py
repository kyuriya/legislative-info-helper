import os
import json
import argparse
import torch
from transformers import MarianMTModel, MarianTokenizer


def initial_translation_model(model_name="Helsinki-NLP/opus-mt-ko-en"):
    """
    번역 모델과 토크나이저를 초기화
    
    Args:
        model_name (str): 사용할 번역 모델의 이름 (기본값: 'Helsinki-NLP/opus-mt-ko-en')
        
    Returns:
        tuple: 초기화된 tokenizer, model
    """
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Translation model loaded on {device}")
    return tokenizer, model, device


def translate_to_eng(batch_texts, tokenizer, model, device):
    """
    한국어 키워드의 배치를 영어로 번역
    
    Args:
        batch_texts (list): 번역할 텍스트의 리스트
        tokenizer: 번역 모델의 토크나이저
        model: 번역 모델
        device (str): 'cuda' 또는 'cpu'
        
    Returns:
        list: 번역된 영어 키워드의 리스트
    """
    inputs = tokenizer(batch_texts, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    outputs = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return translated_texts


def translate_terminology(data, tokenizer, model, device, batch_size=32):
    """
    terminology 필드를 영어로 번역하여 terminology_en 필드에 추가
    
    Args:
        data (list): JSON 데이터의 리스트
        tokenizer: 번역 모델의 토크나이저
        model: 번역 모델
        device (str): 'cuda' 또는 'cpu'
        batch_size (int): 배치 크기 (기본값: 32)
        
    Returns:
        list: 번역된 데이터 리스트
    """
    processed_data = []
    terminology_texts = [item.get("terminology", "") for item in data]  # 모든 용어 추출
    
    for i in range(0, len(terminology_texts), batch_size):
        batch_texts = terminology_texts[i:i + batch_size]
        translated_batch = translate_to_eng(batch_texts, tokenizer, model, device)
        
        for j, translated_keywords in enumerate(translated_batch):
            idx = i + j
            data[idx]["terminology_en"] = translated_keywords
            processed_data.append(data[idx])
    
    return processed_data


def remove_duplicate_keywords(data):
    """
    terminology_en 필드에서 중복된 키워드를 제거하고 고유한 키워드만 유지
    
    Args:
        data (list): JSON 데이터의 리스트
        
    Returns:
        list: 중복 키워드가 제거된 데이터 리스트
    """
    for item in data:
        terminology_en = item.get("terminology_en", "")
        if terminology_en:
            # 콤마(,)를 기준으로 분리한 뒤, 고유값만 유지
            unique_terms = list(set(terminology_en.split(", ")))
            # 다시 정렬 후 문자열로 병합
            item["terminology_en"] = ", ".join(sorted(unique_terms))
    return data


def save_to_json(data, output_file):
    """
    Args:
        data (list): 저장할 데이터
        output_file (str): 저장할 파일 경로
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data successfully saved to {output_file}")


def load_json_file(input_file):
    """
    Args:
        input_file (str): 불러올 JSON 파일의 경로
        
    Returns:
        list: 로드된 JSON 데이터의 리스트
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def main(input_file, output_file):

    print("Loading input data...")
    data = load_json_file(input_file)
    
    print("Initializing translation model...")
    tokenizer, model, device = initial_translation_model()
    
    print("Translating terminology to English...")
    translated_data = translate_terminology(data, tokenizer, model, device)
    
    print("Removing duplicate English keywords in terminology_en...")
    final_data = remove_duplicate_keywords(translated_data)
    
    print(f"Saving final data to {output_file}...")
    save_to_json(final_data, output_file)
    print("Data processing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Terminology Translation and Preprocessing Script")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the preprocessed JSON file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the final preprocessed JSON file')

    args = parser.parse_args()
    
    main(
        input_file=args.input_file, 
        output_file=args.output_file
    )
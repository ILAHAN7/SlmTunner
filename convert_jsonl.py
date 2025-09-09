import json

input_file = 'result.jsonl'
output_file = 'converted_plain.jsonl'

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        if line.strip():  # 빈 줄 무시
            data = json.loads(line)
            # 구조적 정보 보존: JSON 문자열로 직렬화
            new_data = {
                "prompt": json.dumps(data["input"], ensure_ascii=False),
                "completion": json.dumps(data["answer"], ensure_ascii=False)
            }
            f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')

print(f"변환 완료: {output_file} 파일이 생성되었습니다.")

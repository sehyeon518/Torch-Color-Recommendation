import json

# JSON 파일 경로
json_file_path = 'palette_and_image.jsonl'

def check_duplicates(json_data):
    seen_urls = set()

    for entry in json_data:
        src = entry.get('src')

        if src in seen_urls:
            print(f'Duplicate found for URL: {src}')
        else:
            seen_urls.add(src)

def remove_duplicates(json_data):
    seen_urls = set()
    unique_entries = []

    for entry in json_data:
        src = entry.get('src')

        if src not in seen_urls:
            seen_urls.add(src)
            unique_entries.append(entry)

    return unique_entries

# JSON 파일 읽기
with open(json_file_path, 'r') as json_file:
    try:
        json_data = [json.loads(line) for line in json_file.readlines()]
        print(len(json_data))
        unique_data = remove_duplicates(json_data)
        print(len(unique_data))
        # 중복이 제거된 데이터를 새로운 JSON 파일에 저장
        with open(json_file_path, 'w') as output_file:
            for entry in unique_data:
                json.dump(entry, output_file)
                output_file.write('\n')

        print('Duplicates removed, unique data saved to palette_and_image.json')
    except json.JSONDecodeError as e:
        print(f'Error decoding JSON: {e}')
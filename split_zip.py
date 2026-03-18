import os

def split_file(file_path, chunk_size_mb=50):
    chunk_size = chunk_size_mb * 1024 * 1024
    file_size = os.path.getsize(file_path)

    with open(file_path, 'rb') as f:
        chunk_num = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            chunk_filename = f"{file_path}.part{chunk_num}"
            with open(chunk_filename, 'wb') as chunk_file:
                chunk_file.write(chunk)

            print(f"Created: {chunk_filename}")
            chunk_num += 1

    print(f"\n✅ Split complete: {chunk_num} parts created.")


# Example usage
split_file(r"D:\\algo-dashboard\\algo-strategies\\filtered_data_new.zip", chunk_size_mb=90)
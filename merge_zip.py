import os

def join_file(output_file, parts_prefix):
    parts = sorted(
        [p for p in os.listdir() if p.startswith(parts_prefix)],
        key=lambda x: int(x.split("part")[-1])
    )

    with open(output_file, 'wb') as outfile:
        for part in parts:
            with open(part, 'rb') as infile:
                outfile.write(infile.read())
            print(f"Added: {part}")

    print(f"\n✅ File reconstructed: {output_file}")


# Example usage
join_file("reconstructed.zip", "bigfile.zip.part")
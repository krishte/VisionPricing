import os
import re

# chat gpt heheheha
def replace_first_number_with_offset(filename, offset):
    with open(filename, 'r') as file:
        content = file.read()
        # Use regular expression to find the first number
        match = re.search(r'\d+', content)
        if match:
            # Get the first number and add the offset
            original_number = int(match.group())
            new_number = original_number + offset
            # Replace the first number with the new one
            modified_content = re.sub(r'\d+', str(new_number), content, count=1)
            # Write the modified content back to the file
            with open(filename, 'w') as file:
                file.write(modified_content)
            print(f"Modified {filename}: Replaced {original_number} with {new_number}")

folder_path = 'datasets/aleksa/labels'

# Offset to add to the first number
offset = 40

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        replace_first_number_with_offset(file_path, offset)

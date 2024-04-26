import os

directory = 'PATH TO FILES'

for name in os.listdir(directory):
    filename = name.split(".")[0]
    extension = name.split(".")[1]
    if '_' in filename:
        parts = filename.split('_')
        # for example renamed wp_2.jpg to 2_wp.jpg
        if len(parts) == 2:
            number = parts[1]
            word = parts[0]
            # Check if the filename needs renaming
            if not word.isdigit() and number.isdigit():
                new_filename = f"{number}_{word}"
                # Rename the file
                os.rename(os.path.join(directory, name), os.path.join(directory, new_filename + "."+extension))
                print(f"Renamed {name} to {new_filename + "."+extension}")

        # for example Renamed augmented_wp_19.png to 19_augmented_wp.png
        if len(parts) == 3:
            word1 = parts[0]
            word2 = parts[1]
            word3 = parts[2]
            new_filename = filename
        
            # Check if 'number' is a valid number
            if word2.isdigit():
                new_filename = f"{word2}_{word1}_{word3}"
            elif word3.isdigit():
                new_filename = f"{word3}_{word1}_{word2}"

            os.rename(os.path.join(directory, name), os.path.join(directory, new_filename + "."+extension))
            print(f"Renamed {name} to {new_filename + "."+extension}")

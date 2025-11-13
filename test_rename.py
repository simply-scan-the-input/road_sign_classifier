import os

test_dir = "/content/road_sign_classifier/fotki_znaków"  # adjust if your path is different

for folder in os.listdir(test_dir):
    old_path = os.path.join(test_dir, folder)
    if os.path.isdir(old_path):
        try:
            class_num = int(folder.split("_")[-1])
        except ValueError:
            print(f"Skipping {folder} (no valid number)")
            continue

        new_name = f"{class_num:05d}"
        new_path = os.path.join(test_dir, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {folder} → {new_name}")

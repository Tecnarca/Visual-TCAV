import os

# Target directory
directory = "/home/tecnarca/PycharmProjects/Visual-TCAV/edited_images/leopard_unAsparagus"

for filename in os.listdir(directory):
    if "edited" in filename or "_" in filename:
        old_path = os.path.join(directory, filename)
        new_filename = filename.replace("edited", "")
        new_filename = filename.replace("_", "")
        new_path = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")

print("✅ Renaming complete!")

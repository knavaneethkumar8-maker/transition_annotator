import json

file_path = "file_status.json"

# Load JSON
with open(file_path, "r") as f:
    data = json.load(f)

# Add "verified": False only if missing
for file_name in data:
    if "verified" not in data[file_name]:
        data[file_name]["verified"] = False

# Save back
with open(file_path, "w") as f:
    json.dump(data, f, indent=2)

print("Added 'verified': false only where missing.")
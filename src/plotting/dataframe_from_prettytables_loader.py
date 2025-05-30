import re
import pandas as pd

def load_prettytables_as_dataframe(file_path):
    # Read the content of the file
    text_content = file_path.read_text()
    # Use the previously loaded content to extract structured data into a DataFrame
    lines = [line.strip() for line in text_content.strip().split("\n") if "|" in line]

    # Find all blocks of tables separated by model/class headers
    tables = []
    current_model = current_class = current_examples = ""
    data_rows = []

    for line in lines:
        if "Model:" in line and "Class:" in line:
            if data_rows:
                tables.append((current_model, current_class, current_examples, data_rows))
                data_rows = []
            parts = line.strip("|").split(";")
            current_model = parts[0].split(":")[1].strip()
            current_class = parts[1].split(":")[1].strip()
            current_examples = parts[2].split(":")[1].strip() if len(parts) > 2 else ""
        elif "+-" in line or "[" in line:
            parts = [part.strip() for part in line.strip("|").split("|")]
            if parts[0]:
                current_concept = parts[0]
            else:
                parts[0] = current_concept
            # Parse mean and std
            mean_std_match = re.match(r"([\d.eE+-]+)\s*\+-\s*([\d.eE+-]+)", parts[2])
            mean, std = (float(mean_std_match.group(1)), float(mean_std_match.group(2))) if mean_std_match else (None, None)
            # Parse CI
            ci_bounds = eval(parts[3])
            ci_low, ci_high = float(ci_bounds[0]), float(ci_bounds[1])
            data_rows.append([current_model, current_class, current_examples, parts[0], parts[1], mean, std, ci_low, ci_high])

    # Append the last table
    if data_rows:
        tables.append((current_model, current_class, current_examples, data_rows))

    # Flatten all rows into one dataframe
    all_rows = [row for (_, _, _, table_rows) in tables for row in table_rows]
    df = pd.DataFrame(all_rows, columns=["Model", "Class", "Examples", "Concept", "Layer", "Mean", "Std", "CI Low", "CI High"])
    return df
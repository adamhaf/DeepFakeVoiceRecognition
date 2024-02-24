import csv

def remove_inthewild_from_filepath(csv_file_path, output_file_path):
    try:
        with open(csv_file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames

            # Ensure that 'filepath' is a valid field in the CSV
            if 'filepath' not in fieldnames:
                raise ValueError("Column 'filepath' not found in the CSV file.")

            rows = []
            for row in reader:
                if 'filepath' in row:
                    # Remove "InTheWild/" from the 'filepath' column value
                    row['filepath'] = row['filepath'].replace("InTheWild/", "")
                rows.append(row)

        with open(output_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print("Operation completed successfully. Updated CSV written to:", output_file_path)

    except FileNotFoundError:
        print("File not found:", csv_file_path)
    except Exception as e:
        print("An error occurred:", str(e))

# Example usage:
input_csv = "CSVs/InTheWild.csv"  # Replace with your input CSV file path
output_csv = "CSVs/InTheWild1.csv"  # Replace with the desired output CSV file path
remove_inthewild_from_filepath(input_csv, output_csv)

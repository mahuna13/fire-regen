import os
from pathlib import Path
import zipfile
import re


def unzip_dnbr_files(dir_path: str):
    directory = os.fsencode(dir_path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".zip"):
            filename_no_ext = filename[:-4]
            zip_file = os.path.join(dir_path, filename)
            unzip_directory = os.path.join(dir_path, filename_no_ext)

            print(zip_file)
            # Create the directory where to unzip.
            Path(unzip_directory).mkdir(exist_ok=True)

            # Unzip the content inside the new directory.
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                files_in_archive = zip_ref.namelist()
                r = re.compile(f"^{filename_no_ext}.*_dnbr\.tif")
                matches = list(filter(r.match, files_in_archive))
                if (len(matches) != 1):
                    print("No dnbr file.")
                else:
                    dnbr_file = matches[0]
                    zip_ref.extract(dnbr_file, path=unzip_directory)


def unzip_all_dnbr_files(mtbs_path: str):
    directory = os.fsencode(mtbs_path)
    for year in os.listdir(directory):
        unzip_dnbr_files(os.fsdecode(os.path.join(directory, year)))

# from llama_cpp import Llama
import os
import time

import requests

from auto_lca.shared.util import upload_blob

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class FileManager:
    # Class to extract download and upload .pdfs
    @classmethod
    def download_pdf_with_session(
        cls, session, pdf_url, publication_page_url, save_path
    ):
        """
        Downloads a protected PDF using the SAME session that opened the publication page.
        """

        headers = {
            **REQUEST_HEADERS,
            "Referer": publication_page_url,
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "document",
        }

        s = time.time()
        response = session.get(pdf_url, stream=True, timeout=15, headers=headers)
        response.raise_for_status()
        print("Get PDF:", time.time() - s, "seconds")

        s = time.time()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        print("Write PDF:", time.time() - s, "seconds")

        print("Saved →", save_path)

    @classmethod
    def download_pdf(cls, url, save_path):
        """
        Download a PDF from a given URL and save it to the specified path.

        Args:
            url (str): The url of the pdf file.
            save_path (str): The path to save the downloaded pdf.
        """
        # Download the PDF
        s = time.time()
        response = requests.get(url, stream=True, timeout=5, headers=REQUEST_HEADERS)
        response.raise_for_status()
        e = time.time()
        print(f"Get paper took: {e-s}")

        s = time.time()
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        e = time.time()
        print(f"Download paper took: {e-s}")

        print(f"PDF downloaded successfully to {save_path}")

        return None

    @classmethod
    def upload_pdf_to_cloud(
        cls, url, destination_blob_name, bucket_name="ingest-paper-pdfs"
    ):
        """
        Download a PDF from a given URL and save it to the specified path.

        Args:
            url (str): The url of the pdf file.
            destination_blob_name (str): The pdf name.

        Returns:
            bool: True if the file was downloaded successfully, False otherwise.
        """

        # Create a temporary file to store the downloaded PDF locally
        temp_file = os.path.join(
            os.getcwd(), f"/tmp/{os.path.basename(destination_blob_name)}"
        )
        cls.download_pdf(url, temp_file)

        # Upload the file to Google Cloud Storage
        s = time.time()
        blob_url = upload_blob(bucket_name, destination_blob_name, temp_file)
        print(
            f"PDF uploaded successfully to GCS bucket '{bucket_name}' as '{destination_blob_name}'"
        )
        e = time.time()
        print(f"Upload paper took: {e-s}")

        # # Optional: Make the file publicly accessible
        # blob.make_public()
        # print(f"File is publicly accessible at {blob.public_url}")
        # return blob.public_url # TODO Add cloud storage to pdf
        return temp_file, blob_url

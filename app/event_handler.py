"""
Event Handler for GCS Upload Events
Processes CloudEvents from Eventarc when files are uploaded to the bucket
"""

import sys
import os
# Inject parent directory into sys.path to enable root package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pathlib import Path
from typing import Dict, Any

from flask import Flask, request, jsonify
from google.cloud import storage

from config.config_loader import get_config_loader
from pipeline.document_processor import DocumentProcessor
from pipeline.mmkg_builder import MMKGBuilder
from graph.spanner_graph import SpannerGraphManager
from utils.logger import get_logger

logger = get_logger(__name__)

app = Flask(__name__)

def process_uploaded_file(bucket_name: str, file_path: str) -> Dict[str, Any]:
    """
    Process uploaded file and build MMKG.
    """
    logger.info(f"Processing file: gs://{bucket_name}/{file_path}")

    try:
        config_loader = get_config_loader()
        bucket_config = config_loader.get_bucket_config()
        upload_folder = bucket_config.get("upload_folder", "uploads/")

        if not file_path.startswith(upload_folder):
            logger.info(f"Skipping file not in uploads folder: {file_path}")
            return {
                "success": False,
                "message": f"File not in uploads folder ({upload_folder})",
                "skipped": True
            }

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)

        temp_dir = Path("/tmp/enterprise_rag_events/")
        temp_dir.mkdir(exist_ok=True)

        file_name = Path(file_path).name
        temp_file = temp_dir / file_name

        logger.info(f"Downloading to: {temp_file}")
        blob.download_to_filename(str(temp_file))

        doc_processor = DocumentProcessor()
        mmkg_builder = MMKGBuilder()
        graph_manager = SpannerGraphManager()

        logger.info("Processing document...")
        result = doc_processor.process_document(str(temp_file))

        logger.info("Building MMKG...")
        mmkg_data = mmkg_builder.build_mmkg(result["pages"])

        logger.info("Inserting into graph database...")
        graph_manager.insert_entities(mmkg_data["entities"])
        graph_manager.insert_relationships(mmkg_data["relationships"])

        processed_folder = bucket_config.get("processed_folder", "processed/")
        processed_path = f"{processed_folder}{file_name}.json"
        processed_blob = bucket.blob(processed_path)

        processed_blob.upload_from_string(
            json.dumps(mmkg_data, indent=2),
            content_type="application/json"
        )
        logger.info(f"Results saved to: gs://{bucket_name}/{processed_path}")

        temp_file.unlink()

        return {
            "success": True,
            "file_path": file_path,
            "entity_count": mmkg_data["metadata"]["entity_count"],
            "relationship_count": mmkg_data["metadata"]["relationship_count"],
            "processed_path": processed_path
        }
    except Exception as e:
        logger.error(f"Failed to process file: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }

@app.route("/", methods=["POST"])
def handle_cloud_event():
    """Handle CloudEvent from Eventarc."""
    try:
        event = request.get_json()
        if not event:
            logger.warning("Received empty event")
            return jsonify({"success": False, "error": "Empty event"}), 400

        logger.info(f"Received CloudEvent: {event.get('type')}")
        event_type = event.get("type")
        data = event.get("data", {})

        if event_type != "google.cloud.storage.object.v1.finalized":
            logger.info(f"Ignoring event type: {event_type}")
            return jsonify({"success": True, "message": "Event type ignored"}), 200

        bucket_name = data.get("bucket")
        file_path = data.get("name")

        if not bucket_name or not file_path:
            logger.error("Missing bucket or file name in event")
            return jsonify({"success": False, "error": "Missing bucket or file"}), 400

        result = process_uploaded_file(bucket_name, file_path)
        if result.get("skipped"):
            return jsonify(result), 200

        if result.get("success"):
            return jsonify(result), 200
        else:
            return jsonify(result), 500
    except Exception as e:
        logger.error(f"Error handling CloudEvent: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

"""
Background sync: upload lineart to Cloudinary, create AIcentre_Image__c in Salesforce.

Runs entirely in a background thread so the drawing pipeline is never blocked.
All errors are logged and swallowed — this module must never affect the drawing.
"""

import os
import time
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import cloudinary
import cloudinary.uploader
import requests

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each attempt


class CloudSync:
    """Uploads lineart to Cloudinary and creates a Salesforce record."""

    def __init__(self):
        self._sf_access_token: Optional[str] = None
        self._sf_instance_url: Optional[str] = None
        self._sf_user_id: Optional[str] = None
        self._configured = False

    def initialize(self) -> bool:
        """Check that all required env vars are present and configure Cloudinary."""
        cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME")
        cloud_key = os.environ.get("CLOUDINARY_API_KEY")
        cloud_secret = os.environ.get("CLOUDINARY_API_SECRET")
        sf_client_id = os.environ.get("SF_CLIENT_ID")
        sf_client_secret = os.environ.get("SF_CLIENT_SECRET")

        missing = []
        if not cloud_name:
            missing.append("CLOUDINARY_CLOUD_NAME")
        if not cloud_key:
            missing.append("CLOUDINARY_API_KEY")
        if not cloud_secret:
            missing.append("CLOUDINARY_API_SECRET")
        if not sf_client_id:
            missing.append("SF_CLIENT_ID")
        if not sf_client_secret:
            missing.append("SF_CLIENT_SECRET")

        if missing:
            logger.warning(
                f"CloudSync disabled — missing env vars: {', '.join(missing)}"
            )
            return False

        cloudinary.config(
            cloud_name=cloud_name,
            api_key=cloud_key,
            api_secret=cloud_secret,
            secure=True,
        )

        self._configured = True
        logger.info("CloudSync initialized")
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync_in_background(self, lineart_path: str, prompt: str) -> None:
        """Fire-and-forget background sync. Safe to call even if not configured."""
        if not self._configured:
            return

        path = Path(lineart_path)
        if not path.exists() or path.stat().st_size == 0:
            logger.warning(f"CloudSync skipped — file not ready: {lineart_path}")
            return

        thread = threading.Thread(
            target=self._sync,
            args=(lineart_path, prompt),
            daemon=True,
        )
        thread.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sync(self, lineart_path: str, prompt: str) -> None:
        """Upload to Cloudinary then create the Salesforce record."""
        try:
            image_url = self._upload_to_cloudinary(lineart_path)
            if not image_url:
                return

            self._create_salesforce_record(image_url, prompt)
        except Exception as e:
            logger.error(f"CloudSync unexpected error: {e}")

    def _upload_to_cloudinary(self, file_path: str) -> Optional[str]:
        """Upload the lineart image and return its secure URL."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    f"Cloudinary upload attempt {attempt}/{MAX_RETRIES}..."
                )
                result = cloudinary.uploader.upload(
                    file_path,
                    folder="ai-centre-robot-arm",
                    resource_type="image",
                )
                url = result.get("secure_url")
                if url:
                    logger.info(f"Cloudinary upload successful: {url}")
                    return url
                raise RuntimeError("No secure_url in Cloudinary response")
            except Exception as e:
                logger.warning(f"Cloudinary attempt {attempt} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)

        logger.error("Cloudinary upload failed after all retries")
        return None

    # ------------------------------------------------------------------
    # Salesforce helpers
    # ------------------------------------------------------------------

    def _authenticate_salesforce(self) -> bool:
        """Obtain an access token via Client Credentials flow."""
        login_url = os.environ.get("SF_LOGIN_URL", "https://login.salesforce.com")
        token_url = f"{login_url}/services/oauth2/token"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    f"Salesforce auth attempt {attempt}/{MAX_RETRIES}..."
                )
                resp = requests.post(
                    token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": os.environ["SF_CLIENT_ID"],
                        "client_secret": os.environ["SF_CLIENT_SECRET"],
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                body = resp.json()

                self._sf_access_token = body["access_token"]
                self._sf_instance_url = body["instance_url"]

                # Fetch the running user's ID from the identity endpoint
                id_url = body.get("id")
                if id_url:
                    self._sf_user_id = id_url.rstrip("/").split("/")[-1]

                logger.info("Salesforce authentication successful")
                return True

            except Exception as e:
                logger.warning(f"Salesforce auth attempt {attempt} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)

        logger.error("Salesforce authentication failed after all retries")
        return False

    def _create_salesforce_record(self, image_url: str, prompt: str) -> bool:
        """Create an AIcentre_Image__c record."""
        if not self._sf_access_token:
            if not self._authenticate_salesforce():
                return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record = {
            "Name": f"Robot Arm - {timestamp}",
            "Image_URL__c": image_url,
            "Prompt__c": prompt,
            "Type__c": "Robot Arm",
            "Status__c": "Completed",
        }
        if self._sf_user_id:
            record["User_Id__c"] = self._sf_user_id

        url = (
            f"{self._sf_instance_url}"
            "/services/data/v62.0/sobjects/AIcentre_Image__c"
        )
        headers = {
            "Authorization": f"Bearer {self._sf_access_token}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    f"Salesforce record creation attempt {attempt}/{MAX_RETRIES}..."
                )
                resp = requests.post(
                    url, json=record, headers=headers, timeout=30
                )

                # Re-auth on 401 then retry
                if resp.status_code == 401 and attempt < MAX_RETRIES:
                    logger.info("Access token expired, re-authenticating...")
                    self._sf_access_token = None
                    if not self._authenticate_salesforce():
                        logger.error("Re-auth failed, aborting record creation")
                        return False
                    headers["Authorization"] = (
                        f"Bearer {self._sf_access_token}"
                    )
                    url = (
                        f"{self._sf_instance_url}"
                        "/services/data/v62.0/sobjects/AIcentre_Image__c"
                    )
                    time.sleep(RETRY_BACKOFF * attempt)
                    continue

                if not resp.ok:
                    logger.error(
                        f"Salesforce API error {resp.status_code}: {resp.text}"
                    )
                    resp.raise_for_status()
                body = resp.json()
                record_id = body.get("id", "unknown")
                logger.info(
                    f"Salesforce record created: {record_id}"
                )
                return True

            except Exception as e:
                self._sf_access_token = None
                logger.warning(
                    f"Salesforce record attempt {attempt} failed: {e}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)

        logger.error("Salesforce record creation failed after all retries")
        return False

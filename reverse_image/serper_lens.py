"""
Serper Lens API module for reverse image search.
Uses Google Lens through Serper API to find similar images.
"""
import asyncio
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path
import aiohttp
from loguru import logger


class SerperLensResult:
    """Represents a single result from Serper Lens search."""

    def __init__(
        self,
        title: str,
        source: str,
        link: str,
        image_url: str,
        thumbnail_url: Optional[str] = None,
    ):
        self.title = title
        self.source = source
        self.link = link  # Page URL
        self.image_url = image_url  # Image URL
        self.thumbnail_url = thumbnail_url

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SerperLensResult":
        """Create SerperLensResult from API response dictionary."""
        return cls(
            title=data.get("title", ""),
            source=data.get("source", ""),
            link=data.get("link", ""),
            image_url=data.get("imageUrl", ""),
            thumbnail_url=data.get("thumbnailUrl", ""),
        )


class SerperLensClient:
    """Client for Serper Lens API (Google Lens reverse image search)."""

    def __init__(self, api_key: str, max_results: int = 20):
        """
        Initialize Serper Lens client.

        Args:
            api_key: Serper API key
            max_results: Maximum number of results to retrieve (default: 20)
        """
        self.api_key = api_key
        self.max_results = max_results
        self.base_url = "https://google.serper.dev/lens"

    def _encode_image_to_base64(self, image_path: Path) -> str:
        """
        Encode local image file to base64 string.

        Args:
            image_path: Path to local image file

        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode("utf-8")

    async def _upload_to_imgbb(
        self, image_path: Path, api_key: str, expiration: int = 600
    ) -> str:
        """
        Upload local image to ImgBB and get public URL.

        Args:
            image_path: Path to local image file
            api_key: ImgBB API key
            expiration: Expiration time in seconds (default: 600)

        Returns:
            Public URL of uploaded image
        """
        image_b64 = self._encode_image_to_base64(image_path)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.imgbb.com/1/upload",
                data={
                    "key": api_key,
                    "image": image_b64,
                    "expiration": expiration,
                },
            ) as response:
                result = await response.json()

        if result.get("success"):
            return result["data"]["url"].strip()
        else:
            error_msg = result.get("error", {}).get("message", "Unknown error")
            raise Exception(f"ImgBB upload failed: {error_msg}")

    async def search(
        self,
        image_path: Path,
        imgbb_api_key: str,
    ) -> List[SerperLensResult]:
        """
        Perform reverse image search using Serper Lens API.

        Args:
            image_path: Path to local image file
            imgbb_api_key: ImgBB API key for uploading image

        Returns:
            List of SerperLensResult objects
        """
        logger.info(f"Uploading local image to ImgBB: {image_path}")
        public_url = await self._upload_to_imgbb(image_path, imgbb_api_key)
        logger.info(f"Image uploaded successfully: {public_url}")

        logger.info("Calling Serper Google Lens API...")
        results = await self._search_by_url(public_url)
        logger.info(f"Retrieved {len(results)} results")

        return results

    async def _search_by_url(self, image_url: str) -> List[SerperLensResult]:
        """
        Search using an image URL.

        Args:
            image_url: Public URL of the image

        Returns:
            List of SerperLensResult objects
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {"url": image_url}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url, headers=headers, json=payload
            ) as response:
                response_text = await response.text()

        try:
            data = eval(response_text)  # Parse response
        except Exception as e:
            logger.error(f"Failed to parse Serper response: {e}")
            logger.debug(f"Response text: {response_text}")
            return []

        # Extract organic results
        organic_results = data.get("organic", [])

        # Limit to max_results
        limited_results = organic_results[: self.max_results]

        # Convert to SerperLensResult objects
        results = [SerperLensResult.from_dict(item) for item in limited_results]

        return results


from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.helpers import detect_file_encodings


from typing import Any, Dict, List, Optional


class CustomTextLoader(BaseLoader):
    """Load plain text and metadata dictionary."""

    def __init__(
        self,
        plain_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,
    ):
        """Initialize with plain text and metadata."""
        self.plain_text = plain_text
        self.metadata = metadata or {}
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding

    def load(self) -> List[Document]:
        """Load from plain text."""
        text = self.plain_text
        if self.encoding:
            try:
                text = text.encode(self.encoding).decode(self.encoding)
            except UnicodeDecodeError as e:
                if self.autodetect_encoding:
                    detected_encodings = detect_file_encodings(text)
                    for encoding in detected_encodings:
                        # logger.debug("Trying encoding: ", encoding)
                        try:
                            text = text.encode(encoding).decode(encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                else:
                    raise RuntimeError(f"Error loading plain text") from e

        metadata = self.metadata
        return [Document(page_content=text, metadata=metadata)]
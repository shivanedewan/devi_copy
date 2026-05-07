# ... (at the top of your file with other imports)


# ... (inside your @app.get("/api/documents/{file_path:path}") endpoint)
# i have to get bs4 and urlib

# Add these imports at the top of your Python file
import base64
import io
from bs4 import BeautifulSoup # pip install beautifulsoup4
from urllib.parse import unquote # To handle URL-encoded filenames if any
import os
import shutil # For shutil.which to find libreoffice
import mimetypes
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime
import zipfile
import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Form, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse,StreamingResponse
from fastapi.security import OAuth2PasswordBearer
import logging
import json
from jose import JWTError, jwt


# new
from pydantic import BaseModel

from fastapi.staticfiles import StaticFiles

# ... (your existing imports: os, shutil, mimetypes, subprocess, tempfile, Path, etc.)
# ... (FastAPI imports, Settings, logger, etc.)


logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level. Can be DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format of the log messages
    handlers=[
        logging.FileHandler('app.log'),  # Log to a file named 'app.log'
        logging.StreamHandler()           # Also print to the console
    ]
)


logger = logging.getLogger(__name__)


# Elasticsearch configuration
ES_HOST = "http://192.168.10.234:9200"
ES_INDEX = "document_169"
LOG_INDEX = "logging"



idx_url="http://192.168.10.234:9200/document_169"

# --- Auth Configuration ---
SECRET_KEY = "your-secret-key-keep-it-secret" # In production, use env var
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# --- Auth Models & Utils ---
class User(BaseModel):
    username: str

async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    query_token: Optional[str] = Query(None, alias="token")
):
    final_token = token or query_token

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not final_token:
        raise credentials_exception

    try:
        payload = jwt.decode(final_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return User(username=username)

# --- Async Logging ---
async def log_user_action(username: str, action: str, details: Dict[str, Any]):
    log_entry = {
        "username": username,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "action": action,
        "details": details
    }
    try:
        async with httpx.AsyncClient() as client:
            url = f"{ES_HOST}{LOG_INDEX}/_doc"
            await client.post(url, json=log_entry)
    except Exception as e:
        logger.error(f"Failed to log action: {e}")

ALLOWED_DOCUMENT_ROOTS: List[Path] = [
    Path("/opt/ALL_CASES").resolve(),  
    # Path("/another/allowed/path").resolve(),
]
if not ALLOWED_DOCUMENT_ROOTS or not ALLOWED_DOCUMENT_ROOTS[0].is_dir(): # Basic check
    print("CRITICAL WARNING: ALLOWED_DOCUMENT_ROOTS is not configured correctly or paths do not exist.")
    # In a real app, you might want to prevent startup if this isn't set.

# Try to find LibreOffice or SOffice executable. Allow override via environment variable.
LIBREOFFICE_PATH = os.getenv("LIBREOFFICE_PATH", shutil.which("libreoffice") or shutil.which("soffice"))

if not LIBREOFFICE_PATH:
    print(
        "WARNING: LibreOffice binary ('libreoffice' or 'soffice') not found in PATH or via LIBREOFFICE_PATH env var. "
        "DOC/DOCX to HTML conversion will fail."
    )

app = FastAPI() # Your FastAPI app instance
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this when you run on production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)


# --- Helper Functions ---
def is_path_secure_and_valid(client_path_str: str, allowed_roots: List[Path]) -> Optional[Path]:
    """
    Validates if the client-provided absolute path is:
    1. Syntactically an absolute path.
    2. Resolves to a real file.
    3. Falls within one of the `allowed_roots`.
    Returns the resolved Path object if valid, None otherwise.
    Raises HTTPException for specific failure reasons.
    """
    if not client_path_str or not client_path_str.startswith("/"): # Or os.path.isabs for OS-agnostic check
        raise HTTPException(status_code=400, detail="Invalid path: Not an absolute path.")

    try:
        # Resolve the path (handles '..', symlinks, etc.) and ensure it exists (strict=True)
        resolved_path = Path(client_path_str).resolve(strict=True)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found at: {client_path_str}")
    except Exception as e: # Catch other resolution errors (e.g., permission issues during resolve)
        print(f"Error resolving path '{client_path_str}': {e}")
        raise HTTPException(status_code=500, detail="Error processing file path.")

    if not resolved_path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {client_path_str}")

    # Security Check: Ensure the resolved path is within an allowed root
    is_allowed = False
    for root in allowed_roots:
        try:
            # Check if root is an ancestor of resolved_path or they are the same
            if resolved_path == root or root in resolved_path.parents:
                is_allowed = True
                break
        except Exception: # Should not happen with pre-resolved paths, but for safety
            continue
    
    if not is_allowed:
        print(f"SECURITY ALERT: Unauthorized access attempt to '{resolved_path}' (original: '{client_path_str}')")
        raise HTTPException(status_code=403, detail="Access to this file path is forbidden.")

    return resolved_path


def get_static_html_year_folder(document_date: int | None) -> str:
    """
    Maps unix timestamp to static HTML folder name.
    """

    if not document_date or document_date == 0:
        return "0"

    if document_date > 10_000_000_000:
        document_date = int(document_date / 1000)

    if document_date >= 1735669800:
        return "2025"

    if 1704047400 <= document_date < 1735669800:
        return "2024"

    if 1672511400 <= document_date < 1704047400:
        return "2023"

    if 1577817000 <= document_date < 1672511400:
        return "2020-22"

    return "before-2020"





def embed_images_as_base64(html_content: str, base_path: Path) -> str:
    """
    Parses HTML content, finds local image references, and embeds them as Base64.
    Args:
        html_content: The HTML string.
        base_path: The directory where the HTML file and its associated images are located.
    Returns:
        HTML string with images embedded as Base64.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    images_embedded_count = 0
    for img_tag in soup.find_all('img'):
        src = img_tag.get('src')
        if src and not src.startswith(('http:', 'https:', 'data:')):
            # It's a relative path, resolve it against the HTML file's location
            # LibreOffice might create subdirectories for images, e.g., "_html_images"
            # or put them in the same directory.
            
            # Handle URL-encoded filenames from LibreOffice (e.g., %20 for space)
            image_relative_path_decoded = unquote(src)
            image_path = (base_path / image_relative_path_decoded).resolve()

            # Security check: ensure image_path is still within base_path
            # This is important if `src` could somehow contain `../` etc.
            if not str(image_path).startswith(str(base_path.resolve())):
                logger.warning(f"Skipping image due to potential path traversal: {src} from base {base_path}")
                continue

            if image_path.is_file():
                try:
                    with open(image_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    # Guess MIME type for the data URI
                    mime_type, _ = mimetypes.guess_type(image_path)
                    mime_type = mime_type or 'image/png' # Default if guess fails

                    img_tag['src'] = f"data:{mime_type};base64,{encoded_string}"
                    images_embedded_count += 1
                    # logger.debug(f"Embedded image: {image_path}")
                except Exception as e:
                    logger.error(f"Error embedding image {image_path}: {e}")
            else:
                logger.warning(f"Image not found for embedding: {image_path} (original src: {src})")
    
    if images_embedded_count > 0:
        logger.info(f"Successfully embedded {images_embedded_count} images as Base64.")
    return str(soup)


# *** UPDATED *** convert_to_html_with_libreoffice
def convert_to_html_with_libreoffice(source_file: Path, output_dir: Path, embed_images: bool = True) -> str: # Returns HTML string
    """
    Converts a document to HTML using LibreOffice.
    Optionally embeds images as Base64.
    Raises exceptions on failure. Returns the HTML content as a string.
    """
    if not LIBREOFFICE_PATH:
        raise RuntimeError("LibreOffice binary path is not configured or found.")
    if not source_file.exists():
        raise FileNotFoundError(f"Source file for conversion does not exist: {source_file}")

    # LibreOffice will create files in output_dir.
    # We expect an HTML file and potentially image files/subdirectories.
    
    cmd = [
        LIBREOFFICE_PATH, "--headless", "--nolockcheck", "--nodefault",
        "--norestore", "--invisible",
        "--convert-to", "html:HTML (StarWriter)", # This filter usually handles images better
        # Forcing a specific HTML export filter might be needed if images are problematic
        # e.g. "html:XHTML Writer File" might behave differently with images
        "--outdir", str(output_dir), str(source_file)
    ]

    try:
        process = subprocess.run(cmd, check=True, timeout=120, capture_output=True, text=True, errors='replace')
        if process.stderr:
             logger.info(f"LibreOffice STDERR for {source_file.name} conversion: {process.stderr[:1000]}") # Log stderr
    except subprocess.TimeoutExpired as e:
        error_msg = f"LibreOffice conversion timed out for {source_file.name}. Stderr: {e.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    except subprocess.CalledProcessError as e:
        error_msg = (f"LibreOffice conversion failed for {source_file.name} (exit code {e.returncode}).\n"
                     f"Stderr: {e.stderr[:1000]}\nStdout: {e.stdout[:1000]}")
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error during LibreOffice execution for {source_file.name}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    # Find the generated HTML file. LibreOffice typically names it based on the source stem.
    # It might also create subfolders like 'filename_html_SOMEHASH' or just 'filename.html'
    
    # Try common naming patterns first
    expected_html_filename = source_file.stem + ".html"
    html_file_path = output_dir / expected_html_filename
    
    if not html_file_path.is_file():
        # Fallback: search for any .html file in the output directory
        # This is less precise but can catch cases where LO uses a different naming scheme
        html_files_in_output = list(output_dir.glob("*.html"))
        if not html_files_in_output:
            # Also check one level deep for subdirectories LO might create
            for item in output_dir.iterdir():
                if item.is_dir():
                    html_files_in_output.extend(list(item.glob("*.html")))
            
            if not html_files_in_output:
                error_msg = f"LibreOffice conversion for {source_file.name} seemed to succeed, but no HTML output file found in {output_dir} or its immediate subdirectories."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        html_file_path = html_files_in_output[0] # Take the first one found
        logger.warning(f"Expected HTML file '{expected_html_filename}' not found. Using first found: '{html_file_path.name}'")

    logger.info(f"LibreOffice successfully converted '{source_file.name}' to '{html_file_path}'.")
    
    with open(html_file_path, "r", encoding="utf-8", errors="replace") as f:
        html_content = f.read()

    if embed_images:
        logger.info(f"Attempting to embed images for {html_file_path.name} from base directory {html_file_path.parent}")
        # The base path for resolving relative image URLs is the directory containing the HTML file.
        html_content_with_embedded_images = embed_images_as_base64(html_content, html_file_path.parent)
        return html_content_with_embedded_images
    
    return html_content

# *** UPDATED *** /api/documents/{file_path:path} endpoint
@app.get("/api/documents/{file_path:path}")
async def get_document(
    file_path: str,
    background_tasks: BackgroundTasks,
    action: str = Query("view", enum=["view", "download"]),
    document_date: Optional[int] = Query(None), # Added document_date
    user: User = Depends(get_current_user)
):
    file_path = unquote(file_path)
    try:
        valid_src_path: Path = is_path_secure_and_valid(file_path, ALLOWED_DOCUMENT_ROOTS) # Your validation function
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during path validation for '{file_path}': {e}")
        raise HTTPException(status_code=500, detail="Server error during path validation.")

    file_extension = valid_src_path.suffix.lower()
    mime_type, _ = mimetypes.guess_type(valid_src_path)
    mime_type = mime_type or "application/octet-stream"

    # Log the action asynchronously
    background_tasks.add_task(
        log_user_action,
        user.username,
        f"{action}_document",
        {"system_path": file_path, "document_date": document_date}
    )

    try:
        if action == "download":
            return FileResponse(
                path=valid_src_path, media_type=mime_type, filename=valid_src_path.name,
                headers={"Content-Disposition": f'attachment; filename="{valid_src_path.name}"'}
            )
        
        elif action == "view":
            # --- DOC/DOCX to HTML (with embedded images) ---
            if file_extension in [".doc", ".docx"]:
                # --- NEW LOGIC: Check for pre-converted static HTML ---
                if document_date:
                     try:
                        folder = get_static_html_year_folder(document_date)
                        # Construct static path: /bpt/ALL_CASES/html/{YEAR}/{DOC_NAME}.html
                        # user specified /bpt/ALL_CASES, but ALLOWED_DOCUMENT_ROOTS is /opt/ALL_CASES. 
                        # I will assume the user meant /bpt based on their prompt.
                        static_html_root = Path("/bpt/ALL_CASES/html")
                        static_file_path = static_html_root / folder / f"{valid_src_path.stem}.html"
                        
                        if static_file_path.exists():
                             logger.info(f"Serving static HTML for {valid_src_path.name} from {static_file_path}")
                             with open(static_file_path, "r", encoding="utf-8", errors="replace") as f:
                                 html_content = f.read()
                             return HTMLResponse(content=html_content)
                        else:
                             logger.info(f"Static HTML not found at {static_file_path}. Falling back to dynamic conversion.")
                     except Exception as e:
                         logger.error(f"Error checking/serving static HTML: {e}. Falling back to dynamic conversion.")

                # Fallback to dynamic conversion if static not found or simple bypass
                if LIBREOFFICE_PATH:
                    with tempfile.TemporaryDirectory(prefix="lo_html_convert_") as tmp_dir_str:
                        tmp_output_dir = Path(tmp_dir_str)
                        try:
                            # Call the updated function that returns HTML string
                            html_content_with_embedded_images = convert_to_html_with_libreoffice(
                                valid_src_path, 
                                tmp_output_dir,
                                embed_images=True # Ensure this is true
                            )
                            return HTMLResponse(content=html_content_with_embedded_images)
                        except (RuntimeError, FileNotFoundError) as conversion_error:
                            logger.error(f"HTML conversion failed for {valid_src_path.name}: {conversion_error}")
                            return HTMLResponse(
                                content=f"<html><body><h1>Preview Unavailable</h1><p>Could not generate a preview for {valid_src_path.name}. Error: {str(conversion_error)[:200]}...</p><p>You can try downloading the file directly.</p></body></html>",
                                status_code=500
                            )
            
            # --- PPT/PPTX to PDF ---
            # *** NEW ***
            elif file_extension in [".ppt", ".pptx"] and LIBREOFFICE_PATH:
                with tempfile.TemporaryDirectory(prefix="lo_pdf_convert_") as tmp_dir_str:
                    tmp_output_dir = Path(tmp_dir_str)
                    try:
                        # LibreOffice command for PDF conversion
                        cmd_pdf = [
                            LIBREOFFICE_PATH, "--headless", "--nolockcheck", "--nodefault",
                            "--norestore", "--invisible",
                            "--convert-to", "pdf:writer_pdf_Export", # Specific PDF export filter
                            "--outdir", str(tmp_output_dir), str(valid_src_path)
                        ]
                        process_pdf = subprocess.run(cmd_pdf, check=True, timeout=180, capture_output=True, text=True, errors='replace')
                        if process_pdf.stderr:
                            logger.info(f"LibreOffice PDF conversion STDERR for {valid_src_path.name}: {process_pdf.stderr[:1000]}")

                        pdf_filename = valid_src_path.stem + ".pdf"
                        converted_pdf_path = tmp_output_dir / pdf_filename
                        expected_pdf=tmp_output_dir/(valid_src_path.stem+".pdf")
                        
                        if not converted_pdf_path.is_file():
                            # Fallback: search for any .pdf file
                            pdf_files = list(tmp_output_dir.glob("*.pdf"))
                            if not pdf_files:
                                raise FileNotFoundError("LibreOffice PDF conversion failed: No PDF file found.")
                            converted_pdf_path = pdf_files[0]
                            expected_pdf=pdf_files[0]
                            logger.warning(f"Expected PDF file '{pdf_filename}' not found. Using first found: '{converted_pdf_path.name}'")
                        

                        pdf_bytes=expected_pdf.read_bytes()
                        logger.info(f"Successfully converted '{valid_src_path.name}' to PDF: '{converted_pdf_path}'")
                        # return FileResponse(
                        #     path=converted_pdf_path,
                        #     media_type="application/pdf",
                        #     filename=valid_src_path.stem + ".pdf", # Suggest a filename for the browser
                        #     content_disposition_type="inline" # Tell browser to display inline
                        # )

                    
                       # HIGHLIGHTED CHANGE SECTION: PPT/PPTX Conversion Error Handling
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) as conversion_error:
                        logger.error(f"PDF conversion failed for {valid_src_path.name}: {conversion_error}")
                        
                        # Instead of returning the original file, return an HTML error page
                        # You can make this HTML more elaborate if needed
                        error_html_content = f"""
                        <html>
                            <head><title>Preview Error</title></head>
                            <body style="font-family: sans-serif; padding: 20px;">
                                <h1>PDF Preview Unavailable</h1>
                                <p>Could not generate a PDF preview for the file: <strong>{valid_src_path.name}</strong>.</p>
                                <p>Reason: {str(conversion_error)[:250]}...</p>
                                <p>You can try downloading the original file instead.</p>
                                <!-- Optional: Add a direct link to download the original file -->
                                <!-- This would require knowing the base URL or constructing it carefully -->
                                <!-- <p><a href="/api/documents/{file_path}?action=download">Download Original File ({valid_src_path.name})</a></p> -->
                                <p><button onclick="window.close()">Close Tab</button></p>
                            </body>
                        </html>
                        """
                        return HTMLResponse(content=error_html_content, status_code=500)

                    return StreamingResponse(
                        content=io.BytesIO(pdf_bytes),
                        media_type="application/pdf",
                        headers={
                            "Content-Disposition": f'inline; filename="{valid_src_path.stem}.pdf"'
                        }
                    )
                    # END OF HIGHLIGHTED CHANGE SECTION
            
            elif file_extension == ".html":
                return FileResponse(path=valid_src_path, media_type="text/html", content_disposition_type="inline")

            elif mime_type and (mime_type.startswith("image/") or mime_type.startswith("text/") or mime_type == "application/pdf"):
                return FileResponse(path=valid_src_path, media_type=mime_type, content_disposition_type="inline")
            
            else: # Other types
                logger.info(f"Attempting inline view for '{valid_src_path.name}' (MIME: {mime_type}). Browser may download.")
                return FileResponse(path=valid_src_path, media_type=mime_type, content_disposition_type="inline", filename=valid_src_path.name)

    except PermissionError:
        logger.error(f"Permission denied accessing file: {valid_src_path}")
        raise HTTPException(status_code=403, detail="Server does not have permission to access this file.")
    except FileNotFoundError: # Should be caught by validation, but as safeguard
        raise HTTPException(status_code=404, detail="File not found during action processing.")
    except Exception as e:
        logger.exception(f"Unexpected error during '{action}' for '{valid_src_path.name}'")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred.")



@app.get("/download_all")
async def download_all(
    background_tasks: BackgroundTasks,
    parent_app_id: str = Query(..., description="Parent AppId to fetch documents for"),
    user: User = Depends(get_current_user)
):
    MAX_ATTACHMENT_RESULTS = 1000

    background_tasks.add_task(
        log_user_action,
        user.username,
        "download_all",
        {"parent_app_id": parent_app_id}
    )

    # Elasticsearch query: get attachments + main document
    query = {
        "query": {
            "bool": {
                "should": [
                    {"term": {"ParentProphecyId.keyword": parent_app_id}},  # attachments
                    {"term": {"ProphecyId.keyword": parent_app_id}},        # main doc
                ]
            }
        },
        "_source": ["SystemPath", "ProphecyId", "ParentProphecyId"],
        "size": MAX_ATTACHMENT_RESULTS + 1
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            url = f"{idx_url}/_search"
            response = await client.post(url, json=query)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Elasticsearch error: {response.status_code} - {response.text}"
                )

        data = response.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            raise HTTPException(status_code=404, detail="No documents found for given parent_app_id")

        # Extract file paths
        file_paths = []
        for hit in hits:
            src = hit.get("_source", {})
            if "SystemPath" in src:
                file_paths.append(src["SystemPath"])

        if not file_paths:
            raise HTTPException(status_code=404, detail="No valid SystemPath found for given parent_app_id")

        # Create in-memory ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in file_paths:
                try:
                    valid_src_path: Path = is_path_secure_and_valid(file_path, ALLOWED_DOCUMENT_ROOTS)
                    if valid_src_path.exists():
                        zipf.write(valid_src_path, arcname=valid_src_path.name)
                except Exception as e:
                    logger.error(f"Skipping file {file_path}: {e}")

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{parent_app_id}_documents.zip"'}
        )

    except Exception as e:
        logger.error(f"Error in /download_all for parent_app_id={parent_app_id}: {e}")
        raise HTTPException(status_code=500, detail="Server error during document download")




@app.post("/api/download_multiple")
async def download_multiple(
    background_tasks: BackgroundTasks,
    prophecy_ids: str = Form(...),
    user: User = Depends(get_current_user)
):
    """
    Accepts a JSON string list of Prophecy IDs, retrieves their system paths from
    Elasticsearch using httpx, packages the files into a ZIP archive, and streams it.
    """
    if not prophecy_ids:
        raise HTTPException(status_code=400, detail="No Prophecy IDs provided.")

    try:
        # 1. Parse the incoming list of IDs from the form data
        list_of_ids = json.loads(prophecy_ids)
        if not isinstance(list_of_ids, list) or not list_of_ids:
            raise HTTPException(status_code=400, detail="Invalid format: Prophecy IDs must be a non-empty array.")
        logger.info(f"Received request to download {len(list_of_ids)} documents.")

        background_tasks.add_task(
            log_user_action,
            user.username,
            "download_multiple",
            {"prophecy_ids": list_of_ids, "count": len(list_of_ids)}
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Could not parse Prophecy IDs. Must be a valid JSON array.")

    # 2. Build a single, efficient Elasticsearch query
    query = {
        "query": {
            "terms": {
                "ProphecyId.keyword": list_of_ids
            }
        },
        "_source": ["SystemPath"], # Also fetch FileName for clarity
        "size": len(list_of_ids) # Ensure we get all requested documents
    }

    # 3. Execute the query using httpx
    hits = []
    search_url = f"{ES_HOST}/{ES_INDEX}/_search"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(search_url, json=query)
            # Raise an exception for 4xx or 5xx status codes
            response.raise_for_status()
            
            response_data = response.json()
            hits = response_data.get('hits', {}).get('hits', [])

    except httpx.HTTPStatusError as e:
        # Error response from Elasticsearch (e.g., 404 Not Found, 400 Bad Request)
        logger.error(f"Elasticsearch returned an error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=502, detail=f"Error communicating with the search server: {e.response.text}")
    except httpx.RequestError as e:
        # Network-related error (e.g., DNS failure, connection refused)
        logger.error(f"Could not connect to Elasticsearch at {search_url}: {e}")
        raise HTTPException(status_code=503, detail="The search service is currently unavailable.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Elasticsearch query: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while fetching data.")

    if not hits:
        raise HTTPException(status_code=404, detail="None of the requested documents were found.")

    # 4. Validate paths and collect a list of valid files to be zipped
    valid_file_paths: List[Path] = []
    for hit in hits:
        source = hit.get('_source', {})
        system_path_str = source.get('SystemPath')

        if not system_path_str:
            logger.warning(f"Document with ID {hit['_id']} is missing a SystemPath.")
            continue

        try:
            doc_path = system_path_str

            # ★★★ SECURITY CHECK: Ensure the resolved path is inside the allowed directory ★★★
            # if not doc_path.is_relative_to(ALLOWED_DOWNLOAD_DIRECTORY):
            #     logger.error(f"SECURITY ALERT: Attempt to access a forbidden path: {doc_path}")
            #     continue # Silently skip this file

            
            valid_file_paths.append(doc_path)
            # else:
            #     logger.warning(f"Path for file  does not exist or is not a file: {doc_path}")

        except Exception as e:
            logger.error(f"Error processing path '{system_path_str}': {e}")
            continue

    if not valid_file_paths:
         raise HTTPException(status_code=404, detail="Found document records, but none of the corresponding files could be accessed on the server.")

    # 5. Create a ZIP archive in an in-memory buffer
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_path in valid_file_paths:
            # Use `arcname` to store the file with just its name, not the full server path
            try:
                valid_src_path: Path = is_path_secure_and_valid(file_path, ALLOWED_DOCUMENT_ROOTS)
                if valid_src_path.exists():
                    zip_file.write(valid_src_path, arcname=valid_src_path.name)
            except Exception as e:
                logger.error(f"Skipping file {file_path}: {e}")
            
    
    # Reset the buffer's cursor to the beginning for reading
    zip_buffer.seek(0)

    # 6. Stream the ZIP file as the final response
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    zip_filename = f"documents_{timestamp}.zip"

    logger.info(f"Streaming {len(valid_file_paths)} files in '{zip_filename}'")

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'}
    )



# ---------------------------------------
# ? NEW ENDPOINTS: Prophecy Tool
# ---------------------------------------

@app.get("/api/prophecy/{prophecy_id}")
async def get_prophecy_by_id(
    prophecy_id: str
):
    query = {
        "query": {
            "term": {
                "ProphecyId.keyword": prophecy_id
            }
        },
        "size": 1
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"{idx_url}/_search"
            response = await client.post(url, json=query)

            if response.status_code != 200:
                 raise HTTPException(status_code=500, detail=f"ES error: {response.text}")

            data = response.json()
            hits = data.get("hits", {}).get("hits", [])

            if not hits:
                return JSONResponse(content={"found": False}, status_code=404)

            hit = hits[0]
            source = hit["_source"]
            
            return {
                "found": True,
                "prophecy_id": source.get("ProphecyId"),
                "system_path": source.get("SystemPath"),
                "access_branches": source.get("Access_Branches", ""), # Make sure this matches ES field
                "es_id": hit["_id"],
                "index": hit["_index"],
                "source": source # return full source for debugging/display
            }

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail="Elasticsearch unavailable")


class AccessBranchesUpdate(BaseModel):
    prophecy_id: str
    access_branches: str

@app.post("/api/update-access-branches")
async def update_access_branches(
    update_data: AccessBranchesUpdate
):
    # 1. First find the document to get its ID (if not provided, but we use strict query for safety)
    # Actually we can use update_by_query or find then update. 
    # Let's find first to be safe and get the _id.
    
    # ... Or better, since we know ProphecyId is the key we want to match:

    search_query = {
        "query": {
            "term": {
                "ProphecyId.keyword": update_data.prophecy_id
            }
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # A. Search
            url = f"{idx_url}/_search"
            resp = await client.post(url, json=search_query)
            if resp.status_code != 200:
                 raise HTTPException(status_code=500, detail="ES Search failed")
            
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            
            if not hits:
                raise HTTPException(status_code=404, detail="ProphecyId not found")
            
            doc_id = hits[0]["_id"]
            
            # B. Update using _update API
            update_url = f"{idx_url}/_update/{doc_id}"
            update_body = {
                "doc": {
                    "Access_Branches": update_data.access_branches
                }
            }
            
            update_resp = await client.post(update_url, json=update_body)
            
            if update_resp.status_code not in [200, 201]:
                raise HTTPException(status_code=500, detail=f"Update failed: {update_resp.text}")
                
            return {"message": "Updated successfully", "new_value": update_data.access_branches}

    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Elasticsearch unavailable")

# Optional: A generic exception handler for cleaner error responses
# @app.exception_handler(Exception)
# async def generic_exception_handler(request: Request, exc: Exception):
#     logger.error(f"An unhandled exception occurred for request {request.url}: {exc}", exc_info=True)
#     return JSONResponse(
#         status_code=500,
#         content={"detail": "An unexpected internal server error occurred."},
#     )

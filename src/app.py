import gradio as gr
import json
import requests
import asyncio
import aiohttp
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import asdict
from huggingface_hub import HfApi, SpaceInfo
from sentence_transformers import SentenceTransformer
import torch
import re
import logging
from urllib.parse import urlparse
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPSpaceFinder:
    def __init__(self):
        """Initialize the MCP Space Finder with necessary models and API."""
        self.api = HfApi()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.spaces_cache = None
        self.embeddings_cache = None
        self.verified_mcp_cache = {}  # Cache for MCP verification results
        self.last_update = None

    def normalize_schema(self, schema_data: Any) -> Dict:
        """
        Normalize schema data to ensure it's always a dictionary.
        Some MCP servers return a list of tools directly, others return a dict with 'tools' key.
        """
        if schema_data is None:
            return {"tools": []}
        
        if isinstance(schema_data, list):
            # If it's a list, assume it's a list of tools
            return {"tools": schema_data}
        
        if isinstance(schema_data, dict):
            # If it's already a dict, return as-is
            return schema_data
        
        # If it's something else, return empty structure
        logger.warning(f"Unexpected schema data type: {type(schema_data)}")
        return {"tools": []}

    async def verify_mcp_server(self, space_id: str, timeout: int = 10) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Verify if a space actually has a working MCP server by checking the schema endpoint.

        Args:
            space_id: The space ID (e.g., 'author/space-name')
            timeout: Request timeout in seconds

        Returns:
            Tuple of (is_working, mcp_url, schema_info)
        """
        # Check cache first
        if space_id in self.verified_mcp_cache:
            cached_result = self.verified_mcp_cache[space_id]
            # Cache for 1 hour to avoid too many requests
            if time.time() - cached_result.get('timestamp', 0) < 3600:
                return cached_result.get('is_working', False), cached_result.get('mcp_url'), cached_result.get('schema')

        # Construct the MCP server URL
        mcp_url = f"https://{space_id.replace('/', '-')}.hf.space/gradio_api/mcp/sse"
        schema_url = f"https://{space_id.replace('/', '-')}.hf.space/gradio_api/mcp/schema"

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                # Try to get the MCP schema
                async with session.get(schema_url) as response:
                    if response.status == 200:
                        try:
                            raw_schema_data = await response.json()
                            # Normalize the schema data
                            schema_data = self.normalize_schema(raw_schema_data)
                            # Cache the successful result
                            self.verified_mcp_cache[space_id] = {
                                'is_working': True,
                                'mcp_url': mcp_url,
                                'schema': schema_data,
                                'timestamp': time.time()
                            }
                            return True, mcp_url, schema_data
                        except Exception as e:
                            logger.warning(f"Failed to parse schema for {space_id}: {e}")

                # If schema doesn't work, try the SSE endpoint
                async with session.get(mcp_url) as response:
                    if response.status == 200:
                        # Cache as working but without schema
                        self.verified_mcp_cache[space_id] = {
                            'is_working': True,
                            'mcp_url': mcp_url,
                            'schema': None,
                            'timestamp': time.time()
                        }
                        return True, mcp_url, None

        except Exception as e:
            logger.debug(f"MCP verification failed for {space_id}: {e}")

        # Cache the failed result
        self.verified_mcp_cache[space_id] = {
            'is_working': False,
            'mcp_url': None,
            'schema': None,
            'timestamp': time.time()
        }
        return False, None, None

    def get_mcp_spaces_from_hub(self, force_refresh: bool = False) -> List[SpaceInfo]:
        """
        Fetch MCP-capable spaces using HuggingFace's official MCP filter.

        Args:
            force_refresh: Force refresh of cached data

        Returns:
            List of MCP-capable SpaceInfo objects
        """
        now = datetime.now(timezone.utc)
        if (self.spaces_cache is None or force_refresh or
            (self.last_update and (now - self.last_update).total_seconds() > 1800)):  # 30 min cache

            logger.info("Fetching MCP spaces from HuggingFace Hub using official filter...")

            try:
                # Use the official MCP filter - this is much more reliable
                mcp_spaces = list(self.api.list_spaces(
                    full=True,
                    limit=500,
                    filter="mcp-server"  # Official HF MCP filter
                ))

                # Also get some popular Gradio spaces that might have MCP
                gradio_spaces = list(self.api.list_spaces(
                    full=True,
                    limit=200,
                    sort="likes",
                    filter="gradio"
                ))

                # Combine and deduplicate
                all_spaces = {}
                for space in mcp_spaces + gradio_spaces:
                    if hasattr(space, 'sdk') and space.sdk == 'gradio':
                        all_spaces[space.id] = space

                self.spaces_cache = list(all_spaces.values())
                self.last_update = now

                logger.info(f"Found {len(self.spaces_cache)} potential MCP spaces")

                # Generate embeddings for semantic search
                if self.spaces_cache:
                    space_descriptions = []
                    for space in self.spaces_cache:
                        # Create rich description for embedding
                        desc_parts = [
                            space.id,
                            getattr(space, 'title', ''),
                            ' '.join(space.tags or []),
                        ]
                        # Add card data if available
                        if space.card_data:
                            try:
                                card_dict = asdict(space.card_data)
                                desc_parts.extend([
                                    card_dict.get('title', ''),
                                    ' '.join(card_dict.get('tags', []) or [])
                                ])
                            except Exception as e:
                                logger.warning(f"Failed to process card data for {space.id}: {e}")

                        desc = ' '.join(filter(None, desc_parts))
                        space_descriptions.append(desc)

                    self.embeddings_cache = self.model.encode(space_descriptions, convert_to_tensor=True)

            except Exception as e:
                logger.error(f"Failed to fetch spaces: {e}")
                self.spaces_cache = []

        return self.spaces_cache or []

    def format_space_for_agent(self, space: SpaceInfo, mcp_url: str = None, schema: Dict = None, is_verified: bool = False) -> Dict[str, Any]:
        """
        Format space information for code agents with comprehensive metadata.

        Args:
            space: SpaceInfo object to format
            mcp_url: Verified MCP server URL
            schema: MCP schema information if available
            is_verified: Whether the MCP server was verified as working

        Returns:
            Dictionary with structured space information
        """
        # Calculate URLs
        if not mcp_url:
            mcp_url = f"https://{space.id.replace('/', '-')}.hf.space/gradio_api/mcp/sse"

        web_url = f"https://huggingface.co/spaces/{space.id}"
        direct_url = f"https://{space.id.replace('/', '-')}.hf.space"

        # Helper function to make datetime timezone-aware for calculations
        def make_aware(dt):
            if dt is None:
                return None
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        # Get timezone-aware current time
        now_aware = datetime.now(timezone.utc)
        created_aware = make_aware(space.created_at)
        modified_aware = make_aware(space.last_modified)

        # Ensure schema is a dictionary and handle both dict and None cases
        normalized_schema = self.normalize_schema(schema) if schema else {"tools": []}

        # Extract comprehensive metadata
        space_data = {
            # Basic Information
            "space_id": space.id,
            "author": getattr(space, 'author', 'unknown'),
            "title": getattr(space, 'title', space.id.split('/')[-1]),

            # MCP Server Information
            "mcp_server_url": mcp_url,
            "mcp_verified": is_verified,
            "mcp_schema_available": schema is not None,
            "mcp_tools_count": len(normalized_schema.get('tools', [])),

            # URLs
            "web_interface_url": web_url,
            "direct_app_url": direct_url,
            "huggingface_url": web_url,

            # Technical Details
            "sdk": getattr(space, 'sdk', 'gradio'),
            "python_version": None,
            "sdk_version": None,

            # Popularity & Stats
            "likes": getattr(space, 'likes', 0),
            "trending_score": getattr(space, 'trending_score', 0),
            "downloads": getattr(space, 'downloads', 0),

            # Timestamps
            "created_at": space.created_at.isoformat() if space.created_at else None,
            "last_modified": space.last_modified.isoformat() if space.last_modified else None,
            "age_days": (now_aware - created_aware).days if created_aware else None,
            "last_update_days": (now_aware - modified_aware).days if modified_aware else None,

            # Access & Status
            "private": getattr(space, 'private', False),
            "disabled": getattr(space, 'disabled', False),
            "gated": getattr(space, 'gated', False),

            # Content & Relationships
            "tags": space.tags or [],
            "models": getattr(space, 'models', []),
            "datasets": getattr(space, 'datasets', []),

            # Additional Metadata
            "host": getattr(space, 'host', None),
            "subdomain": getattr(space, 'subdomain', None),
        }

        # Add card data if available
        if space.card_data:
            try:
                card_dict = asdict(space.card_data)
                # Extract useful card data
                space_data.update({
                    "python_version": card_dict.get('python_version'),
                    "sdk_version": card_dict.get('sdk_version'),
                    "app_file": card_dict.get('app_file'),
                    "license": card_dict.get('license'),
                    "duplicated_from": card_dict.get('duplicated_from'),
                })

                # Add all non-null card data
                space_data["card_data"] = {k: v for k, v in card_dict.items() if v is not None}
            except Exception as e:
                logger.warning(f"Failed to process card data for space {space.id}: {e}")
                space_data["card_data"] = {}

        # Add MCP schema information if available
        if schema:
            try:
                tools = normalized_schema.get('tools', [])
                # Ensure tools is a list and handle cases where individual tools might not be dicts
                safe_tools = []
                tool_names = []
                capabilities = []
                
                for tool in tools:
                    if isinstance(tool, dict):
                        safe_tools.append(tool)
                        tool_names.append(tool.get('name', 'unnamed'))
                        capabilities.append(tool.get('description', 'no description'))
                    else:
                        logger.warning(f"Unexpected tool format in schema for {space.id}: {tool}")
                
                space_data["mcp_schema"] = {
                    "tools": safe_tools,
                    "tool_names": tool_names,
                    "capabilities": capabilities,
                }
            except Exception as e:
                logger.warning(f"Failed to process MCP schema for space {space.id}: {e}")
                space_data["mcp_schema"] = {
                    "tools": [],
                    "tool_names": [],
                    "capabilities": [],
                }

        return space_data

    async def search_mcp_spaces(
        self,
        query: str = "",
        max_results: int = 10,
        min_likes: int = 0,
        author_filter: str = "",
        tag_filter: str = "",
        sort_by: str = "relevance",
        created_after: str = "",
        include_private: bool = False,
        verify_mcp: bool = True,
        min_age_days: int = 0,
        max_age_days: int = 365
    ) -> str:
        """
        Search and filter MCP-capable spaces with verification and comprehensive filtering.

        Args:
            query: Search query for semantic matching
            max_results: Maximum number of results to return
            min_likes: Minimum number of likes
            author_filter: Filter by author (partial match)
            tag_filter: Filter by tag (comma-separated)
            sort_by: Sort results by 'relevance', 'likes', 'created', 'modified', 'verified'
            created_after: Filter spaces created after this date (YYYY-MM-DD)
            include_private: Include private spaces
            verify_mcp: Actually verify MCP servers are working (slower but more accurate)
            min_age_days: Minimum age in days
            max_age_days: Maximum age in days

        Returns:
            JSON string with search results formatted for code agents
        """
        try:
            spaces = self.get_mcp_spaces_from_hub()

            if not spaces:
                return json.dumps({
                    "status": "error",
                    "message": "No MCP-capable spaces found",
                    "results": []
                })

            logger.info(f"Starting search with {len(spaces)} spaces")

            # Apply filters
            filtered_spaces = []
            for space in spaces:
                # Skip private spaces unless requested
                if not include_private and getattr(space, 'private', False):
                    continue

                # Skip disabled spaces
                if getattr(space, 'disabled', False):
                    continue

                # Filter by minimum likes
                if getattr(space, 'likes', 0) < min_likes:
                    continue

                # Filter by author
                if author_filter and author_filter.lower() not in getattr(space, 'author', '').lower():
                    continue

                # Filter by tags
                if tag_filter:
                    required_tags = [t.strip().lower() for t in tag_filter.split(',')]
                    space_tags = [t.lower() for t in (space.tags or [])]
                    if not any(req_tag in ' '.join(space_tags) for req_tag in required_tags):
                        continue

                # Filter by creation date
                if created_after:
                    try:
                        cutoff_date = datetime.fromisoformat(created_after)
                        # Make cutoff_date timezone-aware if it isn't already
                        if cutoff_date.tzinfo is None:
                            cutoff_date = cutoff_date.replace(tzinfo=timezone.utc)
                        # Make space creation date timezone-aware for comparison
                        space_created = space.created_at
                        if space_created and space_created.tzinfo is None:
                            space_created = space_created.replace(tzinfo=timezone.utc)
                        if space_created and space_created < cutoff_date:
                            continue
                    except ValueError:
                        pass  # Invalid date format, skip filter

                # Filter by age
                if space.created_at:
                    space_created = space.created_at
                    if space_created.tzinfo is None:
                        space_created = space_created.replace(tzinfo=timezone.utc)
                    age_days = (datetime.now(timezone.utc) - space_created).days
                    if age_days < min_age_days or age_days > max_age_days:
                        continue

                filtered_spaces.append(space)

            logger.info(f"After filtering: {len(filtered_spaces)} spaces")

            # Verify MCP servers if requested (in batches to avoid overwhelming servers)
            verified_results = []
            if verify_mcp and filtered_spaces:
                logger.info("Verifying MCP servers...")
                batch_size = 5  # Process in small batches
                for i in range(0, len(filtered_spaces), batch_size):
                    batch = filtered_spaces[i:i+batch_size]
                    tasks = [self.verify_mcp_server(space.id) for space in batch]
                    verification_results = await asyncio.gather(*tasks, return_exceptions=True)

                    for space, result in zip(batch, verification_results):
                        if isinstance(result, Exception):
                            logger.warning(f"Verification error for {space.id}: {result}")
                            verified_results.append((space, False, None, None))
                        else:
                            is_working, mcp_url, schema = result
                            verified_results.append((space, is_working, mcp_url, schema))

                    # Small delay between batches to be respectful
                    if i + batch_size < len(filtered_spaces):
                        await asyncio.sleep(1)
            else:
                # No verification, just mark all as unverified
                verified_results = [(space, None, None, None) for space in filtered_spaces]

            # Semantic search and ranking
            results_with_scores = []

            if query and self.embeddings_cache is not None and self.spaces_cache:
                # Semantic search
                query_embedding = self.model.encode(query, convert_to_tensor=True)

                # Find indices of filtered spaces in original list
                space_to_index = {space.id: i for i, space in enumerate(self.spaces_cache)}
                filtered_indices = [space_to_index[space.id] for space, _, _, _ in verified_results if space.id in space_to_index]

                # Calculate similarities for filtered spaces
                if filtered_indices:
                    filtered_embeddings = self.embeddings_cache[filtered_indices]
                    cosine_scores = torch.nn.functional.cosine_similarity(
                        query_embedding.unsqueeze(0), filtered_embeddings
                    )

                    for (space, is_verified, mcp_url, schema), score in zip(verified_results, cosine_scores):
                        # Boost score for verified MCP servers
                        adjusted_score = float(score)
                        if is_verified:
                            adjusted_score += 0.2  # Boost verified servers
                        results_with_scores.append((space, is_verified, mcp_url, schema, adjusted_score))
            else:
                # No semantic search, use like-based scoring
                for space, is_verified, mcp_url, schema in verified_results:
                    # Score based on likes and verification
                    score = getattr(space, 'likes', 0) / 100.0
                    if is_verified:
                        score += 0.5  # Significant boost for verified servers
                    results_with_scores.append((space, is_verified, mcp_url, schema, score))

            # Sort results
            if sort_by == "relevance":
                results_with_scores.sort(key=lambda x: x[4], reverse=True)
            elif sort_by == "likes":
                results_with_scores.sort(key=lambda x: getattr(x[0], 'likes', 0), reverse=True)
            elif sort_by == "created":
                results_with_scores.sort(
                    key=lambda x: x[0].created_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True
                )
            elif sort_by == "modified":
                results_with_scores.sort(
                    key=lambda x: x[0].last_modified or datetime.min.replace(tzinfo=timezone.utc), reverse=True
                )
            elif sort_by == "verified":
                results_with_scores.sort(key=lambda x: (x[1] is True, x[4]), reverse=True)

            # Format results for agents
            formatted_results = []
            verified_count = 0
            for space, is_verified, mcp_url, schema, score in results_with_scores[:max_results]:
                try:
                    space_info = self.format_space_for_agent(space, mcp_url, schema, is_verified)
                    space_info["relevance_score"] = round(score, 4)
                    formatted_results.append(space_info)
                    if is_verified:
                        verified_count += 1
                except Exception as e:
                    logger.warning(f"Failed to format space {space.id}: {e}")
                    continue

            return json.dumps({
                "status": "success",
                "query": query,
                "filters_applied": {
                    "min_likes": min_likes,
                    "author_filter": author_filter,
                    "tag_filter": tag_filter,
                    "created_after": created_after,
                    "include_private": include_private,
                    "verify_mcp": verify_mcp,
                    "min_age_days": min_age_days,
                    "max_age_days": max_age_days,
                },
                "stats": {
                    "total_spaces_searched": len(spaces),
                    "spaces_after_filtering": len(filtered_spaces),
                    "results_returned": len(formatted_results),
                    "verified_mcp_servers": verified_count,
                    "verification_enabled": verify_mcp,
                },
                "results": formatted_results
            }, indent=2)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return json.dumps({
                "status": "error",
                "message": f"Search failed: {str(e)}",
                "results": []
            })


# Initialize the finder
finder = MCPSpaceFinder()


def search_mcp_spaces(
    query: str,
    max_results: int,
    min_likes: int,
    author_filter: str,
    tag_filter: str,
    sort_by: str,
    created_after: str,
    include_private: bool,
    verify_mcp: bool,
    min_age_days: int,
    max_age_days: int
) -> str:
    """
    Search for MCP-capable spaces on HuggingFace.
    
    Args:
        query: Search query for semantic matching
        max_results: Maximum number of results to return
        min_likes: Minimum number of likes required
        author_filter: Filter by author (partial match)
        tag_filter: Filter by tags (comma-separated)
        sort_by: Sort by relevance, likes, created, modified, or verified
        created_after: Filter spaces created after this date (YYYY-MM-DD)
        include_private: Include private spaces in results
        verify_mcp: Actually verify MCP endpoints work
        min_age_days: Minimum age in days
        max_age_days: Maximum age in days
        
    Returns:
        JSON string with search results
    """
    # Run the async function
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        finder.search_mcp_spaces(
            query=query,
            max_results=max_results,
            min_likes=min_likes,
            author_filter=author_filter,
            tag_filter=tag_filter,
            sort_by=sort_by,
            created_after=created_after,
            include_private=include_private,
            verify_mcp=verify_mcp,
            min_age_days=min_age_days,
            max_age_days=max_age_days
        )
    )


# Create the Gradio interface
with gr.Blocks(title="🚀 Enhanced HuggingFace MCP Space Finder") as demo:
    gr.Markdown("""
    # 🚀 Enhanced HuggingFace MCP Space Finder
    
    **The most advanced tool for finding working MCP servers on HuggingFace Spaces!**

    ### 🎯 **Key Features:**
    - **✅ Real Verification**: Actually tests MCP endpoints to ensure they work (HTTP 200 status)
    - **🎯 Official MCP Filter**: Uses HuggingFace's native `mcp-server` filter for accuracy
    - **🔍 Semantic Search**: AI-powered search using sentence transformers
    - **📊 Rich Metadata**: Complete space information including age, popularity, and technical details
    - **🤖 Agent-Ready**: Returns structured JSON that code agents can immediately use
    - **⚡ Smart Caching**: Caches verification results to avoid overwhelming servers

    ### 📋 **Perfect for Code Agents:**
    - **Direct MCP URLs**: Ready-to-use server endpoints
    - **Verification Status**: Know which servers actually work
    - **Complete Metadata**: Creation dates, update times, popularity metrics
    - **Tool Information**: Available MCP tools and capabilities
    - **Quality Scoring**: Relevance and reliability scores

    ### 🛠 **How It Works:**
    1. Fetches spaces using HF's official MCP filter
    2. Applies your custom filtering criteria
    3. Verifies MCP servers are actually responding (optional)
    4. Returns ranked, verified results with complete metadata

    **⚠️ Tip**: Enable "Verify MCP Servers" for the most accurate results (takes longer but ensures working endpoints)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="🔍 Search Query",
                placeholder="e.g., 'sentiment analysis', 'image generation', 'text processing'",
                info="Semantic search across space names, titles, and tags",
                value=""
            )
            
            with gr.Row():
                max_results = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=15,
                    step=1,
                    label="📊 Max Results",
                    info="Maximum number of spaces to return"
                )
                min_likes = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=5,
                    step=1,
                    label="👍 Minimum Likes",
                    info="Filter spaces with at least this many likes"
                )
            
            with gr.Row():
                author_filter = gr.Textbox(
                    label="👤 Author Filter",
                    placeholder="e.g., 'huggingface', 'microsoft'",
                    info="Filter by author name (partial match)",
                    value=""
                )
                tag_filter = gr.Textbox(
                    label="🏷️ Tag Filter",
                    placeholder="e.g., 'nlp,computer-vision,mcp-server'",
                    info="Filter by tags (comma-separated)",
                    value=""
                )
            
            with gr.Row():
                sort_by = gr.Dropdown(
                    choices=["relevance", "likes", "created", "modified", "verified"],
                    value="verified",
                    label="📈 Sort By",
                    info="How to sort the results (verified = working MCP servers first)"
                )
                created_after = gr.Textbox(
                    label="📅 Created After",
                    placeholder="2024-01-01",
                    info="Show only spaces created after this date (YYYY-MM-DD)",
                    value=""
                )
            
            with gr.Row():
                include_private = gr.Checkbox(
                    label="🔒 Include Private Spaces",
                    value=False,
                    info="Include private spaces in results"
                )
                verify_mcp = gr.Checkbox(
                    label="✅ Verify MCP Servers",
                    value=True,
                    info="Actually test MCP endpoints (slower but more accurate)"
                )
            
            with gr.Row():
                min_age_days = gr.Slider(
                    minimum=0,
                    maximum=30,
                    value=0,
                    step=1,
                    label="⏰ Min Age (Days)",
                    info="Minimum age in days (0 = any age)"
                )
                max_age_days = gr.Slider(
                    minimum=1,
                    maximum=1000,
                    value=365,
                    step=1,
                    label="📆 Max Age (Days)",
                    info="Maximum age in days"
                )
            
            search_btn = gr.Button("🔍 Search MCP Spaces", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            result_output = gr.Code(
                language="json",
                label="🤖 MCP Server Results",
               # info="JSON format optimized for code agents with verified MCP endpoints",
                lines=25
            )
    
    # Examples section
    gr.Markdown("### 📚 **Example Searches:**")
    examples = gr.Examples(
        examples=[
            ["sentiment analysis", 5, 5, "", "nlp", "verified", "", False, True, 0, 365],
            ["image generation", 3, 10, "", "computer-vision,art", "likes", "2024-01-01", False, True, 0, 180],
            ["chatbot", 10, 0, "huggingface", "mcp-server", "modified", "", False, True, 0, 365],
            ["", 20, 5, "", "mcp-server", "verified", "", False, True, 0, 365],  # All verified MCP servers
        ],
        inputs=[query_input, max_results, min_likes, author_filter, tag_filter, sort_by, created_after, include_private, verify_mcp, min_age_days, max_age_days],
        outputs=result_output,
        fn=search_mcp_spaces,
        cache_examples=False,
    )
    
    # Event handler
    search_btn.click(
        search_mcp_spaces,
        inputs=[query_input, max_results, min_likes, author_filter, tag_filter, sort_by, created_after, include_private, verify_mcp, min_age_days, max_age_days],
        outputs=result_output
    )
    
    # Additional information
    gr.Markdown("""
    ---
    ### 🔧 **For Developers:**
    
    **MCP URL Format:** `https://SPACE-ID.hf.space/gradio_api/mcp/sse`
    
    **Claude Desktop Config Example:**
    ```json
    {
      "mcpServers": {
        "gradio": {
          "command": "npx",
          "args": [
            "mcp-remote",
            "https://your-space.hf.space/gradio_api/mcp/sse"
          ]
        }
      }
    }
    ```
    
    **Direct URL Access:** Some clients support direct SSE connections:
    ```json
    {
      "mcpServers": {
        "gradio": {
          "url": "https://your-space.hf.space/gradio_api/mcp/sse"
        }
      }
    }
    ```
    
    ### 🐛 **Troubleshooting:**
    - If MCP verification fails, try disabling it for faster results
    - Some spaces may be temporarily unavailable during builds
    - Use `mcp-remote` for better compatibility with Claude Desktop
    - Check the space's status page if connection issues persist
    """)

# Launch with MCP server support and better error handling
if __name__ == "__main__":
    try:
        # Launch with proper error handling for ASGI issues
        demo.launch(
            mcp_server=True,
            debug=True,  # Set to False to reduce ASGI issues
            share=True,  # Set to True if you want a public link
           # server_name="0.0.0.0",
           # server_port=7860,
           # show_error=False,
           # enable_queue=True,  # Enable queue for better stability
           # max_size=10,  # Limit queue size
           # enable_queue=True,
           # max_size=20,
           # show_error=False,  # Reduce error display that can cause ASGI issues
           # prevent_thread_lock=False

        )
    except Exception as e:
        logger.error(f"Failed to launch with MCP server: {e}")
        logger.info("Falling back to regular Gradio app without MCP server...")
        # Fallback: launch without MCP server if there are issues
        demo.launch(
            debug=True,
            share=True,
         #   server_name="0.0.0.0", 
         #   server_port=7860,
            show_error=True
        )
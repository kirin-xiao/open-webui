import logging

import requests
from open_webui.retrieval.web.main import SearchResult

log = logging.getLogger(__name__)

EXA_API_BASE = 'https://api.exa.ai'


def search_exa(
    api_key: str,
    query: str,
    count: int,
    filter_list: list[str] | None = None,
    max_content_length: int | None = None,
) -> list[SearchResult]:
    """Search using Exa Search API and return the results as a list of SearchResult objects.

    Args:
        api_key (str): A Exa Search API key
        query (str): The query to search for
        count (int): Number of results to return
        filter_list (list[str] | None): List of domains to filter results by
        max_content_length (int | None): Safety cap on the highlights snippet; None leaves it uncapped.
    """
    log.info('Searching with Exa for query: %s', query)

    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

    payload = {
        'query': query,
        'numResults': count or 5,
        'includeDomains': filter_list,
        # Use Exa's token-efficient highlights instead of the full page text,
        # which can be hundreds of KB per result when the web loader is bypassed.
        'contents': {'highlights': True},
        'type': 'auto',  # Use the auto search type (keyword or neural)
    }

    try:
        response = requests.post(f'{EXA_API_BASE}/search', headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        def get_snippet(result: dict) -> str:
            highlights = result.get('highlights') or []
            if isinstance(highlights, list):
                snippet = '\n'.join(str(h) for h in highlights).strip()
            else:
                snippet = str(highlights).strip()
            if not snippet:
                snippet = (result.get('text') or '').strip()
            if max_content_length is not None:
                snippet = snippet[:max_content_length]
            return snippet

        results = data['results']
        log.info('Found %s results', len(results))
        return [
            SearchResult(
                link=result['url'],
                title=result['title'],
                snippet=get_snippet(result),
            )
            for result in results
        ]
    except Exception as e:
        log.error(f'Error searching Exa: {e}')
        return []

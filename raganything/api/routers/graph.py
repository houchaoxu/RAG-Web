"""Knowledge graph endpoints — entities and relationships."""

import logging
from collections import Counter
from fastapi import APIRouter, HTTPException, Query
from raganything.api.services.rag_service import rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])


async def _get_graph():
    rag = rag_service.rag
    if rag.lightrag is None:
        await rag._ensure_lightrag_initialized()
    if rag.lightrag is None:
        raise HTTPException(status_code=503, detail="RAG not initialized")
    return rag.lightrag.chunk_entity_relation_graph


@router.get("/stats")
async def graph_stats():
    """Return entity/relation counts and entity type breakdown."""
    try:
        graph = await _get_graph()
        nodes = await graph.get_all_nodes()
        edges = await graph.get_all_edges()
        type_counts = Counter(n.get("entity_type", "UNKNOWN") for n in nodes)
        return {
            "entity_count": len(nodes),
            "relation_count": len(edges),
            "entity_types": dict(type_counts),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities")
async def list_entities(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search: str = Query("", description="Filter by entity name (case-insensitive substring)"),
):
    """List entities with optional search and pagination."""
    try:
        graph = await _get_graph()
        nodes = await graph.get_all_nodes()

        if search:
            q = search.lower()
            nodes = [n for n in nodes if q in (n.get("entity_id") or "").lower()]

        total = len(nodes)
        start = (page - 1) * page_size
        page_nodes = nodes[start : start + page_size]

        entities = [
            {
                "entity_id": n.get("entity_id", n.get("id", "")),
                "entity_type": n.get("entity_type", "UNKNOWN"),
                "description": n.get("description", ""),
                "source_id": n.get("source_id", ""),
                "file_path": n.get("file_path", ""),
                "created_at": n.get("created_at"),
            }
            for n in page_nodes
        ]
        return {"entities": entities, "total": total, "page": page, "page_size": page_size}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/relations")
async def list_relations(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search: str = Query("", description="Filter by source/target/description (case-insensitive substring)"),
):
    """List relationships with optional search and pagination."""
    try:
        graph = await _get_graph()
        edges = await graph.get_all_edges()

        if search:
            q = search.lower()
            edges = [
                e for e in edges
                if q in (e.get("source") or "").lower()
                or q in (e.get("target") or "").lower()
                or q in (e.get("description") or "").lower()
            ]

        total = len(edges)
        start = (page - 1) * page_size
        page_edges = edges[start : start + page_size]

        relations = [
            {
                "source": e.get("source", ""),
                "target": e.get("target", ""),
                "description": e.get("description", ""),
                "keywords": e.get("keywords", ""),
                "weight": e.get("weight", 1.0),
                "source_id": e.get("source_id", ""),
                "file_path": e.get("file_path", ""),
                "created_at": e.get("created_at"),
            }
            for e in page_edges
        ]
        return {"relations": relations, "total": total, "page": page, "page_size": page_size}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Color palette for entity types
_TYPE_COLORS = {
    "person": "#f87171",
    "organization": "#60a5fa",
    "method": "#34d399",
    "concept": "#a78bfa",
    "artifact": "#fbbf24",
    "data": "#38bdf8",
    "content": "#fb923c",
    "location": "#4ade80",
    "event": "#e879f9",
    "UNKNOWN": "#9ca3af",
}


@router.get("/network")
async def graph_network(
    max_nodes: int = Query(500, ge=50, le=2000, description="Max nodes to return"),
):
    """Return full graph in vis-network format for visualization."""
    try:
        graph = await _get_graph()
        all_nodes = await graph.get_all_nodes()
        all_edges = await graph.get_all_edges()

        # Limit nodes
        truncated = len(all_nodes) > max_nodes
        nodes_subset = all_nodes[:max_nodes]
        node_ids = {n.get("entity_id", n.get("id", "")) for n in nodes_subset}

        # Build degree map for sizing
        degree = {}
        for e in all_edges:
            src, tgt = e.get("source", ""), e.get("target", "")
            if src in node_ids:
                degree[src] = degree.get(src, 0) + 1
            if tgt in node_ids:
                degree[tgt] = degree.get(tgt, 0) + 1

        vis_nodes = []
        for n in nodes_subset:
            nid = n.get("entity_id", n.get("id", ""))
            etype = n.get("entity_type", "UNKNOWN")
            deg = degree.get(nid, 0)
            vis_nodes.append({
                "id": nid,
                "label": nid,
                "group": etype,
                "title": f"{nid}\nType: {etype}\nConnections: {deg}",
                "value": deg,
                "color": _TYPE_COLORS.get(etype, _TYPE_COLORS["UNKNOWN"]),
            })

        # Only include edges where both endpoints exist in our node set
        vis_edges = []
        for e in all_edges:
            src, tgt = e.get("source", ""), e.get("target", "")
            if src in node_ids and tgt in node_ids:
                desc = (e.get("description") or "").split("<SEP>")[0][:80]
                vis_edges.append({
                    "from": src,
                    "to": tgt,
                    "title": desc,
                    "value": e.get("weight", 1),
                })

        return {
            "nodes": vis_nodes,
            "edges": vis_edges,
            "truncated": truncated,
            "total_nodes": len(all_nodes),
            "total_edges": len(all_edges),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

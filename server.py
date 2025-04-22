"""Server routes for the RealtimeNodes extension"""
import json
from aiohttp import web
from server import PromptServer

# Dictionary to store workflow-specific keys
# Format: { workflow_id: { key: value, ... }, ... }
workflow_states = {}

@PromptServer.instance.routes.get('/realtimenodes/state/{workflow_id}')
async def get_workflow_state(request):
    """Get the current state for a specific workflow"""
    workflow_id = request.match_info['workflow_id']
    state = workflow_states.get(workflow_id, {})
    return web.json_response(state)

@PromptServer.instance.routes.get('/realtimenodes/state/{workflow_id}/{key}')
async def get_state_value(request):
    """Get a specific state value for a workflow"""
    workflow_id = request.match_info['workflow_id']
    key = request.match_info['key']
    
    workflow_state = workflow_states.get(workflow_id, {})
    value = workflow_state.get(key, None)
    
    return web.json_response({"value": value})

@PromptServer.instance.routes.post('/realtimenodes/state/{workflow_id}/{key}')
async def set_state_value(request):
    """Set a specific state value for a workflow"""
    workflow_id = request.match_info['workflow_id']
    key = request.match_info['key']
    
    # Get request body
    body = await request.json()
    value = body.get("value")
    
    # Ensure workflow exists in states
    if workflow_id not in workflow_states:
        workflow_states[workflow_id] = {}
    
    # Store the value
    workflow_states[workflow_id][key] = value
    
    return web.json_response({"success": True})

@PromptServer.instance.routes.delete('/realtimenodes/state/{workflow_id}/{key}')
async def delete_state_value(request):
    """Delete a specific state value for a workflow"""
    workflow_id = request.match_info['workflow_id']
    key = request.match_info['key']
    
    if workflow_id in workflow_states and key in workflow_states[workflow_id]:
        del workflow_states[workflow_id][key]
        return web.json_response({"success": True})
    
    return web.json_response({"success": False, "error": "Key not found"}, status=404)

@PromptServer.instance.routes.delete('/realtimenodes/state/{workflow_id}')
async def clear_workflow_state(request):
    """Clear all state for a specific workflow"""
    workflow_id = request.match_info['workflow_id']
    
    if workflow_id in workflow_states:
        workflow_states[workflow_id] = {}
        return web.json_response({"success": True})
    
    return web.json_response({"success": False, "error": "Workflow not found"}, status=404) 
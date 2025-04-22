/**
 * State API for interacting with the server-side state storage
 */

// Get the ComfyUI API instance
const api = window.comfyAPI?.api?.api;

// State API for workflow-specific state
export const stateApi = {
    /**
     * Get a value from the state by key
     * @param {string} workflowId - The workflow ID
     * @param {string} key - The key to retrieve
     * @returns {Promise<any>} - The value or null if not found
     */
    async getValue(workflowId, key) {
        try {
            const response = await api.fetchApi(`/realtimenodes/state/${workflowId}/${key}`);
            const data = await response.json();
            return data.value;
        } catch (error) {
            console.error("Error getting state value:", error);
            return null;
        }
    },

    /**
     * Set a value in the state by key
     * @param {string} workflowId - The workflow ID
     * @param {string} key - The key to set
     * @param {any} value - The value to store
     * @returns {Promise<boolean>} - Success status
     */
    async setValue(workflowId, key, value) {
        try {
            const response = await api.fetchApi(`/realtimenodes/state/${workflowId}/${key}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ value })
            });
            const data = await response.json();
            return data.success;
        } catch (error) {
            console.error("Error setting state value:", error);
            return false;
        }
    },

    /**
     * Delete a value from the state by key
     * @param {string} workflowId - The workflow ID
     * @param {string} key - The key to delete
     * @returns {Promise<boolean>} - Success status
     */
    async deleteValue(workflowId, key) {
        try {
            const response = await api.fetchApi(`/realtimenodes/state/${workflowId}/${key}`, {
                method: 'DELETE'
            });
            const data = await response.json();
            return data.success;
        } catch (error) {
            console.error("Error deleting state value:", error);
            return false;
        }
    },

    /**
     * Get all state values for a workflow
     * @param {string} workflowId - The workflow ID
     * @returns {Promise<Object>} - All state values
     */
    async getAllValues(workflowId) {
        try {
            const response = await api.fetchApi(`/realtimenodes/state/${workflowId}`);
            return await response.json();
        } catch (error) {
            console.error("Error getting all state values:", error);
            return {};
        }
    },

    /**
     * Clear all state values for a workflow
     * @param {string} workflowId - The workflow ID
     * @returns {Promise<boolean>} - Success status
     */
    async clearAllValues(workflowId) {
        try {
            const response = await api.fetchApi(`/realtimenodes/state/${workflowId}`, {
                method: 'DELETE'
            });
            const data = await response.json();
            return data.success;
        } catch (error) {
            console.error("Error clearing state values:", error);
            return false;
        }
    }
}; 